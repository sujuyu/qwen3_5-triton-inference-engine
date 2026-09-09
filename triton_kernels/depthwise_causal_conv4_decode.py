"""Gated DeltaNet 的 depthwise causal Conv4 decode 版(单 token + conv state).

背景
----
conv 的定义是 `y[t,c] = silu(sum_{r=0..3} w[c,r] * x[t+r-3, c])`, 算 y[t] 要用到
x[t-3..t]. decode 时只有 x[t] 是新的, 前三个必须从 cache 取 -- 这就是 conv state.

注意它和 delta rule 的 recurrent state 是**两个独立的 cache**, GDN 每层都要:

    conv state       [4,6144]     BF16   本文件维护
    recurrent state  [16,128,128] FP32   gdn_recurrent_decode 维护

conv state 存的是 **conv 的输入**, 也就是 `in_proj_qkv` 的输出, 不是 conv 的输出,
也不是 SiLU 之后的值. 18 层合计只有 0.84 MiB.

约定与参考实现一致(transformers 的 `causal_conv1d_update`, state_len=conv_kernel_size=4):
更新后 `state[:,c]` 恰好是 `x[t-3..t]`, 所以点积不用再做下标偏移.

接口
----
    x:      [1,D] BF16      新 token 的 in_proj_qkv 输出, 模型里 D=6144
    state:  [4,D] BF16      原地更新, 必须 contiguous
    weight: [4,D] BF16      必须 contiguous, 用 conv_weight_for_decode() 从
                            checkpoint 的 [D,1,4] 转换
    out:    [1,D] BF16      含 SiLU

运算
----
    state[:,c] = concat(state[1:,c], x[c])      # 左移一格, 新值放末尾
    acc  = sum_{r=0..3} weight[r,c] * state[r,c]   # FP32 累加
    y[c] = acc * sigmoid(acc)                      # SiLU

为什么是 [4,D] 而不是 [D,4]
--------------------------
kernel 里一个线程负责一个 channel. 取第 k 个 tap 时:

    [D,4]: 线程 d 访问元素 d*4+k -- 相邻线程相隔 4 个元素, 非合并
    [4,D]: 线程 d 访问元素 k*D+d -- 相邻线程地址连续, 完全合并

GPU 取显存的最小单位是 32 字节(16 个 BF16).`[D,4]` 下这 16 个数只有 4 个是本次
要用的(利用率 25%), `[4,D]` 下全部用上. A100 实测(CUDA Graph, D=6144):

    布局                      BLOCK=256   BLOCK=512   BLOCK=1024
    state[D,4] weight[D,4]     1.88us      2.48us      5.14us
    state[4,D] weight[D,4]     1.68us      1.95us      3.27us
    state[4,D] weight[4,D]     1.60us      1.69us      1.89us   ← 采用

除了更快, [4,D] 对 BLOCK_D 几乎不敏感, autotune 才有实际选择空间; [D,4] 在大 block
下退化严重, 之前 autotune 总是选最小的 256 就是在补偿这一点.

**两个张量都必须是真正 contiguous 的 [4,D], 不能是 [D,4] 的转置 view** -- 转置 view
的内存布局仍是 [D,4], 合并访问的好处全部消失. wrapper 里有 assert 挡着.

decode 因此需要一份独立于 prefill 的权重副本(prefill 用 [D,1,4]). 代价是
6144*4*2 bytes * 18 层 = 0.86 MiB, 可以忽略.

纯 memory-bound, 一个 CTA 处理一段 channel 即可.
"""

import torch

import triton
import triton.language as tl


CONV_KERNEL_SIZE = 4


autotune_configs = [
    triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_D": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_D": 2048}, num_warps=8, num_stages=1),
]


@triton.autotune(
    configs=autotune_configs,
    key=["D"],
    # state 会被就地改写, 不加这个的话 autotune 反复试 config 会把 state 推进多次.
    # 与 gdn_recurrent_decode 同样的处理.
    restore_value=["state_ptr"],
)
@triton.jit
def _depthwise_causal_conv4_decode_triton(
    x_ptr,  # [B,D] BF16
    stride_x_b: tl.constexpr,
    stride_x_d: tl.constexpr,
    state_ptr,  # [B,4,D] BF16, 原地更新; 每条序列一份独立状态
    stride_state_b: tl.constexpr,
    stride_state_d: tl.constexpr,
    stride_state_k: tl.constexpr,
    weight_ptr,  # [4,D] BF16, **所有序列共用**, 不加 batch 偏移
    stride_w_d: tl.constexpr,
    stride_w_k: tl.constexpr,
    out_ptr,  # [B,D] BF16
    stride_o_b: tl.constexpr,
    stride_o_d: tl.constexpr,
    D,
    K: tl.constexpr,  # = CONV_KERNEL_SIZE = 4
    BLOCK_D: tl.constexpr,
):
    # grid = (B, cdiv(D, BLOCK_D)). batch 维只影响基址: conv state 是每条序列
    # 独立的(各自的 x[t-3..t-1]), 而 weight 是共用的.
    pid_b, pid = tl.program_id(0), tl.program_id(1)
    x_ptr = x_ptr + pid_b * stride_x_b
    state_ptr = state_ptr + pid_b * stride_state_b
    out_ptr = out_ptr + pid_b * stride_o_b

    offset_d = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offset_d < D

    x = tl.load(x_ptr + offset_d * stride_x_d, mask=mask, other=0.0).to(tl.float32)
    s1 = tl.load(state_ptr + offset_d * stride_state_d + 1 * stride_state_k, mask=mask, other=0.0).to(tl.float32) # x[t-3]
    s2 = tl.load(state_ptr + offset_d * stride_state_d + 2 * stride_state_k, mask=mask, other=0.0).to(tl.float32) # x[t-2]
    s3 = tl.load(state_ptr + offset_d * stride_state_d + 3 * stride_state_k, mask=mask, other=0.0).to(tl.float32) # x[t-1]

    w0 = tl.load(weight_ptr + offset_d * stride_w_d + 0 * stride_w_k, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(weight_ptr + offset_d * stride_w_d + 1 * stride_w_k, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(weight_ptr + offset_d * stride_w_d + 2 * stride_w_k, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(weight_ptr + offset_d * stride_w_d + 3 * stride_w_k, mask=mask, other=0.0).to(tl.float32)

    # 4 个 tap 的求和就在这一行里完成, acc 已经是 [BLOCK_D], 后面不要再 reduce
    acc = (s1 * w0 + s2 * w1 + s3 * w2 + x * w3)

    # 写回state: 读的下标比写的下标各大 1, 就是左移一格
    tl.store(
        state_ptr + offset_d * stride_state_d + 0 * stride_state_k,
        s1.to(tl.bfloat16), mask=mask
    )
    tl.store(
        state_ptr + offset_d * stride_state_d + 1 * stride_state_k,
        s2.to(tl.bfloat16), mask=mask
    )
    tl.store(
        state_ptr + offset_d * stride_state_d + 2 * stride_state_k,
        s3.to(tl.bfloat16), mask=mask
    )
    tl.store(
        state_ptr + offset_d * stride_state_d + 3 * stride_state_k,
        x.to(tl.bfloat16), mask=mask
    )

    acc = acc * tl.sigmoid(acc) # SiLU, 与 prefill 版写法一致; acc 已是 FP32

    tl.store(
        out_ptr + offset_d * stride_o_d, acc.to(tl.bfloat16), mask=mask
    )


@torch.library.triton_op(
    "wy_lib::depthwise_causal_conv4_decode",
    mutates_args=("state",),
)
def depthwise_causal_conv4_decode(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    assert x.ndim == 2, "decode 每条序列一个 token, x 是 [B,D]"
    assert state.ndim == 3 and weight.ndim == 2
    assert x.dtype == torch.bfloat16 and state.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    assert x.device == state.device == weight.device

    batch, hidden_dim = x.shape
    assert state.shape == (batch, CONV_KERNEL_SIZE, hidden_dim), (
        f"state 应为 [B,4,D]={(batch, CONV_KERNEL_SIZE, hidden_dim)},"
        f"实际 {tuple(state.shape)}"
    )
    assert weight.shape == (CONV_KERNEL_SIZE, hidden_dim), (
        f"weight 应为 [4,D]={(CONV_KERNEL_SIZE, hidden_dim)}, 实际 {tuple(weight.shape)};"
        "用 conv_weight_for_decode() 从 checkpoint 的 [D,1,4] 转换"
    )
    # 必须是真正 contiguous 的 [4,D]. 传 [D,4] 的转置 view 也能算出正确结果,
    # 但内存布局仍是 [D,4], 合并访问的好处全部消失(大 block 下慢 2.7 倍).
    assert state.is_contiguous() and weight.is_contiguous(), (
        "state/weight 必须是 contiguous 的 [B,4,D] / [4,D], 不能是 [D,4] 的转置 view -- "
        "那样访存不合并, 本 kernel 换布局的意义就没了"
    )

    out = torch.empty_like(x)

    def grid(meta):
        return (batch, triton.cdiv(hidden_dim, meta["BLOCK_D"]))

    torch.library.wrap_triton(_depthwise_causal_conv4_decode_triton)[grid](
        x_ptr=x,
        stride_x_b=x.stride(0),
        stride_x_d=x.stride(1),
        state_ptr=state,
        # [B,4,D] 布局: channel 维是连续的那一维, tap 维跨度为 D.
        # kernel body 完全不用改, 只是这几个 stride 的来源换了.
        stride_state_b=state.stride(0),
        stride_state_d=state.stride(2),
        stride_state_k=state.stride(1),
        weight_ptr=weight,
        stride_w_d=weight.stride(1),
        stride_w_k=weight.stride(0),
        out_ptr=out,
        stride_o_b=out.stride(0),
        stride_o_d=out.stride(1),
        D=hidden_dim,
        K=CONV_KERNEL_SIZE,
    )
    return out


@torch.library.register_fake("wy_lib::depthwise_causal_conv4_decode")
def _depthwise_causal_conv4_decode_fake(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


def call_depthwise_causal_conv4_decode_triton(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return depthwise_causal_conv4_decode(x, state, weight)


def conv_state_from_prefill(x: torch.Tensor) -> torch.Tensor:
    """prefill 的 conv 输入 [T,D] -> decode 的初始 conv state [4,D], contiguous.

    取最后 4 行; T < 4 时**上方**补零(对应原 [D,4] 布局的左侧), 与参考实现的
    `F.pad(states, (padding_length, 0), value=0)` 一致.
    """
    token_num, hidden_dim = x.shape
    state = torch.zeros(
        (CONV_KERNEL_SIZE, hidden_dim), dtype=x.dtype, device=x.device
    )
    take = min(token_num, CONV_KERNEL_SIZE)
    state[CONV_KERNEL_SIZE - take :] = x[-take:]  # [4,D] 布局下不用转置
    return state


# --------------------------------------------------- 打包 prefill 的 conv state

# 与 decode kernel 共用一套 BLOCK_D 候选. 它们都是纯搬运的 memory-bound kernel,
# 只是这个多一个 K=4 的行维. **所有候选都整除 D=6144**(= 2048 * 3), 所以下面
# 不需要 D 维的 mask -- wrapper 里有断言兜底. 与 gdn_recurrent_prefill_sequential
# 里 "All autotune candidates divide value_dim" 是同一个处理.
conv_state_pack_autotune_configs = [
    triton.Config({"BLOCK_D": bd}, num_warps=w, num_stages=1)
    for bd, w in ((256, 4), (512, 4), (1024, 4), (1024, 8), (2048, 8))
]


@triton.autotune(
    configs=conv_state_pack_autotune_configs,
    key=["D"],
    # **不需要 restore_value**, 尽管 state 是就地写的: 判据是幂等性.
    # 这里纯粹是"从只读的 qkv 里挑几行抄进 state", 写一次和写一百次结果相同.
    # 对比上面的 decode kernel, 那个是 state = f(state) 的读-改-写, 必须 restore.
)
@triton.jit
def _conv_state_from_packed_prefill_triton(
    x_ptr,  # [total_tokens, D] BF16, 打包后的 conv 输入(in_proj_qkv 的输出)
    stride_x_t: tl.constexpr,
    stride_x_d: tl.constexpr,
    state_ptr,  # [B, K, D] BF16, 原地写
    stride_state_b: tl.constexpr,
    stride_state_k: tl.constexpr,
    stride_state_d: tl.constexpr,
    cu_seqlens_ptr,  # [B+1] INT32, 打包偏移
    D: tl.constexpr,
    K: tl.constexpr,  # = CONV_KERNEL_SIZE = 4
    BLOCK_D: tl.constexpr,
):
    """一次抽出 B 条序列各自的末 K 行, 写进 decode 用的 conv state.

    grid = (B, cdiv(D, BLOCK_D)), 一个 program 管一条序列的一段 channel.
    用 PyTorch 写出来:

        b, pid_d = program_id(0), program_id(1)
        start = cu_seqlens[b]
        end   = cu_seqlens[b + 1]

        i = arange(0, K)              # 目标行号 0..3
        t = end - K + i               # 源行号: 本条序列的最后 K 行
        m = t >= start                # 长度不足 K 时, 上方那几行无效
        d = pid_d * BLOCK_D + arange(0, BLOCK_D)

        x = load(x[t, d], mask=m[:, None], other=0.0)   # [K, BLOCK_D]
        state[b, i, d] = x

    替代的是 `conv_state_from_prefill` 逐条调用: 那边每条要 3 个算子
    (torch.zeros + 切片赋值 + 外面的 copy_), 18 个 GDN 层 x B 条, B=32 时约
    1700 次发射 -- 而 prefill 是 CPU 分发受限的, 这部分全额计入首 token 延迟.

    三个要点:

    1. **store 不能带 `m` 的 mask.** 补零语义靠的是"load 时 other=0.0, store 时全写".
       如果 store 也按 m mask 掉, 上方那几行就保留**上一轮的残留值**而不是零.
       原来那个实现是靠 `torch.zeros` 保证的, 换成原地写之后这个保证就转移到了
       store 上. 这个错误只在 T < K 且 cache 被复用时才显形, reset 之后第一次跑
       不会暴露 -- 自测里专门有 lengths=[1] / [2,3] 这样的用例.

    2. **补零补在上方**, 对应参考实现的 `F.pad(states, (padding_length, 0), value=0)`,
       顺序不能反 -- 更新后 `state[:, c]` 必须恰好是 `x[t-3..t]`, decode kernel 的
       点积才不用做下标偏移.

    3. **`end - K + i` 会算出负数**(start=0, 序列长 1 时 t = -3..0). 有 mask 就不会
       被解引用, 但要知道地址表达式本身是负的.

    D 维不需要 mask, 理由见上面 autotune configs 的注释.
    """
    pid_b, pid_d = tl.program_id(0), tl.program_id(1)

    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    start = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
    seq_len = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64) - start

    offset_t = start + seq_len - K + tl.arange(0, K)
    valid_t = offset_t >= start 

    x = tl.load(x_ptr + offset_t[:, None] * stride_x_t + offset_d[None, :] * stride_x_d, 
            mask = valid_t[:, None], 
            other = 0.0
        )
    tl.store(
        state_ptr + pid_b * stride_state_b + tl.arange(0, K)[:, None] * stride_state_k + offset_d[None, :] * stride_state_d, x
    )


@torch.library.triton_op(
    "wy_lib::conv_state_from_packed_prefill", mutates_args=("state",)
)
def conv_state_from_packed_prefill(
    state: torch.Tensor,
    x: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    """`conv_state_from_prefill` 的批量版, 直接写进 cache, 不返回新张量.

    `cu_seqlens` 传 `[0, T]` 就退化成单序列, 所以非打包路径也能用 --
    B=1 时同样把 3 次发射降到 1 次.
    """
    total_tokens, hidden_dim = x.shape
    batch = cu_seqlens.shape[0] - 1
    assert state.shape == (batch, CONV_KERNEL_SIZE, hidden_dim), (
        f"state 形状 {tuple(state.shape)} 与 "
        f"(B={batch}, K={CONV_KERNEL_SIZE}, D={hidden_dim}) 不符"
    )
    assert state.dtype == x.dtype
    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.ndim == 1
    assert x.stride(1) == 1, "x 的 channel 维必须连续"
    # 所有 BLOCK_D 候选都整除 D, kernel 里才能省掉 D 维的 mask
    assert all(
        hidden_dim % c.kwargs["BLOCK_D"] == 0
        for c in conv_state_pack_autotune_configs
    ), f"D={hidden_dim} 不能被全部 BLOCK_D 候选整除, kernel 里需要加 D 维 mask"

    def grid(meta):
        return (batch, triton.cdiv(hidden_dim, meta["BLOCK_D"]))

    torch.library.wrap_triton(_conv_state_from_packed_prefill_triton)[grid](
        x_ptr=x,
        stride_x_t=x.stride(0),
        stride_x_d=x.stride(1),
        state_ptr=state,
        stride_state_b=state.stride(0),
        stride_state_k=state.stride(1),
        stride_state_d=state.stride(2),
        cu_seqlens_ptr=cu_seqlens,
        D=hidden_dim,
        K=CONV_KERNEL_SIZE,
    )


@torch.library.register_fake("wy_lib::conv_state_from_packed_prefill")
def _conv_state_from_packed_prefill_fake(state, x, cu_seqlens) -> None:
    return None


def conv_weight_for_decode(weight: torch.Tensor) -> torch.Tensor:
    """checkpoint 的 [D,1,4] 或 [D,4] -> decode 用的 contiguous [4,D].

    必须 `.contiguous()`: 转置 view 的内存布局仍是 [D,4], 访存不合并.
    在权重加载时调一次即可, 运行时零开销; 18 层多占 0.86 MiB.
    """
    if weight.ndim == 3:
        weight = weight.squeeze(1)
    assert weight.ndim == 2 and weight.shape[1] == CONV_KERNEL_SIZE
    return weight.transpose(0, 1).contiguous()


def _torch_reference(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (out, new_state). 参考实现不原地改 state, 便于对拍时保留旧值.

    state/weight 均为 [4,D].
    """
    # 左移一格, 新值放末尾; 之后 new_state[:,c] 就是 x[t-3..t]
    new_state = torch.cat([state[1:], x], dim=0)
    acc = (new_state.float() * weight.float()).sum(dim=0)
    out = torch.nn.functional.silu(acc).to(x.dtype).unsqueeze(0)
    return out, new_state


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from triton_kernels.depthwise_causal_conv4_prefill import (
        depthwise_causal_conv4_prefill,
    )

    torch.manual_seed(0)
    hidden_dim = 6144

    # ---- 第 1 步: 先验证参考实现本身对不对 -------------------------------
    # 判据是"prefill 前 n 个 token, 拿 conv state, 再逐 token decode 剩下的"
    # 必须与整段 prefill 完全一致. 这一步不依赖 Triton kernel, 现在就能跑.
    print("=== 参考实现 vs prefill kernel ===")
    for token_num, prefix in ((1, 0), (2, 1), (3, 1), (4, 2), (17, 5), (65, 33)):
        x = torch.randn(
            (token_num, hidden_dim), dtype=torch.bfloat16, device="cuda"
        )
        weight = torch.randn(
            (hidden_dim, 1, CONV_KERNEL_SIZE), dtype=torch.bfloat16, device="cuda"
        )

        expected = depthwise_causal_conv4_prefill(x, weight)

        prefix_out = (
            depthwise_causal_conv4_prefill(x[:prefix], weight)
            if prefix > 0
            else torch.empty((0, hidden_dim), dtype=x.dtype, device=x.device)
        )
        weight_dec = conv_weight_for_decode(weight)
        state = conv_state_from_prefill(x[:prefix])
        parts = [prefix_out]
        for t in range(prefix, token_num):
            step_out, state = _torch_reference(x[t : t + 1], state, weight_dec)
            parts.append(step_out)
        actual = torch.cat(parts, dim=0)

        err = (actual.float() - expected.float()).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print(f"  T={token_num:>3} prefix={prefix:>2}  max_abs_error={err:.8f}")
    print("参考实现与 prefill kernel 一致. \n")

    # ---- 第 2 步: Triton kernel vs 参考实现 -------------------------------
    print("=== Triton kernel vs 参考实现 ===")
    for token_num, prefix in ((4, 2), (17, 5), (65, 33)):
        x = torch.randn(
            (token_num, hidden_dim), dtype=torch.bfloat16, device="cuda"
        )
        weight = torch.randn(
            (hidden_dim, 1, CONV_KERNEL_SIZE),
            dtype=torch.bfloat16,
            device="cuda",
        )
        expected = depthwise_causal_conv4_prefill(x, weight)

        weight_dec = conv_weight_for_decode(weight)
        # kernel 现在带 batch 维, 自测用 B=1
        state = conv_state_from_prefill(x[:prefix]).unsqueeze(0)  # [1,4,D]
        parts = [depthwise_causal_conv4_prefill(x[:prefix], weight)]
        for t in range(prefix, token_num):
            # 注意这个 op 会就地改写 state
            parts.append(
                call_depthwise_causal_conv4_decode_triton(
                    x[t : t + 1], state, weight_dec
                )
            )
        actual = torch.cat(parts, dim=0)

        err = (actual.float() - expected.float()).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        assert torch.isfinite(actual).all()
        print(
            f"  T={token_num:>3} prefix={prefix:>2}  max_abs_error={err:.8f}  "
            f"best_config={_depthwise_causal_conv4_decode_triton.best_config}"
        )

    # 通用尺寸, 确认没有把 D=6144 写死; 顺带覆盖 D 不是 BLOCK_D 整数倍的 mask 路径
    x = torch.randn((1, 1000), dtype=torch.bfloat16, device="cuda")
    state = torch.randn(
        (CONV_KERNEL_SIZE, 1000), dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(
        (CONV_KERNEL_SIZE, 1000), dtype=torch.bfloat16, device="cuda"
    )
    expected_out, expected_state = _torch_reference(x, state.clone(), weight)
    state_b = state.unsqueeze(0).contiguous()  # kernel 带 batch 维, 这里 B=1
    actual_out = call_depthwise_causal_conv4_decode_triton(x, state_b, weight)
    torch.testing.assert_close(actual_out, expected_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_b[0], expected_state)
    print("  D=1000 通用尺寸通过, 且 state 已就地更新")

    # 转置 view 数值上也对, 但布局仍是 [D,4], 访存不合并, 必须被 assert 挡住
    bad = torch.randn((1000, CONV_KERNEL_SIZE), dtype=torch.bfloat16,
                      device="cuda").transpose(0, 1)
    assert bad.shape == (CONV_KERNEL_SIZE, 1000) and not bad.is_contiguous()
    try:
        call_depthwise_causal_conv4_decode_triton(x, state_b, bad)
        raise SystemExit("转置 view 没有被 assert 挡住")
    except AssertionError:
        print("  转置 view 被正确拒绝")

    # ---- 第 4 步: 打包版 conv state 抽取 ---------------------------------
    # 判据是**逐字节相等**: 这个 kernel 只搬运不计算, 有一点差异就是下标算错了.
    # 基准是逐条调用 conv_state_from_prefill, 也就是它替代掉的那个循环.
    print("\n=== conv_state_from_packed_prefill vs 逐条 conv_state_from_prefill ===")
    for lengths in ([19], [1], [2, 3], [4, 5], [7, 100, 1, 64], [128] * 8, [1, 1, 1, 1000]):
        batch, total = len(lengths), sum(lengths)
        x = torch.randn((total, 6144), dtype=torch.bfloat16, device="cuda")
        cu = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0)), dtype=torch.int32, device="cuda"
        )
        offsets = [0]
        for n in lengths:
            offsets.append(offsets[-1] + n)
        ref = torch.stack(
            [conv_state_from_prefill(x[offsets[b] : offsets[b + 1]]) for b in range(batch)]
        )
        # **初值故意非零**: T < 4 时上方要补零, 如果 kernel 漏写了那几行(比如给
        # store 也加了 mask), 用零初值是看不出来的 -- 会和"补零"的正确结果重合.
        got = torch.full((batch, CONV_KERNEL_SIZE, 6144), 7.0,
                         dtype=torch.bfloat16, device="cuda")
        conv_state_from_packed_prefill(got, x, cu)
        ok = torch.equal(got, ref)
        print(f"  lengths={str(lengths):<26} {'逐字节相同' if ok else '! 不一致'}")
        assert ok, (got - ref).abs().max().item()

    print("All depthwise causal conv4 decode tests passed.")
