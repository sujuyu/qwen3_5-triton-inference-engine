"""Gated DeltaNet 的 depthwise causal Conv4 decode 版（单 token + conv state）。

背景
----
conv 的定义是 `y[t,c] = silu(sum_{r=0..3} w[c,r] * x[t+r-3, c])`，算 y[t] 要用到
x[t-3..t]。decode 时只有 x[t] 是新的，前三个必须从 cache 取——这就是 conv state。

注意它和 delta rule 的 recurrent state 是**两个独立的 cache**，GDN 每层都要：

    conv state       [6144,4]     BF16   本文件维护
    recurrent state  [16,128,128] FP32   gdn_recurrent_decode 维护

conv state 存的是 **conv 的输入**，也就是 `in_proj_qkv` 的输出，不是 conv 的输出，
也不是 SiLU 之后的值。18 层合计只有 0.84 MiB。

约定与参考实现一致（transformers 的 `causal_conv1d_update`，state_len=conv_kernel_size=4）：
更新后 `state[c,:]` 恰好是 `x[t-3..t]`，所以点积不用再做下标偏移。

接口
----
    x:      [1,D] BF16      新 token 的 in_proj_qkv 输出，模型里 D=6144
    state:  [D,4] BF16      原地更新
    weight: [D,1,4] 或 [D,4] BF16    与 prefill 版共用同一份权重
    out:    [1,D] BF16      含 SiLU

运算
----
    state[c,:] = concat(state[c,1:], x[c])      # 左移一格，新值放末尾
    acc  = sum_{r=0..3} weight[c,r] * state[c,r]   # FP32 累加
    y[c] = acc * sigmoid(acc)                      # SiLU

纯 memory-bound，一个 CTA 处理一段 channel 即可。
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
    # state 会被就地改写，不加这个的话 autotune 反复试 config 会把 state 推进多次。
    # 与 gdn_recurrent_decode 同样的处理。
    restore_value=["state_ptr"],
)
@triton.jit
def _depthwise_causal_conv4_decode_triton(
    x_ptr,  # [1,D] BF16
    stride_x_d: tl.constexpr,
    state_ptr,  # [D,4] BF16，原地更新
    stride_state_d: tl.constexpr,
    stride_state_k: tl.constexpr,
    weight_ptr,  # [D,4] BF16
    stride_w_d: tl.constexpr,
    stride_w_k: tl.constexpr,
    out_ptr,  # [1,D] BF16
    stride_o_d: tl.constexpr,
    D,
    K: tl.constexpr,  # = CONV_KERNEL_SIZE = 4
    BLOCK_D: tl.constexpr,
):

    pid = tl.program_id(0)
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

    # 4 个 tap 的求和就在这一行里完成，acc 已经是 [BLOCK_D]，后面不要再 reduce
    acc = (s1 * w0 + s2 * w1 + s3 * w2 + x * w3)

    # 写回state：读的下标比写的下标各大 1，就是左移一格
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

    acc = acc * tl.sigmoid(acc) # SiLU，与 prefill 版写法一致；acc 已是 FP32

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
    assert x.ndim == 2 and x.shape[0] == 1, "decode 每次只处理一个 token"
    assert state.ndim == 2
    assert weight.ndim in (2, 3)
    assert x.dtype == torch.bfloat16 and state.dtype == torch.bfloat16
    assert weight.dtype == torch.bfloat16
    assert x.device == state.device == weight.device

    hidden_dim = x.shape[1]
    assert state.shape == (hidden_dim, CONV_KERNEL_SIZE)
    if weight.ndim == 3:
        assert weight.shape == (hidden_dim, 1, CONV_KERNEL_SIZE)
        weight = weight.squeeze(1)
    else:
        assert weight.shape == (hidden_dim, CONV_KERNEL_SIZE)

    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(hidden_dim, meta["BLOCK_D"]),)

    torch.library.wrap_triton(_depthwise_causal_conv4_decode_triton)[grid](
        x_ptr=x,
        stride_x_d=x.stride(1),
        state_ptr=state,
        stride_state_d=state.stride(0),
        stride_state_k=state.stride(1),
        weight_ptr=weight,
        stride_w_d=weight.stride(0),
        stride_w_k=weight.stride(1),
        out_ptr=out,
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
    """prefill 的 conv 输入 [T,D] -> decode 的初始 conv state [D,4]。

    取最后 4 行；T < 4 时**左侧**补零，与参考实现的
    `F.pad(states, (padding_length, 0), value=0)` 一致。
    """
    token_num, hidden_dim = x.shape
    state = torch.zeros(
        (hidden_dim, CONV_KERNEL_SIZE), dtype=x.dtype, device=x.device
    )
    take = min(token_num, CONV_KERNEL_SIZE)
    state[:, CONV_KERNEL_SIZE - take :] = x[-take:].T
    return state


def _torch_reference(
    x: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (out, new_state)。参考实现不原地改 state，便于对拍时保留旧值。"""
    if weight.ndim == 3:
        weight = weight.squeeze(1)
    # 左移一格，新值放末尾；之后 new_state[c,:] 就是 x[t-3..t]
    new_state = torch.cat([state[:, 1:], x.transpose(0, 1)], dim=-1)
    acc = (new_state.float() * weight.float()).sum(dim=-1)
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

    # ---- 第 1 步：先验证参考实现本身对不对 -------------------------------
    # 判据是「prefill 前 n 个 token，拿 conv state，再逐 token decode 剩下的」
    # 必须与整段 prefill 完全一致。这一步不依赖 Triton kernel，现在就能跑。
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
        state = conv_state_from_prefill(x[:prefix])
        parts = [prefix_out]
        for t in range(prefix, token_num):
            step_out, state = _torch_reference(x[t : t + 1], state, weight)
            parts.append(step_out)
        actual = torch.cat(parts, dim=0)

        err = (actual.float() - expected.float()).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print(f"  T={token_num:>3} prefix={prefix:>2}  max_abs_error={err:.8f}")
    print("参考实现与 prefill kernel 一致。\n")

    # ---- 第 2 步：Triton kernel vs 参考实现 -------------------------------
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

        state = conv_state_from_prefill(x[:prefix])
        parts = [depthwise_causal_conv4_prefill(x[:prefix], weight)]
        for t in range(prefix, token_num):
            # 注意这个 op 会就地改写 state
            parts.append(
                call_depthwise_causal_conv4_decode_triton(
                    x[t : t + 1], state, weight
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

    # 通用尺寸，确认没有把 D=6144 写死
    x = torch.randn((1, 1000), dtype=torch.bfloat16, device="cuda")
    state = torch.randn(
        (1000, CONV_KERNEL_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    weight = torch.randn(
        (1000, CONV_KERNEL_SIZE), dtype=torch.bfloat16, device="cuda"
    )
    expected_out, expected_state = _torch_reference(x, state.clone(), weight)
    actual_out = call_depthwise_causal_conv4_decode_triton(x, state, weight)
    torch.testing.assert_close(actual_out, expected_out, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state, expected_state)
    print("  D=1000 通用尺寸通过，且 state 已就地更新")

    print("All depthwise causal conv4 decode tests passed.")
