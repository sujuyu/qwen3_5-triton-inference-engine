"""带 KV cache 的 GQA decode（单 token query 对全部历史 K/V）。

这是 3.3 `gqa_attention_without_kvcache_casual` 的 decode 版，也是切换到增量
decode 所缺的最后一块（GDN 的两个 cache kernel 已就位，见 3.8b / 3.10b）。

与 prefill 版的三个关键差别
--------------------------
1. **不需要 causal mask**。cache 里的每一个位置都在新 token 之前（或就是它自己），
   全部都该被 attend。prefill 那种下三角 mask 在这里是多余的。
2. **query 只有一行**，`[H_q, D]`。所以没有 Q 方向的分块，只在 T 方向做 online softmax。
3. **K/V 来自 cache 而不是参数**，且新 token 的 K/V 要先追加进 cache。

cache 布局：`[H_kv, T_max, D]`，整块预分配
-----------------------------------------
不做 paging。paging 解决的四个问题（多序列碎片、continuous batching、前缀共享、
beam search 分叉）当前一个都不存在——batch 恒为 1、greedy、单序列。将来要加也很便宜：
kernel 里 T 方向本来就是分块遍历，插 paging 只是在循环里多一次块表查询。

选 `[H_kv, T_max, D]` 而不是 `[H_kv, D, T_max]` 或 `[T_max, H_kv, D]`：

    读：一次取 k_cache[h, t0:t0+BLOCK_T, :]，D 维连续，每行 512B 全部用满
    写：新 token 每个 head 写 256 个连续值，一次连续写

另外两种布局的写入都是跨步的。KV cache 在 8K 上下文是 96 MiB，**远超 A100 的 40MB
L2**，所以这里的合并访问是实打实的 DRAM 带宽，不像 conv state 那样能靠 cache 兜底
（见 3.8b 的实测：装得下 cache 时布局无差别，超出后差 2.82 倍）。

显存：6 层 × 2 KV head × 256 dim × 2 字节 × 2(K+V) = 12 KiB/token。
8K 上下文 96 MiB，32K 384 MiB，batch=1 下都可以接受。

**cache 里存的必须是 RoPE 之后的 K。** 参考实现 `Qwen3_5Attention.forward` 的顺序是
先 `apply_rotary_pos_emb` 再 `past_key_values.update`。存 RoPE 前的值、每步重新旋转
是错的——历史 token 的 position 不会变。

接口
----
    q:       [H_q, D]        BF16   新 token 的 query，已过 q_norm 和 RoPE
    k_new:   [H_kv, D]       BF16   新 token 的 key，已过 k_norm 和 RoPE
    v_new:   [H_kv, D]       BF16
    k_cache: [H_kv,T_max,D]  BF16   原地追加
    v_cache: [H_kv,T_max,D]  BF16   原地追加
    past_len: int                   追加位置 = 追加前的历史长度
    out:     [H_q, D]        BF16

运算（GROUP = H_q // H_kv = 4）
------------------------------
    追加：k_cache[h_kv, past_len, :] = k_new[h_kv, :]，v 同理
    S = past_len + 1
    对每个 query head h_q，取 h_kv = h_q // GROUP：
        score[t] = dot(q[h_q,:], k_cache[h_kv,t,:]) * D^-0.5      t = 0..S-1
        p        = softmax_fp32(score)
        out[h_q] = sum_t p[t] * v_cache[h_kv,t,:]

追加放在 python wrapper 里而不是 kernel 里
------------------------------------------
如果在 kernel 里追加，同一个 h_kv 会被 GROUP 个 program 同时写同一个位置（写的值相同，
数据上无害），但紧接着又要读回这个位置——跨 program 的写后读没有可见性保证，需要
fence。放在 wrapper 里用一次 slice 赋值最简单也最容易验证。

代价是每层多两次 PyTorch copy 的 launch。想融进 kernel 的话正确做法是：循环只读
`[0, past_len)`，新 token 的 k/v 直接从寄存器参与 online softmax 的最后一步，
完全不经过 cache 读回。这是后续的融合点，不是第一版该做的事。

已知的性能问题：grid 是 `(H_q,)` = 8 个 CTA
-------------------------------------------
A100 有 108 个 SM，8 个 CTA 只能用上 7%。长上下文时每个 CTA 要串行扫完整个 T。
正确的做法是 flash-decoding 式的 split-K：把 T 切成若干段并行算局部 (m, l, acc)，
再用第二个 kernel 归约。第一版先把正确性打通，split-K 作为后续项。
"""

import torch

import triton
import triton.language as tl


# 写完 _gqa_attention_decode_triton 的 body 之后改成 True，__main__ 的第 2 段测试
# 就会跑起来。判断放在 python wrapper 里而不是 kernel 里：`raise` 不是合法的
# Triton AST 节点。
KERNEL_IMPLEMENTED = False


autotune_configs = [
    triton.Config({"BLOCK_T": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_T": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_T": 128}, num_warps=8, num_stages=2),
]


def _seq_bucket(seq_len: int) -> int:
    """seq_len 每步都在涨，直接进 autotune key 会导致每步重新调优。分桶。"""
    if seq_len <= 64:
        return 64
    if seq_len <= 256:
        return 256
    if seq_len <= 1024:
        return 1024
    if seq_len <= 4096:
        return 4096
    return 4097


@triton.autotune(
    configs=autotune_configs,
    key=["H_Q", "D", "GROUP", "S_BUCKET"],
)
@triton.jit
def _gqa_attention_decode_triton(
    q_ptr,  # [H_q, D] BF16
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    k_cache_ptr,  # [H_kv, T_max, D] BF16，只读（追加已在 wrapper 里做完）
    stride_kc_h: tl.constexpr,
    stride_kc_t: tl.constexpr,
    stride_kc_d: tl.constexpr,
    v_cache_ptr,  # [H_kv, T_max, D] BF16，只读
    stride_vc_h: tl.constexpr,
    stride_vc_t: tl.constexpr,
    stride_vc_d: tl.constexpr,
    out_ptr,  # [H_q, D] BF16
    stride_o_h: tl.constexpr,
    stride_o_d: tl.constexpr,
    seq_len,  # = past_len + 1，运行时值
    scale,  # = D ** -0.5，FP32
    H_Q: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,  # H_q // H_kv
    S_BUCKET: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # TODO(kernel): 留给你写。一个 program 负责一个 query head。
    #
    #   pid_h = tl.program_id(0)              # query head，0..H_Q-1
    #   pid_kv = pid_h // GROUP               # 对应的 KV head，GQA 不复制 K/V
    #   offset_d = tl.arange(0, D)
    #   q = tl.load(q_ptr + pid_h*stride_q_h + offset_d*stride_q_d).to(tl.float32)
    #
    # online softmax，在 T 方向分块。三个累加量都用 FP32：
    #   m   标量，running max
    #   l   标量，running sum of exp
    #   acc [D]，running 加权和
    #
    #   m = -inf; l = 0.0; acc = tl.zeros([D], tl.float32)
    #   for t0 in tl.range(0, seq_len, BLOCK_T):
    #       offset_t = t0 + tl.arange(0, BLOCK_T)
    #       valid = offset_t < seq_len
    #       k = tl.load(k_cache_ptr + pid_kv*stride_kc_h
    #                   + offset_t[:,None]*stride_kc_t + offset_d[None,:]*stride_kc_d,
    #                   mask=valid[:,None], other=0.0).to(tl.float32)     # [BLOCK_T, D]
    #       s = tl.sum(q[None,:] * k, axis=1) * scale                      # [BLOCK_T]
    #       s = tl.where(valid, s, -float('inf'))
    #
    #       m_new = tl.maximum(m, tl.max(s, axis=0))
    #       alpha = tl.exp(m - m_new)          # 旧的累加量要按新 max 重新缩放
    #       p = tl.exp(s - m_new)              # [BLOCK_T]
    #       p = tl.where(valid, p, 0.0)
    #
    #       v = tl.load(v_cache_ptr + ... 同样的地址结构 ...)               # [BLOCK_T, D]
    #       acc = acc * alpha + tl.sum(p[:,None] * v, axis=0)
    #       l = l * alpha + tl.sum(p, axis=0)
    #       m = m_new
    #
    #   out = acc / l
    #
    # 注意：
    # - **不需要 causal mask**，cache 里所有位置都该被 attend；唯一的 mask 是
    #   offset_t < seq_len 的越界保护。
    # - 第一个 tile 时 m = -inf，`tl.exp(-inf - m_new)` 要能给出 0 而不是 NaN。
    #   如果 seq_len 内所有 s 都是 -inf 才会出问题，实际不会发生（至少有一个有效位置）。
    # - GQA 不要 materialize repeat_kv，直接用 pid_h // GROUP 索引。
    # - q 只有 [D] 一行，用 tl.sum(q[None,:] * k, axis=1) 而不是 tl.dot——
    #   BLOCK_T×D 对 1×D 的乘加不构成有效的 MMA 形状。
    tl.static_assert(D % 16 == 0)


@torch.library.triton_op(
    "wy_lib::gqa_attention_decode",
    mutates_args=("k_cache", "v_cache"),
)
def gqa_attention_decode(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    past_len: int,
) -> torch.Tensor:
    if not KERNEL_IMPLEMENTED:
        raise NotImplementedError(
            "_gqa_attention_decode_triton 的 body 还没写；"
            "写完后把本文件顶部的 KERNEL_IMPLEMENTED 改成 True"
        )
    assert q.ndim == 2 and k_new.ndim == 2 and v_new.ndim == 2
    assert k_cache.ndim == 3 and v_cache.ndim == 3
    assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16
    assert v_cache.dtype == torch.bfloat16
    assert k_new.dtype == torch.bfloat16 and v_new.dtype == torch.bfloat16

    num_q_heads, head_dim = q.shape
    num_kv_heads, max_len, cache_dim = k_cache.shape
    assert cache_dim == head_dim
    assert v_cache.shape == k_cache.shape
    assert k_new.shape == (num_kv_heads, head_dim)
    assert v_new.shape == (num_kv_heads, head_dim)
    assert num_q_heads % num_kv_heads == 0
    assert 0 <= past_len < max_len, f"past_len={past_len} 超出 cache 容量 {max_len}"
    assert triton.next_power_of_2(head_dim) == head_dim

    # 追加放在这里而不是 kernel 里：kernel 里同一个 KV head 会被 GROUP 个 program
    # 同时写、随即又读回，跨 program 的写后读没有可见性保证。详见模块 docstring。
    k_cache[:, past_len, :] = k_new
    v_cache[:, past_len, :] = v_new
    seq_len = past_len + 1

    out = torch.empty_like(q)

    torch.library.wrap_triton(_gqa_attention_decode_triton)[(num_q_heads,)](
        q_ptr=q,
        stride_q_h=q.stride(0),
        stride_q_d=q.stride(1),
        k_cache_ptr=k_cache,
        stride_kc_h=k_cache.stride(0),
        stride_kc_t=k_cache.stride(1),
        stride_kc_d=k_cache.stride(2),
        v_cache_ptr=v_cache,
        stride_vc_h=v_cache.stride(0),
        stride_vc_t=v_cache.stride(1),
        stride_vc_d=v_cache.stride(2),
        out_ptr=out,
        stride_o_h=out.stride(0),
        stride_o_d=out.stride(1),
        seq_len=seq_len,
        scale=head_dim**-0.5,
        H_Q=num_q_heads,
        D=head_dim,
        GROUP=num_q_heads // num_kv_heads,
        S_BUCKET=_seq_bucket(seq_len),
    )
    return out


@torch.library.register_fake("wy_lib::gqa_attention_decode")
def _gqa_attention_decode_fake(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    past_len: int,
) -> torch.Tensor:
    return torch.empty_like(q)


def call_gqa_attention_decode_triton(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    past_len: int,
) -> torch.Tensor:
    return gqa_attention_decode(q, k_new, v_new, k_cache, v_cache, past_len)


def allocate_kv_cache(
    num_kv_heads: int,
    max_len: int,
    head_dim: int,
    device="cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """整块预分配 [H_kv, T_max, D] 的 K/V cache。

    K 和 V 分成两个张量而不是合并成 [2,H,T,D]：少一层 stride，接口更直白。
    """
    shape = (num_kv_heads, max_len, head_dim)
    return (
        torch.zeros(shape, dtype=torch.bfloat16, device=device),
        torch.zeros(shape, dtype=torch.bfloat16, device=device),
    )


def kv_cache_from_prefill(
    k: torch.Tensor,
    v: torch.Tensor,
    max_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """prefill 的 K/V `[H_kv, T, D]`（RoPE 之后）-> 预分配好的 cache。

    prefill 那边拿到的通常是 `[1,H_kv,T,D]`，squeeze 掉 batch 维再传进来。
    """
    num_kv_heads, token_num, head_dim = k.shape
    assert v.shape == k.shape
    assert token_num <= max_len
    k_cache, v_cache = allocate_kv_cache(
        num_kv_heads, max_len, head_dim, device=k.device
    )
    k_cache[:, :token_num, :] = k
    v_cache[:, :token_num, :] = v
    return k_cache, v_cache


def _torch_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """q `[H_q,D]`，cache `[H_kv,T_max,D]` -> out `[H_q,D]`。FP32 softmax。"""
    num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[0]
    group = num_q_heads // num_kv_heads

    k = k_cache[:, :seq_len, :].float()  # [H_kv, S, D]
    v = v_cache[:, :seq_len, :].float()
    q32 = q.float()

    out = torch.empty((num_q_heads, head_dim), dtype=torch.float32, device=q.device)
    for h in range(num_q_heads):
        hk = h // group  # GQA：不复制 K/V，直接映射
        score = (k[hk] @ q32[h]) * (head_dim**-0.5)  # [S]，无需 causal mask
        prob = torch.softmax(score, dim=-1)
        out[h] = prob @ v[hk]
    return out.to(q.dtype)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from triton_kernels.gqa_attention_without_kvcache_casual import (
        gqa_attention_without_kvcache_casual,
    )

    torch.manual_seed(0)
    H_Q, H_KV, D = 8, 2, 256
    MAX_LEN = 512

    def run_prefill_then_decode(token_num, prefix, decode_fn):
        """prefill 前 prefix 个 token，再逐 token decode 剩下的，拼成完整输出。

        判据是它必须等于对整段直接做 causal prefill 的结果。
        """
        q = torch.randn((1, H_Q, token_num, D), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((1, H_KV, token_num, D), dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)

        expected = gqa_attention_without_kvcache_casual(q, k, v)[0]  # [H_Q,T,D]

        parts = []
        if prefix > 0:
            parts.append(
                gqa_attention_without_kvcache_casual(
                    q[:, :, :prefix], k[:, :, :prefix], v[:, :, :prefix]
                )[0]
            )
        k_cache, v_cache = kv_cache_from_prefill(
            k[0, :, :prefix], v[0, :, :prefix], MAX_LEN
        )
        for t in range(prefix, token_num):
            out = decode_fn(
                q[0, :, t, :].contiguous(),
                k[0, :, t, :].contiguous(),
                v[0, :, t, :].contiguous(),
                k_cache,
                v_cache,
                t,
            )
            parts.append(out.unsqueeze(1))  # [H_Q,1,D]
        actual = torch.cat(parts, dim=1)
        return actual, expected

    def _reference_decode(q, k_new, v_new, k_cache, v_cache, past_len):
        k_cache[:, past_len, :] = k_new
        v_cache[:, past_len, :] = v_new
        return _torch_reference(q, k_cache, v_cache, past_len + 1)

    # ---- 第 1 步：参考实现 vs prefill kernel（不依赖 Triton kernel，现在就能跑）----
    print("=== 参考实现 vs prefill kernel ===")
    for token_num, prefix in ((1, 0), (2, 1), (17, 5), (65, 33), (129, 64)):
        actual, expected = run_prefill_then_decode(token_num, prefix, _reference_decode)
        err = (actual.float() - expected.float()).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print(f"  T={token_num:>4} prefix={prefix:>3}  max_abs_error={err:.8f}")
    print("参考实现与 prefill kernel 一致。\n")

    # ---- 第 2 步：Triton kernel vs 参考实现 -------------------------------
    print("=== Triton kernel vs 参考实现 ===")
    try:
        for token_num, prefix in ((1, 0), (17, 5), (65, 33), (129, 64), (257, 128)):
            actual, expected = run_prefill_then_decode(
                token_num, prefix, call_gqa_attention_decode_triton
            )
            err = (actual.float() - expected.float()).abs().max().item()
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            assert torch.isfinite(actual).all()
            print(
                f"  T={token_num:>4} prefix={prefix:>3}  max_abs_error={err:.8f}  "
                f"best_config={_gqa_attention_decode_triton.best_config}"
            )

        # cache 里 past_len 之后的位置不该被读到：填成 NaN 也必须不影响结果
        q = torch.randn((H_Q, D), dtype=torch.bfloat16, device="cuda")
        k_new = torch.randn((H_KV, D), dtype=torch.bfloat16, device="cuda")
        v_new = torch.randn_like(k_new)
        k_cache, v_cache = allocate_kv_cache(H_KV, MAX_LEN, D)
        k_cache[:, :10].normal_()
        v_cache[:, :10].normal_()
        clean = call_gqa_attention_decode_triton(q, k_new, v_new, k_cache, v_cache, 10)
        k_cache[:, 11:] = float("nan")
        v_cache[:, 11:] = float("nan")
        dirty = call_gqa_attention_decode_triton(q, k_new, v_new, k_cache, v_cache, 10)
        torch.testing.assert_close(clean, dirty)
        assert torch.isfinite(dirty).all()
        print("  cache 尾部填 NaN 不影响结果（越界保护正确）")

        print("All GQA attention decode tests passed.")
    except NotImplementedError as exc:
        print(f"  跳过：{exc}")
        print("  填完 _gqa_attention_decode_triton 的 body 后重跑本文件。")
