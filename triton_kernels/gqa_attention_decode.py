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

按 KV head 分块，grid = `(H_kv,)` = 2
-------------------------------------
与 prefill 版同一个选择：一个 program 负责一个 KV head 及其 GROUP 个 Q head，
KV 只读一遍。按 Q head 分块的话 grid=(8,)，每个 KV head 会被读 4 遍。

**grid 必须是 `num_kv_heads` 而不是 `num_q_heads`。** kernel 里
`offset_h = pid * GROUP + tl.arange(0, GROUP)`，用 8 起 grid 的话 pid=2..7 会
越界读 cache、并越界写 `out`（写到第 31 行，而 out 只有 8 行），踩坏分配器里
相邻的张量——表现为"数据相关的错值"，最后变成 illegal memory access。

下一步必须做 split-K（不是可选优化）
-------------------------------------
代价是 CTA 只有 2 个。实测（A100，流式读 96 MiB）：

    CTA 数     2 warp     4 warp     8 warp    16 warp
        2       4GB/s      8GB/s     21GB/s     46GB/s
       32      62GB/s    122GB/s    308GB/s    628GB/s
      108     206GB/s    393GB/s    877GB/s   1186GB/s
      432     675GB/s    998GB/s   1171GB/s   1195GB/s   ← 饱和约 1170 GB/s

**2 个 CTA 即使开 16 warp 也只有饱和带宽的 4%**——天花板由"只占 2 个 SM"决定，
加 warp 补不回来。带宽大致正比于在飞的 warp 总数（约 1.4 GB/s per warp），
要打满需要 ~100 个 CTA（8 warp）或 ~50 个（16 warp）。

split-K：把 T 切成 num_splits 段，grid 变成 `(H_kv, num_splits)`，每段算局部
(m_i, l_i, acc)，再用第二个 kernel 归约：

    m   = max_i m_i
    l   = sum_i l_i * exp(m_i - m)
    acc = sum_i acc_i * exp(m_i - m)
    out = acc / l

num_splits 取 50~200（即 100~400 个 CTA）就能进饱和区。按 6 层合计估算：

    T       不切 T（2 CTA/8 warp）    split-K      倍数
    512            300 us              39 us       7.7x
    2048           1.2 ms              42 us        29x
    8192           4.8 ms              92 us        52x

对照：每个 decode step 必须读一遍全部 1.4 GiB 权重 ≈ 1.23 ms。不切 T 的话
T=2048 时 attention 就和整个模型的权重读取一样贵了。

本文件这个不切 T 的版本是 split-K 的对拍基准——跨 split 的 m/l 重缩放是 bug
高发区，错了往往只偏一点点，没有基准很难发现。

另：autotune 的 num_warps 目前只到 8。decode 是纯 memory-bound 且 CTA 数很少，
warp 数影响很大（上表 2 CTA 那行 4 warp 8GB/s vs 16 warp 46GB/s），建议加到 16。
"""

import torch

import triton
import triton.language as tl


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
    # 按照kv的head切分block 减少对kv的读取
    # 副作用是会造成decode阶段cta数量不足 这个矛盾在后续的迭代kernel里面使用split-k进行弥补
    pid_h = tl.program_id(0)

    # kv cache偏移消除head维度
    k_cache_ptr = k_cache_ptr + pid_h * stride_kc_h
    v_cache_ptr = v_cache_ptr + pid_h * stride_vc_h

    offset_h = pid_h * GROUP + tl.arange(0, GROUP)
    offset_d = tl.arange(0, D)
    q = tl.load(q_ptr + offset_h[:, None] * stride_q_h + offset_d[None, :] * stride_q_d) # q不需要mask [GROUP, D]

    acc = tl.zeros([GROUP, D], dtype = tl.float32) # V 的加权和 必须从 0 起
    m_i = tl.zeros([GROUP], dtype = tl.float32) - float('inf') # 局部最大值
    l_i = tl.zeros([GROUP], dtype = tl.float32) # 分母累加和

    for t0 in tl.range(0, seq_len, BLOCK_T):
        offset_t = t0 + tl.arange(0, BLOCK_T)
        k = tl.load(
            k_cache_ptr + offset_t[None, :] * stride_kc_t + offset_d[:, None] * stride_kc_d, 
            mask = offset_t[None, :] < seq_len,
            other = 0.0
        ) # [D, BLOCK_T] 载入过程中完成转置
        qk = tl.dot(q, k) * scale # [GROUP, BLOCK_T]
        qk = tl.where(offset_t[None, :] < seq_len, qk, -float('inf')) # mask越界位置

        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None]) # 这边不需要再次对p执行tl.where 因为对-inf取exp本身等于0 m_i_new不存在整行被mask的情况

        v = tl.load(
            v_cache_ptr + offset_t[:, None] * stride_vc_t + offset_d[None, :] * stride_vc_d,
            mask = offset_t[:, None] < seq_len,
            other = 0.0
        )

        # p 是 FP32、v 是 BF16，tl.dot 要求两个操作数同 dtype，这里显式降到 BF16
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v) # [GROUP, D]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        m_i = m_i_new 

    out = acc / l_i[:, None]

    # 写回
    # element_ty 只有指针类型才有，要取 out_ptr 的而不是 out 这个值的
    tl.store(out_ptr + offset_h[:, None] * stride_o_h + offset_d[None, :] * stride_o_d, out.to(out_ptr.dtype.element_ty))



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

    # 按 KV head 分块：一个 program 负责一个 KV head 及其 GROUP 个 Q head。
    # 这里必须是 num_kv_heads——kernel 里 offset_h = pid * GROUP + arange(GROUP)，
    # 用 num_q_heads 起 grid 会让 pid>=num_kv_heads 的 program 越界读 cache、越界写 out。
    torch.library.wrap_triton(_gqa_attention_decode_triton)[(num_kv_heads,)](
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
