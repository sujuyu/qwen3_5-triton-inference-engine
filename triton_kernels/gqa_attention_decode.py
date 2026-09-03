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
    pos_ptr,  # [1] INT64，当前已缓存的 token 数（= past_len），**放在显存里**
    scale,  # = D ** -0.5，FP32
    H_Q: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,  # H_q // H_kv
    S_BUCKET: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # seq_len 从显存读而不是当标量参数传，是为了让这个 kernel 能被 CUDA Graph 捕获。
    # CUDA Graph 在 capture 时会把标量 kernel 参数**烧进** launch 配置，replay 时用的
    # 永远是 capture 那一刻的值；而 decode 每步 seq_len 都在变，图就废了。实测：
    #     标量参数：capture 时 n=100，replay 传 300/777 得到的仍是 100
    #     显存 + tl.load：replay 得到 300/777，正确
    # 指针地址在整个生命周期不变，所以图始终有效；改的只是那 8 个字节的内容。
    # 代价是每次多一个 4/8 字节的 global load，L2 常驻可以忽略。
    seq_len = tl.load(pos_ptr).to(tl.int32) + 1

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
    pos: torch.Tensor,
    seq_bucket: int,
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
    assert pos.dtype == torch.int64 and pos.numel() == 1
    assert triton.next_power_of_2(head_dim) == head_dim

    # 追加放在这里而不是 kernel 里：kernel 里同一个 KV head 会被 GROUP 个 program
    # 同时写、随即又读回，跨 program 的写后读没有可见性保证。详见模块 docstring。
    #
    # 用 index_copy_ 而不是 `k_cache[:, past_len, :] = k_new`：后者的下标是 python int，
    # CUDA Graph capture 时会把偏移烧进 copy kernel，replay 永远写同一行。
    # index_copy_ 的下标来自显存，kernel 执行时才读，所以可以被捕获且 replay 正确
    # （已实测）。
    k_cache.index_copy_(1, pos, k_new.unsqueeze(1))
    v_cache.index_copy_(1, pos, v_new.unsqueeze(1))

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
        pos_ptr=pos,
        scale=head_dim**-0.5,
        H_Q=num_q_heads,
        D=head_dim,
        GROUP=num_q_heads // num_kv_heads,
        # seq_bucket 只影响 autotune 选哪个 config，不参与任何计算。它必须是
        # host 侧的 python int（autotune 的 key 只能是标量）。在 CUDA Graph 下
        # config 冻结在 capture 那一刻——这是可接受的，因为选错 config 只是慢一点，
        # 不会算错。correctness 全部由显存里的 pos 决定。
        S_BUCKET=_seq_bucket(seq_bucket),
    )
    return out


@torch.library.register_fake("wy_lib::gqa_attention_decode")
def _gqa_attention_decode_fake(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: torch.Tensor,
    seq_bucket: int,
) -> torch.Tensor:
    return torch.empty_like(q)


def call_gqa_attention_decode_triton(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    past_len,
) -> torch.Tensor:
    """past_len 可以是 int（测试/非 graph 路径）或 [1] INT64 显存张量（graph 路径）。"""
    pos = _as_position(past_len, q.device)
    # seq_bucket 只用于 autotune 选 config。传 int 时直接用；传张量时无法在不同步的
    # 前提下读出它，退而用 cache 容量作为上界——config 选得保守一点，但不影响正确性。
    hint = past_len + 1 if isinstance(past_len, int) else k_cache.shape[1]
    return gqa_attention_decode(q, k_new, v_new, k_cache, v_cache, pos, hint)


def allocate_position(device="cuda") -> torch.Tensor:
    """[1] INT64，保存"已缓存的 token 数"（= past_len），**必须放在显存里**。

    为什么不用 python int：CUDA Graph 在 capture 时会把标量参数和切片下标烧进
    launch 配置，replay 时永远用 capture 那一刻的值。实测对照——

        cache[:, past_len, :] = k_new   （past_len 是 python int）
            replay 三次都写第 0 行
        cache.index_copy_(1, pos, k_new)（pos 是显存张量）
            replay 三次分别写第 0/1/2 行

    dtype 用 INT64 是因为 index_copy_ 要求 index 为 long；kernel 里 load 之后
    立刻 .to(tl.int32)，避免后续 int64 运算。

    **图内自增**：把 `pos.add_(1)` 也捕获进图，每次 replay 位置自动前进，
    host 侧一行都不用碰。注意这让图变成有状态的——换 prompt 或 prefill 之后
    必须显式 `pos.zero_()` 复位；capture 过程本身（warmup + 正式捕获）也会把
    pos 推进好几格，捕获完同样要复位。
    """
    return torch.zeros(1, dtype=torch.int64, device=device)


def _as_position(past_len, device) -> torch.Tensor:
    """int 或张量 -> [1] INT64 显存张量。

    传 int 时会临时分配并做一次 host->device 拷贝，**只适合测试和非 graph 路径**；
    真实 runner 应该持有一个 allocate_position() 的张量全程复用。
    """
    if isinstance(past_len, torch.Tensor):
        assert past_len.dtype == torch.int64 and past_len.numel() == 1
        return past_len
    return torch.tensor([past_len], dtype=torch.int64, device=device)


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


# ===========================================================================
# split-K 版本：把 T 切成 MAX_SPLITS 段并行算，再用第二个 kernel 归约
# ===========================================================================
#
# 为什么是两个 kernel 而不是 tl.atomic_add
# ---------------------------------------
# online softmax 的归约不是简单求和：acc 和 l 都要先按**全局** max 重新缩放才能相加。
# atomic_add 只能无条件累加；就算用 atomic_max 求出全局 m，此前已累加进去的 acc
# 是按旧 max 缩放的，事后无法追溯修正——要修正就得再扫一遍，那本质就是第二个 kernel。
#
# 唯一能让 atomic 成立的办法是放弃减 max、直接算 exp(s) 再纯加。数学上可行，
# 但丢掉数值稳定性（fp32 里 exp(s) 在 s>88 溢出），而这正是 online softmax 要消除的
# 隐患。另外 atomic 的累加顺序不定会破坏结果可复现性，而本项目的验收方法
# （逐算子对拍 + 分叉步的 top-2 间距）依赖确定性。
#
# num_splits 为什么恒等于 MAX_SPLITS
# ----------------------------------
# 更多 split 没有收益——CTA 数过了 ~432 带宽就饱和了（见模块 docstring 的实测表），
# 所以 split 数天然有上界。固定成常数还有个更强的理由：**cudagraph 要求 grid 在
# capture 时固定**，而 cudagraph 是把 CPU 开销归零的唯一途径（HANDOFF 8.3）。
# num_splits 随 seq_len 变的话每步都要重新 capture，等于白做。
#
# 于是 scratch buffer 也是定长的，不随 token 数增长：
#
#     m_partial   [H_q, MAX_SPLITS]      FP32     4 KB
#     l_partial   [H_q, MAX_SPLITS]      FP32     4 KB
#     acc_partial [H_q, MAX_SPLITS, D]   FP32     1 MB
#
# 构造 runner 时分配一次，6 层和所有 decode step 全程复用。seq_len 增长时变的是
# 每个 split 内部循环的长度（chunk = cdiv(seq_len, MAX_SPLITS)），不是 buffer 大小。
# 这与 KV cache 本身预分配到 T_max 是同一个思路。
#
# 代价：seq_len 小时大部分 split 空转。**处理方式是完全不用分支**：
# `tl.range(start, end, BLOCK_T)` 在 start >= end 时零次迭代，循环体不执行；
# store 无条件做，写的就是初始值 m=-inf, l=0, acc=0，正是空 split 应有的值。
# combine 里 exp(-inf - m) = 0，这些 split 自动贡献 0，不需要特判。
#
# 实测过"用 if 包住 store 以跳过空 split 的写"，结论是不要这么做：
#
#     seq_len   活跃split   无条件 store   if 包住 store     差异
#           7          7        4.74us       4.73us       0.1%
#         128        128        7.22us       9.10us     -26.1%   ← if 版慢
#        1024        128        9.24us       9.21us       0.4%
#        8192        128       14.14us      14.15us      -0.0%
#
# seq_len=128 时所有 split 都活跃、分支恒为真，if 版仍慢 26%——分支本身干扰了
# codegen；而它想省的写流量在 seq_len=7 上根本没测出收益。
#
# chunk 和 seq_len 都是**运行时标量**，不要做成 tl.constexpr——它们随 seq_len 变，
# 做成 constexpr 就是每个取值触发一次重编译（T_BUCKET 跨桶那几十秒已经吃过一次）。
# 只有 MAX_SPLITS（用于 buffer 索引的上界）是 constexpr。

MAX_SPLITS = 128


split_autotune_configs = [
    triton.Config({"BLOCK_T": tile}, num_warps=warps, num_stages=stages)
    for tile in [16, 32, 64]
    for warps in [4, 8]
    for stages in [1, 2]
]


@triton.autotune(
    configs=split_autotune_configs,
    key=["H_Q", "D", "GROUP", "S_BUCKET"],
    # 不需要 restore_value：kernel 只写 scratch，且每次写的值相同（幂等）。
    # 与 conv4_decode / gdn_recurrent_decode 不同——那两个的 state 会向前推进，
    # autotune 反复试 config 会把状态推多次，所以必须 restore。
)
@triton.jit
def _gqa_attention_decode_split_triton(
    q_ptr,  # [H_q, D] BF16
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    k_cache_ptr,  # [H_kv, T_max, D] BF16，只读
    stride_kc_h: tl.constexpr,
    stride_kc_t: tl.constexpr,
    stride_kc_d: tl.constexpr,
    v_cache_ptr,  # [H_kv, T_max, D] BF16，只读
    stride_vc_h: tl.constexpr,
    stride_vc_t: tl.constexpr,
    stride_vc_d: tl.constexpr,
    m_partial_ptr,  # [H_q, MAX_SPLITS] FP32
    stride_mp_h: tl.constexpr,
    stride_mp_s: tl.constexpr,
    l_partial_ptr,  # [H_q, MAX_SPLITS] FP32
    stride_lp_h: tl.constexpr,
    stride_lp_s: tl.constexpr,
    acc_partial_ptr,  # [H_q, MAX_SPLITS, D] FP32
    stride_ap_h: tl.constexpr,
    stride_ap_s: tl.constexpr,
    stride_ap_d: tl.constexpr,
    pos_ptr,  # [1] INT64，理由同不切 T 的版本
    scale,
    H_Q: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    MAX_SPLITS_C: tl.constexpr,
    S_BUCKET: tl.constexpr,
    BLOCK_T: tl.constexpr,
):

    # seq_len 和 chunk 都在 kernel 内从 pos 算出来，host 侧一个都不用传——
    # 传标量的话 CUDA Graph replay 会用 capture 时的旧值。理由详见不切 T 的版本。
    seq_len = tl.load(pos_ptr).to(tl.int32) + 1
    chunk = (seq_len + MAX_SPLITS_C - 1) // MAX_SPLITS_C

    pid_h, pid_s = tl.program_id(0), tl.program_id(1)

    k_cache_ptr = k_cache_ptr + pid_h * stride_kc_h
    v_cache_ptr = v_cache_ptr + pid_h * stride_vc_h

    start = pid_s * chunk
    end = tl.minimum(start + chunk, seq_len)

    offset_h = pid_h * GROUP + tl.arange(0, GROUP)
    offset_d = tl.arange(0, D)

    m_i = tl.zeros([GROUP], tl.float32) - float('inf')
    acc = tl.zeros([GROUP, D], tl.float32)
    l_i = tl.zeros([GROUP], tl.float32)

    q = tl.load(q_ptr + offset_h[:, None] * stride_q_h + offset_d[None, :] * stride_q_d)

    for t0 in tl.range(start, end, BLOCK_T):
        offset_t = t0 + tl.arange(0, BLOCK_T)
        k = tl.load(
            k_cache_ptr + offset_t[None, :] * stride_kc_t + offset_d[:, None] * stride_kc_d,
            mask=offset_t[None, :] < end,
            other = 0.0
        )
        qk = tl.dot(q, k) * scale
        qk = tl.where(offset_t[None, :] < end, qk, -float('inf'))
        m_i_new = tl.maximum(m_i, tl.max(qk, axis = 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        v = tl.load(
            v_cache_ptr + offset_t[:, None] * stride_vc_t + offset_d[None, :] * stride_vc_d,
            mask=offset_t[:, None] < end,
            other = 0.0
        )

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    tl.store(
        m_partial_ptr + offset_h * stride_mp_h + pid_s * stride_mp_s, m_i
    )
    tl.store(
        l_partial_ptr + offset_h * stride_lp_h + pid_s * stride_lp_s, l_i
    )
    tl.store(
        acc_partial_ptr + offset_h[:, None] * stride_ap_h + pid_s * stride_ap_s + offset_d[None, :] * stride_ap_d, acc
    )

combine_autotune_configs = [
    triton.Config({"BLOCK_D": block_d}, num_warps=warps, num_stages=1)
    for block_d in [32, 64, 128]
    for warps in [2, 4, 8]
]


@triton.autotune(
    configs=combine_autotune_configs,
    key=["D", "MAX_SPLITS_C"],
)
@triton.jit
def _gqa_attention_decode_combine_triton(
    m_partial_ptr,  # [H_q, MAX_SPLITS] FP32
    stride_mp_h: tl.constexpr,
    stride_mp_s: tl.constexpr,
    l_partial_ptr,  # [H_q, MAX_SPLITS] FP32
    stride_lp_h: tl.constexpr,
    stride_lp_s: tl.constexpr,
    acc_partial_ptr,  # [H_q, MAX_SPLITS, D] FP32
    stride_ap_h: tl.constexpr,
    stride_ap_s: tl.constexpr,
    stride_ap_d: tl.constexpr,
    out_ptr,  # [H_q, D] BF16
    stride_o_h: tl.constexpr,
    stride_o_d: tl.constexpr,
    pos_ptr,  # [1] INT64，num_active 由它算出，理由同 split kernel
    D: tl.constexpr,
    MAX_SPLITS_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # H_q维度上肯定可以切 但是这么点CTA肯定吃不满sm
    # 在D维度上也进行切分 主要是增加m和l的读取量 但是m l本身很小 L2cache完全可以兜住 代价很小 

    # num_active 必须与 split kernel 用同一个 pos 算，否则两边对"哪些 split 有效"的
    # 理解会不一致，读到上一次残留的局部量。这里刻意重算而不是从 host 传，
    # 就是为了消除这个不一致的可能。
    seq_len = tl.load(pos_ptr).to(tl.int32) + 1
    chunk = (seq_len + MAX_SPLITS_C - 1) // MAX_SPLITS_C
    num_active = (seq_len + chunk - 1) // chunk

    pid_h, pid_d = tl.program_id(0), tl.program_id(1)
    offset_s = tl.arange(0, MAX_SPLITS_C)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    m_all = tl.load(
        m_partial_ptr + pid_h * stride_mp_h + offset_s * stride_mp_s,
        mask = offset_s < num_active, 
        other = -float('inf')
    ) # [MAX_SPLITS]
    l_all = tl.load(
        l_partial_ptr + pid_h * stride_lp_h + offset_s * stride_lp_s,
        mask = offset_s < num_active,
        other = 0.0
    ) # [MAX_SPLITS]
    acc_all = tl.load(
        acc_partial_ptr + pid_h * stride_ap_h + offset_s[:, None] * stride_ap_s + offset_d[None, :] * stride_ap_d,
        mask = offset_s[:, None] < num_active,
        other = 0.0
    )

    m = tl.max(m_all, axis = -1) # 单独head下这是一个标量单值
    alpha = tl.exp(m_all - m) # [MAX_SPLITS]
    l = tl.sum(l_all * alpha, axis = -1) # 单独head下这是一个标量单值
    
    acc_all = tl.sum(acc_all * alpha[:, None], axis = 0) # [D]
    acc_all /= l
    
    tl.store(
        out_ptr + pid_h * stride_o_h + offset_d * stride_o_d,
        acc_all.to(out_ptr.dtype.element_ty)
    )


@torch.library.triton_op(
    "wy_lib::gqa_attention_decode_split",
    mutates_args=("k_cache", "v_cache", "m_partial", "l_partial", "acc_partial"),
)
def gqa_attention_decode_split(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    m_partial: torch.Tensor,
    l_partial: torch.Tensor,
    acc_partial: torch.Tensor,
    pos: torch.Tensor,
    seq_bucket: int,
) -> None:
    num_q_heads, head_dim = q.shape
    num_kv_heads, max_len, _ = k_cache.shape
    assert m_partial.shape == (num_q_heads, MAX_SPLITS)
    assert l_partial.shape == (num_q_heads, MAX_SPLITS)
    assert acc_partial.shape == (num_q_heads, MAX_SPLITS, head_dim)
    assert m_partial.dtype == l_partial.dtype == acc_partial.dtype == torch.float32
    assert pos.dtype == torch.int64 and pos.numel() == 1

    # 追加与不切 T 的版本同理：放在 wrapper 里，且用 index_copy_ 而非 python int 下标，
    # 这样 CUDA Graph replay 时写入位置才会跟着 pos 走。
    k_cache.index_copy_(1, pos, k_new.unsqueeze(1))
    v_cache.index_copy_(1, pos, v_new.unsqueeze(1))

    torch.library.wrap_triton(_gqa_attention_decode_split_triton)[
        (num_kv_heads, MAX_SPLITS)
    ](
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
        m_partial_ptr=m_partial,
        stride_mp_h=m_partial.stride(0),
        stride_mp_s=m_partial.stride(1),
        l_partial_ptr=l_partial,
        stride_lp_h=l_partial.stride(0),
        stride_lp_s=l_partial.stride(1),
        acc_partial_ptr=acc_partial,
        stride_ap_h=acc_partial.stride(0),
        stride_ap_s=acc_partial.stride(1),
        stride_ap_d=acc_partial.stride(2),
        pos_ptr=pos,
        scale=head_dim**-0.5,
        H_Q=num_q_heads,
        D=head_dim,
        GROUP=num_q_heads // num_kv_heads,
        MAX_SPLITS_C=MAX_SPLITS,
        S_BUCKET=_seq_bucket(seq_bucket),  # 仅影响 config 选择，不参与计算
    )


@torch.library.register_fake("wy_lib::gqa_attention_decode_split")
def _gqa_attention_decode_split_fake(
    q, k_new, v_new, k_cache, v_cache, m_partial, l_partial, acc_partial, pos, seq_bucket
) -> None:
    return None


@torch.library.triton_op("wy_lib::gqa_attention_decode_combine", mutates_args=())
def gqa_attention_decode_combine(
    m_partial: torch.Tensor,
    l_partial: torch.Tensor,
    acc_partial: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    num_q_heads, _, head_dim = acc_partial.shape
    out = torch.empty(
        (num_q_heads, head_dim), dtype=torch.bfloat16, device=acc_partial.device
    )

    def grid(meta):
        return (num_q_heads, head_dim // meta["BLOCK_D"])

    torch.library.wrap_triton(_gqa_attention_decode_combine_triton)[grid](
        m_partial_ptr=m_partial,
        stride_mp_h=m_partial.stride(0),
        stride_mp_s=m_partial.stride(1),
        l_partial_ptr=l_partial,
        stride_lp_h=l_partial.stride(0),
        stride_lp_s=l_partial.stride(1),
        acc_partial_ptr=acc_partial,
        stride_ap_h=acc_partial.stride(0),
        stride_ap_s=acc_partial.stride(1),
        stride_ap_d=acc_partial.stride(2),
        out_ptr=out,
        stride_o_h=out.stride(0),
        stride_o_d=out.stride(1),
        pos_ptr=pos,
        D=head_dim,
        MAX_SPLITS_C=MAX_SPLITS,
    )
    return out


@torch.library.register_fake("wy_lib::gqa_attention_decode_combine")
def _gqa_attention_decode_combine_fake(m_partial, l_partial, acc_partial, pos):
    return torch.empty(
        (acc_partial.shape[0], acc_partial.shape[2]),
        dtype=torch.bfloat16,
        device=acc_partial.device,
    )


def allocate_split_scratch(
    num_q_heads: int, head_dim: int, device="cuda"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """定长 scratch，构造 runner 时分配一次，所有层和所有 decode step 复用。

    大小只与 MAX_SPLITS 有关，**不随 token 数增长**。
    """
    f32 = dict(dtype=torch.float32, device=device)
    return (
        torch.zeros((num_q_heads, MAX_SPLITS), **f32),
        torch.zeros((num_q_heads, MAX_SPLITS), **f32),
        torch.zeros((num_q_heads, MAX_SPLITS, head_dim), **f32),
    )


def call_gqa_attention_decode_split_triton(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    past_len,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """接口与不切 T 的 `call_gqa_attention_decode_triton` 一致，可直接互换。

    scratch 传 None 时临时分配——**只用于测试**，实际 runner 应该复用一份。
    """
    if scratch is None:
        scratch = allocate_split_scratch(q.shape[0], q.shape[1], device=q.device)
    m_partial, l_partial, acc_partial = scratch
    pos = _as_position(past_len, q.device)
    hint = past_len + 1 if isinstance(past_len, int) else k_cache.shape[1]
    gqa_attention_decode_split(
        q, k_new, v_new, k_cache, v_cache, m_partial, l_partial, acc_partial, pos, hint
    )
    # combine 自己从同一个 pos 算 num_active，不从 host 传——两边必须用同一个来源，
    # 否则对"哪些 split 有效"的理解可能不一致。
    return gqa_attention_decode_combine(m_partial, l_partial, acc_partial, pos)


def _torch_reference_split(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """按 MAX_SPLITS 分段算局部量再归约的 PyTorch 参考实现。

    结果必须与 `_torch_reference` 完全一致——这条本身就是对归约公式的自检。
    """
    num_q_heads, head_dim = q.shape
    group = num_q_heads // k_cache.shape[0]
    chunk = -(-seq_len // MAX_SPLITS)

    m_partial = torch.full((num_q_heads, MAX_SPLITS), -float("inf"), device=q.device)
    l_partial = torch.zeros((num_q_heads, MAX_SPLITS), device=q.device)
    acc_partial = torch.zeros((num_q_heads, MAX_SPLITS, head_dim), device=q.device)

    for h in range(num_q_heads):
        hk = h // group
        for s in range(MAX_SPLITS):
            start, end = s * chunk, min((s + 1) * chunk, seq_len)
            if start >= end:
                continue  # 空 split 保持 m=-inf, l=0, acc=0
            k = k_cache[hk, start:end].float()
            v = v_cache[hk, start:end].float()
            score = (k @ q[h].float()) * (head_dim**-0.5)
            m = score.max()
            p = torch.exp(score - m)
            m_partial[h, s] = m
            l_partial[h, s] = p.sum()
            acc_partial[h, s] = p @ v

    m = m_partial.max(dim=1, keepdim=True).values  # [H_q,1]
    rescale = torch.exp(m_partial - m)  # 空 split -> exp(-inf) = 0
    l = (l_partial * rescale).sum(dim=1, keepdim=True)
    acc = (acc_partial * rescale[:, :, None]).sum(dim=1)
    return (acc / l).to(q.dtype)


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

    # ---- 第 3 步：split-K 的归约公式自检（不依赖 Triton kernel）-----------
    # 分段算局部量再归约，必须与一次性算完全一致。这条能抓住 m/l 重缩放写错。
    print("\n=== split-K 归约公式 vs 一次性算（纯 PyTorch）===")
    for seq_len in (1, 7, 64, 65, 200, 511):
        q = torch.randn((H_Q, D), dtype=torch.bfloat16, device="cuda")
        kc, vc = allocate_kv_cache(H_KV, MAX_LEN, D)
        kc[:, :seq_len].normal_()
        vc[:, :seq_len].normal_()
        a = _torch_reference(q, kc, vc, seq_len)
        b = _torch_reference_split(q, kc, vc, seq_len)
        err = (a.float() - b.float()).abs().max().item()
        torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)
        active = min(MAX_SPLITS, -(-seq_len // (-(-seq_len // MAX_SPLITS))))
        print(f"  seq_len={seq_len:>4} 活跃split={active:>4}/{MAX_SPLITS}  max_abs_error={err:.8f}")
    print("归约公式正确。\n")

    # ---- 第 4 步：split-K Triton kernel vs 不切 T 的版本 ------------------
    # 不切 T 的版本已经对过 prefill kernel，这里拿它当基准。
    print("=== split-K kernel vs 不切 T 的版本 ===")
    try:
        for token_num, prefix in ((1, 0), (17, 5), (65, 33), (129, 64), (257, 128)):
            actual, expected = run_prefill_then_decode(
                token_num, prefix, call_gqa_attention_decode_split_triton
            )
            err = (actual.float() - expected.float()).abs().max().item()
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            assert torch.isfinite(actual).all()
            print(
                f"  T={token_num:>4} prefix={prefix:>3}  max_abs_error={err:.8f}  "
                f"split_cfg={_gqa_attention_decode_split_triton.best_config}"
            )

        # 空 split 必须写零值而不是留下上一次的残留：先用长序列填脏 scratch，
        # 再用短序列跑，结果必须正确。
        scratch = allocate_split_scratch(H_Q, D)
        q = torch.randn((H_Q, D), dtype=torch.bfloat16, device="cuda")
        kn = torch.randn((H_KV, D), dtype=torch.bfloat16, device="cuda")
        vn = torch.randn_like(kn)
        kc, vc = allocate_kv_cache(H_KV, MAX_LEN, D)
        kc[:, :400].normal_()
        vc[:, :400].normal_()
        call_gqa_attention_decode_split_triton(q, kn, vn, kc, vc, 400, scratch)
        kc2, vc2 = allocate_kv_cache(H_KV, MAX_LEN, D)
        kc2[:, :5].normal_()
        vc2[:, :5].normal_()
        short = call_gqa_attention_decode_split_triton(q, kn, vn, kc2, vc2, 5, scratch)
        ref = _torch_reference(q, kc2, vc2, 6)
        torch.testing.assert_close(short, ref, rtol=2e-2, atol=2e-2)
        print("  脏 scratch 复用后短序列仍正确（空 split 确实写了零值）")

        # ---- 第 5 步：CUDA Graph 捕获 + 连续 replay ----------------------
        # 这是整套显存标量改造的真正判据：图捕获一次，之后每步只 replay，
        # 位置靠图内的 pos.add_(1) 自动前进，host 侧不碰任何标量。
        print("\n=== CUDA Graph capture + replay ===")
        token_num, prefix = 40, 8
        q_all = torch.randn((1, H_Q, token_num, D), dtype=torch.bfloat16, device="cuda")
        k_all = torch.randn((1, H_KV, token_num, D), dtype=torch.bfloat16, device="cuda")
        v_all = torch.randn_like(k_all)
        expected = gqa_attention_without_kvcache_casual(q_all, k_all, v_all)[0]

        kc, vc = kv_cache_from_prefill(
            k_all[0, :, :prefix], v_all[0, :, :prefix], MAX_LEN
        )
        scratch = allocate_split_scratch(H_Q, D)
        pos = allocate_position()
        # 输入槽：图捕获的是"读这几个固定地址"，每步只更新内容
        q_slot = torch.empty((H_Q, D), dtype=torch.bfloat16, device="cuda")
        k_slot = torch.empty((H_KV, D), dtype=torch.bfloat16, device="cuda")
        v_slot = torch.empty_like(k_slot)
        out_slot = torch.empty((H_Q, D), dtype=torch.bfloat16, device="cuda")

        def one_step():
            o = call_gqa_attention_decode_split_triton(
                q_slot, k_slot, v_slot, kc, vc, pos, scratch
            )
            out_slot.copy_(o)
            pos.add_(1)  # 图内自增：replay 时位置自动前进

        # 捕获前先 warmup（Triton JIT / autotune 必须在捕获之外完成）
        q_slot.copy_(q_all[0, :, prefix, :])
        k_slot.copy_(k_all[0, :, prefix, :])
        v_slot.copy_(v_all[0, :, prefix, :])
        for _ in range(3):
            one_step()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                one_step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            one_step()

        # 捕获过程本身把 pos 推进了好几格、也污染了 cache，必须复位。
        # 这是"图有状态"的直接后果，真实 runner 要有显式的 reset。
        pos.zero_()
        kc.zero_()
        vc.zero_()
        kc[:, :prefix] = k_all[0, :, :prefix]
        vc[:, :prefix] = v_all[0, :, :prefix]
        pos.fill_(prefix)

        parts = [
            gqa_attention_without_kvcache_casual(
                q_all[:, :, :prefix], k_all[:, :, :prefix], v_all[:, :, :prefix]
            )[0]
        ]
        for t in range(prefix, token_num):
            q_slot.copy_(q_all[0, :, t, :])
            k_slot.copy_(k_all[0, :, t, :])
            v_slot.copy_(v_all[0, :, t, :])
            graph.replay()  # host 侧只有这一次调用，不传任何标量
            parts.append(out_slot.clone().unsqueeze(1))
        actual = torch.cat(parts, dim=1)

        err = (actual.float() - expected.float()).abs().max().item()
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        assert int(pos.item()) == token_num, f"pos 应前进到 {token_num}，实际 {pos.item()}"
        print(f"  捕获 1 次，replay {token_num - prefix} 次，pos 自动 {prefix} -> {pos.item()}")
        print(f"  与整段 causal prefill 一致，max_abs_error={err:.8f}")

        print("All GQA attention decode split-K tests passed.")
    except NotImplementedError as exc:
        print(f"  跳过：{exc}")
    except Exception as exc:
        if "静态" in str(exc) or "static_assert" in str(exc).lower():
            raise
        print(f"  split-K kernel 尚未实现或有误：{type(exc).__name__}: {str(exc)[:160]}")
        print("  填完两个 body 后重跑本文件。")
