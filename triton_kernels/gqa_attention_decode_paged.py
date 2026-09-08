"""Paged KV cache 版的 GQA decode attention（split-K）。

和 `gqa_attention_decode.py` 的关系
==================================
在线 softmax、split-K 的切分与归约、`pos` 进显存这些**全部不变**，
唯一的差别是 **K/V 的地址怎么算**：整块 cache 是「按 token 线性扫」，
paged 是「先查页表拿物理页号，再在页内偏移」。

所以：
- `combine` 只改了一行 `num_active` 的算法（见下），逻辑照抄；
- `split` 的在线 softmax 部分逐行照抄，只有两处 `tl.load` 的地址变了。

为什么要 paged
--------------
**batch=1 时 paged 没有任何好处，只有额外的一层间接寻址。** 它是为凑批做的铺垫：

- 整块方案下每条请求都要按 `max_len` 预分配，而实际长度差异很大，浪费严重；
- 多条请求之间没法共享（比如共同的 system prompt 前缀）；
- 变长请求在一块连续 buffer 里没法紧凑排布。

paged 把这三件事都解决了，代价是 kernel 里多一次页表查找。

物理布局
--------
```text
k_cache / v_cache   [num_pages, H_kv, PAGE, D]   BF16
block_table         [max_pages]                  INT32   逻辑页 -> 物理页
```

固定 (page, head) 时 `[PAGE, D]` 是完全连续的 16 x 256 x 2B = 8 KiB，
和整块方案里 `k_cache[h, t0:t0+BLOCK_T, :]` 的访存形态一致，
所以 kernel 内层的载入模式不用改，只是基址不同。

一个页放下所有 head：`H_kv * PAGE * D * 2B = 2*16*256*2 = 16 KiB`（K 和 V 各一份）。

PAGE = 16 的取舍
----------------
选 16 是为了让 **BLOCK_T 恰好等于 PAGE**，于是每次循环迭代只覆盖一个页，
**页表只需要查一次、而且查出来是个标量**：

```python
for lp in range(page_start, page_end):
    phys = tl.load(block_table_ptr + lp)      # 标量，一次
    k = tl.load(k_cache_ptr + phys * stride_p + ...)
```

如果 BLOCK_T > PAGE，一个 tile 会跨多个页，就得 gather 一个页号向量再逐元素算地址，
复杂度和寄存器压力都上一个台阶。0.8B 的上下文不会很长，16 够用。

代价是 BLOCK_T 不再能被 autotune 选（整块版本会在 16/32/64/128 里挑，长序列时选 32）。
实测整块版本 seq=4095 时 BLOCK_T=16 vs 32 的差距在 3% 以内，可以接受。

split 的切分改成按页对齐
------------------------
整块版本按 token 均分：`chunk = cdiv(seq_len, MAX_SPLITS)`，切点落在页中间的话
一个 tile 会跨页。paged 改成按页均分：

```python
num_pages_used  = cdiv(seq_len, PAGE)
pages_per_split = cdiv(num_pages_used, MAX_SPLITS)
page_start = pid_s * pages_per_split
page_end   = min(page_start + pages_per_split, num_pages_used)
```

**`combine` 里的 `num_active` 必须用同一套公式**，否则两边对「哪些 split 有效」
的理解不一致，会读到上一次残留的局部量。这是整块版本里已经踩过并写进注释的坑，
这里同样刻意在 kernel 内重算而不是从 host 传。
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl


# 页大小。改这个值需要同步检查：
#   1. split kernel 里 BLOCK_T == PAGE_SIZE 的假设；
#   2. tl.arange 要求 2 的幂。
PAGE_SIZE = 16

# 与整块版本保持一致：split 数固定，grid 才不会随 seq_len 变，CUDA Graph 才能复用
# 同一张图。理由见 gqa_attention_decode.py 里「num_splits 为什么恒等于 MAX_SPLITS」。
MAX_SPLITS = 128


# --------------------------------------------------------------------------- 分配

def allocate_paged_kv_cache(
    num_pages: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """`[num_pages, H_kv, PAGE, D]` 的 K/V 池。"""
    shape = (num_pages, num_kv_heads, PAGE_SIZE, head_dim)
    k = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    v = torch.zeros(shape, dtype=torch.bfloat16, device=device)
    return k, v


# `allocate_block_table` 在 `engine/paging.py`。
#
# 这个文件里的其他 allocator（`allocate_paged_kv_cache`、`allocate_split_scratch`）
# 留在这里，是因为它们和 kernel **强耦合**：前者的四维布局必须与下面四个 stride
# 参数一一对应，后者的形状绑死在 MAX_SPLITS 上，改 kernel 就得改它们，放一起才不会
# 漏改。而 block_table 只是一个 INT32 向量，与 kernel 的唯一约束是 dtype，
# 属于分页管理而不是 kernel 的事。
#
# 反过来 `physical_rows` 留在这里没得选：`paged_kv_append` / `paged_kv_fill` 要用它，
# 而 `triton_kernels/` 不依赖 `engine/`（每个 kernel 文件都能单独 `python xxx.py`
# 自测），把它挪进 paging.py 会造成循环 import。


def physical_rows(
    block_table: torch.Tensor,
    token_index: torch.Tensor,
    num_kv_heads: int,
) -> torch.Tensor:
    """逻辑 token 下标 -> `k_cache.view(-1, D)` 里的物理行号，形状 `[T, H]`。

    ```text
    row(t, h) = (block_table[t // PAGE] * H + h) * PAGE + (t % PAGE)
    ```

    全部是显存上的张量运算，没有 `.item()`，所以可以进 CUDA Graph。
    """
    logical_page = token_index // PAGE_SIZE
    slot = token_index % PAGE_SIZE
    phys = block_table.index_select(0, logical_page.to(torch.int32)).to(torch.int64)
    heads = torch.arange(num_kv_heads, device=block_table.device, dtype=torch.int64)
    return (phys[:, None] * num_kv_heads + heads[None, :]) * PAGE_SIZE + slot[:, None]


paged_append_autotune_configs = [
    triton.Config({}, num_warps=w, num_stages=1) for w in (1, 2, 4)
]


@triton.autotune(
    configs=paged_append_autotune_configs,
    key=["D"],
    # **不需要 restore_value**，尽管 k_cache/v_cache 是就地写的。
    #
    # 判据是**幂等性**，不是「是否原地写」：这里的目标地址完全由只读输入
    # （pos、block_table）决定，写进去的值也来自只读输入，所以写一次和写一百次的
    # 最终状态相同。autotune 反复试 config 不会留下错误状态。
    #
    # 对比 `depthwise_causal_conv4_decode` 和 `gdn_recurrent_decode`：它们是
    # `state = f(state)` 的读-改-写，每跑一次状态就往前推一格，autotune 试 N 个
    # config 就推 N 格，所以必须 `restore_value=["state_ptr"]`。
    #
    # **如果哪天给这个 kernel 加上「顺便把 pos 自增」之类的行为，它立刻就不幂等，
    # 必须回来加 restore_value。** 这类错误是静默的：autotune 只在第一次遇到某个
    # key 时跑，表现成「某次运行的头几个 token 不对、后面都对」，极难定位。
    #
    # 另外注意 `restore_value` 和 wrapper 上的 `mutates_args` 是两回事：
    #   restore_value   triton.autotune 语义，面向 benchmark，只有非幂等才需要
    #   mutates_args    torch.library 语义，面向编译器，幂等与否都必须写
    # 后者少了的话，torch.compile 会把这个返回 None 的 op 当成纯函数——实测会被
    # 直接 DCE 掉（kernel 根本不执行），或者被 CSE 合并、被重排到读操作之后。
)
@triton.jit
def _paged_kv_append_triton(
    k_cache_ptr,  # [num_pages, H, PAGE, D] BF16，原地写
    stride_kc_p: tl.constexpr,
    stride_kc_h: tl.constexpr,
    stride_kc_s: tl.constexpr,
    stride_kc_d: tl.constexpr,
    v_cache_ptr,  # [num_pages, H, PAGE, D] BF16，原地写
    stride_vc_p: tl.constexpr,
    stride_vc_h: tl.constexpr,
    stride_vc_s: tl.constexpr,
    stride_vc_d: tl.constexpr,
    k_new_ptr,  # [H, D] BF16
    stride_kn_h: tl.constexpr,
    stride_kn_d: tl.constexpr,
    v_new_ptr,  # [H, D] BF16
    stride_vn_h: tl.constexpr,
    stride_vn_d: tl.constexpr,
    block_table_ptr,  # [max_pages] INT32
    pos_ptr,  # [1] INT64
    PAGE: tl.constexpr,
    D: tl.constexpr,
):
    """把新 token 的 K/V 写进它所在的页。grid = (H,)，一个 program 管一个 KV head。

    TODO(用户填)。用 PyTorch 写出来就是这几行：

        h    = program_id(0)
        p    = pos[0]                       # INT64 标量
        lp   = p // PAGE                    # 逻辑页号
        slot = p % PAGE                     # 页内槽位
        phys = block_table[lp]              # 物理页号，INT32 标量

        d = arange(0, D)
        k_cache[phys, h, slot, d] = k_new[h, d]
        v_cache[phys, h, slot, d] = v_new[h, d]

    几个要点：

    1. **K 和 V 放在同一个 kernel 里。** 两者布局相同，目标地址只需要算一次用两次；
       拆成两个 kernel 就要多付一次约 4us 的启动成本，而这个 kernel 总共才搬 2 KiB。

    2. **`phys` 要提到 INT64 再参与地址运算。** `block_table` 是 INT32，
       `phys * stride_kc_p` 在页数大时会溢出。当前规模离上限还远，但这是廉价的保险。

    3. **D 维不需要 mask。** head_dim=256 是 2 的幂，`tl.arange(0, D)` 正好覆盖一行，
       wrapper 里有 `next_power_of_2(D) == D` 的断言兜底。

    4. **不要在这里做边界检查。** `pos` 是否越过已分配的页，是 host 侧调度的责任
       （`SequencePages.reserve`）；kernel 里加分支只会拖慢它，而且报错时机也太晚。

    扩到 batch 时这段循环体一个字不用改，只是上面多一层偏移：

        pid_b = program_id(0);  h = program_id(1)          # grid = (B, H)
        p     = tl.load(pos_ptr + pid_b)
        phys  = tl.load(block_table_ptr + pid_b * stride_bt_b + lp)
    """
    pid_h = tl.program_id(0)
    p = tl.load(pos_ptr).to(tl.int64)
    lp = p // PAGE
    slot = p % PAGE
    phys = tl.load(block_table_ptr + lp).to(tl.int64) # 物理页号
    # 因为head维度的存在 每个page cache的维度视作[H PAGE D]
    k_cache_ptr = k_cache_ptr + phys * stride_kc_p + pid_h * stride_kc_h + slot * stride_kc_s
    v_cache_ptr = v_cache_ptr + phys * stride_vc_p + pid_h * stride_vc_h + slot * stride_vc_s

    offset_d = tl.arange(0, D)
    k_new = tl.load(k_new_ptr + pid_h * stride_kn_h + offset_d * stride_kn_d) # [D]
    v_new = tl.load(v_new_ptr + pid_h * stride_vn_h + offset_d * stride_vn_d) # [D]

    tl.store(
        k_cache_ptr + offset_d * stride_kc_d, k_new
    )
    tl.store(
        v_cache_ptr + offset_d * stride_vc_d, v_new
    )



@torch.library.triton_op(
    "wy_lib::paged_kv_append", mutates_args=("k_cache", "v_cache")
)
def paged_kv_append(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    block_table: torch.Tensor,
    pos: torch.Tensor,
) -> None:
    """一个 kernel 完成追加。替代下面 `paged_kv_append_torch` 那条算子链。"""
    num_kv_heads, head_dim = k_new.shape
    assert k_cache.shape[1] == num_kv_heads and k_cache.shape[3] == head_dim
    assert k_cache.shape[2] == PAGE_SIZE
    assert v_cache.shape == k_cache.shape
    assert k_new.shape == v_new.shape
    assert k_cache.dtype == v_cache.dtype == torch.bfloat16
    assert k_new.dtype == v_new.dtype == torch.bfloat16
    assert block_table.dtype == torch.int32
    assert pos.dtype == torch.int64 and pos.numel() == 1
    assert triton.next_power_of_2(head_dim) == head_dim

    torch.library.wrap_triton(_paged_kv_append_triton)[(num_kv_heads,)](
        k_cache_ptr=k_cache,
        stride_kc_p=k_cache.stride(0),
        stride_kc_h=k_cache.stride(1),
        stride_kc_s=k_cache.stride(2),
        stride_kc_d=k_cache.stride(3),
        v_cache_ptr=v_cache,
        stride_vc_p=v_cache.stride(0),
        stride_vc_h=v_cache.stride(1),
        stride_vc_s=v_cache.stride(2),
        stride_vc_d=v_cache.stride(3),
        k_new_ptr=k_new,
        stride_kn_h=k_new.stride(0),
        stride_kn_d=k_new.stride(1),
        v_new_ptr=v_new,
        stride_vn_h=v_new.stride(0),
        stride_vn_d=v_new.stride(1),
        block_table_ptr=block_table,
        pos_ptr=pos,
        PAGE=PAGE_SIZE,
        D=head_dim,
    )


@torch.library.register_fake("wy_lib::paged_kv_append")
def _paged_kv_append_fake(k_cache, v_cache, k_new, v_new, block_table, pos) -> None:
    return None


@torch.library.custom_op(
    "wy_lib::paged_kv_append_torch",
    mutates_args=("k_cache", "v_cache"),
    device_types="cuda",
)
def paged_kv_append_torch(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    block_table: torch.Tensor,
    pos: torch.Tensor,
) -> None:
    """`paged_kv_append` 的 PyTorch 版，**只用于对拍**，不要放进热路径。

    它在功能上完全正确，但 `physical_rows` 会被展开成一长串逐元素算子，实测
    **12 个 kernel、23.5us**——而真正搬运的数据只有 H*D*2B*2 = 2 KiB。
    这就是为什么要专门写一个 Triton kernel。

    追加放在 attention kernel 之外（而不是塞进 split kernel）的理由和整块版本一样：
    同一个 KV head 会被 GROUP 个 program 同时写、随即又读回，跨 program 的写后读
    没有可见性保证。
    """
    num_kv_heads, head_dim = k_new.shape
    assert k_cache.shape[1] == num_kv_heads and k_cache.shape[3] == head_dim
    assert v_cache.shape == k_cache.shape
    assert k_new.shape == v_new.shape
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert pos.dtype == torch.int64 and pos.numel() == 1

    rows = physical_rows(block_table, pos, num_kv_heads).reshape(-1)  # [H]
    k_cache.view(-1, head_dim).index_copy_(0, rows, k_new)
    v_cache.view(-1, head_dim).index_copy_(0, rows, v_new)


@paged_kv_append_torch.register_fake
def _paged_kv_append_torch_fake(
    k_cache, v_cache, k_new, v_new, block_table, pos
) -> None:
    return None


def paged_kv_fill(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_prefill: torch.Tensor,
    v_prefill: torch.Tensor,
    block_table: torch.Tensor,
) -> None:
    """prefill 用：把整段 `[H, T, D]` 的 K/V 散布到各自的页里。

    这就是「填充既往 KV」在 paged 下的变化：整块方案是一次连续 `copy_`，
    paged 因为逻辑上连续的 token 可能落在物理上不相邻的页，必须走 scatter。
    prefill 不在 CUDA Graph 里，所以这里没有标量冻结的顾虑，但下标同样全在显存上算。
    """
    num_kv_heads, token_num, head_dim = k_prefill.shape
    assert v_prefill.shape == k_prefill.shape
    tokens = torch.arange(token_num, device=k_cache.device, dtype=torch.int64)
    rows = physical_rows(block_table, tokens, num_kv_heads)  # [T, H]
    # k_prefill 是 [H,T,D]，rows 是 [T,H]；转置后两边都按 h-major 展平，一一对应
    flat_rows = rows.t().reshape(-1)
    k_cache.view(-1, head_dim).index_copy_(0, flat_rows, k_prefill.reshape(-1, head_dim))
    v_cache.view(-1, head_dim).index_copy_(0, flat_rows, v_prefill.reshape(-1, head_dim))


def allocate_split_scratch(
    num_q_heads: int, head_dim: int, *, device: torch.device | str = "cuda"
):
    """和整块版本完全一样的 scratch，可以直接共用同一份。"""
    m = torch.zeros((num_q_heads, MAX_SPLITS), dtype=torch.float32, device=device)
    l = torch.zeros((num_q_heads, MAX_SPLITS), dtype=torch.float32, device=device)
    acc = torch.zeros(
        (num_q_heads, MAX_SPLITS, head_dim), dtype=torch.float32, device=device
    )
    return m, l, acc


# --------------------------------------------------------------------------- split

# BLOCK_T 被钉死等于 PAGE_SIZE（理由见模块 docstring），所以这里只剩 warps/stages 可调。
paged_split_autotune_configs = [
    triton.Config({}, num_warps=w, num_stages=s)
    for w in (2, 4, 8)
    for s in (1, 2, 3)
]


@triton.autotune(configs=paged_split_autotune_configs, key=["H_Q", "D", "GROUP", "S_BUCKET"])
@triton.jit
def _paged_gqa_attention_decode_split_triton(
    q_ptr,  # [H_q, D] BF16
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    k_cache_ptr,  # [num_pages, H_kv, PAGE, D] BF16，只读
    stride_kc_p: tl.constexpr, # 每个page占用的地址大小
    stride_kc_h: tl.constexpr,
    stride_kc_s: tl.constexpr,
    stride_kc_d: tl.constexpr,
    v_cache_ptr,  # [num_pages, H_kv, PAGE, D] BF16，只读
    stride_vc_p: tl.constexpr,
    stride_vc_h: tl.constexpr,
    stride_vc_s: tl.constexpr,
    stride_vc_d: tl.constexpr,
    block_table_ptr,  # [max_pages] INT32，逻辑页 -> 物理页
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
    pos_ptr,  # [1] INT64
    scale,
    H_Q: tl.constexpr,
    D: tl.constexpr,
    GROUP: tl.constexpr,
    MAX_SPLITS_C: tl.constexpr,
    PAGE: tl.constexpr,
    S_BUCKET: tl.constexpr,
):
    # ------------------------------------------------------------------ 切分
    # seq_len 从显存里的 pos 算，host 一个标量都不传——CUDA Graph replay 会用
    # capture 时的旧值。切分改成按页对齐，这样每次迭代恰好覆盖一个完整的页。
    seq_len = tl.load(pos_ptr).to(tl.int32) + 1
    num_pages_used = (seq_len + PAGE - 1) // PAGE
    pages_per_split = (num_pages_used + MAX_SPLITS_C - 1) // MAX_SPLITS_C

    pid_h, pid_s = tl.program_id(0), tl.program_id(1)

    page_start = pid_s * pages_per_split
    page_end = tl.minimum(page_start + pages_per_split, num_pages_used)

    offset_h = pid_h * GROUP + tl.arange(0, GROUP)
    offset_d = tl.arange(0, D)
    offset_s = tl.arange(0, PAGE)

    m_i = tl.zeros([GROUP], tl.float32) - float("inf")
    l_i = tl.zeros([GROUP], tl.float32)
    acc = tl.zeros([GROUP, D], tl.float32)

    q = tl.load(q_ptr + offset_h[:, None] * stride_q_h + offset_d[None, :] * stride_q_d)

    # 本 program 负责的 KV head 在页内的基址偏移
    k_head_ptr = k_cache_ptr + pid_h * stride_kc_h
    v_head_ptr = v_cache_ptr + pid_h * stride_vc_h

    offset_s = tl.arange(0, PAGE)
    for lp in tl.range(page_start, page_end):
        # page-attention下 TILE_Tj即为page-size
        # 一次载入page-size的连续数据空间
        phys = tl.load(block_table_ptr + lp).to(tl.int64)
        k = tl.load(
            k_head_ptr + phys * stride_kc_p + 
            offset_s[None, :] * stride_kc_s + offset_d[:, None] * stride_kc_d
        )
        v = tl.load(
            v_head_ptr + phys * stride_vc_p + 
            offset_s[:, None] * stride_vc_s + offset_d[None, :] * stride_vc_d
        )

        qk = tl.dot(q, k) * scale # [GROUP, PAGE]
        qk = tl.where(lp * PAGE + offset_s[None, :] < seq_len, qk, -float("inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis = 1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])

        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v) # [GROUP, D]
        l_i = l_i * alpha + tl.sum(p, axis = 1)
        m_i = m_i_new

    # ------------------------------------------------------------------ 落盘
    tl.store(m_partial_ptr + offset_h * stride_mp_h + pid_s * stride_mp_s, m_i)
    tl.store(l_partial_ptr + offset_h * stride_lp_h + pid_s * stride_lp_s, l_i)
    tl.store(
        acc_partial_ptr
        + offset_h[:, None] * stride_ap_h
        + pid_s * stride_ap_s
        + offset_d[None, :] * stride_ap_d,
        acc,
    )


# -------------------------------------------------------------------------- combine

paged_combine_autotune_configs = [
    triton.Config({"BLOCK_D": block_d}, num_warps=warps, num_stages=1)
    for block_d in (32, 64, 128)
    for warps in (2, 4)
]


@triton.autotune(configs=paged_combine_autotune_configs, key=["D", "MAX_SPLITS_C"])
@triton.jit
def _paged_gqa_attention_decode_combine_triton(
    m_partial_ptr,
    stride_mp_h: tl.constexpr,
    stride_mp_s: tl.constexpr,
    l_partial_ptr,
    stride_lp_h: tl.constexpr,
    stride_lp_s: tl.constexpr,
    acc_partial_ptr,
    stride_ap_h: tl.constexpr,
    stride_ap_s: tl.constexpr,
    stride_ap_d: tl.constexpr,
    out_ptr,
    stride_o_h: tl.constexpr,
    stride_o_d: tl.constexpr,
    pos_ptr,
    D: tl.constexpr,
    MAX_SPLITS_C: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """与整块版本逐行相同，**只有 num_active 的算法换成按页对齐**。

    这三行必须和 split kernel 里的切分用同一套公式，否则两边对「哪些 split 有效」
    的理解不一致，会读到上一次残留的局部量。刻意在 kernel 内重算而不是从 host 传。
    """
    seq_len = tl.load(pos_ptr).to(tl.int32) + 1
    num_pages_used = (seq_len + PAGE - 1) // PAGE
    pages_per_split = (num_pages_used + MAX_SPLITS_C - 1) // MAX_SPLITS_C
    num_active = (num_pages_used + pages_per_split - 1) // pages_per_split

    pid_h, pid_d = tl.program_id(0), tl.program_id(1)
    offset_s = tl.arange(0, MAX_SPLITS_C)
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    m_all = tl.load(
        m_partial_ptr + pid_h * stride_mp_h + offset_s * stride_mp_s,
        mask=offset_s < num_active,
        other=-float("inf"),
    )
    l_all = tl.load(
        l_partial_ptr + pid_h * stride_lp_h + offset_s * stride_lp_s,
        mask=offset_s < num_active,
        other=0.0,
    )
    acc_all = tl.load(
        acc_partial_ptr
        + pid_h * stride_ap_h
        + offset_s[:, None] * stride_ap_s
        + offset_d[None, :] * stride_ap_d,
        mask=offset_s[:, None] < num_active,
        other=0.0,
    )

    m = tl.max(m_all, axis=-1)
    alpha = tl.exp(m_all - m)
    l = tl.sum(l_all * alpha, axis=-1)
    acc_all = tl.sum(acc_all * alpha[:, None], axis=0) / l

    tl.store(
        out_ptr + pid_h * stride_o_h + offset_d * stride_o_d,
        acc_all.to(out_ptr.dtype.element_ty),
    )


# --------------------------------------------------------------------------- wrapper

@torch.library.triton_op(
    "wy_lib::paged_gqa_attention_decode_split",
    mutates_args=("m_partial", "l_partial", "acc_partial"),
)
def paged_gqa_attention_decode_split(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    m_partial: torch.Tensor,
    l_partial: torch.Tensor,
    acc_partial: torch.Tensor,
    pos: torch.Tensor,
    seq_bucket: int,
) -> None:
    num_q_heads, head_dim = q.shape
    num_pages, num_kv_heads, page, cache_dim = k_cache.shape
    assert page == PAGE_SIZE and cache_dim == head_dim
    assert v_cache.shape == k_cache.shape
    assert q.dtype == torch.bfloat16 and k_cache.dtype == torch.bfloat16
    assert block_table.dtype == torch.int32
    assert m_partial.shape == (num_q_heads, MAX_SPLITS)
    assert acc_partial.shape == (num_q_heads, MAX_SPLITS, head_dim)
    assert pos.dtype == torch.int64 and pos.numel() == 1
    assert num_q_heads % num_kv_heads == 0

    torch.library.wrap_triton(_paged_gqa_attention_decode_split_triton)[
        (num_kv_heads, MAX_SPLITS)
    ](
        q_ptr=q,
        stride_q_h=q.stride(0),
        stride_q_d=q.stride(1),
        k_cache_ptr=k_cache,
        stride_kc_p=k_cache.stride(0),
        stride_kc_h=k_cache.stride(1),
        stride_kc_s=k_cache.stride(2),
        stride_kc_d=k_cache.stride(3),
        v_cache_ptr=v_cache,
        stride_vc_p=v_cache.stride(0),
        stride_vc_h=v_cache.stride(1),
        stride_vc_s=v_cache.stride(2),
        stride_vc_d=v_cache.stride(3),
        block_table_ptr=block_table,
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
        PAGE=PAGE_SIZE,
        S_BUCKET=_seq_bucket(seq_bucket),
    )


@torch.library.register_fake("wy_lib::paged_gqa_attention_decode_split")
def _paged_split_fake(
    q, k_cache, v_cache, block_table, m_partial, l_partial, acc_partial, pos, seq_bucket
) -> None:
    return None


@torch.library.triton_op("wy_lib::paged_gqa_attention_decode_combine", mutates_args=())
def paged_gqa_attention_decode_combine(
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

    torch.library.wrap_triton(_paged_gqa_attention_decode_combine_triton)[grid](
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
        PAGE=PAGE_SIZE,
    )
    return out


@torch.library.register_fake("wy_lib::paged_gqa_attention_decode_combine")
def _paged_combine_fake(m_partial, l_partial, acc_partial, pos) -> torch.Tensor:
    num_q_heads, _, head_dim = acc_partial.shape
    return torch.empty(
        (num_q_heads, head_dim), dtype=torch.bfloat16, device=acc_partial.device
    )


def _seq_bucket(seq_len: int) -> int:
    """与整块版本同一套分桶：seq_len 每步都在涨，直接进 autotune key 会每步重调。"""
    if seq_len <= 64:
        return 64
    if seq_len <= 256:
        return 256
    if seq_len <= 1024:
        return 1024
    if seq_len <= 4096:
        return 4096
    return 4097


def call_paged_gqa_attention_decode_triton(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    pos: torch.Tensor,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """追加 + split + combine，接口与整块版本对齐（多一个 block_table）。"""
    if scratch is None:
        scratch = allocate_split_scratch(q.shape[0], q.shape[1], device=q.device)
    m_partial, l_partial, acc_partial = scratch

    paged_kv_append(k_cache, v_cache, k_new, v_new, block_table, pos)
    # seq_bucket 只用于 autotune 选 config，不参与任何计算。传 cache 容量做上界，
    # 避免每步换桶；CUDA Graph 下它在 capture 时就冻结了。
    hint = k_cache.shape[0] * PAGE_SIZE
    paged_gqa_attention_decode_split(
        q, k_cache, v_cache, block_table, m_partial, l_partial, acc_partial, pos, hint
    )
    return paged_gqa_attention_decode_combine(m_partial, l_partial, acc_partial, pos)


# --------------------------------------------------------------------------- 参考实现

def torch_paged_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """逐页取回、拼成整块、再做一次普通的 GQA decode attention。

    只用于对拍：把 paged 的寻址和 attention 本身解耦，
    这样 kernel 出问题时能立刻分清是地址算错了还是 softmax 写错了。
    """
    num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group = num_q_heads // num_kv_heads

    tokens = torch.arange(seq_len, device=q.device, dtype=torch.int64)
    rows = physical_rows(block_table, tokens, num_kv_heads)  # [T, H]
    k = k_cache.view(-1, head_dim)[rows.t().reshape(-1)].view(num_kv_heads, seq_len, head_dim)
    v = v_cache.view(-1, head_dim)[rows.t().reshape(-1)].view(num_kv_heads, seq_len, head_dim)

    k = k.repeat_interleave(group, dim=0).float()  # [H_q, T, D]
    v = v.repeat_interleave(group, dim=0).float()
    scores = (q.float().unsqueeze(1) @ k.transpose(1, 2)).squeeze(1) * head_dim**-0.5
    probs = torch.softmax(scores, dim=-1)
    return (probs.unsqueeze(1) @ v).squeeze(1).to(q.dtype)
