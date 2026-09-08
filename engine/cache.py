"""三类 cache 的分配与生命周期管理。

Qwen3.5 的 24 层里，18 层 GDN 每层要两个 cache，6 层 full attention 每层要一对
KV cache——**三类，不是一类**：

    conv state       [4, 6144]      BF16   GDN，depthwise Conv4 要回看 3 个位置
    recurrent state  [16, 128, 128] FP32   GDN，delta rule 的状态
    K/V cache        [2, T_max, 256] BF16 ×2  full attention

前两个是**定长**的（跟上下文长度无关），只有 KV cache 随 T_max 线性增长：

    conv       18 层 × 4 × 6144 × 2B                    = 0.84 MiB
    recurrent  18 层 × 16 × 128 × 128 × 4B              = 18.00 MiB
    KV         6 层 × 2 × T_max × 256 × 2B × 2(K+V)     = 12 KiB/token
               T_max=8192 -> 96 MiB，32768 -> 384 MiB

位置只用一个显存张量 `pos`，三类 cache 共用——它同时是 KV cache 的写入下标、
attention 的 seq_len 来源。conv state 和 recurrent state 是"就地推进"的，
不需要知道自己在第几步，所以不用位置。

**为什么位置放显存**：CUDA Graph capture 会把标量参数和切片下标烧进 launch 配置，
replay 永远用 capture 那一刻的值。详见 `triton_kernels/gqa_attention_decode.py`
顶部那段长注释。

**reset() 不是可选的**。图是有状态的：capture 过程本身（warmup + 正式捕获）会把
pos 推进好几格、也会污染 cache；换 prompt 时更是必须清空。所以这里把复位做成显式
方法，而不是指望调用方记得。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.loader import AttnLayerWeights, GDNLayerWeights, TextWeights
from engine.paging import PagePool, SequencePages
from triton_kernels.depthwise_causal_conv4_decode import CONV_KERNEL_SIZE
from triton_kernels.gqa_attention_decode import (
    allocate_kv_cache,
    allocate_position,
    allocate_split_scratch,
)
from triton_kernels.gqa_attention_decode_paged import (
    PAGE_SIZE,
    allocate_paged_kv_cache,
)


@dataclass
class DecodeCaches:
    """一次生成过程中的全部可变状态。"""

    # 位置：已缓存的 token 数。KV cache 的写入下标 + attention 的 seq_len 来源。
    pos: torch.Tensor  # [1] INT64，显存

    # 按层号索引；非对应类型的层为 None，这样层号可以直接当下标用
    conv_states: list[torch.Tensor | None]  # GDN: [4, 6144] BF16
    recurrent_states: list[torch.Tensor | None]  # GDN: [16,128,128] FP32
    k_caches: list[torch.Tensor | None]  # attn: [2, T_max, 256] BF16
    v_caches: list[torch.Tensor | None]

    # split-K 的 scratch，所有 attention 层共用一份（串行执行，用完即弃）
    split_scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    max_len: int

    # ---- paged 专用；整块方案下都是 None ----
    #
    # **页表只有一张，六个 attention 层共用。** 它们的 cache 形状完全一致，
    # 逻辑页 i 在每一层都映射到同一个物理页 j，所以没必要一层一张。
    # 六层各有自己的 k_caches[i] / v_caches[i] 页池。
    pages: SequencePages | None = None
    page_pool: PagePool | None = None

    @property
    def paged(self) -> bool:
        return self.pages is not None

    @property
    def block_table(self) -> torch.Tensor | None:
        return None if self.pages is None else self.pages.block_table

    def reserve(self, token_num: int) -> None:
        """确保能放下 `token_num` 个 token。整块方案下退化成一次容量断言。

        paged 下这是 host 侧的事，必须在会写到那个位置的 forward **之前**调用；
        写完再发现越界就晚了（`index_copy_` 会在 device 上 assert，报错点离原因很远）。
        """
        assert token_num <= self.max_len, (
            f"需要 {token_num} 个 token 的容量，超出 cache 上限 {self.max_len}"
        )
        if self.pages is not None:
            self.pages.reserve(token_num, self.page_pool)

    def reset(self) -> None:
        """清空全部状态。换 prompt、或 CUDA Graph 捕获之后必须调用。

        捕获过程本身会执行若干次 warmup，把 pos 推进、把 cache 写脏——
        这是"图有状态"的直接后果，忘了复位会得到静悄悄的错误结果。
        """
        self.pos.zero_()
        for group in (self.conv_states, self.recurrent_states, self.k_caches, self.v_caches):
            for t in group:
                if t is not None:
                    t.zero_()
        for t in self.split_scratch:
            t.zero_()
        if self.pages is not None:
            # 把页还回池子。严格说不清页表也能正确——kernel 只遍历
            # `lp < cdiv(seq_len, PAGE)`，而 seq_len 来自刚归零的 pos，
            # 残留的旧页号永远读不到。但还页是必须的，否则池子会漏。
            self.pages.release(self.page_pool)

    def memory_bytes(self) -> dict[str, int]:
        def total(group):
            return sum(t.numel() * t.element_size() for t in group if t is not None)

        return {
            "conv": total(self.conv_states),
            "recurrent": total(self.recurrent_states),
            "kv": total(self.k_caches) + total(self.v_caches),
            "scratch": sum(t.numel() * t.element_size() for t in self.split_scratch),
        }


def allocate_caches(w: TextWeights, max_len: int, *, paged: bool = False) -> DecodeCaches:
    """按层类型分配三类 cache。一次分配，整个生成过程复用。

    `paged=True` 时 attention 的 KV 改用分页布局。**batch=1 下它没有任何显存收益**
    ——同样的 T_max 下两种方案占的字节数完全一样，paged 还多一层间接寻址。
    它是为凑批做的铺垫：整块方案下 B 条请求要按 `B * max_len` 预留，
    paged 只要 `sum(实际长度)`，而且请求之间能共享前缀。

    GDN 的两个 cache 与 paged 无关：它们是定长的，和上下文长度没关系。
    """
    device = w.device
    conv_dim = w.linear_num_heads * w.linear_head_dim * 3  # q+k+v = 6144

    # 页数按 max_len 算，再留一页余量给「最后一个 token 正好落在新页开头」的情况
    num_pages = (max_len + PAGE_SIZE - 1) // PAGE_SIZE + 1
    pool = PagePool(num_pages) if paged else None
    pages = SequencePages(num_pages, device=device) if paged else None

    conv_states: list[torch.Tensor | None] = []
    recurrent_states: list[torch.Tensor | None] = []
    k_caches: list[torch.Tensor | None] = []
    v_caches: list[torch.Tensor | None] = []

    for layer in w.layers:
        if isinstance(layer, GDNLayerWeights):
            conv_states.append(
                torch.zeros(
                    (CONV_KERNEL_SIZE, conv_dim), dtype=torch.bfloat16, device=device
                )
            )
            recurrent_states.append(
                torch.zeros(
                    (w.linear_num_heads, w.linear_head_dim, w.linear_head_dim),
                    dtype=torch.float32,  # delta rule 的状态必须 FP32
                    device=device,
                )
            )
            k_caches.append(None)
            v_caches.append(None)
        else:
            assert isinstance(layer, AttnLayerWeights)
            conv_states.append(None)
            recurrent_states.append(None)
            if paged:
                # 每层一份页池，但页表只有一张（见 DecodeCaches.pages 的注释）
                k, v = allocate_paged_kv_cache(
                    num_pages, w.num_key_value_heads, w.head_dim, device=device
                )
            else:
                k, v = allocate_kv_cache(
                    w.num_key_value_heads, max_len, w.head_dim, device=device
                )
            k_caches.append(k)
            v_caches.append(v)

    return DecodeCaches(
        pos=allocate_position(device),
        conv_states=conv_states,
        recurrent_states=recurrent_states,
        k_caches=k_caches,
        v_caches=v_caches,
        split_scratch=allocate_split_scratch(
            w.num_attention_heads, w.head_dim, device=device
        ),
        max_len=max_len,
        pages=pages,
        page_pool=pool,
    )
