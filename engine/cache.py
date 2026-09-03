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
replay 永远用 capture 那一刻的值。详见 HANDOFF 8.8 和
`triton_kernels/gqa_attention_decode.py` 顶部那段长注释。

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
from triton_kernels.depthwise_causal_conv4_decode import CONV_KERNEL_SIZE
from triton_kernels.gqa_attention_decode import (
    allocate_kv_cache,
    allocate_position,
    allocate_split_scratch,
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

    def memory_bytes(self) -> dict[str, int]:
        def total(group):
            return sum(t.numel() * t.element_size() for t in group if t is not None)

        return {
            "conv": total(self.conv_states),
            "recurrent": total(self.recurrent_states),
            "kv": total(self.k_caches) + total(self.v_caches),
            "scratch": sum(t.numel() * t.element_size() for t in self.split_scratch),
        }


def allocate_caches(w: TextWeights, max_len: int) -> DecodeCaches:
    """按层类型分配三类 cache。一次分配，整个生成过程复用。"""
    device = w.device
    conv_dim = w.linear_num_heads * w.linear_head_dim * 3  # q+k+v = 6144

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
    )
