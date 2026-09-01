import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl
import math


'''
针对qwen3.5 0.6b的GQA Attention实现 q_head=8 kv_head=2
第一步暂时不支持kv cache, 直接做带casual的full-attention
head维度上分为按照Q和KV的num_head切分两种方式 
前者并行度大 需要L2 cache降低对global memory的读取压力 
后者对于kv的读取量减少 同时启动的block数量也会降低
'''

autotune_configs = [
    triton.Config(
        {
            "BLOCK_Q_S": block_q_s,
            "TILE_KV_S": tile_kv_s,
        },
        num_warps=num_warps,
        num_stages=2,
    )
    for block_q_s in [16, 32]
    for tile_kv_s in [32, 64]
    for num_warps in [2, 4]
]
@triton.autotune(
    configs=autotune_configs,
    key=["d_model", "group_size"],
)
@triton.jit
def _gqa_attention_without_kvcache_casual_triton(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d, 
    stride_k_b, stride_k_h, stride_k_s, stride_k_d, 
    stride_v_b, stride_v_h, stride_v_s, stride_v_d, 
    stride_o_b, stride_o_h, stride_o_s, stride_o_d, 
    q_seq_len, kv_seq_len,
    sm_scale: tl.constexpr, 
    d_model: tl.constexpr, 
    group_size: tl.constexpr,
    BLOCK_Q_S: tl.constexpr,
    TILE_KV_S: tl.constexpr
):
    # 不考虑kv cache的情况下 需要对q的完整序列维度进行计算
    q_id = tl.program_id(0)
    # 按照kv 的 num head 起block
    head_id = tl.program_id(1)
    batch_id = tl.program_id(2)

    q_base_ptr = q_ptr + batch_id * stride_q_b + head_id * group_size * stride_q_h
    o_base_ptr = o_ptr + batch_id * stride_o_b + head_id * group_size * stride_o_h
    # 对kv直接执行batch head维度消除
    k_base_ptr = k_ptr + batch_id * stride_k_b + head_id * stride_k_h 
    v_base_ptr = v_ptr + batch_id * stride_v_b + head_id * stride_v_h

    offset_d = tl.arange(0, d_model)
    offset_q_h = tl.arange(0, group_size)
    offset_q_s = q_id * BLOCK_Q_S + tl.arange(0, BLOCK_Q_S)

    # [group_size, BLOCK_Q_S, d_model]
    q = tl.load(
        q_base_ptr + offset_q_h[:, None, None] * stride_q_h + offset_q_s[None, :, None] * stride_q_s + offset_d[None, None, :] * stride_q_d, 
        mask=offset_q_s[None, :, None] < q_seq_len,
        other = 0.0
    )

    k_block_ptr = tl.make_block_ptr(
        base = k_base_ptr,
        shape = (d_model, kv_seq_len),
        strides = (stride_k_d, stride_k_s),
        offsets = (0, 0),
        block_shape = (d_model, TILE_KV_S),
        order = (0, 1)
    )

    v_block_ptr = tl.make_block_ptr(
        base = v_base_ptr, 
        shape = (kv_seq_len, d_model), 
        strides = (stride_v_s, stride_v_d), 
        offsets = (0, 0),
        block_shape = (TILE_KV_S, d_model),
        order = (1, 0)
    )

    m_i = tl.zeros([group_size, BLOCK_Q_S], dtype = tl.float32) - float('inf')
    l_i = tl.zeros([group_size, BLOCK_Q_S], dtype = tl.float32)
    acc = tl.zeros([group_size, BLOCK_Q_S, d_model], dtype = tl.float32)

    qk_scale = sm_scale * 1.4426950408889634

    q = (q * qk_scale).to(tl.bfloat16)

    for start_kv_s in tl.range(0, q_id * BLOCK_Q_S + BLOCK_Q_S, TILE_KV_S):
        # [TILE_KV_S, d_model]
        k = tl.load(k_block_ptr, boundary_check = (1,)).to(tl.bfloat16)
        v = tl.load(v_block_ptr, boundary_check = (0, )).to(tl.bfloat16)

        qk = tl.dot(
            q, tl.broadcast_to(k[None, :, :], (group_size, d_model, TILE_KV_S))
        )

        offset_kv_s = start_kv_s + tl.arange(0, TILE_KV_S)
        causal_mask = (
            offset_kv_s[None, None, :]
            <= offset_q_s[None, :, None]
        )

        # tl.where 不会原地修改，必须赋值
        qk = tl.where(causal_mask, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, axis = -1))
        alpha = tl.math.exp2(m_i - m_i_new)

        p = tl.math.exp2(qk - m_i_new[:, :, None])

        acc *= alpha[:, :, None]
        acc += tl.dot(p.to(tl.bfloat16), tl.broadcast_to(v[None, :, :], (group_size, TILE_KV_S, d_model)))

        l_i = l_i * alpha + tl.sum(p, axis = -1)
        m_i = m_i_new

        k_block_ptr = tl.advance(k_block_ptr, (0, TILE_KV_S))
        v_block_ptr = tl.advance(v_block_ptr, (TILE_KV_S, 0))

    acc = acc / l_i[:, :, None]
    tl.store(
        o_base_ptr + offset_q_h[:, None, None] * stride_o_h + offset_q_s[None, :, None] * stride_o_s + offset_d[None, None, :] * stride_o_d, 
        acc.to(tl.bfloat16), 
        mask=offset_q_s[None, :, None] < q_seq_len,
    )

@torch.library.triton_op(
    "wy_lib::gqa_attention_without_kvcache_casual",
    mutates_args=(),
)
def gqa_attention_without_kvcache_casual(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16

    batch_size, num_q_head, seq_len_q, d_model = q.shape
    kv_seq_len = k.shape[2]
    kv_num_head = k.shape[1]

    def grid(META):
        return (triton.cdiv(seq_len_q, META['BLOCK_Q_S']), kv_num_head, batch_size)
    o = torch.empty_like(q)

    torch.library.wrap_triton(_gqa_attention_without_kvcache_casual_triton)[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=o,
        stride_q_b=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_s=q.stride(2),
        stride_q_d=q.stride(3),
        stride_k_b=k.stride(0),
        stride_k_h=k.stride(1),
        stride_k_s=k.stride(2),
        stride_k_d=k.stride(3),
        stride_v_b=v.stride(0),
        stride_v_h=v.stride(1),
        stride_v_s=v.stride(2),
        stride_v_d=v.stride(3),
        stride_o_b=o.stride(0),
        stride_o_h=o.stride(1),
        stride_o_s=o.stride(2),
        stride_o_d=o.stride(3),
        q_seq_len=seq_len_q,
        kv_seq_len=kv_seq_len,
        sm_scale=1.0 / math.sqrt(d_model),
        d_model=d_model,
        group_size=num_q_head // kv_num_head
    )

    return o 


@torch.library.register_fake("wy_lib::gqa_attention_without_kvcache_casual")
def _gqa_attention_without_kvcache_casual_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(q)


def call_gqa_attention_without_kvcache_casual_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    return gqa_attention_without_kvcache_casual(q, k, v)


if __name__ == "__main__":
    batch_size = 1
    num_q_head = 8
    num_kv_head = 2
    seq_len_q = 128
    seq_len_kv = 128
    d_model = 256

    q = torch.randn(batch_size, num_q_head, seq_len_q, d_model, dtype=torch.bfloat16, device='cuda')
    k = torch.randn(batch_size, num_kv_head, seq_len_kv, d_model, dtype=torch.bfloat16, device='cuda')
    v = torch.randn(batch_size, num_kv_head, seq_len_kv, d_model, dtype=torch.bfloat16, device='cuda')

    o = call_gqa_attention_without_kvcache_casual_triton(q, k, v)
    print(o.shape)
    print(o)
    o_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, enable_gqa=True)
    print(torch.max(torch.abs(o - o_sdpa)))
