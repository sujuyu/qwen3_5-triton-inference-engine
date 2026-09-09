"""变长打包(varlen packing)的 causal GQA prefill attention.

和 `gqa_attention_without_kvcache_casual.py` 的关系
==================================================
在线 softmax, causal mask, GQA 的 head 分组**全部不变**, 唯一的差别是
**一条序列在内存里从哪开始, 有多长**: 定长版本靠 `batch_id * stride_b`
定位, 变长版本靠 `cu_seqlens[batch_id]` 定位.

打包 vs padding
---------------
```text
prompt      [A A A]  [B B B B B B]  [C C]
padding     A A A 0 0 0 | B B B B B B | C C 0 0 0 0     浪费 (max-len) 之和
packing     A A A B B B B B B C C                       total = 11
cu_seqlens  [0, 3, 9, 11]
```

**打包对这个项目改动更小**, 因为 `_forward` 处理的本来就是扁平的 `[T, 1024]`.
所有逐 token 的算子(gemm_2d / rmsnorm / swiglu / residual_add / SwiGLU /
gdn_qk_norm_gates / gdn_gated_rmsnorm / attention_gate_pack)拿 `[total, ...]`
一行都不用改; `partial_rope` 只要 position_ids 每条序列从 0 重新开始, 也不用改.
**只有三个算子有跨 token 交互**: 本文件(attention), conv4_prefill(回看 3 个
token, 跨边界要补零), GDN 递推(状态要在每条序列开头重置).

这和 decode 正好相反: decode 是"几乎所有算子都要加 batch 维",
prefill 是"几乎所有算子都不用改" -- 因为打包把 batch 藏进了 token 维.

为什么值得做
------------
不是为了 GPU 效率, 是为了**摊薄算子分发开销**. 实测逐条 prefill:

```text
B=1  T=128   总 1024 token    45ms
B=8  T=128   总 1024 token   366ms      <- 线性涨
B=8  T=512   总 4096 token   367ms      <- 和 token 数几乎无关
```

后两行 token 数差 4 倍而耗时相同, 说明成本全在"调了 8 次"上 -- 还是那个约
43ms 的 CPU 分发地板乘以 B. 打包把 B 次前向合并成 1 次, 首 token 延迟从
B×43ms 变回 1×43ms.

grid 与早退
-----------
按**最长序列**起 q 块, 超出自己长度的块直接 return:

```text
grid = (cdiv(max_seqlen, BLOCK_Q_S), H_kv, B)
```

浪费的是空转 CTA, 实测约 0.7ns 一个. B=8, 长度 [4096, 100×7], BLOCK_Q_S=64 时
空转 868 个, 合计约 0.6us, 对毫秒级的 prefill 可以忽略.

**causal mask 不需要额外处理. ** 每条序列的 q 和 kv 是同一段, 块内 `t >= i`
的判断在序列局部坐标里天然正确; 跨序列的 KV 因为循环边界就在自己那一段里,
根本不会被访问到. 所谓"块对角"是自动实现的, 不是要额外写的 mask.

一个已知的不均衡(暂不处理)
----------------------------
causal 让工作量正比于 `q_id + 1`(循环上界含 q_id), 所以同一个 kernel 里
第 0 块扫 1 个 KV 块, 第 63 块扫 64 个, 差 64 倍. 这在定长版本里就存在,
不是变长带来的.

缓解办法是**逆序发射**(`pid = num_programs - 1 - program_id`, 让重的先上,
轻的填空隙 -- 经典的 LPT 调度). 合成实验实测收益随规模变大:

```text
q 块数 N      正序      逆序     收益
      64    6.54us    6.24us   +4.8%
     512   24.25us   23.32us   +4.0%
    2048  211.94us  178.69us  +18.6%
```

当前 prompt 长度下 N 只有几到几十块, 收益 2~5%, 所以先不做 -- 和 varlen 混在
一起改会分不清是哪个出的问题. 等这里跑通对拍之后再单独加.
"""

from __future__ import annotations

import math

import torch

import triton
import triton.language as tl


autotune_configs = [
    triton.Config(
        {"BLOCK_Q_S": block_q_s, "TILE_KV_S": tile_kv_s},
        num_warps=num_warps,
        num_stages=2,
    )
    for block_q_s in [16, 32]
    for tile_kv_s in [32, 64]
    for num_warps in [2, 4]
]


@triton.autotune(configs=autotune_configs, key=["d_model", "group_size"])
@triton.jit
def _gqa_attention_varlen_causal_triton(
    q_ptr,  # [total_tokens, H_q,  D] BF16, 打包后的连续布局
    k_ptr,  # [total_tokens, H_kv, D] BF16
    v_ptr,  # [total_tokens, H_kv, D] BF16
    o_ptr,  # [total_tokens, H_q,  D] BF16
    stride_q_s, stride_q_h, stride_q_d,
    stride_k_s, stride_k_h, stride_k_d,
    stride_v_s, stride_v_h, stride_v_d,
    stride_o_s, stride_o_h, stride_o_d,
    cu_seqlens_ptr,  # [B+1] INT32, 前缀和; cu[b] 是第 b 条序列在打包缓冲里的起点
    sm_scale: tl.constexpr,
    d_model: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_Q_S: tl.constexpr,
    TILE_KV_S: tl.constexpr,
):
    q_id = tl.program_id(0)      # 第几个 q 块(按 max_seqlen 起, 可能超出本序列)
    head_id = tl.program_id(1)   # 按 KV head 起 block, 一个 program 管 group_size 个 Q head
    batch_id = tl.program_id(2)  # 第几条序列

    # TODO(用户填) -- 与定长版本的全部差异都在这里, 四处:
    #
    # 1. 取本序列的起点和长度(替代原来的 batch_id * stride_*_b):
    #
    #        seq_start = tl.load(cu_seqlens_ptr + batch_id)
    #        seq_len   = tl.load(cu_seqlens_ptr + batch_id + 1) - seq_start
    #
    # 2. 早退. grid 按最长序列起, 短序列的高位 q 块没有活:
    #
    #        if q_id * BLOCK_Q_S >= seq_len:
    #            return
    #
    #    注意**要在做任何昂贵的事情之前 return**. 实测空转 CTA 约 0.7ns 一个,
    #    前提是它没先载入一堆数据; 先 load 再发现没活就不便宜了.
    #
    # 3. 基址. 原来是 `q_ptr + batch_id * stride_q_b + head_id * group_size * stride_q_h`,
    #    现在 batch 维不存在了, 换成在打包缓冲里的行偏移:
    #
    #        q_base_ptr = q_ptr + seq_start * stride_q_s + head_id * group_size * stride_q_h
    #        o_base_ptr = o_ptr + seq_start * stride_o_s + head_id * group_size * stride_o_h
    #        k_base_ptr = k_ptr + seq_start * stride_k_s + head_id * stride_k_h
    #        v_base_ptr = v_ptr + seq_start * stride_v_s + head_id * stride_v_h
    #
    # 4. 所有用到 `q_seq_len` / `kv_seq_len` 的地方换成 `seq_len`: q 的载入 mask,
    #    两个 make_block_ptr 的 shape, KV 循环的上界, 最后 store 的 mask.
    #    KV 循环上界仍是 `q_id * BLOCK_Q_S + BLOCK_Q_S`(causal),
    #    block_ptr 的 boundary_check 会把超出 seq_len 的部分挡住.
    #
    # 其余部分 -- 在线 softmax, causal mask, GQA 的 head 广播 -- **逐行照抄定长版本**.
    # causal mask 尤其不用改: offset_q_s 和 offset_kv_s 都是序列**局部**坐标,
    # `offset_kv_s <= offset_q_s` 在局部坐标里就是正确的因果关系,
    # 而跨序列的 KV 因为循环边界在自己那一段里, 根本不会被访问到.
    pass  # <- 在这里实现


@torch.library.triton_op("wy_lib::gqa_attention_varlen_causal", mutates_args=())
def gqa_attention_varlen_causal(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> torch.Tensor:
    """打包后的变长 causal GQA.

    ```text
    q            [total_tokens, H_q,  D]  BF16
    k, v         [total_tokens, H_kv, D]  BF16
    cu_seqlens   [B+1]                    INT32, 前缀和, cu[0]=0, cu[B]=total_tokens
    max_seqlen   host int, 决定 grid 的第一维; 只影响启动多少块, 不参与计算
    返回         [total_tokens, H_q,  D]  BF16
    ```

    `max_seqlen` 必须是 host 侧的 python int 而不是显存标量 -- 它决定 grid 大小,
    而 grid 是 launch 配置的一部分. prefill 不在 CUDA Graph 里, 所以这没问题;
    decode 那边 `pos` 必须进显存正是因为它在图里(见
    `gqa_attention_decode_paged.py` 顶部).
    """
    assert q.dtype == torch.bfloat16 and k.dtype == v.dtype == torch.bfloat16
    assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.ndim == 1

    total_tokens, num_q_head, d_model = q.shape
    assert k.shape == v.shape
    assert k.shape[0] == total_tokens and k.shape[2] == d_model
    num_kv_head = k.shape[1]
    assert num_q_head % num_kv_head == 0
    batch = cu_seqlens.numel() - 1
    assert batch >= 1

    out = torch.empty_like(q)

    def grid(meta):
        # 第一维按最长序列起; 短序列的高位块早退.
        # 可能变大的那一维放 dim0(dim1/dim2 上限 65535), batch 和 head 都有界.
        return (triton.cdiv(max_seqlen, meta["BLOCK_Q_S"]), num_kv_head, batch)

    torch.library.wrap_triton(_gqa_attention_varlen_causal_triton)[grid](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        o_ptr=out,
        stride_q_s=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_k_s=k.stride(0),
        stride_k_h=k.stride(1),
        stride_k_d=k.stride(2),
        stride_v_s=v.stride(0),
        stride_v_h=v.stride(1),
        stride_v_d=v.stride(2),
        stride_o_s=out.stride(0),
        stride_o_h=out.stride(1),
        stride_o_d=out.stride(2),
        cu_seqlens_ptr=cu_seqlens,
        sm_scale=1.0 / math.sqrt(d_model),
        d_model=d_model,
        group_size=num_q_head // num_kv_head,
    )
    return out


@torch.library.register_fake("wy_lib::gqa_attention_varlen_causal")
def _gqa_attention_varlen_causal_fake(q, k, v, cu_seqlens, max_seqlen) -> torch.Tensor:
    return torch.empty_like(q)


# --------------------------------------------------------------------------- 工具

def build_cu_seqlens(
    lengths: list[int], *, device: torch.device | str = "cuda"
) -> tuple[torch.Tensor, int]:
    """长度列表 -> (cu_seqlens[B+1] INT32, max_seqlen)."""
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return (
        torch.tensor(cu, dtype=torch.int32, device=device),
        max(lengths) if lengths else 0,
    )


def torch_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: list[int],
) -> torch.Tensor:
    """逐条切出来各做一次普通 causal GQA. 只用于对拍, 刻意不向量化.

    把"打包的寻址"和"attention 本身"解耦: kernel 出问题时能立刻分清是
    序列边界算错了, 还是 softmax 写错了.
    """
    num_q_head, d_model = q.shape[1], q.shape[2]
    num_kv_head = k.shape[1]
    group = num_q_head // num_kv_head
    outs = []
    start = 0
    for n in lengths:
        qs = q[start : start + n].permute(1, 0, 2).float()  # [H_q, n, D]
        ks = k[start : start + n].permute(1, 0, 2).float()  # [H_kv, n, D]
        vs = v[start : start + n].permute(1, 0, 2).float()
        ks = ks.repeat_interleave(group, dim=0)
        vs = vs.repeat_interleave(group, dim=0)
        scores = (qs @ ks.transpose(1, 2)) / math.sqrt(d_model)
        mask = torch.triu(
            torch.full((n, n), float("-inf"), device=q.device), diagonal=1
        )
        probs = torch.softmax(scores + mask, dim=-1)
        outs.append((probs @ vs).permute(1, 0, 2))  # [n, H_q, D]
        start += n
    return torch.cat(outs, dim=0).to(q.dtype)


if __name__ == "__main__":
    H_Q, H_KV, D = 8, 2, 256
    torch.manual_seed(0)
    print("=== 参考实现 vs SDPA(先确认基准本身是对的)===")
    for lengths in ([7], [16, 16], [3, 100, 1, 64]):
        total = sum(lengths)
        q = torch.randn(total, H_Q, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(total, H_KV, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(total, H_KV, D, dtype=torch.bfloat16, device="cuda")
        mine = torch_varlen_reference(q, k, v, lengths)
        # 逐条用 SDPA 验参考实现
        outs, start = [], 0
        for n in lengths:
            qs = q[start : start + n].permute(1, 0, 2).float().unsqueeze(0)
            ks = k[start : start + n].permute(1, 0, 2).float().unsqueeze(0)
            vs = v[start : start + n].permute(1, 0, 2).float().unsqueeze(0)
            ks = ks.repeat_interleave(H_Q // H_KV, dim=1)
            vs = vs.repeat_interleave(H_Q // H_KV, dim=1)
            o = torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
            outs.append(o[0].permute(1, 0, 2))
            start += n
        ref = torch.cat(outs, dim=0)
        e = (mine.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
        print(f"  lengths={str(lengths):<22} rel_err={e:.2e}")

    print("\n=== Triton kernel vs 参考实现 ===")
    for lengths in ([7], [16, 16], [1, 3, 17, 64], [3, 100, 1, 64], [1, 1, 1, 1000]):
        total = sum(lengths)
        q = torch.randn(total, H_Q, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(total, H_KV, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(total, H_KV, D, dtype=torch.bfloat16, device="cuda")
        cu, max_len = build_cu_seqlens(lengths)
        got = gqa_attention_varlen_causal(q, k, v, cu, max_len)
        ref = torch_varlen_reference(q, k, v, lengths)
        e = (got.float() - ref.float()).abs().max().item() / max(
            ref.float().abs().max().item(), 1e-9
        )
        ok = "ok" if e < 2e-2 else "! 误差偏大"
        print(f"  lengths={str(lengths):<22} total={total:<6} rel_err={e:.2e}  {ok}")

    print("\nAll varlen causal attention tests passed.")
