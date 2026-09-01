"""Qwen3.5-0.8B 24 层文本前向，全部用本仓库的 Triton kernel。

当前策略是**完整重算**：每生成一个 token，就对增长后的整个序列重跑一次 forward。
GDN 的 recurrent state 在单次 forward 内从零开始按 token 顺序推进，forward 之间不保留。
KV cache、conv state cache 和增量 decode 等 kernel 齐了之后再切（见 HANDOFF.md 第 8 节）。

布局要点（踩过的坑都在这）：

- `q_proj` 已在 loader 里拆成 Q / gate 两个权重，所以这里两个 GEMM 输出都是连续的
  [T,2048]，view(T,8,256) 也连续，能直接喂给要求 contiguous 的 qwen_rmsnorm。
- `attention_gate_pack` 的两个参数都按 [B,H,T,D] 索引。gate 的内存布局是 [B,T,H,D]，
  所以要 permute 成 [B,H,T,D] 的 view 传进去，而不是直接传 [B,T,H,D] 形状的张量。
- GDN 的 conv 输出按 [0:2048|2048:4096|4096:6144] 连续三段切 q/k/v。切片后是 strided
  view，除 qwen_rmsnorm 外的 kernel 都 stride-aware，不需要 contiguous 拷贝。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.loader import AttnLayerWeights, GDNLayerWeights, TextWeights, load_text_weights
from triton_kernels.attention_gate_pack import attention_gate_pack
from triton_kernels.depthwise_causal_conv4_prefill import depthwise_causal_conv4_prefill
from triton_kernels.embedding_gather import embedding_gather
from triton_kernels.gdn_gated_rmsnorm import gdn_gated_rmsnorm
from triton_kernels.gdn_qk_norm_gates import gdn_qk_norm_gates
from triton_kernels.gdn_recurrent_prefill import gdn_recurrent_prefill_sequential
from triton_kernels.gemm_2d import gemm_2d
from triton_kernels.gqa_attention_without_kvcache_casual import (
    gqa_attention_without_kvcache_casual,
)
from triton_kernels.partial_rope import partial_rope
from triton_kernels.qwen_rmsnorm import qwen_rmsnorm
from triton_kernels.residual_add import residual_add
from triton_kernels.swiglu import swiglu
from triton_kernels.vocab_argmax import lm_head_argmax


# tokenizer_config 把 <|im_end|> 设为 EOS，text config 里还有 <|endoftext|>。
# 停止 token 由调用层决定，这里只给默认值。
DEFAULT_STOP_IDS = (248046, 248044)


class Qwen35Runner:
    def __init__(self, weights: TextWeights):
        self.w = weights
        self.eps = weights.rms_norm_eps
        self.device = weights.device

        # inv_freq[j] = 1 / theta^(2j/R)，与参考实现的 compute_default_rope_parameters 一致。
        r = weights.rotary_dim
        self.inv_freq = (
            1.0
            / (
                weights.rope_theta
                ** (torch.arange(0, r, 2, dtype=torch.float32, device=self.device) / r)
            )
        ).contiguous()
        assert self.inv_freq.shape == (r // 2,)

        self._position_ids: torch.Tensor | None = None

    def _positions(self, token_num: int) -> torch.Tensor:
        """[1,T] int32。纯文本下三个 MRoPE 轴相同，退化为普通 RoPE。"""
        if self._position_ids is None or self._position_ids.shape[1] < token_num:
            self._position_ids = torch.arange(
                max(token_num, 64), dtype=torch.int32, device=self.device
            ).unsqueeze(0)
        return self._position_ids[:, :token_num].contiguous()

    # ---------------------------------------------------------------- mixers

    def _gdn(self, h: torch.Tensor, w: GDNLayerWeights, trace: dict | None, tag: str):
        token_num = h.shape[0]
        heads = self.w.linear_num_heads
        head_dim = self.w.linear_head_dim
        key_dim = heads * head_dim

        qkv = gemm_2d(h, w.in_proj_qkv)  # [T,6144]
        z = gemm_2d(h, w.in_proj_z)  # [T,2048]
        a = gemm_2d(h, w.in_proj_a)  # [T,16]
        b = gemm_2d(h, w.in_proj_b)  # [T,16]

        conv = depthwise_causal_conv4_prefill(qkv, w.conv1d)  # [T,6144]，含 SiLU

        # 连续三段切 q|k|v，各自 view 成 [T,H,D]。切片是 strided view，
        # 但最后一维连续，view 合法；下游 kernel 都 stride-aware。
        q = conv[:, 0:key_dim].view(token_num, heads, head_dim)
        k = conv[:, key_dim : 2 * key_dim].view(token_num, heads, head_dim)
        v = conv[:, 2 * key_dim : 3 * key_dim].view(token_num, heads, head_dim)

        q_n, k_n, beta, g = gdn_qk_norm_gates(q, k, a, b, w.a_log, w.dt_bias)
        core, state = gdn_recurrent_prefill_sequential(q_n, k_n, v, beta, g)

        normed = gdn_gated_rmsnorm(core, z.view(token_num, heads, head_dim), w.norm)
        out = gemm_2d(normed.view(token_num, key_dim), w.out_proj)  # [T,1024]

        if trace is not None:
            trace[f"{tag}.in_proj_qkv"] = qkv
            trace[f"{tag}.in_proj_z"] = z
            trace[f"{tag}.in_proj_a"] = a
            trace[f"{tag}.in_proj_b"] = b
            trace[f"{tag}.conv"] = conv
            trace[f"{tag}.q_pre_norm"] = q
            trace[f"{tag}.k_pre_norm"] = k
            trace[f"{tag}.v"] = v
            trace[f"{tag}.q_norm"] = q_n
            trace[f"{tag}.k_norm"] = k_n
            trace[f"{tag}.beta"] = beta
            trace[f"{tag}.g"] = g
            trace[f"{tag}.core_attn_out"] = core
            trace[f"{tag}.final_state"] = state
            trace[f"{tag}.gated_norm"] = normed
            trace[f"{tag}.out_proj"] = out
        return out

    def _attention(
        self, h: torch.Tensor, w: AttnLayerWeights, trace: dict | None, tag: str
    ):
        token_num = h.shape[0]
        nh = self.w.num_attention_heads
        nkv = self.w.num_key_value_heads
        d = self.w.head_dim

        q_raw = gemm_2d(h, w.q_proj_q)  # [T,2048] 连续
        gate = gemm_2d(h, w.q_proj_gate)  # [T,2048] 连续
        k_raw = gemm_2d(h, w.k_proj)  # [T,512]
        v = gemm_2d(h, w.v_proj)  # [T,512]

        # view 后连续，满足 qwen_rmsnorm 的 contiguous 要求。
        q = qwen_rmsnorm(q_raw.view(token_num, nh, d), w.q_norm, self.eps)
        k = qwen_rmsnorm(k_raw.view(token_num, nkv, d), w.k_norm, self.eps)

        pos = self._positions(token_num)
        # [T,H,D] -> [1,H,T,D]
        q4 = partial_rope(
            q.unsqueeze(0).permute(0, 2, 1, 3), pos, self.inv_freq, self.w.rotary_dim
        )
        k4 = partial_rope(
            k.unsqueeze(0).permute(0, 2, 1, 3), pos, self.inv_freq, self.w.rotary_dim
        )
        v4 = v.view(token_num, nkv, d).unsqueeze(0).permute(0, 2, 1, 3)

        ctx = gqa_attention_without_kvcache_casual(q4, k4, v4)  # [1,H,T,D]

        # gate 内存布局是 [T,H,D]；permute 成 [1,H,T,D] 的 view 再传。
        gate4 = gate.view(1, token_num, nh, d).permute(0, 2, 1, 3)
        packed = attention_gate_pack(ctx, gate4)  # [T,2048] 连续
        out = gemm_2d(packed, w.o_proj)  # [T,1024]

        if trace is not None:
            trace[f"{tag}.q_proj_q"] = q_raw
            trace[f"{tag}.q_proj_gate"] = gate
            trace[f"{tag}.k_proj"] = k_raw
            trace[f"{tag}.v_proj"] = v
            trace[f"{tag}.q_after_norm"] = q
            trace[f"{tag}.k_after_norm"] = k
            trace[f"{tag}.rope_q"] = q4
            trace[f"{tag}.rope_k"] = k4
            trace[f"{tag}.ctx"] = ctx
            trace[f"{tag}.packed"] = packed
            trace[f"{tag}.out_proj"] = out
        return out

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        trace: dict | None = None,
        trace_layers: set[int] | None = None,
    ) -> torch.Tensor:
        """input_ids: [T] int32/int64 -> final norm 之后的 hidden [T,1024] BF16。"""
        assert input_ids.ndim == 1
        if trace_layers is None:
            trace_layers = {0, 3}

        x = embedding_gather(input_ids, self.w.embed_tokens)  # [T,1024]
        if trace is not None:
            trace["embed"] = x

        for i, layer in enumerate(self.w.layers):
            tag = f"layer{i:02d}"
            inner = trace if (trace is not None and i in trace_layers) else None

            residual = x
            h = qwen_rmsnorm(x, layer.input_layernorm, self.eps)
            if inner is not None:
                inner[f"{tag}.input_layernorm"] = h

            if isinstance(layer, GDNLayerWeights):
                h = self._gdn(h, layer, inner, tag)
            else:
                h = self._attention(h, layer, inner, tag)
            x = residual_add(h, residual)

            residual = x
            h = qwen_rmsnorm(x, layer.post_attention_layernorm, self.eps)
            if inner is not None:
                inner[f"{tag}.post_attention_layernorm"] = h
            h = swiglu(gemm_2d(h, layer.mlp.gate_proj), gemm_2d(h, layer.mlp.up_proj))
            h = gemm_2d(h, layer.mlp.down_proj)
            x = residual_add(h, residual)

            if trace is not None:
                trace[f"{tag}.out"] = x

        x = qwen_rmsnorm(x, self.w.norm, self.eps)
        if trace is not None:
            trace["final_norm"] = x
        return x

    def next_token(self, input_ids: torch.Tensor) -> int:
        """完整重算一次，返回下一个 greedy token。"""
        hidden = self.forward(input_ids)
        return int(lm_head_argmax(hidden, self.w.embed_tokens).item())

    def generate(
        self,
        input_ids: list[int] | torch.Tensor,
        max_new_tokens: int = 32,
        stop_ids: tuple[int, ...] | None = DEFAULT_STOP_IDS,
        verbose: bool = False,
    ) -> list[int]:
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=self.device)
        ids = input_ids.to(torch.int32)

        generated: list[int] = []
        for step in range(max_new_tokens):
            token = self.next_token(ids)
            generated.append(token)
            if verbose:
                print(f"  step {step:3d}  token {token}", flush=True)
            if stop_ids and token in stop_ids:
                break
            ids = torch.cat(
                [ids, torch.tensor([token], dtype=torch.int32, device=self.device)]
            )
        return generated


def build_runner(model_dir: str = "Qwen3.5-0.8B", device: str = "cuda") -> Qwen35Runner:
    return Qwen35Runner(load_text_weights(model_dir, device))


if __name__ == "__main__":
    import time

    from tokenizers import Tokenizer

    runner = build_runner()

    prompt = "你好，请简单介绍一下自己。"
    rendered = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    tokenizer = Tokenizer.from_file("Qwen3.5-0.8B/tokenizer.json")
    ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    print(f"prompt tokens: {len(ids)}")

    start = time.time()
    generated = runner.generate(ids, max_new_tokens=32)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"generated ({elapsed:.1f}s, {elapsed / max(len(generated), 1):.2f}s/token):")
    print(generated)
    print(tokenizer.decode(generated, skip_special_tokens=False))
