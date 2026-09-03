"""Qwen3.5-0.8B 24 层文本前向，全部用本仓库的 Triton kernel。

三条生成路径：

    generate()          不带 cache：每生成一个 token 就对增长后的整个序列重跑一次
                        forward，每 token O(T)、总共 O(T^2)。只用作对拍基准。
    generate_cached()   prefill 一次填好三类 cache，之后逐 token 增量 decode。
    generate_graphed()  同上，且把整个 decode step 捕获成一张 CUDA Graph。
                        demo 默认走这条，实测 42.0 -> 4.4 ms/token。

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
from triton_kernels.depthwise_causal_conv4_decode import (
    conv_state_from_prefill,
    depthwise_causal_conv4_decode,
)
from triton_kernels.gdn_recurrent_prefill import (
    call_gdn_recurrent_prefill_chunked_triton,
    gdn_recurrent_decode,
    gdn_recurrent_prefill_sequential,
)
from triton_kernels.gqa_attention_decode import (
    call_gqa_attention_decode_split_triton,
)
from engine.cache import DecodeCaches, allocate_caches
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

# GDN prefill 走 chunk-64 并行路径的最小 token 数，低于它用 sequential 递推。
# 依据见 `_gdn` 里的实测表。注意这个阈值**不能**照搬 kernel 级的交叉点。
GDN_CHUNKED_PREFILL_MIN_TOKENS = 2048


class Qwen35Runner:
    """完整重算前向。默认开 torch.compile。

    compile 的收益和代价（A100 实测）：

    - eager 41.3 ms/forward，compile 后 6.4 ms，约 6.5x。eager 下 92% 的时间是
      torch.library 的分发开销，跟 T 几乎无关（T=19 到 257 都是 41-45ms）。
    - 启动成本：冷缓存（每个代码版本第一次）约 50s + 65s；Inductor 磁盘缓存命中后
      每个进程约 2.3s + 2.5s ≈ 4.8s。之后 T 变化不再重编译，除非跨过 kernel 的
      T_BUCKET 边界（1/16/17/64/65/128/129 附近），那时会再编一次。
    - **盈亏平衡约 137 次 forward**（4.8s ÷ 每次省 35ms）。短生成反而更慢：
      32 token 大约 5.0s，eager 只要 1.3s。一次性跑几十个 token 的场景请传
      compile=False。
    - **compiled 与 eager 的结果不是 bit-identical**，相对差约 1.1%（BF16 量级）。
      这足以在 top-1/top-2 极接近时翻转 argmax。传 compile=False 可拿到与
      Hugging Face 逐 token 一致的那条路径，逐算子对拍必须用它。
    """

    def __init__(self, weights: TextWeights, compile: bool = True):
        self.w = weights
        self.eps = weights.rms_norm_eps
        self.device = weights.device
        self._compiled = torch.compile(self._forward) if compile else None
        self._compile_announced = not compile

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
        """[1,T] int32。纯文本下三个 MRoPE 轴相同，退化为普通 RoPE。

        **必须在 compile 区域之外调用**：这里按 Python 状态分支并可能重新分配，
        放进图里会让 T 每增长一次就 guard 失败重编译。按 2 的幂扩容，避免
        序列变长时频繁重分配。
        """
        if self._position_ids is None or self._position_ids.shape[1] < token_num:
            capacity = 1 << max(token_num - 1, 63).bit_length()
            self._position_ids = torch.arange(
                capacity, dtype=torch.int32, device=self.device
            ).unsqueeze(0)
        return self._position_ids[:, :token_num]

    # ---------------------------------------------------------------- mixers

    def _gdn(
        self,
        h: torch.Tensor,
        w: GDNLayerWeights,
        trace: dict | None,
        tag: str,
        fill: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
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

        # 两条 GDN prefill 路径，按 T 选。sequential 在 T 上是串行递推、时间线性
        # 增长；chunked 把 T 切成 64 的块并行处理，时间几乎是平的。
        #
        # **但 kernel 级的交叉点和模型级的交叉点差一个数量级**，这一条值得记住：
        #
        #     T       kernel 级（单层 GDN）      模型级（整个 prefill）
        #             seq      chunk            seq       chunk
        #      512   0.702ms  0.224ms  3.1x    43.1ms   49.4ms  0.87x
        #     1536      —        —              48.2ms   49.2ms  0.98x
        #     2048   2.801ms  0.682ms  4.1x    66.9ms   47.7ms  1.40x
        #     3072      —        —             106.0ms   48.3ms  2.20x
        #
        # 差异来自两处，都不在 kernel 里：
        #   1. chunked 是三个 kernel，sequential 是一个。18 个 GDN 层就是 36 次
        #      额外的 op 分发。
        #   2. prefill 在 T < 2048 时是 CPU 分发受限的（短 prompt 约 43ms 是个
        #      地板，与序列长度几乎无关），GPU 那点差距根本露不出来，反倒是这
        #      36 次分发被全额计入。
        #
        # 所以阈值按模型级的 2048 取，不是 kernel 级的 192。等 prefill 本身不再
        # 卡在 CPU 上（进 CUDA Graph 或让它走 compile 路径），这个阈值应该下调。
        if token_num >= GDN_CHUNKED_PREFILL_MIN_TOKENS:
            core, state = call_gdn_recurrent_prefill_chunked_triton(q_n, k_n, v, beta, g)
        else:
            core, state = gdn_recurrent_prefill_sequential(q_n, k_n, v, beta, g)

        normed = gdn_gated_rmsnorm(core, z.view(token_num, heads, head_dim), w.norm)
        out = gemm_2d(normed.view(token_num, key_dim), w.out_proj)  # [T,1024]

        if fill is not None:
            conv_state, recurrent_state = fill
            # conv state 取的是 conv 的**输入** qkv 的最后 4 行，不是 conv 输出。
            # T < 4 时 conv_state_from_prefill 会在上方补零。
            conv_state.copy_(conv_state_from_prefill(qkv))
            recurrent_state.copy_(state)

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
        self,
        h: torch.Tensor,
        w: AttnLayerWeights,
        pos: torch.Tensor,
        trace: dict | None,
        tag: str,
        fill: tuple[torch.Tensor, torch.Tensor] | None = None,
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

        if fill is not None:
            k_cache, v_cache = fill
            # 存 RoPE **之后**的 K；v4 是 permute 出来的 view，copy_ 会处理 stride
            k_cache[:, :token_num, :].copy_(k4[0])
            v_cache[:, :token_num, :].copy_(v4[0])

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

    # ------------------------------------------------------- decode 版 mixer
    #
    # 与 prefill 版逐行对应，只有三处不同：
    #   1. T 恒为 1，所有 [T,...] 变成 [1,...]
    #   2. conv 和 delta rule 换成 decode 版，从 cache 续算而不是从零重算
    #   3. attention 从 KV cache 读历史，位置由显存里的 pos 决定
    # 其余（投影、切分、norm、gate、o_proj）完全共用同一套 kernel 和布局约定。

    def _gdn_decode(
        self,
        h: torch.Tensor,
        w: GDNLayerWeights,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
    ):
        heads = self.w.linear_num_heads
        head_dim = self.w.linear_head_dim
        key_dim = heads * head_dim

        qkv = gemm_2d(h, w.in_proj_qkv)  # [1,6144]
        z = gemm_2d(h, w.in_proj_z)
        a = gemm_2d(h, w.in_proj_a)
        b = gemm_2d(h, w.in_proj_b)

        # conv_state 存的是 conv 的**输入**（in_proj_qkv 的输出），不是输出也不是
        # SiLU 之后的值。这个 op 会就地推进 state。权重要用 [4,D] 那份。
        conv = depthwise_causal_conv4_decode(qkv, conv_state, w.conv1d_decode)

        q = conv[:, 0:key_dim].view(1, heads, head_dim)
        k = conv[:, key_dim : 2 * key_dim].view(1, heads, head_dim)
        v = conv[:, 2 * key_dim : 3 * key_dim].view(1, heads, head_dim)

        q_n, k_n, beta, g = gdn_qk_norm_gates(q, k, a, b, w.a_log, w.dt_bias)
        # 同样就地推进 [16,128,128] 的 FP32 状态
        core = gdn_recurrent_decode(q_n, k_n, v, beta, g, recurrent_state)

        normed = gdn_gated_rmsnorm(core, z.view(1, heads, head_dim), w.norm)
        return gemm_2d(normed.view(1, key_dim), w.out_proj)

    def _attention_decode(
        self,
        h: torch.Tensor,
        w: AttnLayerWeights,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos: torch.Tensor,
        scratch,
    ):
        nh, nkv, d = (
            self.w.num_attention_heads,
            self.w.num_key_value_heads,
            self.w.head_dim,
        )

        q_raw = gemm_2d(h, w.q_proj_q)  # [1,2048]
        gate = gemm_2d(h, w.q_proj_gate)
        k_raw = gemm_2d(h, w.k_proj)  # [1,512]
        v_raw = gemm_2d(h, w.v_proj)

        q = qwen_rmsnorm(q_raw.view(1, nh, d), w.q_norm, self.eps)
        k = qwen_rmsnorm(k_raw.view(1, nkv, d), w.k_norm, self.eps)

        # RoPE 的 position 就是 pos 本身（新 token 的下标 = 已缓存的数量）。
        # 传显存张量而不是 python int——CUDA Graph 下标量会被冻结。
        pos_ids = pos.view(1, 1)
        q4 = partial_rope(
            q.unsqueeze(0).permute(0, 2, 1, 3), pos_ids, self.inv_freq, self.w.rotary_dim
        )
        k4 = partial_rope(
            k.unsqueeze(0).permute(0, 2, 1, 3), pos_ids, self.inv_freq, self.w.rotary_dim
        )

        # 存进 cache 的必须是 RoPE **之后**的 K——历史 token 的 position 不会变，
        # 每步重新旋转是错的。
        ctx = call_gqa_attention_decode_split_triton(
            q4[0, :, 0, :].contiguous(),
            k4[0, :, 0, :].contiguous(),
            v_raw.view(nkv, d).contiguous(),
            k_cache,
            v_cache,
            pos,
            scratch,
        )  # [H_q, D]

        # T=1 时 pack 退化成逐元素乘 + 重排，仍复用 prefill 那个 kernel
        gate4 = gate.view(1, 1, nh, d).permute(0, 2, 1, 3)  # [1,H,1,D]
        packed = attention_gate_pack(ctx.view(1, nh, 1, d), gate4)  # [1,2048]
        return gemm_2d(packed, w.o_proj)

    def decode_step(
        self,
        token_id: torch.Tensor,
        caches: DecodeCaches,
        trace: dict | None = None,
    ) -> torch.Tensor:
        """单 token 前向。token_id: [1] int32；返回 final norm 之后的 [1,1024]。

        **不推进 pos**——推进放在调用方（或 CUDA Graph 末尾），因为 attention
        需要"写入位置 = 当前 pos"，推进必须在整个 forward 之后。
        """
        x = embedding_gather(token_id, self.w.embed_tokens)  # [1,1024]

        for i, layer in enumerate(self.w.layers):
            residual = x
            h = qwen_rmsnorm(x, layer.input_layernorm, self.eps)
            if isinstance(layer, GDNLayerWeights):
                h = self._gdn_decode(
                    h, layer, caches.conv_states[i], caches.recurrent_states[i]
                )
            else:
                h = self._attention_decode(
                    h,
                    layer,
                    caches.k_caches[i],
                    caches.v_caches[i],
                    caches.pos,
                    caches.split_scratch,
                )
            x = residual_add(h, residual)

            residual = x
            h = qwen_rmsnorm(x, layer.post_attention_layernorm, self.eps)
            h = swiglu(gemm_2d(h, layer.mlp.gate_proj), gemm_2d(h, layer.mlp.up_proj))
            h = gemm_2d(h, layer.mlp.down_proj)
            x = residual_add(h, residual)

            if trace is not None:
                trace[f"layer{i:02d}.out"] = x

        return qwen_rmsnorm(x, self.w.norm, self.eps)

    # ---------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        trace: dict | None = None,
        trace_layers: set[int] | None = None,
    ) -> torch.Tensor:
        """input_ids: [T] int32/int64 -> final norm 之后的 hidden [T,1024] BF16。

        传了 trace 就走 eager——往 dict 里塞中间量没法编译，而且逐算子对拍本来
        就要的是与 Hugging Face 一致的那条路径。
        """
        assert input_ids.ndim == 1
        pos = self._positions(input_ids.shape[0])

        if trace is None and self._compiled is not None:
            if not self._compile_announced:
                self._compile_announced = True
                print(
                    "[runner] torch.compile 启动中：热缓存约 4.8s，冷缓存约 2 分钟。"
                    "之后 6.4ms/forward（eager 41ms）。生成不到 ~137 个 token 时"
                    "eager 更快，传 compile=False 即可。",
                    file=sys.stderr,
                    flush=True,
                )
            return self._compiled(input_ids, pos, None, None)
        return self._forward(input_ids, pos, trace, trace_layers)

    def _forward(
        self,
        input_ids: torch.Tensor,
        pos: torch.Tensor,
        trace: dict | None,
        trace_layers: set[int] | None,
        caches: DecodeCaches | None = None,
    ) -> torch.Tensor:
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
                h = self._gdn(
                    h, layer, inner, tag,
                    fill=None if caches is None
                    else (caches.conv_states[i], caches.recurrent_states[i]),
                )
            else:
                h = self._attention(
                    h, layer, pos, inner, tag,
                    fill=None if caches is None
                    else (caches.k_caches[i], caches.v_caches[i]),
                )
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


    # -------------------------------------------------- prefill + decode 路径

    def prefill(self, input_ids: torch.Tensor, caches: DecodeCaches) -> torch.Tensor:
        """整段 prefill，顺带把三类 cache 填好。返回 final norm 后的 [T,1024]。

        走的就是完整重算那条路径，只是每层多写一次 cache——所以 prefill 的数值
        与 `forward()` 完全一致，不需要额外对拍。

        **必须先 reset**：cache 里可能有上一轮的残留。
        """
        assert input_ids.ndim == 1
        token_num = input_ids.shape[0]
        assert token_num <= caches.max_len, (
            f"prompt 长度 {token_num} 超出 cache 容量 {caches.max_len}"
        )
        pos = self._positions(token_num)
        hidden = self._forward(input_ids, pos, None, None, caches=caches)
        # 位置推进到"已缓存 token_num 个"，下一个 decode 写在下标 token_num
        caches.pos.fill_(token_num)
        return hidden

    def generate_cached(
        self,
        input_ids: list[int] | torch.Tensor,
        max_new_tokens: int = 32,
        stop_ids: tuple[int, ...] | None = DEFAULT_STOP_IDS,
        max_len: int | None = None,
        caches: DecodeCaches | None = None,
    ) -> list[int]:
        """prefill 一次 + 逐 token 增量 decode。

        与 `generate()`（每步对整个序列重算）的区别只在于用不用 cache；两者的
        greedy 结果应当一致，`tests/test_decode_parity.py` 就是对拍这一点。
        """
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=self.device)
        ids = input_ids.to(torch.int32)
        prompt_len = ids.shape[0]

        if caches is None:
            need = max_len or (prompt_len + max_new_tokens + 8)
            caches = allocate_caches(self.w, need)
        assert prompt_len + max_new_tokens <= caches.max_len, (
            f"prompt {prompt_len} + 生成 {max_new_tokens} 超出 cache 容量 "
            f"{caches.max_len}"
        )
        caches.reset()

        hidden = self.prefill(ids, caches)
        token = int(lm_head_argmax(hidden, self.w.embed_tokens).item())

        generated = [token]
        slot = torch.empty(1, dtype=torch.int32, device=self.device)
        for _ in range(max_new_tokens - 1):
            if stop_ids and token in stop_ids:
                break
            slot.fill_(token)
            hidden = self.decode_step(slot, caches)
            # pos 的推进放在 forward 之后：attention 需要"写入位置 = 当前 pos"
            caches.pos.add_(1)
            token = int(lm_head_argmax(hidden, self.w.embed_tokens).item())
            generated.append(token)
        return generated

    def generate_graphed(
        self,
        input_ids: list[int] | torch.Tensor,
        max_new_tokens: int = 32,
        stop_ids: tuple[int, ...] | None = DEFAULT_STOP_IDS,
        max_len: int | None = None,
        decoder: "GraphedDecoder | None" = None,
    ) -> list[int]:
        """prefill + CUDA Graph 化的增量 decode。结果与 generate_cached 一致。

        decoder 传 None 时会当场分配 cache 并捕获一次图；反复调用应该复用同一个
        GraphedDecoder，捕获只需一次（捕获本身要跑几次 warmup，不便宜）。
        """
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=self.device)
        ids = input_ids.to(torch.int32)

        if decoder is None:
            need = max_len or (ids.shape[0] + max_new_tokens + 8)
            caches = allocate_caches(self.w, need)
            decoder = GraphedDecoder(self, caches)
            # 捕获前先 prefill，让 warmup 跑在合法的 cache 状态上
            caches.reset()
            self.prefill(ids, caches)
            decoder.capture()

        # host 侧提前拦住：越界的话 index_copy_ 会在 device 上 assert，
        # 报错点离真正原因很远（"CUDA error: device-side assert triggered"），很难查。
        need_len = ids.shape[0] + max_new_tokens
        assert need_len <= decoder.caches.max_len, (
            f"prompt {ids.shape[0]} + 生成 {max_new_tokens} = {need_len} "
            f"超出 cache 容量 {decoder.caches.max_len}；"
            f"重新分配一个更大的 GraphedDecoder，或减少 max_new_tokens"
        )

        # capture 或上一轮生成把 cache 写脏了，这里必须重新来过
        decoder.caches.reset()
        hidden = self.prefill(ids, decoder.caches)
        token = int(lm_head_argmax(hidden, self.w.embed_tokens).item())

        generated = [token]
        for _ in range(max_new_tokens - 1):
            if stop_ids and token in stop_ids:
                break
            token = decoder.step(token)
            generated.append(token)
        return generated



class GraphedDecoder:
    """把一整个 decode step 包进 CUDA Graph：24 层前向 + argmax + pos 自增。

    为什么值得做
    ------------
    eager 下一步 decode 有约 400 次 op 调用，每次约 90us 的 torch.library 分发开销
    GPU 实际工作只有几毫秒。实测：

        eager 逐步                    42.04 ms/token
        graph replay                   4.41 ms/token    9.5x
        graph replay + 每步 .item()    4.24 ms/token    9.9x

    **每步 .item() 的同步开销测下来是 0**（在噪声内）——GPU 那 4.4ms 的工作足够长，
    同步完全被掩盖。所以不需要"批量 replay N 步再统一读回 token"那种取舍，
    每步照常判停止条件即可。

    图内闭环
    --------
    关键是 `tok_slot.copy_(tok_out)`：把本步 argmax 的结果直接写回输入槽。
    于是连续 replay 就自动逐 token 生成，host 侧每步只需要 `graph.replay()`
    加一次读回来判停止。位置也在图内 `pos.add_(1)` 自增，同样不用 host 介入。

    有状态，所以必须 reset
    ----------------------
    capture 过程本身（warmup + 正式捕获）会执行若干次 one_step，把 pos 推进、
    把三类 cache 写脏。所以 `capture()` 之后、每次新 prompt 之前都必须重新
    prefill。本类把这个约束封进 `run()`，调用方不用记。

    局限
    ----
    - autotune 选中的 config 冻结在 capture 那一刻。实测代价很小：固定 seq_len、
      只改"冻结了哪个 config"时差异是 2.7%（短 prompt 下捕获、seq=4095 时 replay：
      15.72 -> 16.15us），而 attention decode 只占一步的 2.2%。所以单张图够用；
      要更好就每个 seq_bucket 捕获一张并共享内存池。
      别拿"不同 seq_len 之间的耗时跨度"当这个代价——那里面大部分是序列变长
      本身的成本，多少张图都省不掉。
    - grid 也冻结，但本项目的 decode kernel grid 全部与 seq_len 无关，天然满足。
    """

    def __init__(self, runner: "Qwen35Runner", caches: DecodeCaches):
        self.runner = runner
        self.caches = caches
        dev = runner.device
        # 输入输出槽都在图外分配，地址固定；图只录它们的地址常量
        self.tok_slot = torch.zeros(1, dtype=torch.int32, device=dev)
        self.tok_out = torch.zeros((), dtype=torch.int64, device=dev)
        self.graph: torch.cuda.CUDAGraph | None = None

    def _one_step(self) -> None:
        hidden = self.runner.decode_step(self.tok_slot, self.caches)
        self.tok_out.copy_(lm_head_argmax(hidden, self.runner.w.embed_tokens))
        # pos 必须在 forward 之后推进：attention 要"写入位置 = 当前 pos"
        self.caches.pos.add_(1)
        self.tok_slot.copy_(self.tok_out)  # 闭环：本步输出即下步输入

    def capture(self, warmup: int = 3) -> None:
        """捕获一次。之后 caches 处于脏状态，调用方必须重新 prefill。"""
        for _ in range(warmup):
            self._one_step()
        torch.cuda.synchronize()
        # capture 必须在 side stream 上预热，否则 CUDA 会拒绝
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                self._one_step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._one_step()

    def step(self, prev_token: int | None = None) -> int:
        """replay 一步。prev_token 传 None 表示沿用图内闭环写回的值。"""
        assert self.graph is not None, "先调用 capture()"
        if prev_token is not None:
            self.tok_slot.fill_(prev_token)
        self.graph.replay()
        return int(self.tok_out.item())


def build_runner(
    model_dir: str = "Qwen3.5-0.8B",
    device: str = "cuda",
    compile: bool = True,
) -> Qwen35Runner:
    return Qwen35Runner(load_text_weights(model_dir, device), compile=compile)


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
