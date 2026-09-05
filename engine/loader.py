"""Qwen3.5-0.8B 文本主干权重加载器。

只加载 `model.language_model.*`，显式跳过 `model.visual.*` 和 `mtp.*`。
张量名、形状和 dtype 已对照 model.safetensors.index.json 核实。

两处不是原样搬运的地方：

1. `q_proj.weight [4096,1024]` 拆成 `q_proj_q` 和 `q_proj_gate` 各 [2048,1024]。
   checkpoint 里 Q 和 gate 是按 head 交错的（第 h 个 head 占 [h*512, h*512+512)，
   前 256 行是 Q、后 256 行是 gate）。不拆的话，运行时切出来的 Q 是 strided view，
   而 qwen_rmsnorm 要求 contiguous。拆开后两个输出都是连续的 [T,2048]，
   view(T,8,256) 也连续，现有 kernel 一行不用改。

2. `conv1d.weight [6144,1,4]` 存两份：prefill 用 `[6144,4]`（squeeze 后仍连续），
   decode 用 `[4,6144]` contiguous。后者是 `depthwise_causal_conv4_decode` 要的布局
   ——decode 时一个线程负责一个 channel，`[4,D]` 下相邻线程地址连续、访存合并，
   `[D,4]` 下相隔 4 个元素。必须是真正 contiguous 的转置结果，不能是 view，
   否则内存布局没变、合并的好处全没了。多占 18 层 × 48 KiB = 0.86 MiB。
   详见 `triton_kernels/depthwise_causal_conv4_decode.py`。

A_log 和 linear_attn.norm.weight 在 checkpoint 里就是 FP32，这里断言而不是转换——
如果哪天上游改成 BF16 存，静默 cast 会掩盖问题。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


TEXT_PREFIX = "model.language_model."
SKIP_PREFIXES = ("model.visual.", "mtp.")

# 来自 README.md 1.3 的校验值。
EXPECTED_TEXT_PARAMS = 752_393_024
EXPECTED_NUM_LAYERS = 24
EXPECTED_NUM_GDN_LAYERS = 18
EXPECTED_NUM_ATTN_LAYERS = 6
FULL_ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23)


def _fuse_rows(parts: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """把若干个 `[N_i, K]` 的权重沿输出维拼成一块，返回整块和指向它的行 view。

    为什么要拼
    ----------
    decode 时 M 恒为 1，这些 GEMV 共享同一个输入 h，所以
    `y_i = h @ W_i.T` 可以合成 `Y = h @ cat(W_i).T` 再切输出，数学上是恒等的。
    收益不是省字节也不是省 FLOPs（两者完全不变），而是**每消掉一个 kernel 就省
    约 4us 的固定成本**：其中约 1.9us 是 GPU 侧的 grid 分发与完成信号（CUDA Graph
    也压不掉），其余约 2us 是 ramp-up 和 drain——kernel 开头第一次 K 循环的访存没有
    任何东西可以与之重叠，结尾则是最后几个 CTA 收尾时 SM 空转，而同一 stream 里
    相邻 kernel 之间有隐式屏障，前一个的 drain 没法和后一个的 ramp 重叠。

    实测每组的收益都严格等于「消掉的 kernel 数 × 约 4us」，与被消掉的是大矩阵还是
    小矩阵无关：GDN 消 3 个省 11.92us，attn 消 3 个省 13.33us，MLP 消 1 个省 3.89us。

    为什么返回 view 而不是留着原张量
    --------------------------------
    **按行切一块 `[N,K]` 的连续张量，得到的仍然是连续张量**（stride 保持 (K,1)）。
    所以 prefill 那条路径继续用这些 view 时，拿到的东西和融合前完全一样，代码一行
    不用改，也不多占一点显存。

    反过来，**输出的切片就只在 M=1 时连续**：`[1,8224][:, 0:6144]` 的 stride 是
    (8224,1)，但长度为 1 的维不参与连续性判定，所以是连续的；`[64,8224]` 同样切法
    就不是了。而 `qwen_rmsnorm` 是全仓库唯一要求 contiguous 的 kernel，attn 的
    q_norm/k_norm 正好用它。这就是融合只用在 decode 路径、prefill 保持原样的原因。
    """
    fused = torch.cat(parts, dim=0).contiguous()
    views: list[torch.Tensor] = []
    offset = 0
    for part in parts:
        views.append(fused[offset : offset + part.shape[0]])
        offset += part.shape[0]
    return fused, views


@dataclass(frozen=True)
class MLPWeights:
    gate_proj: torch.Tensor  # [3584,1024] BF16，gate_up 的行 view
    up_proj: torch.Tensor  # [3584,1024] BF16，gate_up 的行 view
    down_proj: torch.Tensor  # [1024,3584] BF16
    gate_up: torch.Tensor  # [7168,1024] BF16，decode 用整块打一次 GEMV


@dataclass(frozen=True)
class GDNLayerWeights:
    """Gated DeltaNet 层，18 个。"""

    input_layernorm: torch.Tensor  # [1024] BF16
    post_attention_layernorm: torch.Tensor  # [1024] BF16
    # 下面四个都是 in_proj_fused 的行 view，prefill 用；decode 用整块
    in_proj_qkv: torch.Tensor  # [6144,1024] BF16
    in_proj_z: torch.Tensor  # [2048,1024] BF16
    in_proj_a: torch.Tensor  # [16,1024] BF16
    in_proj_b: torch.Tensor  # [16,1024] BF16
    in_proj_fused: torch.Tensor  # [8224,1024] BF16 = cat(qkv, z, a, b)
    conv1d: torch.Tensor  # [6144,4] BF16，prefill 用
    conv1d_decode: torch.Tensor  # [4,6144] BF16 contiguous，decode 用
    a_log: torch.Tensor  # [16] FP32
    dt_bias: torch.Tensor  # [16] BF16
    norm: torch.Tensor  # [128] FP32，direct-weight gated RMSNorm
    out_proj: torch.Tensor  # [1024,2048] BF16
    mlp: MLPWeights

    layer_type: str = "linear_attention"


@dataclass(frozen=True)
class AttnLayerWeights:
    """Full attention 层，6 个（层号 3/7/11/15/19/23）。"""

    input_layernorm: torch.Tensor  # [1024] BF16
    post_attention_layernorm: torch.Tensor  # [1024] BF16
    # 下面四个都是 qkvg_fused 的行 view，prefill 用；decode 用整块
    q_proj_q: torch.Tensor  # [2048,1024] BF16，从 q_proj 拆出的 Q 部分
    q_proj_gate: torch.Tensor  # [2048,1024] BF16，从 q_proj 拆出的 gate 部分
    k_proj: torch.Tensor  # [512,1024] BF16
    v_proj: torch.Tensor  # [512,1024] BF16
    qkvg_fused: torch.Tensor  # [5120,1024] BF16 = cat(q, gate, k, v)
    q_norm: torch.Tensor  # [256] BF16
    k_norm: torch.Tensor  # [256] BF16
    o_proj: torch.Tensor  # [1024,2048] BF16
    mlp: MLPWeights

    layer_type: str = "full_attention"


@dataclass(frozen=True)
class TextWeights:
    embed_tokens: torch.Tensor  # [248320,1024] BF16，同时作为 LM head
    norm: torch.Tensor  # [1024] BF16
    layers: tuple[GDNLayerWeights | AttnLayerWeights, ...]

    hidden_size: int
    num_layers: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    rotary_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    linear_num_heads: int
    linear_head_dim: int

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.device


def _split_q_proj(q_proj: torch.Tensor, num_heads: int, head_dim: int):
    """[4096,1024] -> ([2048,1024] Q, [2048,1024] gate)，两者都 contiguous。

    行布局是 head-major：第 h 个 head 的 Q 占行 [h*2D, h*2D+D)，gate 占 [h*2D+D, (h+1)*2D)。
    拆完保持 head 顺序，因此输出 [T,2048] view(T,H,D) 与参考实现的 [T,H,D] 一致。
    """
    out_features, in_features = q_proj.shape
    assert out_features == num_heads * head_dim * 2, (
        f"q_proj 行数 {out_features} != num_heads*head_dim*2 = {num_heads * head_dim * 2}"
    )
    per_head = q_proj.view(num_heads, head_dim * 2, in_features)
    q = per_head[:, :head_dim, :].reshape(num_heads * head_dim, in_features)
    gate = per_head[:, head_dim:, :].reshape(num_heads * head_dim, in_features)
    return q.contiguous(), gate.contiguous()


class _TensorSource:
    """按名字取张量，并记录读过哪些、跳过哪些，便于收尾断言。"""

    def __init__(self, model_dir: Path, device: str):
        self.model_dir = model_dir
        self.device = device
        index_path = model_dir / "model.safetensors.index.json"
        self.weight_map: dict[str, str] = json.loads(
            index_path.read_text(encoding="utf-8")
        )["weight_map"]
        self._handles: dict[str, object] = {}
        self.read_keys: set[str] = set()

    def _handle(self, shard: str):
        if shard not in self._handles:
            path = self.model_dir / shard
            self._handles[shard] = safe_open(
                str(path), framework="pt", device=self.device
            )
        return self._handles[shard]

    def get(
        self,
        name: str,
        *,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert name in self.weight_map, f"checkpoint 里没有 {name}"
        tensor = self._handle(self.weight_map[name]).get_tensor(name)
        assert tuple(tensor.shape) == shape, (
            f"{name} 形状 {tuple(tensor.shape)} != 预期 {shape}"
        )
        assert tensor.dtype == dtype, (
            f"{name} dtype {tensor.dtype} != 预期 {dtype}；"
            "不要静默 cast，先确认 checkpoint 是否变了"
        )
        self.read_keys.add(name)
        return tensor

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()


def _load_mlp(src: _TensorSource, prefix: str, hidden: int, inter: int) -> MLPWeights:
    bf16 = torch.bfloat16
    # gate 和 up 吃同一个输入，融合成 [7168,1024] 一次算完；down 的输入是 swiglu
    # 的输出，融合不进来。
    gate_up, (gate, up) = _fuse_rows([
        src.get(f"{prefix}mlp.gate_proj.weight", shape=(inter, hidden), dtype=bf16),
        src.get(f"{prefix}mlp.up_proj.weight", shape=(inter, hidden), dtype=bf16),
    ])
    return MLPWeights(
        gate_proj=gate,
        up_proj=up,
        down_proj=src.get(f"{prefix}mlp.down_proj.weight", shape=(hidden, inter), dtype=bf16),
        gate_up=gate_up,
    )


def load_text_weights(
    model_dir: str | Path = "Qwen3.5-0.8B",
    device: str = "cuda",
) -> TextWeights:
    """加载文本主干。视觉塔和 MTP 完全不读，不占显存。"""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)

    hidden = text_config["hidden_size"]
    inter = text_config["intermediate_size"]
    num_layers = text_config["num_hidden_layers"]
    vocab_size = text_config["vocab_size"]
    num_heads = text_config["num_attention_heads"]
    num_kv_heads = text_config["num_key_value_heads"]
    head_dim = text_config["head_dim"]
    lin_heads = text_config["linear_num_value_heads"]
    lin_head_dim = text_config["linear_value_head_dim"]
    layer_types = text_config["layer_types"]

    rope = text_config.get("rope_scaling") or text_config.get("rope_parameters") or {}
    rope_theta = rope.get("rope_theta", text_config.get("rope_theta"))
    partial = rope.get("partial_rotary_factor", text_config.get("partial_rotary_factor", 1.0))
    rotary_dim = int(head_dim * partial)

    assert num_layers == EXPECTED_NUM_LAYERS, f"层数 {num_layers} != {EXPECTED_NUM_LAYERS}"
    assert len(layer_types) == num_layers
    assert text_config["linear_num_key_heads"] == lin_heads, (
        "linear K/V head 数不同，需要 repeat_interleave；当前 runner 假定比值为 1"
    )
    assert text_config["tie_word_embeddings"] is True, "LM head 应与 embedding 共享"

    bf16, fp32 = torch.bfloat16, torch.float32
    src = _TensorSource(model_dir, device)

    assert not any("lm_head" in k for k in src.weight_map), (
        "checkpoint 出现了独立 lm_head，与 tie_word_embeddings 矛盾"
    )

    try:
        embed_tokens = src.get(
            f"{TEXT_PREFIX}embed_tokens.weight", shape=(vocab_size, hidden), dtype=bf16
        )
        final_norm = src.get(f"{TEXT_PREFIX}norm.weight", shape=(hidden,), dtype=bf16)

        layers: list[GDNLayerWeights | AttnLayerWeights] = []
        for i, layer_type in enumerate(layer_types):
            p = f"{TEXT_PREFIX}layers.{i}."
            input_ln = src.get(f"{p}input_layernorm.weight", shape=(hidden,), dtype=bf16)
            post_ln = src.get(
                f"{p}post_attention_layernorm.weight", shape=(hidden,), dtype=bf16
            )
            mlp = _load_mlp(src, p, hidden, inter)

            if layer_type == "full_attention":
                assert i in FULL_ATTENTION_LAYERS, f"层 {i} 是 full_attention，但不在预期层号内"
                q_proj = src.get(
                    f"{p}self_attn.q_proj.weight",
                    shape=(num_heads * head_dim * 2, hidden),
                    dtype=bf16,
                )
                q_proj_q, q_proj_gate = _split_q_proj(q_proj, num_heads, head_dim)
                # q / gate / k / v 都吃同一个 h，融合成 [5120,1024]。顺序即输出的
                # 切片顺序，改这里必须同步改 runner._attention_decode。
                qkvg_fused, (q_proj_q, q_proj_gate, k_proj, v_proj) = _fuse_rows([
                    q_proj_q,
                    q_proj_gate,
                    src.get(
                        f"{p}self_attn.k_proj.weight",
                        shape=(num_kv_heads * head_dim, hidden),
                        dtype=bf16,
                    ),
                    src.get(
                        f"{p}self_attn.v_proj.weight",
                        shape=(num_kv_heads * head_dim, hidden),
                        dtype=bf16,
                    ),
                ])
                layers.append(
                    AttnLayerWeights(
                        input_layernorm=input_ln,
                        post_attention_layernorm=post_ln,
                        q_proj_q=q_proj_q,
                        q_proj_gate=q_proj_gate,
                        k_proj=k_proj,
                        v_proj=v_proj,
                        qkvg_fused=qkvg_fused,
                        q_norm=src.get(
                            f"{p}self_attn.q_norm.weight", shape=(head_dim,), dtype=bf16
                        ),
                        k_norm=src.get(
                            f"{p}self_attn.k_norm.weight", shape=(head_dim,), dtype=bf16
                        ),
                        o_proj=src.get(
                            f"{p}self_attn.o_proj.weight",
                            shape=(hidden, num_heads * head_dim),
                            dtype=bf16,
                        ),
                        mlp=mlp,
                    )
                )
            else:
                assert layer_type == "linear_attention", f"未知层类型 {layer_type}"
                assert i not in FULL_ATTENTION_LAYERS
                key_dim = lin_heads * lin_head_dim
                conv1d = src.get(
                    f"{p}linear_attn.conv1d.weight", shape=(key_dim * 3, 1, 4), dtype=bf16
                )
                # qkv / z / a / b 都吃同一个 h，融合成 [8224,1024]。顺序即输出的
                # 切片顺序，改这里必须同步改 runner._gdn_decode。
                in_proj_fused, (in_qkv, in_z, in_a, in_b) = _fuse_rows([
                    src.get(
                        f"{p}linear_attn.in_proj_qkv.weight",
                        shape=(key_dim * 3, hidden),
                        dtype=bf16,
                    ),
                    src.get(
                        f"{p}linear_attn.in_proj_z.weight",
                        shape=(key_dim, hidden),
                        dtype=bf16,
                    ),
                    src.get(
                        f"{p}linear_attn.in_proj_a.weight",
                        shape=(lin_heads, hidden),
                        dtype=bf16,
                    ),
                    src.get(
                        f"{p}linear_attn.in_proj_b.weight",
                        shape=(lin_heads, hidden),
                        dtype=bf16,
                    ),
                ])
                layers.append(
                    GDNLayerWeights(
                        input_layernorm=input_ln,
                        post_attention_layernorm=post_ln,
                        in_proj_qkv=in_qkv,
                        in_proj_z=in_z,
                        in_proj_a=in_a,
                        in_proj_b=in_b,
                        in_proj_fused=in_proj_fused,
                        conv1d=conv1d.squeeze(1).contiguous(),
                        conv1d_decode=conv1d.squeeze(1)
                        .transpose(0, 1)
                        .contiguous(),  # [4,D]，必须 contiguous 不能是 view
                        a_log=src.get(
                            f"{p}linear_attn.A_log", shape=(lin_heads,), dtype=fp32
                        ),
                        dt_bias=src.get(
                            f"{p}linear_attn.dt_bias", shape=(lin_heads,), dtype=bf16
                        ),
                        norm=src.get(
                            f"{p}linear_attn.norm.weight",
                            shape=(lin_head_dim,),
                            dtype=fp32,
                        ),
                        out_proj=src.get(
                            f"{p}linear_attn.out_proj.weight",
                            shape=(hidden, key_dim),
                            dtype=bf16,
                        ),
                        mlp=mlp,
                    )
                )
    finally:
        src.close()

    _verify(src, layers)

    return TextWeights(
        embed_tokens=embed_tokens,
        norm=final_norm,
        layers=tuple(layers),
        hidden_size=hidden,
        num_layers=num_layers,
        vocab_size=vocab_size,
        rms_norm_eps=text_config["rms_norm_eps"],
        rope_theta=float(rope_theta),
        rotary_dim=rotary_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        linear_num_heads=lin_heads,
        linear_head_dim=lin_head_dim,
    )


def _verify(src: _TensorSource, layers: list) -> None:
    """收尾断言：层数分布、参数量、以及"跳过是有意的"。"""
    num_gdn = sum(1 for x in layers if isinstance(x, GDNLayerWeights))
    num_attn = sum(1 for x in layers if isinstance(x, AttnLayerWeights))
    assert num_gdn == EXPECTED_NUM_GDN_LAYERS, f"GDN 层 {num_gdn} != {EXPECTED_NUM_GDN_LAYERS}"
    assert num_attn == EXPECTED_NUM_ATTN_LAYERS, (
        f"full-attention 层 {num_attn} != {EXPECTED_NUM_ATTN_LAYERS}"
    )

    # checkpoint 里所有 language_model 张量都必须被读到，一个不漏。
    all_text_keys = {k for k in src.weight_map if k.startswith(TEXT_PREFIX)}
    missed = all_text_keys - src.read_keys
    assert not missed, f"有 {len(missed)} 个文本主干张量没被加载：{sorted(missed)[:5]}"

    # 跳过的必须只有视觉塔和 MTP。
    unread = set(src.weight_map) - src.read_keys
    unexpected = {k for k in unread if not k.startswith(SKIP_PREFIXES)}
    assert not unexpected, f"跳过了预期之外的张量：{sorted(unexpected)[:5]}"

    # 参数量按 q_proj 拆分前的原始张量计（拆分不改变总量）。
    total = 0
    for name in src.read_keys:
        shape = _shape_of(src, name)
        n = 1
        for d in shape:
            n *= d
        total += n
    assert total == EXPECTED_TEXT_PARAMS, (
        f"文本主干参数量 {total:,} != 预期 {EXPECTED_TEXT_PARAMS:,}"
    )


_SHAPE_CACHE: dict[int, dict[str, tuple[int, ...]]] = {}


def _shape_of(src: _TensorSource, name: str) -> tuple[int, ...]:
    """从 safetensors header 读形状，不再把张量搬一遍。"""
    key = id(src)
    if key not in _SHAPE_CACHE:
        import struct

        shapes: dict[str, tuple[int, ...]] = {}
        for shard in set(src.weight_map.values()):
            with open(src.model_dir / shard, "rb") as f:
                header_len = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(header_len))
            for k, v in header.items():
                if k != "__metadata__":
                    shapes[k] = tuple(v["shape"])
        _SHAPE_CACHE[key] = shapes
    return _SHAPE_CACHE[key][name]


if __name__ == "__main__":
    import time

    start = time.time()
    weights = load_text_weights()
    elapsed = time.time() - start

    num_gdn = sum(1 for x in weights.layers if isinstance(x, GDNLayerWeights))
    num_attn = sum(1 for x in weights.layers if isinstance(x, AttnLayerWeights))
    bytes_on_gpu = torch.cuda.memory_allocated()

    print(f"加载耗时       {elapsed:.2f}s")
    print(f"device         {weights.device}")
    print(f"GDN / attn 层  {num_gdn} / {num_attn}")
    print(f"embed_tokens   {tuple(weights.embed_tokens.shape)} {weights.embed_tokens.dtype}")
    print(f"rope_theta     {weights.rope_theta:,.0f}")
    print(f"rotary_dim     {weights.rotary_dim}")
    print(f"显存占用       {bytes_on_gpu / 2**30:.3f} GiB")

    attn0 = weights.layers[3]
    print(
        f"layer3 q_proj  Q{tuple(attn0.q_proj_q.shape)} "
        f"gate{tuple(attn0.q_proj_gate.shape)} "
        f"contiguous={attn0.q_proj_q.is_contiguous() and attn0.q_proj_gate.is_contiguous()}"
    )
    gdn0 = weights.layers[0]
    print(
        f"layer0 conv1d  prefill{tuple(gdn0.conv1d.shape)} "
        f"decode{tuple(gdn0.conv1d_decode.shape)}"
        f"(contig={gdn0.conv1d_decode.is_contiguous()}) | "
        f"A_log {gdn0.a_log.dtype} | norm {gdn0.norm.dtype} | dt_bias {gdn0.dt_bias.dtype}"
    )
    print("\n全部断言通过。")
