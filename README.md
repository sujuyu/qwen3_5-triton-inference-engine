# Qwen3.5-0.8B 专用推理引擎：结构与算子清单

> 调研日期：2026-08-02  
> 目标 checkpoint：`Qwen/Qwen3.5-0.8B`，revision `2fc06364715b967f1860aea9cf38778875588b17`  
> 目标：不依赖现成深度学习算子库，自己实现这个 checkpoint 的 BF16 推理。本文只讨论推理，不讨论训练和反向传播。

## 0. 先给结论

1. 这个模型不是普通的 24 层全注意力 Transformer。它的文本主干按 `3 个 Gated DeltaNet + 1 个全注意力` 重复 6 次，也就是：

   - 18 个 Gated DeltaNet 线性注意力层；
   - 6 个 GQA 全注意力层，索引为 `3, 7, 11, 15, 19, 23`；
   - 每层都有一个 1024 → 3584 → 1024 的 SwiGLU MLP。

2. `0.8B` 是语言模型的标称规模，不是 checkpoint 中所有张量之和。官方 Safetensors 实际包含：

   | 部分 | 参数量 | BF16/FP32 权重字节 | GiB |
   |---|---:|---:|---:|
   | 文本主干 | 752,393,024 | 1,504,791,232 | 1.401 |
   | 视觉塔 | 100,592,896 | 201,185,792 | 0.187 |
   | MTP | 20,452,864 | 40,905,728 | 0.038 |
   | 合计 | 873,438,784 | 1,746,882,752 | 1.627 |

   文本主干里有 2,592 个 FP32 参数，即每个 Gated DeltaNet 层的 `A_log[16]` 和 gated RMSNorm 权重 `[128]`；其余主权重为 BF16。

3. 普通的逐 token 生成不需要 `mtp.*`。Hugging Face 的标准前向没有实例化 MTP，vLLM 的普通加载器也显式跳过 `mtp.`。因此第一版引擎可以安全跳过 MTP。

4. 如果第一版只做文本生成，可以同时跳过 `model.visual.*` 和 `mtp.*`，只加载约 1.40 GiB 文本权重。建议第一阶段只支持：

   ```text
   batch = 1 + BF16 + 文本输入 + greedy sampling + KV/state cache
   ```

5. 最难且最值得写成专用融合核的不是 Softmax，而是：

   - Gated DeltaNet 的 depthwise causal Conv1D；
   - Gated Delta Rule 的 recurrent state 更新；
   - decode 阶段的小矩阵 GEMV/GEMM；
   - 只出最后一个 token 的大词表 LM head。

## 1. 模型总览

### 1.1 完整数据流

```text
文本 ──Qwen2 BPE── token ids ──Embedding──────────────┐
                                                     ├─替换/拼接─┐
图像/视频 ─预处理─ Vision Transformer ─ PatchMerger ┘          │
                                                                ▼
                    24 × Decoder Layer
       ┌─────────────────────────────────────────────────┐
       │  RMSNorm                                        │
       │  Gated DeltaNet 或 gated GQA full attention     │
       │  residual add                                    │
       │  RMSNorm → SwiGLU MLP → residual add             │
       └─────────────────────────────────────────────────┘
                                │
                          final RMSNorm
                                │
                 tied embedding / LM-head GEMM
                                │
                     sampling → next token
```

### 1.2 文本配置

| 配置 | 值 |
|---|---:|
| vocabulary | 248,320 |
| hidden size | 1,024 |
| decoder layers | 24 |
| MLP intermediate size | 3,584 |
| native context length | 262,144 |
| RMSNorm epsilon | `1e-6` |
| activation | SiLU |
| full-attention Q heads | 8 |
| full-attention KV heads | 2 |
| head dimension | 256 |
| GQA group ratio | 4 |
| attention bias/dropout | false / 0 |
| RoPE theta | 10,000,000 |
| rotary fraction | 0.25，即每个 head 的前 64 维 |
| MRoPE sections | `[11, 11, 10]` |
| Gated DeltaNet Q/K heads | 16 / 16 |
| Gated DeltaNet V heads | 16 |
| Gated DeltaNet Q/K/V head dim | 128 / 128 / 128 |
| causal Conv1D kernel | 4 |

层类型的精确顺序为：

```text
0 L, 1 L, 2 L, 3 A,
4 L, 5 L, 6 L, 7 A,
8 L, 9 L, 10 L, 11 A,
12 L, 13 L, 14 L, 15 A,
16 L, 17 L, 18 L, 19 A,
20 L, 21 L, 22 L, 23 A
```

其中 `L = linear_attention / Gated DeltaNet`，`A = full_attention`。

### 1.3 参数分布校验值

这些数字适合在权重加载器中作为断言：

| 结构 | 单个参数量 | 数量 |
|---|---:|---:|
| token embedding | 254,279,680 | 1 |
| Gated DeltaNet decoder layer | 21,555,360 | 18 |
| full-attention decoder layer | 18,352,640 | 6 |
| final RMSNorm | 1,024 | 1 |
| vision patch embedding | 1,180,416 | 1 |
| vision learned position embedding | 1,769,472 | 1 |
| vision block | 7,087,872 | 12 |
| vision merger | 12,588,544 | 1 |
| MTP | 20,452,864 | 1 |

权重矩阵在 Safetensors 中采用 PyTorch 习惯的 `[out_features, in_features]` 布局。`lm_head.weight` 没有单独存储，输出头与 `embed_tokens.weight` 共享。

## 2. 文本主干的精确计算

以下形状统一用：

```text
B = batch size
T = 当前 prefill 长度；decode 时 T = 1
D = 1024
```

### 2.1 Embedding 和 decoder 外框

Embedding 是一次按 token id 的行 gather：

```text
W_embed: [248320, 1024]
x = W_embed[input_ids]                     # [B, T, 1024]
```

每个 decoder layer 都是 pre-norm、双 residual：

```text
r = x
h = qwen_rms_norm(x, input_layernorm.weight)
h = token_mixer(h)                         # GDN 或 full attention
x = r + h

r = x
h = qwen_rms_norm(x, post_attention_layernorm.weight)
h = down_proj(silu(gate_proj(h)) * up_proj(h))
x = r + h
```

这里的 `qwen_rms_norm` 不是普通的 `norm(x) * weight`。checkpoint 的权重以 0 为中心，公式是：

```text
qwen_rms_norm(x, w) = x * rsqrt(mean(x²) + 1e-6) * (1 + w)
```

均方、`rsqrt` 和 `(1+w)` 乘法应在 FP32 中完成，再转回输入 dtype。这是最容易导致整网输出漂移的细节之一。

### 2.2 Gated DeltaNet 层：18 层

#### 投影和形状

输入 `x: [B,T,1024]`：

| 权重 | 形状 | 输出 |
|---|---:|---:|
| `in_proj_qkv.weight` | `[6144,1024]` | `[B,T,6144]` |
| `in_proj_z.weight` | `[2048,1024]` | `[B,T,16,128]` |
| `in_proj_b.weight` | `[16,1024]` | `[B,T,16]` |
| `in_proj_a.weight` | `[16,1024]` | `[B,T,16]` |
| `conv1d.weight` | `[6144,1,4]` | depthwise causal Conv1D |
| `out_proj.weight` | `[1024,2048]` | `[B,T,1024]` |

`in_proj_qkv` 的输出先经过 6144 通道、kernel=4、无 bias 的 depthwise causal Conv1D，再做 SiLU，之后才切成：

```text
q: [B,T,16,128]
k: [B,T,16,128]
v: [B,T,16,128]
```

`z`、`a`、`b` 不经过 Conv1D。

#### 门控值

```text
beta = sigmoid(b)                                      # [B,T,16]
g    = -exp(A_log_fp32) * softplus(a_fp32 + dt_bias)   # [B,T,16]
```

其中：

```text
A_log:  [16], FP32
dt_bias:[16], BF16，但计算 g 时转 FP32
```

#### Q/K 归一化与 recurrent delta rule

每个 128 维 head 独立计算：

```text
q = q * rsqrt(sum(q²) + 1e-6)
k = k * rsqrt(sum(k²) + 1e-6)
q = q / sqrt(128)
```

对 token `t`、head `h`，维护状态 `S[h]: [128,128]`：

```text
S = exp(g_t) * S
memory = k_t^T @ S                         # [128]
delta  = beta_t * (v_t - memory)           # [128]
S = S + outer(k_t, delta)                  # [128,128]
o_t = q_t^T @ S                            # [128]
```

这就是最小正确实现所需的核心 Gated Delta Rule。它既可以逐 token 跑 prefill，也可以跑 decode。逐 token prefill 是 O(T) 且容易验证，但 GPU 利用率低；优化版应再实现官方思路的 `chunk_size=64` chunk kernel。

Transformers 的无融合 fallback 会把 q/k/v/beta/g 和 recurrent state 转为 FP32 做上述计算，最后把输出转回原 dtype。第一版若追求与参考实现接近，建议 state 也使用 FP32。

#### Gated RMSNorm

Delta Rule 输出仍是 `[B,T,16,128]`，每个 head 分别执行：

```text
n = o * rsqrt(mean(o²) + 1e-6)
n = n * linear_attn.norm.weight             # 这里没有“1 + weight”
n = n * silu(z_fp32)
```

`linear_attn.norm.weight: [128]` 是 FP32。随后把 16 个 head 拼回 2048 维，再通过 `[1024,2048]` 的 `out_proj`。

#### GDN cache

每个 GDN 层需要两种 cache：

1. Conv state：`[B,6144,4]`，decode 时移位/环形更新后做 4-tap depthwise dot product；
2. recurrent state：`[B,16,128,128]`，推荐 FP32。

batch=1、18 层时：

```text
Conv state（BF16） ≈ 0.844 MiB
recurrent state（FP32） = 18 MiB
```

### 2.3 Full attention 层：6 层

#### gated Q projection

这一层不是标准 Qwen2 attention。它的 Q projection 同时产生 query 和 output gate：

```text
q_proj: [4096,1024]
raw_q:  [B,T,8,512]
(q, gate) = split(raw_q, 256, last_dim)

q:      [B,T,8,256]
gate:   [B,T,8,256] → flatten 为 [B,T,2048]
```

K/V 是 GQA：

```text
k_proj: [512,1024] → [B,T,2,256]
v_proj: [512,1024] → [B,T,2,256]
```

Q 和 K 在每个 256 维 head 上各做一次 Qwen RMSNorm，即仍使用 `(1 + weight)`。

#### partial MRoPE

只有每个 head 的前 64 维应用 rotary，后 192 维原样通过：

```text
rotary_dim = 256 * 0.25 = 64
theta = 10,000,000
```

纯文本输入的 temporal/height/width position 相同，MRoPE 退化为普通 RoPE。多模态输入时，先计算三个轴的频率 `[T,H,W]`，再按 `[11,11,10]` 交错选择 32 个频率，复制成 64 维 cos/sin。不要把 256 个维度全部旋转。

`cos`/`sin` 建议由 FP32 position 和 inverse frequency 计算，再转换到 BF16；对给定位置可以缓存。

#### GQA attention

逻辑上把每个 KV head 服务给 4 个 Q head，不需要真的 materialize `repeat_kv`：

```text
score = (Q @ K^T) / sqrt(256) = (Q @ K^T) / 16
score += causal/padding mask
prob = softmax_fp32(score)
context = prob @ V                          # [B,T,8,256]
context = context * sigmoid(gate)
output = context.reshape(B,T,2048) @ W_o^T  # W_o: [1024,2048]
```

decode 时只需计算新 query 与 cache 中所有 K/V 的点积。prefill 时使用 causal mask。视觉塔的 attention 则是非 causal，不能复用硬编码 causal 的入口。

#### KV cache

仅 6 个 full-attention 层保存 KV：

```text
每层、每 token：K/V × 2 KV heads × 256 dim × BF16
                 = 2 × 2 × 256 × 2 bytes
                 = 2 KiB

六层合计：12 KiB / token / batch item
```

| 上下文长度 | 六层 BF16 KV cache，batch=1 |
|---:|---:|
| 8,192 | 96 MiB |
| 32,768 | 384 MiB |
| 131,072 | 1.5 GiB |
| 262,144 | 3.0 GiB |

这个模型虽然 18 层线性注意力使用定长状态，但 6 层全注意力仍让总 KV cache 随上下文线性增长。

### 2.4 MLP：所有 24 层都有

```text
gate = x @ W_gate^T       # [1024] → [3584]
up   = x @ W_up^T         # [1024] → [3584]
h    = silu(gate) * up
out  = h @ W_down^T       # [3584] → [1024]
```

三个矩阵均无 bias。实现时应把 gate/up GEMM、SiLU 和逐元素乘尽量融合成 SwiGLU kernel。

### 2.5 Final norm、LM head 和采样

24 层后再做一次 Qwen RMSNorm。输出矩阵与 embedding 共享：

```text
logits = last_hidden[1024] @ W_embed^T[1024,248320]
```

decode 时只计算最后一个位置的 logits，不要为历史位置重复计算 248,320 维 logits。即使模型很小，这个大词表 GEMV 仍然很显著。

模型本身只输出 logits。生成引擎还需按产品需求实现：

- greedy：argmax；
- temperature：logits 除法；
- top-k：选择；
- top-p：排序/选择、前缀和；
- multinomial：稳定随机数和累计概率查找；
- 可选 repetition/presence/frequency penalty。

## 3. 视觉塔：要支持图片/视频时才实现

### 3.1 配置

| 配置 | 值 |
|---|---:|
| input channels | 3 |
| patch kernel/stride | temporal 2 × height 16 × width 16 |
| hidden size | 768 |
| depth | 12 |
| heads / head dim | 12 / 64 |
| MLP intermediate | 3,072 |
| learned position table | 2,304 × 768，即 48 × 48 grid |
| spatial merge | 2 × 2 |
| merger output | 1,024，与文本 hidden 对齐 |

### 3.2 图像/视频预处理

如果引擎接收的是官方 processor 已生成的 `pixel_values` 和 `grid_thw`，可以先不实现这部分。如果要从 JPEG/PNG/视频直接开始，则还需要：

- 解码和 RGB 通道转换；
- 按 patch/merge 对齐规则做动态 resize；
- bilinear/bicubic resize；
- `x / 255`、mean=`[0.5,0.5,0.5]`、std=`[0.5,0.5,0.5]` 的 normalize；
- temporal padding/采帧；
- patch 重排和 `grid_thw` 生成；
- chat template 中 image/video placeholder 的展开。

这些属于 processor，不是神经网络 kernel。建议第一版 API 直接接收预处理后的 patch tensor，等数值核心稳定后再补齐。

### 3.3 Patch embedding

官方实现是带 bias 的 Conv3D：

```text
weight: [768,3,2,16,16]
stride = kernel = [2,16,16]
```

由于 stride 等于 kernel、patch 之间不重叠，可把每个 patch 展平为 1,536 维，然后复用带 bias 的 GEMM：

```text
[Npatch,1536] @ [1536,768] + bias[768]
```

无需为这个模型单独做通用 Conv3D 框架。

### 3.4 位置编码

视觉输入同时使用两套位置编码：

1. learned 48×48 position table，经 bilinear interpolation 到当前 patch grid，然后相加；
2. 二维 vision RoPE，作用于 attention Q/K 的完整 64 维 head。

因此还需实现“小表 gather × bilinear weight × reduce”或一个专用位置插值 kernel，以及视觉 2D RoPE。

### 3.5 12 个 ViT block

每个 block：

```text
x = x + MHA(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

MHA：

```text
qkv: 768 → 2304，带 bias
12 heads × 64 dim
非 causal attention
vision 2D RoPE
out projection: 768 → 768，带 bias
```

MLP：

```text
768 → 3072，带 bias
GELU tanh approximation（`gelu_pytorch_tanh`）
3072 → 768，带 bias
```

多张图/多段视频打包时，attention 以 `cu_seqlens` 分段；不同图片之间不能相互 attention。正确性优先版本可以按图片逐个调用非 causal attention。

### 3.6 Patch merger

先对每个 768 维 patch 做普通 LayerNorm，然后把空间相邻的 2×2 patch 合并成 3,072 维：

```text
LayerNorm(768)
reshape/group 4 patches → 3072
Linear 3072 → 3072，带 bias
exact GELU（PyTorch `nn.GELU()` 默认形式）
Linear 3072 → 1024，带 bias
```

注意 merger 的 GELU 与 ViT MLP 的 tanh 近似 GELU 不同。若目标是严格贴近官方结果，merger 需要 `erf` 或数值等价实现。

最终的 1024 维视觉 embedding 通过 masked scatter 替换文本序列中的 `<|image_pad|>` / `<|video_pad|>` embedding，然后整个混合序列进入同一个 24 层文本主干。

## 4. MTP 是否需要实现

checkpoint 确实包含 `mtp.*`，结构从张量名和形状看包括：

- 两个 1024 维 pre-FC norm；
- 拼接后的 `[2048] → [1024]` FC；
- 1 个与 full-attention decoder 相同形状的层；
- final norm；
- 复用主模型 LM head。

但它不在 Hugging Face 普通 `Qwen3_5ForConditionalGeneration.forward()` 的路径中，标准 vLLM Qwen3.5 loader 也使用 `skip_prefixes=["mtp."]`。因此：

- 普通 greedy/sampling：不需要 MTP；
- 想做 multi-token prediction/speculative decoding：才需要 MTP；
- MTP 除了复用 full-attention、MLP、RMSNorm 外，还需要 proposal、验证和 acceptance 调度逻辑；这不应进入第一版正确性里程碑。

## 5. 必须自己实现的算子清单

下面把“算子”分为 primitive kernel 与少量专用 fused kernel。`T` 表示文本最小引擎必需，`V` 表示视觉必需，`O` 表示可选功能。

### 5.1 张量与加载运行时

| 算子/功能 | 范围 | 说明 |
|---|---|---|
| Safetensors header/parser | T | mmap/读取 BF16、FP32，校验 shape、offset、dtype |
| BF16 ↔ FP32 convert | T | 归一化、Softmax、GDN state 等需要 |
| tensor view/reshape | T | 最好是零拷贝 metadata 操作 |
| transpose/permute | T | 可由 kernel 的 stride/layout 支持，避免真实搬运 |
| contiguous/layout pack | T | 只在 GEMM 或 cache 需要时执行 |
| slice/split/chunk | T | QKV、Q/gate 分割 |
| concat | T | MRoPE、小量状态；应尽量融合 |
| gather | T | token embedding、位置表 |
| masked scatter | V | 视觉 embedding 替换 image/video token |
| repeat/interleave | V/host | 位置 id；GQA 不应真实复制 K/V |
| padding/mask build | T | batch/prefill 时使用；batch=1 无 padding 可简化 |

### 5.2 通用数值 kernel

| 算子 | 范围 | 精度/特化建议 |
|---|---|---|
| GEMM/GEMV，无 bias | T | BF16 输入，至少 FP32 accumulate；覆盖文本全部线性层 |
| GEMM/GEMV，带 bias | V | ViT、patch embed、merger |
| batched/head GEMM | T | QKᵀ、PV；也可直接写 fused attention |
| outer product / rank-1 update | T | GDN state 更新 |
| elementwise add/mul/scale/neg | T | residual、gate、衰减 |
| reduce sum/mean/max | T | norm、Softmax、argmax |
| square / rsqrt / sqrt | T | RMSNorm、L2 norm、attention scale |
| exp | T | Softmax、GDN decay |
| sigmoid | T | beta、attention output gate |
| softplus | T | GDN `g` |
| SiLU | T | MLP、Conv 输出、GDN z gate |
| GELU-tanh | V | ViT MLP |
| exact GELU/erf | V | vision merger |
| sin/cos | T | RoPE/MRoPE；可在 host 预计算和缓存 |
| LayerNorm | V | 标准 mean/variance + gamma/beta |
| Qwen RMSNorm `(1+w)` | T | FP32 reduction/multiply |
| direct-weight gated RMSNorm | T | GDN 输出，与上一个不能混用 |
| L2Norm | T | GDN 的 Q/K，epsilon=`1e-6` |
| stable FP32 Softmax | T | 减 max → exp → sum → divide |

### 5.3 模型专用核心 kernel

| 专用 kernel | 范围 | 推荐接口/职责 |
|---|---|---|
| `depthwise_causal_conv4_prefill` | T | 6144 通道，4 taps，Conv 后 SiLU |
| `depthwise_causal_conv4_decode` | T | 更新 `[6144,4]` state 并输出一个 token |
| `gated_delta_recurrent_prefill` | T | 顺序更新 `[16,128,128]` FP32 state |
| `gated_delta_recurrent_decode` | T | 单 token q/k/v/g/beta → state/output |
| `gated_delta_chunk64_prefill` | 性能 O | 正确后再做；减少串行 prefill 开销 |
| `rmsnorm_gated_silu_16x128` | T | GDN output norm、z gate、head 拼接融合 |
| `gqa_attention_prefill` | T | 8 Q heads、2 KV heads、dim=256、causal |
| `gqa_attention_decode` | T | 只对新 token，读 6 层 KV cache |
| `partial_mrope_64` | T | 只旋转 Q/K 前 64 维 |
| `swiglu_1024_3584` | T | gate/up projection 后 activation/mul 融合 |
| `lm_head_248320x1024` | T | last-token-only GEMV，可与 top-k 部分融合 |
| `vision_attention_12x64` | V | 非 causal、按图像分段 |
| `vision_rope_2d_64` | V | 视觉 Q/K 全 head rotary |
| `patch_embed_1536x768` | V | Conv3D 特化成 patch GEMM |
| `patch_merge_4x768` | V | LayerNorm、pack、MLP |

### 5.4 host 侧算法

这些不一定写成 GPU kernel，但完整引擎必须有：

- Qwen2 byte-level BPE tokenizer、官方 pretokenize regex、UTF-8 decode；
- chat template 和 special token 处理；
- 文本 position id、3D MRoPE position id、`rope_delta`；
- causal/padding mask 与 packed sequence metadata；
- KV cache、GDN conv cache、GDN recurrent cache 的分配和生命周期；
- logits processor、停止条件和 sampler；
- image/video processor（若不要求调用者传预处理 tensor）。

聊天场景要注意：tokenizer metadata 把 `<|im_end|>`（248046）设为 EOS，而 text config 中还有 `<|endoftext|>`（248044）。停止 token 应由调用层配置，不能只凭一个硬编码值。

## 6. 推荐的最小 kernel API

不必实现一个通用 tensor framework。只针对该 checkpoint，可以把接口固定为以下形状：

```cpp
// 基础线性代数
gemm_bf16_f32acc(A, B, C, M, N, K, optional_bias);
gemv_bf16_f32acc(x, W_out_in, y, out_dim, in_dim);

// 文本归一化/激活
qwen_rmsnorm_1024(x_bf16, w_bf16, y_bf16, rows, eps);
qwen_rmsnorm_head256(x_bf16, w_bf16, y_bf16, rows, eps);
gdn_rmsnorm_gate_128(x_fp32_or_bf16, w_fp32, z_bf16, y_bf16, rows, eps);
swiglu_3584(gate_bf16, up_bf16, out_bf16, rows);

// GDN
gdn_conv4_prefill(qkv, weight, conv_state, B, T);
gdn_conv4_decode(qkv_one, weight, conv_state, B);
gdn_recurrent_prefill(q, k, v, g, beta, state_fp32, out, B, T);
gdn_recurrent_decode(q, k, v, g, beta, state_fp32, out, B);

// full attention
mrope_partial64(q, k, position_ids, cos_sin_cache);
gqa_prefill_8q_2kv_d256(q, k, v, mask, kv_cache, out, B, T);
gqa_decode_8q_2kv_d256(q, k_new, v_new, kv_cache, out, B, past_len);

// 输出
lm_head_last_token(hidden_1024, tied_embedding, logits_248320);
argmax_or_sample(logits, sampling_config);
```

这种固定形状设计会牺牲通用性，但更符合“只针对 Qwen3.5-0.8B”的目标，也更容易做融合和验证。

## 7. Prefill 与 decode 应分开设计

### 7.1 Prefill

Prefill 的 `T > 1`，特点是：

- 大 GEMM 更容易跑满 GPU；
- 6 个 full-attention 层要做 causal attention；
- 18 个 GDN 层需要从零状态扫描整段序列；
- 正确性版可用逐 token recurrent GDN；性能版应实现 chunk-64 GDN；
- 最后通常只需要最后一行 logits。

如果复刻参考 chunk 算法，内部还会用到 pad-to-64、cumsum、下三角 mask、64×64 小矩阵乘和每 chunk 的状态传播。推荐把它封装成一个 GDN prefill kernel，而不是把这些暴露成大量通用算子。

### 7.2 Decode

Decode 每步 `T=1`，特点是：

- 线性层大多退化成 GEMV，小 batch 下非常 memory-bound；
- GDN 只做固定大小的 Conv state 和 128×128 recurrent state 更新；
- full attention 只新增一组 K/V，但需要读取全部历史 KV；
- LM head 每步读取约 254M 个 embedding 参数，是明显带宽热点；
- kernel launch 数量很容易成为瓶颈，适合做 norm+projection、gate+projection 等融合。

不要让 prefill 与 decode 共用一个充满分支的慢路径；共享权重和数学定义即可。

## 8. 建议实现顺序

### 阶段 A：建立数值 oracle

1. 固定官方 checkpoint revision；
2. 用 Hugging Face/Transformers 保存 tokenizer 输出、每层 hidden、logits 和生成 token；
3. 第一版引擎 API 直接接收 token ids，暂不自己写 tokenizer；
4. 先在 CPU/简单 CUDA kernel 上追求正确，不追求速度。

### 阶段 B：文本 greedy MVP

1. Safetensors loader，只加载 language tensors；
2. BF16/FP32、embedding、GEMM/GEMV；
3. Qwen RMSNorm、SiLU、SwiGLU、residual；
4. GDN Conv4 + sequential recurrent delta rule；
5. 6 层 GQA attention + partial RoPE；
6. 三类 cache；
7. tied LM head + argmax；
8. 对齐逐层输出和连续生成结果。

完成这一阶段，只需文本主干约 1.40 GiB 权重。

### 阶段 C：性能化

1. 把 prefill 改成大 GEMM 和 chunk-64 GDN；
2. 自定义 decode GEMV 与权重布局；
3. fused RMSNorm + projection；
4. fused SwiGLU；
5. fused GDN conv/recurrent/gated norm；
6. fused attention decode；
7. last-token LM head + partial top-k fusion；
8. CUDA Graph 或等价低 launch-overhead 调度。

如果“全部算子自己写”也包括不用 cuBLAS，那么高性能 tiled BF16 GEMM/GEMV 会是整个项目工作量最大的一部分。先做正确的 reference kernel，再针对实际 GPU 架构使用 tensor-core 指令和固定矩阵尺寸特化。

### 阶段 D：完整生成能力

1. Qwen2 BPE tokenizer 和 UTF-8 streaming decoder；
2. chat template；
3. top-k/top-p/temperature/random sampling；
4. batch、padding、连续批处理；
5. prefix/cache 管理。

### 阶段 E：多模态

1. 先接收官方 processor 输出；
2. patch GEMM、ViT、position interpolation、vision RoPE、merger；
3. masked scatter 和 3D MRoPE；
4. 再实现图像/视频解码与动态预处理。

### 阶段 F：可选 MTP

最后再研究 MTP proposal/verification/acceptance；它不影响普通生成正确性。

## 9. 正确性测试清单

### 9.1 权重加载

- 总参数量必须是 `873,438,784`；
- 文本主干必须是 `752,393,024`；
- 恰好 18 个 GDN 层、6 个 full-attention 层；
- `A_log` 和 `linear_attn.norm.weight` 必须按 FP32 读取；
- LM head 必须复用 embedding，不能错误地等待一个不存在的独立权重；
- 文本模式确认 `model.visual.*`、`mtp.*` 被有意跳过，而不是静默漏载主干权重。

### 9.2 单算子

- BF16 conversion 的所有边界值、NaN/Inf；
- GEMM/GEMV 对随机输入与真实权重；
- Qwen RMSNorm 与 direct-weight gated RMSNorm 分开测试；
- SiLU、sigmoid、softplus、exp、GELU、Softmax；
- partial RoPE 只改变前 64 维；
- Conv4 对长度 `1, 2, 3, 4, 63, 64, 65`；
- GDN 一次 prefill 与逐 token cached decode 的结果；
- GQA head 到 KV head 的映射 `q_head / 4`；
- KV cache append 和读取边界。

### 9.3 端到端

- 空 cache 的单 token；
- 短 prompt prefill 后连续生成 32 tokens；
- cached decode 与每步重算全序列对比；
- greedy token 序列与 Transformers oracle 一致；
- 8K 以上上下文检查 cache offset；
- batch/padding 开启后检查 GDN padding state 不被污染；
- 多模态阶段测试不同长宽比、多个图像、图像+文本交错、视频帧。

不要把“BF16 每层绝对误差完全为零”作为唯一标准。参考实现的 chunk GDN、FlashAttention 和 fused kernel 可能改变浮点规约顺序；应同时检查逐层误差、logit 排名和 greedy token 是否稳定。

## 10. 最容易踩错的细节

1. 把模型误当成 24 层普通 self-attention；
2. GDN 的 q/k/v 是 Conv4+SiLU 后再 split；
3. `a/b/z` 不经过 Conv1D；
4. GDN Q/K 用 L2Norm，decoder 入口用 RMSNorm，两者不同；
5. 普通 Qwen RMSNorm 用 `(1+w)`，GDN gated RMSNorm 直接乘 `w`；
6. GDN 的 `g`、recurrent state 和 gated norm 中有关键 FP32 路径；
7. full attention 的 `q_proj` 一半是 Q，一半是 output gate；
8. attention head dim 是 256，但 RoPE 只作用前 64 维；
9. 只有 6 层分配 KV cache，18 个 GDN 层分配另外两类 state；
10. GQA 不要真的复制 K/V 四份；
11. vision attention 是非 causal，文本 attention 是 causal；
12. vision MLP GELU-tanh 与 merger exact GELU 不同；
13. patch merger 是先按 768 做 LayerNorm，再把四个 patch 组成 3072；
14. embedding 与 LM head tied；
15. `mtp.*` 不属于普通生成必经路径；
16. tokenizer/chat EOS 与 text config EOS 的语义要由调用层明确处理。

## 11. 参考来源

- [官方 Qwen3.5-0.8B 模型卡](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [固定 revision 的 config.json](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/config.json)
- [固定 revision 的 Safetensors index](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/model.safetensors.index.json)
- [固定 revision 的 tokenizer config](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/tokenizer_config.json)
- [Transformers Qwen3.5 参考实现，固定 commit](https://github.com/huggingface/transformers/blob/b3a36037d3feb22e3f0174b3dd4248fcc0f0f722/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [vLLM Qwen3.5 推理实现，固定 commit](https://github.com/vllm-project/vllm/blob/0601850791155003afbe5a0d5d086350cada8deb/vllm/model_executor/models/qwen3_5.py)

## 12. 最终的最小实现边界

如果目标只是“让 Qwen3.5-0.8B 文本对话正确跑起来”，最终真正不可删的神经网络算子集合可以压缩为：

```text
BF16/FP32 + GEMM/GEMV + gather
add/mul/reduce/rsqrt/exp/sin/cos
SiLU/sigmoid/softplus/FP32 Softmax
Qwen RMSNorm + GDN gated RMSNorm + L2Norm
depthwise causal Conv1D(kernel=4)
Gated Delta recurrent rule
partial MRoPE
causal GQA attention(8Q/2KV/256)
SwiGLU
argmax/sampling
三类 cache 与 host 调度
```

视觉 LayerNorm、GELU、带 bias 线性层、patch embedding、position interpolation、非 causal attention 和 masked scatter 都可以留到第二阶段；MTP 可以留到最后，甚至永远不实现。
