# Qwen3.5-0.8B Triton 推理引擎交接说明

更新日期：2026-09-01

## 1. 项目目标与当前范围

目标是手写 Triton kernel，实现 Qwen3.5-0.8B 的简化文本推理。

当前阶段明确限定为：

- batch 恒定为 1；
- 仅支持文本，**不实现视觉塔，不实现 MTP**；
- BF16 权重和激活，关键 reduction/状态使用 FP32；
- greedy generation；
- 暂不实现 continuous batching、padding、top-k/top-p sampling。

模型文件位于 `Qwen3.5-0.8B/`。完整模型结构和算子调研见 `README.md`；README 中第 3 节（视觉塔）、第 4 节（MTP）以及算子清单里标记为 `V` 的条目当前都不在范围内。

关于 cache 的现状：kernel 层面 GDN 的 recurrent state cache 路径已经打通（`gdn_recurrent_decode` 原地更新 `[H,128,128]` FP32 state，并已验证「prefill 前缀 + 逐 token decode」与整段 sequential 结果一致）；但 GDN conv state cache 和 full attention 的 KV cache 都还没有 kernel。因此第一版 runner 仍按「每生成一个 token，对增长后的完整序列重新执行一次 forward」实现，等三类 cache 齐备后再切到增量 decode。

无论走哪条路，Gated DeltaNet 在单次 forward 内必须按 token 顺序维护 recurrent state；完整重算时从零状态开始。

## 2. 模型关键结构

文本配置：

```text
vocab_size       = 248320
hidden_size      = 1024
num_layers       = 24
intermediate     = 3584
rms_norm_eps     = 1e-6
dtype            = BF16
```

层类型按以下模式重复 6 次：

```text
linear_attention
linear_attention
linear_attention
full_attention
```

即：

- 18 层 Gated DeltaNet；
- 6 层 full attention，层号为 3、7、11、15、19、23；
- 每层都有 SwiGLU MLP。

Full attention 参数：

```text
Q heads          = 8
KV heads         = 2
head_dim         = 256
GQA group size   = 4
rotary_dim       = 64
rope_theta       = 10_000_000
```

Gated DeltaNet 参数：

```text
Q/K/V heads      = 16
head_dim         = 128
conv kernel      = 4（depthwise，6144 通道）
```

## 3. 已有 kernel

共 13 个文件，全部位于 `triton_kernels/`，全部已有 autotune（少数固定配置）、`torch.library.triton_op` 封装、`register_fake` 和 PyTorch reference，且自测通过。注册的 op 命名空间统一为 `wy_lib::`。

### 3.1 `gemm_2d.py` — `wy_lib::gemm_2d`

```text
x:      [M,K] BF16
weight: [N,K] BF16
out:    [M,N] BF16

out = x @ weight.T
```

FP32 accumulate，要求 `K % 128 == 0`。可以覆盖当前文本模型的投影尺寸。

kernel 自带的自测只覆盖 `(M,K,N)=(1,128,128)`。真实尺寸的覆盖放在 `tests/test_gemm_model_shapes.py`（不改动 kernel 本体）：7 组 `(K,N)` × 4 组 `T`，共 28 组全部通过，相对误差 ≤ 0.46%，与 BF16 输出舍入一致。

```text
K=1024: N ∈ {16, 512, 2048, 3584, 6144}
K=2048: N = 1024
K=3584: N = 1024
T ∈ {1, 19, 65, 129}
```

`N=248320` 的 LM head 不走这个 kernel，由 `vocab_argmax.py` 融合处理。

性能上这个 kernel 是达标的：CUDA Graph 下 105-145 TFLOPS，相当于 cuBLAS 的
1.1-1.2 倍耗时，个别形状还更快。不要凭 eager 下的逐 kernel 计时判断它慢，
详见 8.5。

### 3.2 `qwen_rmsnorm.py` — `wy_lib::qwen_rmsnorm`

接口支持最后一维 256 或 1024：

```text
y = x * rsqrt(mean(x^2) + eps) * (1 + weight)
```

输入和 weight 均为 BF16，reduction 和乘法使用 FP32。

注意：wrapper 有 `assert x.is_contiguous()`，是全仓库唯一一个不接受 stride 的 kernel。Full-attention 的 q projection 曾经与它冲突，现已通过 loader 里拆分 `q_proj` 解决（见第 4 节和 7.1），kernel 本体不用改。

这个 kernel 不能用于 GDN 的 gated RMSNorm，后者见 3.11。

### 3.3 `gqa_attention_without_kvcache_casual.py` — `wy_lib::gqa_attention_without_kvcache_casual`

```text
q:   [B,8,T,256] BF16
k:   [B,2,T,256] BF16
v:   [B,2,T,256] BF16
out: [B,8,T,256] BF16
```

实现 causal GQA 和 online softmax，不支持 KV cache。它只实现 attention 核心，不包括 q/k projection、q/k RMSNorm、RoPE、output gate、o projection。

已在 A100 上对 `T=1,3,17,31,33,63,65,127,129` 与 PyTorch SDPA 比较，最大绝对误差不超过 `0.015625`。

文件名和注释里的 `casual`/`0.6b` 是历史命名问题，实际 head 配置与 Qwen3.5-0.8B 一致。

### 3.4 `partial_rope.py` — `wy_lib::partial_rope`

```text
x:            [B,H,T,D] BF16
position_ids: [B,T] I32/I64
inv_freq:     [R/2] FP32
rotary_dim:   R，偶数且 R <= D
out:          [B,H,T,D] BF16
```

运算：

```text
angle[b,t,j] = position_ids[b,t] * inv_freq[j]

out[...,j]     = x[...,j] * cos(angle) - x[...,j+R/2] * sin(angle)
out[...,j+R/2] = x[...,j] * sin(angle) + x[...,j+R/2] * cos(angle)
out[...,R:D]   = x[...,R:D]
```

Qwen3.5 参数：`D=256`、`R=64`、`inv_freq.shape=[32]`、`inv_freq[j] = 1 / 10_000_000^(2*j/R)`。

kernel 内部根据 position 和 inv_freq 计算 sin/cos，没有保存全长度 cos/sin cache。Q 和 K 分别调用一次。

测试 `(B,H,T,D,R) = (1,8,1,256,64)`、`(1,2,17,256,64)`、`(2,3,65,128,64)`、`(1,4,33,160,96)`，最大绝对误差均为 0。

之前基础版本的旋转下标有问题：`R=64` 时错误地配对了 `0:32` 和 `64:96`；现已修正为 `0:32` 和 `32:64`，并补上 `[R:D)` 的复制。

### 3.5 `attention_gate_pack.py` — `wy_lib::attention_gate_pack`

```text
attention_out: [B,H,T,D] BF16
gate:          [B,H,T,D] BF16   ← 形状，不是内存布局
out:           [B*T,H*D] BF16，连续布局

out[t,h*D+d] = attention_out[b,h,t,d] * sigmoid(gate[b,h,t,d])
```

**两个参数都按 `[B,H,T,D]` 索引**，wrapper 里有 `assert x.shape == gate.shape`。gate 在
上游的自然内存布局是 `[B,T,H,D]`，所以调用方要传 permute 出来的 view，而不是直接传
`[B,T,H,D]` 形状的张量：

```python
gate4 = gate.view(1, T, H, D).permute(0, 2, 1, 3)   # [1,H,T,D]，strided
packed = attention_gate_pack(ctx, gate4)
```

kernel 是 stride-aware 的，permute 不产生拷贝。pack 同时完成 `[B,H,T,D] -> [B*T,H*D]` 的
转置重排，输出直接传给 `o_proj` GEMM。

已使用非连续 `[B,H,T,D]` stride 输入测试 `T=1,3,17,65,129`，最大绝对误差均为 0。Triton 3.7 的 `tl.sigmoid` 要求 FP32，因此 gate 在 kernel 内显式转为 FP32。

### 3.6 `residual_add.py` — `wy_lib::residual_add`

通用 contiguous BF16 逐元素加法，一维 flatten 索引，固定 `BLOCK_SIZE=256`。模型形状及非 1024 对齐形状的测试最大绝对误差均为 0。

### 3.7 `swiglu.py` — `wy_lib::swiglu`

```text
gate, up: [M,N] BF16
out:      [M,N] BF16
out = gate * sigmoid(gate) * up
```

FP32 计算后写回 BF16。模型维度 `N=3584` 及通用 `N=1000` 测试通过，最大绝对误差 `0.00195312`。

用户基础版本原本计算成 `up * sigmoid(gate)`，少了 SiLU 中的 `gate` 因子，且覆盖写入 `up_ptr`；现已修正公式并使用独立 `out_ptr`。

### 3.8 `depthwise_causal_conv4_prefill.py` — `wy_lib::depthwise_causal_conv4_prefill`

```text
x:      [T,D] BF16
weight: [D,1,4] 或 [D,4] BF16
out:    [T,D] BF16
```

depthwise causal Conv4 融合 SiLU。已测试 `T=1,2,3,4,17,65`、模型 `D=6144` 和通用 `D=1000`，最大绝对误差 `0.015625`。

对应的 decode 版本（维护 `[6144,4]` conv state）尚未实现。

### 3.9 `gdn_qk_norm_gates.py` — `wy_lib::gdn_qk_norm_gates`

```text
q,k:     [T,H,D] BF16
a,b:     [T,H] BF16
A_log:   [H] FP32
dt_bias: [H] BF16

q_norm,k_norm: [T,H,D] BF16
beta,g:        [T,H] FP32
```

融合 Q/K L2Norm、Q scale、beta 和稳定 softplus 形式的 g 计算。当前只在 T 维分块，每个 CTA 一次处理完整 H×D；正确性没有问题，但短 T 时 CTA 数少，后续性能阶段可考虑增加 head 分块。

测试覆盖模型 `(H,D)=(16,128)`、`T=1,3,17,65`，以及通用 `(8,64)`；Q/K 最大绝对误差 `6.1035e-05`，beta/g 最大绝对误差 `3.8147e-06`。

输出的 `beta`/`g` 是 FP32，正好对上 3.10 各个 op 的输入要求。

### 3.10 `gdn_recurrent_prefill.py` — GDN delta rule，5 个 op

这个文件是当前最大的一个（约 1540 行），包含三条路径。所有 op 的 q/k/v/beta/g 都接受任意 stride（测试用 `[..., ::2]` view 验证过）。

**(a) 逐 token sequential prefill** — `wy_lib::gdn_recurrent_prefill_sequential`

```text
q,k:  [T,H,DK] BF16      beta,g: [T,H] FP32
v:    [T,H,DV] BF16
->
out:   [T,H,DV] BF16
state: [H,DK,DV] FP32（本次 forward 结束时的最终状态）
```

一个 CTA 负责一个 `(head, v_tile)`，持有 `[DK,BLOCK_V]` 的 state tile，在 T 方向顺序循环。要求 `DV % 128 == 0`、`DK` 为 2 的幂。

**(b) 单 token decode** — `wy_lib::gdn_recurrent_decode`

```text
q,k:   [1,H,DK] BF16     beta,g: [1,H] FP32
v:     [1,H,DV] BF16
state: [H,DK,DV] FP32，原地更新
->
out:   [1,H,DV] BF16
```

声明为 `mutates_args=("state",)`，autotune 用 `restore_value=["state_ptr"]` 避免 tuning 过程污染 state。**调用方要注意这个 op 会就地改写传入的 state。**

**(c) chunk-64 三段式 prefill**（`chunk_size` 固定 64）

```text
wy_lib::gdn_chunk_prepare_wy(k, v, beta, g)
    -> w        [N,H,64,DK] FP32
       u_base   [N,H,64,DV] FP32
       g_cumsum [N,H,64]    FP32

wy_lib::gdn_chunk_state(k, w, u_base, g_cumsum)
    -> delta       [N,H,64,DV] FP32
       chunk_state [N,H,DK,DV] FP32（每个 chunk 的入口状态）
       final_state [H,DK,DV]   FP32

wy_lib::gdn_chunk_output(q, k, delta, g_cumsum, chunk_state)
    -> out [T,H,DV] BF16
```

Python 侧用 `call_gdn_recurrent_prefill_chunked_triton(q,k,v,beta,g) -> (out, final_state)` 把三段串起来，接口与 (a) 完全一致，可以直接互换。

数学基础（写在 `_gdn_chunk_prepare_wy_kernel` 注释里）：单个 chunk 内 delta 满足下三角方程

```text
(I + L) @ delta = beta * v - beta * exp(G) * k @ state_in
L[t,i] = beta[t] * exp(G[t] - G[i]) * <k[t], k[i]>，i < t
P = (I + L)^-1
u_base = P @ (beta * v)
w      = P @ (beta * exp(G) * k)
delta  = u_base - w @ state_in
```

`P` 用前代法逐行求逆（`for row in tl.range(1, BLOCK_T)`）。三段的并行度分别是 `(chunk, head)`、`(head, v_tile)`（chunk 间串行）、`(chunk, head, v_tile)`。

**这条路径当前比 sequential 慢约 20 倍，不能直接切换。** CUDA event 实测（`H=16, DK=DV=128`，
折算成 18 层）：

```text
T=32    sequential  2.29ms   chunked  46.16ms
T=128   sequential  2.34ms   chunked  64.25ms
T=1024  sequential 19.41ms   chunked 351.61ms
```

主要原因在 `prepare_wy`：求 `P=(I+L)^-1` 用的是 `for row in tl.range(1, BLOCK_T)`
的逐行前代法，64 步串行，每步还做一次 `[64,64]` 的全矩阵规约；同时一个 program
一次性 materialize 全部 `64×64` 和 `64×DK` 中间量，寄存器压力很大。注释里自称
「简单正确性版本」，确实只能算正确性版本。

所以"切 chunk-64 提速"这件事**需要重写而不是切换**：至少要把前代法换成分块回代
或 Newton 迭代求逆，并在 DK/DV 上分块。`chunk_state`/`chunk_output` 的 `BLOCK_V`
硬编码为 16、没有 autotune，也是同一批待办。

测试结果（2026-09-01 A100 实测，`H=16, DK=DV=128`）：

```text
prepare_wy       T=1,3,63,64,65,129   max_abs_w 1e-8   max_abs_u 1.4e-6   max_abs_g 1.5e-5
chunked prefill  T=1,3,63,64,65,129   max_abs_out 6.1e-5   max_abs_state 2.2e-6
sequential       T=1,3,17,65          max_abs_out 3.1e-5   max_abs_state 9e-8
prefill(17)+decode(48) 等价性          max_abs_out 3.1e-5   max_abs_state 9e-8
```

chunked 和 sequential 都以逐 token FP32 PyTorch 实现为 reference。三条路径互相一致。

**作者归属**：(a) sequential 和 (b) decode 由用户编写；(c) chunk-64 三段式由 Codex 生成，用户没有逐行审过。它的测试是过的，但如果后续在这条路径上遇到可疑数值，优先怀疑它而不是 sequential；对拍时也应该以 sequential 为基准。本仓库其余 kernel 均为用户编写。

### 3.11 `gdn_gated_rmsnorm.py` — `wy_lib::gdn_gated_rmsnorm`

GDN 输出侧的 direct-weight gated RMSNorm，**没有 `1 + weight`**：

```text
x, z:   [T,H,D] BF16，接受任意 stride
weight: [D] FP32
out:    [T,H,D] BF16，contiguous

n = x * rsqrt(mean(x^2) + 1e-6)
n = n * weight
out = n * silu(z)
```

reduction 和全部乘法在 FP32 中完成。SiLU 用了数值稳定写法（按 `z` 符号分支的 `exp(-|z|)`），测试里显式塞入 `z = ±100` 验证两个尾部都不溢出。

网格为 `(head_num, ceil(T/BLOCK_T))`，head 维不分块，要求 `head_num` 和 `d_model` 都是 2 的幂。

测试 `(T,H,D) = (1,16,128)`、`(3,16,128)`、`(17,16,128)`、`(65,16,128)`、`(7,8,64)`，最大绝对误差 `0.00390625`（出现在 `T=65`，其余为 0）。

### 3.12 `embedding_gather.py` — `wy_lib::embedding_gather`

```text
input_ids: [T] I32/I64，contiguous
weight:    [vocab, D] BF16，contiguous
out:       [T, D] BF16，contiguous
```

按 token id 做行 gather，要求 `D` 是 2 的幂（模型 D=1024）。测试 int32/int64 × `T=1,3,17,65`，并显式覆盖 id=0 和 id=vocab-1，结果与 `F.embedding` 逐位相等。

### 3.13 `vocab_argmax.py` — `wy_lib::lm_head_argmax`

融合 last-token LM head GEMV 与 argmax，**不 materialize 248320 维 logits**：

```text
hidden: [T, 1024] BF16，contiguous，只读取 hidden[T-1]
weight: [248320, 1024] BF16，contiguous（tied embedding）
->
token_id: 标量 INT64
```

两段式：stage 1 每个 program 负责 `GROUP_V=512` 个词，内部按 `TILE_V=16`、`TILE_K=128` 做 FP32 accumulate 的 GEMV，输出局部最大值和全局 index；stage 2 对 `485` 个局部结果做最终 argmax。`248320 = 485 × 512`。常量固定为 `GROUP_V=512 / TILE_V=16 / TILE_K=128`，没有 autotune。

平局规则：相同最大值取较小 index，与 `torch.argmax` 一致；测试里用全零 hidden 验证返回 0。真实尺寸 `(248320,1024)` 测试通过。

## 4. Full-attention 完整数据流

输入 `x: [T,1024]`：

```text
qwen_rmsnorm(x)
  ├─ q_proj: [T,1024] -> [T,4096]
  ├─ k_proj: [T,1024] -> [T,512]
  └─ v_proj: [T,1024] -> [T,512]

raw_q.reshape(T,8,512)
  -> 每个 head 按最后一维切成 q[256] 和 gate[256]

q: [T,8,256] -> qwen_rmsnorm per head
k: [T,2,256] -> qwen_rmsnorm per head
v: [T,2,256]

q/k -> partial_rope(R=64)
q/k/v -> causal GQA -> context [1,8,T,256]
attention_gate_pack(context, gate) -> [T,2048]
o_proj [1024,2048] -> [T,1024]
residual_add
```

**重要布局陷阱（尚未解决）**：`q_proj` 的 `[T,4096]` 不是前 2048 全是 Q、后 2048 全是 gate。必须先 reshape 为 `[T,8,512]`，再对每个 head 的最后一维切 `[0:256]` 和 `[256:512]`。

切出来的 `q` 是 strided view，而 `qwen_rmsnorm` 目前 `assert x.is_contiguous()`。这是整个 runner 里**唯一**的布局冲突点：其余 kernel（`partial_rope`、`gqa_attention`、`attention_gate_pack`、`swiglu`、`gemm_2d`、`conv4`、`gdn_*`）都已经是 stride-aware，`residual_add` 虽然要 contiguous 但它的两个输入天然连续。

注意仅仅"重排 `q_proj` 的行"并不够：即使把全部 Q 排到前 2048 行，`out[:, :2048].view(T,8,256)` 的 stride 仍然是 `(4096,256,1)`，依旧非连续。

**已采用方案 1**（实现在 `engine/loader.py`，见 7.1）：loader 里把 `q_proj.weight` 拆成两个独立张量——按行 gather 出 `q_proj_q [2048,1024]`（行号 `h*512 + i`）和 `q_proj_gate [2048,1024]`（行号 `h*512+256+i`），各做一次 GEMM。两个输出都是连续的 `[T,2048]`，`view(T,8,256)` 连续，现有 kernel 一行不用改。代价是 2 次 GEMM launch 代替 1 次，FLOPs 不变。

当时考虑过但没采用的：显式 `.contiguous()`（多一次运行时拷贝）；给 `qwen_rmsnorm` 加 3D stride 支持（kernel 里已有 `x_stride_m/x_stride_n` 参数，只是 wrapper 硬写成 `d_model` 和 `1`，但单个 `stride_m` 表达不了 `[T,8,256]` 这种两级行结构，要改成 `(t,h,d)` 三维 stride + 二维 grid）。后者更通用，如果哪天想省掉那次额外的 GEMM launch 可以回头做。

K 路径没有这个问题：`k_proj` 输出 `[T,512]` 连续，`view(T,2,256)` 也连续。

## 5. Gated DeltaNet 完整数据流

投影：

```text
in_proj_qkv: 1024 -> 6144
in_proj_z:   1024 -> 2048
in_proj_a:   1024 -> 16
in_proj_b:   1024 -> 16
```

kernel 级串联（输入 `x: [T,1024]`）：

```text
qwen_rmsnorm(x)
  ├─ in_proj_qkv -> [T,6144]
  ├─ in_proj_z   -> [T,2048] -> reshape [T,16,128]
  ├─ in_proj_a   -> [T,16]
  └─ in_proj_b   -> [T,16]

depthwise_causal_conv4_prefill(qkv, conv_weight)   # 含 SiLU
  -> [T,6144] -> split 成 q,k,v: [T,16,128]

gdn_qk_norm_gates(q, k, a, b, A_log, dt_bias)
  -> q_norm,k_norm: [T,16,128] BF16
     beta,g:        [T,16]     FP32

gdn_recurrent_prefill_sequential 或 chunked(q_norm,k_norm,v,beta,g)
  -> out:   [T,16,128] BF16
     state: [16,128,128] FP32

gdn_gated_rmsnorm(out, z, norm_weight_fp32) -> [T,16,128]
  -> reshape [T,2048]
out_proj [1024,2048] -> [T,1024]
residual_add
```

`z`、`a`、`b` 不经过 Conv1D。

核心定义（与 kernel 内实现一致，留作交叉核对）：

```text
Conv:   y[t,c] = silu(sum(r=0..3, weight[c,r] * x[t+r-3,c]))，负下标按零处理

beta = sigmoid(b)
g    = -exp(A_log_fp32) * softplus(a_fp32 + dt_bias)

q = q * rsqrt(sum(q^2) + 1e-6) / sqrt(128)
k = k * rsqrt(sum(k^2) + 1e-6)

S: [128,128] FP32，初始为零
S      = exp(g_t) * S
memory = k_t @ S
delta  = beta_t * (v_t - memory)
S      = S + outer(k_t, delta)
o_t    = q_t @ S

n = o * rsqrt(mean(o^2) + 1e-6)
n = n * norm_weight_fp32          # 没有 1 + weight
n = n * silu(z_fp32)
```

## 6. 数值 oracle 与已核实的结构假设

### 6.1 oracle 的产生方式

参考实现是 `transformers 5.16.1`，装在项目内的独立 venv `.venv-oracle/`（用 `--system-site-packages` 复用系统 torch 2.12.1，没有动主环境）。生成脚本是 `tools/dump_oracle.py`：

```bash
.venv-oracle/bin/python tools/dump_oracle.py
```

产物写到 `oracle/`（已 gitignore，约 11 MB），内容：

```text
meta.pt           input_ids、rendered prompt、生成的 32 个 token
hidden.pt         hidden_00..hidden_24 + layer23_out + final_norm
layer00_gdn.pt    第 0 层（GDN）39 个中间量
layer03_attn.pt   第 3 层（full attention）36 个中间量
logits.pt         最后一个位置的 logits [248320] 和 greedy token
index.json        全部 key 的清单
```

**`hidden_states` 的索引有个坑**：`hidden_00` 是 embedding 输出，`hidden_{i+1}` 是第 i 层的输出——但**只到 i=22**。`hidden_24` 已经是 final norm 之后的值（RMS 从 0.296 跳到 4.105），不是第 23 层的原始输出。所以：

- 第 23 层的原始输出由单独的 hook 存成 `layer23_out`；
- `final_norm` 就等于 `hidden_24`，不要再对它 norm 一次。

（第一版 dump 脚本正是对 `hidden_states[-1]` 又调了一次 `norm`，导致对拍时最后两项显示 90% / 33% 的假误差。）

模型以 `attn_implementation="eager"` 加载，避免落到 flash/sdpa 的融合分支。

固定 prompt 是 `你好，请简单介绍一下自己。`，套 `tokenize_text.py` 里的单轮 non-thinking chat 模板，共 19 个 token。**端到端验收基准**（两次运行完全一致）：

```text
greedy_first_token = 109266
generated = [109266, 6115, 103724, 1167, 16451, 18, 13, 20, 3709, 96336, 110619,
             97793, 113820, 95974, 96359, 103911, 117356, 95726, 96566, 105019,
             98277, 103725, 1710, 95815, 98897, 108696, 98277, 98142, 5205,
             111892, 5205, 104062]
text = "你好！我是 Qwen3.5，由阿里巴巴集团旗下的通义实验室自主研发的超大规模语言模型。我具备强大的语言理解、推理、对话"
```

除模块 forward hook 外，脚本还 patch 了三个模块级自由函数（`causal_conv1d_fn`、`torch_chunk_gated_delta_rule`、`apply_rotary_pos_emb`），因为它们不是 `nn.Module.__call__`，hook 抓不到。所以 conv 输出、delta rule 的 q/k/v/beta/g/final_state、RoPE 前后的 q/k 都有独立锚点——某一层对不上时能直接定位到具体算子。

### 6.2 已核实的结构假设

以下全部**既读了 `modeling_qwen3_5.py` 源码，又用 dump 出来的张量做了数值验证**（带反向对照），不再是调研推测：

| 假设 | 结论 | 证据 |
|---|---|---|
| GDN conv 输出按 `[0:2048\|2048:4096\|4096:6144]` 连续切 q/k/v | 成立 | `torch.split(mixed_qkv,[key_dim,key_dim,value_dim],-1)`；数值 max diff **0.0** |
| conv tap 方向 `y[t,c]=silu(Σ_r w[c,r]·x[t+r-3,c])` | 成立 | 正向 0.28%（BF16 舍入），反向 96.94% |
| `q_proj` view 成 `[T,8,512]` 后每 head 切 `[0:256]=Q`、`[256:512]=gate` | 成立 | 数值 max diff **0.0** |
| gate 与 attention 输出都是 head-major，逐元素乘 sigmoid(gate) | 成立 | 重建 0.5%；不加 gate 3.29、gate 顺序打乱 1.50 作对照 |
| 纯文本下 MRoPE 退化为普通 RoPE | 成立 | `cos[:32]==cos[32:]` 精确相等；与 `cos(pos·inv_freq)` 差 2e-3（BF16） |
| `Qwen3_5RMSNorm` = `x·rsqrt(mean(x²)+eps)·(1+w)`，全 FP32 | 成立 | 源码逐行一致 |
| rotary_dim=64，`inv_freq[j]=1/10⁷^(2j/64)`，配对 (j, j+32)，`[64:256]` 直通 | 成立 | `rotate_half` 在 64 维上切半，`cat((freqs,freqs))` 使 `cos[j]==cos[j+32]` |
| `linear_num_value_heads / linear_num_key_heads = 16/16 = 1` | 成立 | `repeat_interleave` 是 no-op |
| GDN 的 `1/sqrt(128)` scale 只作用于 q，在 delta rule 内部 | 成立 | `scale = 1/query.shape[-1]**0.5`，仅 `query = query * scale` |
| l2norm 用 `sum` 不是 `mean`，eps=1e-6 | 成立 | `rsqrt((x*x).sum(-1)+1e-6)` |
| `g = -exp(A_log_fp32)·softplus(a_fp32+dt_bias)`，全 FP32 | 成立 | 源码一致 |

注意 HF 在 prefill（`seq_len>1` 且无 cache）走的是 `torch_chunk_gated_delta_rule`，即 chunk-64 路径，不是逐 token recurrent。

### 6.3 参考实现比我们精度更低的四处

这几处**不是 bug**，是参考实现在中途 round 到 BF16 而我们全程 FP32。逐层对拍时不能期待 bit-match，要按容差比；而且误差方向是"我们更准"：

1. **`Qwen3_5RMSNormGated`**：FP32 归一化后 `hidden_states.to(input_dtype)` **先转 BF16，再乘 weight**。我们的 `gdn_gated_rmsnorm` 从头到尾 FP32。
2. **GDN 的 l2norm**：在 `.to(torch.float32)` 之前调用，所以 q/k 的 L2 归一化是在 **BF16** 下做的。我们在 FP32。
3. **`beta = b.sigmoid()`**：在 BF16 下算 sigmoid，之后才转 FP32。我们直接 FP32。
4. **RoPE**：`cos/sin` 先 `.to(x.dtype)` 转成 BF16 再做旋转。我们在 kernel 内用 FP32 算 sin/cos。

推论：单算子对拍容差按 BF16 量级设（相对误差 ~1%）；真正该盯死的是 **greedy token 序列完全一致**，以及逐层误差不随层数放大。近似平局时 argmax 可能被这些差异翻转，如果 token 序列在某一步分叉，先看那一步 top-2 的 logit 间距。

## 7. 引擎实现

### 7.1 `engine/loader.py`

`load_text_weights(model_dir, device) -> TextWeights`。只读 `model.language_model.*`，视觉塔和 MTP 的张量根本不 `get_tensor`，不占显存。实测加载 1.43s、1.426 GiB。

权重按层类型分成 `GDNLayerWeights` / `AttnLayerWeights` 两个 frozen dataclass，字段名与 checkpoint 张量名一一对应。

两处不是原样搬运：

1. **`q_proj.weight [4096,1024]` 拆成 `q_proj_q` 和 `q_proj_gate` 各 `[2048,1024]`**，按第 4 节的方案。行布局是 head-major，第 h 个 head 占 `[h*512, h*512+512)`，前 256 行 Q、后 256 行 gate；拆完保持 head 顺序，两个输出都 `.contiguous()`。这样 runner 里 `view(T,8,256)` 连续，能直接喂 `qwen_rmsnorm`。
2. `conv1d.weight [6144,1,4]` squeeze 成 `[6144,4]`。

dtype **只断言不转换**——`A_log` 和 `linear_attn.norm.weight` 必须是 FP32，`dt_bias` 必须是 BF16。静默 cast 会掩盖上游改动。

收尾断言：18 GDN + 6 attn 层；所有 `model.language_model.*` 张量一个不漏地被读到；未读的必须全部落在 `model.visual.` / `mtp.` 前缀内（即"跳过是有意的"）；参数量恰好 `752,393,024`；checkpoint 里不存在独立 `lm_head`。

```bash
python engine/loader.py    # 打印摘要并跑全部断言
```

### 7.2 `engine/runner.py`

`Qwen35Runner`，完整重算路径：

```python
runner = build_runner()                      # 或 Qwen35Runner(load_text_weights())
tokens = runner.generate(input_ids, max_new_tokens=32)
```

- `forward(input_ids, trace=None, trace_layers={0,3}) -> [T,1024]`，返回 final norm 之后的 hidden。传 `trace` 字典即可捞出中间量，逐层对拍就是靠它，不需要在 runner 里插调试代码。
- `next_token()` 完整重算一次并返回 greedy token；`generate()` 套循环，默认停止 token 为 `<|im_end|>`(248046) 和 `<|endoftext|>`(248044)。

实测 19 token prompt 生成 32 token 耗时 35s（约 1.09 s/token）——完整重算 + 逐 token GDN sequential，慢是预期内的，这一版只追求正确性。

```bash
python engine/runner.py    # 端到端跑一遍并打印生成结果
```

### 7.3 对拍结果

```bash
python tests/test_gemm_model_shapes.py   # gemm_2d 真实尺寸，28 组
python tests/test_oracle_parity.py       # 对 oracle 逐算子 + 端到端，49 项
```

`tests/test_oracle_parity.py` 的结果（2026-09-01，A100）：

```text
逐算子检查 49 项，超容差 0 项
最大相对误差            layer15.out = 3.94%
逐层 hidden 相对误差    首层 0.40% -> 末层 1.11%（不发散）
端到端 greedy           32 个 token 与 oracle 逐个相同
```

关键的几个算子级锚点：

```text
layer00 conv(+silu)            0.31%
layer00 delta rule 输出         0.43%
layer00 gated rmsnorm          0.61%
layer03 q_proj 拆分后 Q/gate    0.49% / 0.99%
layer03 rope_q / rope_k        0.89% / 0.69%
layer03 gated pack             0.98%
```

误差量级与 6.3 描述的 BF16 舍入一致，没有出现单点突变。

**"greedy token 全对"是 prompt 相关的，不是不变量。** 在 oracle 那个 19 token 的
prompt 上 32 个 token 全对；但换一个 prompt（`李世民是谁？和朱棣有什么共同点？`），
即使 `compile=False` 也会在第 12 个 token 与 HF 分叉，该步 top1-top2 间距 0.1622
（占 top1 的 0.926%）——落在 6.3 那四处 BF16-vs-FP32 差异造成的 ~1.1% 噪声带内，
属于预期行为而不是 bug。

判断分叉是否可接受，看的是**该步的 top-2 间距相对我们的噪声量级**，而不是 token
是否相同。逐算子对拍（49 项）才是稳定的判据。如果确实需要与 HF 逐 token 一致，
唯一办法是主动把精度降到与参考实现相同——即在 6.3 那四处也 round 到 BF16。
那是一个取舍：贴合参考 vs 更准确，目前选的是后者。

## 8. 性能现状：为什么瓶颈不是计算

这一节的结论推翻了本文档早期版本给出的优化顺序，全部为 A100 实测（绑核到单核、
满 boost 3294MHz；不绑核时干活的核也会 boost，空闲核停在 1500MHz，`lscpu` 显示的
"66%" 是这么来的，不影响测量）。

### 8.1 eager 下 92% 的时间是调用开销

```text
T=19    41.3 ms/forward        T=129   44.6 ms
T=35    41.3 ms                T=257   44.5 ms
T=55    44.0 ms
```

**forward 耗时几乎不随 T 变化。** 一次 forward 有 422 次 op 调用，每次约 90μs，
合计约 38ms —— 占 41ms 的 92%。也就是说当前根本没到 compute bound，
"完整重算浪费 FLOPs" 在这个尺寸下不是问题。

单次 `gemm_2d` 调用的 90μs 构成：

```text
~16 us   实际计算 + 最小分发（以 cuBLAS 同尺寸 15.7us 为参照）
~35 us   Triton 自己的 Python launch 路径（autotune + jit.run + 13 个参数的特化）
~40 us   torch.library 的分发链
```

### 8.2 那 40μs 具体是什么

同一个 trivial 负载，逐层加封装：

```text
(a) 裸 python 函数                 3.32 us
(b) torch.library.Library（低层）   7.25 us   (+3.93)  ← dispatcher 本身
(c) torch.library.custom_op       16.00 us   (+8.75)  ← custom_op 的安全机制
(d) torch.library.triton_op       18.56 us   (+2.56)  ← wrap_triton

custom_op 在 no_grad 下            15.81 us            ← 关梯度也省不掉
```

所以**不是 "torch 分发慢"**——dispatcher 本身只要 3.93μs。贵的是 `custom_op`
每次调用无条件跑的安全机制（`torch/_library/custom_ops.py:374`）：schema 查找 →
`_is_view_op()` → `_c_check_aliasing_constraint`（遍历所有 args/kwargs/results
检查 storage 别名），外面还套 `autograd_impl` → `forward_no_grad` →
`redispatch_boxed` 三层 boxing。参数越多越贵，`gemm_2d` 那种一堆 stride 的签名尤其吃亏。

把 `triton_op` 换成 `custom_op` 只省 2.56μs，解决不了问题，还会让 Inductor
没法 trace 进 kernel。**这 40μs 是进入 compile 生态的门票钱，正确做法是把 compile
用起来让它归零，而不是把门票退掉。**

备选（尚未采用）：低层 `torch.library.Library` 注册只要 7.25μs，比 `triton_op`
便宜 2.6 倍，且照样能配 `register_fake`。代价是 Inductor 只能把 kernel 当黑盒——
对这 13 个手写 kernel 而言本来也不希望它 fuse 进去。如果将来要留在 eager，这是条路。

### 8.3 torch.compile 的实测收益与代价

```text
0 图断点，单图 572 个 op        ← 13 个 kernel 全部 trace 通
eager                41.3 ms
compile(default)      6.4 ms    6.5x
compile(reduce-overhead) 7.7 ms  ← cudagraph 在这个尺寸下反而更慢
```

`register_fake` 是这条路走通的前提，写它的投入在这里兑现。

**启动成本**：

```text
冷缓存（每个代码版本第一次）   50s + 65s
热缓存（每个进程）            2.3s + 2.5s ≈ 4.8s
```

T 第一次变化时会再编一次（切 dynamic shape），之后任意 T 复用同一张图；
只有跨过 kernel 的 `T_BUCKET` 边界（1/16/17/64/65/128/129 附近）才会再编——
因为 `T_BUCKET` 是 `tl.constexpr`，桶变了就是另一份特化。

**盈亏平衡约 137 次 forward**（4.8s ÷ 每次省 35ms）。完整重算下一次 forward
出一个 token，所以生成不到约 137 个 token 时 eager 更快：32 token 大约 5.0s，
eager 只要 1.3s。`Qwen35Runner(compile=False)` 可关。

**数值代价**：compiled 与 eager 的 final hidden 相对差约 1.1%（BF16 量级，
13.9% 的元素差 >0.1）。这在固定 prompt 上的第 28 步翻转了一次 argmax——但那一步
top1-top2 间距只有 0.0067（占 top1 的 0.029%），其余各步是 1.4–3.5。属于模型本身
无所谓的位置上噪声翻硬币，不是 bug。

**推论**：不要把"greedy token 全对"当成唯一验收标准，它在平局处天然是脆的。
`tests/test_oracle_parity.py` 现在会在分叉时自动报告该步的 top-2 间距，
间距 <0.5% 判为近似平局。没有这个数字，下次有人看到 token 不一致会去查一个
根本不存在的 bug。

### 8.4 GPU 侧的真实构成

compile 把 CPU 开销降下来之后，CPU 入队和 GPU 执行几乎完全平衡，GPU 时间开始随 T 增长：

```text
T=200   CPU 入队  9.15 ms | GPU 执行  8.84 ms
T=400   CPU 入队 14.38 ms | GPU 执行 14.23 ms
```

各部分占 forward GPU 时间的比例：

```text
        forward 总    GQA×6层   占比    GDN seq×18层  占比
T=128     20.49ms     0.64ms     3%       2.38ms      12%
T=256     25.54ms     0.71ms     3%       4.49ms      18%
T=512     25.60ms     0.63ms     2%       8.61ms      34%
T=1024    30.19ms     0.68ms     2%      17.07ms      57%
```

两个结论：

- **full attention 只占 2-3%，而且完全不随 T 增长**（0.63-0.71ms 一条平线）。
  给它加 KV cache 不是提速手段——它的价值只在于"增量 decode 的必需件"。
- **GDN sequential 是唯一随 T 显著增长的项**，从 12% 涨到 57%。切到
  `gdn_recurrent_decode` 后从 O(T) 变 O(1)，这是最大的单点收益。

### 8.5 `gemm_2d` 的效率是达标的

用 CUDA Graph 摘掉 CPU 开销后测真实核性能：

```text
        (T,K,N)     triton     cuBLAS      比   triton TFLOPS
(128,1024,6144)     15.3us     13.0us    1.2x          105.2
(512,1024,6144)     44.4us     36.8us    1.2x          145.1
(512,1024,3584)     32.2us     30.5us    1.1x          116.6
(512,3584,1024)     44.7us     26.8us    1.7x           84.1
(1024,1024,3584)    57.5us     60.7us    0.9x          130.7
```

105-145 TFLOPS，与 cuBLAS 相差 1.1-1.2 倍，有一例还更快。**不要去优化它**，
唯一偏弱的是 `K=3584` 那组（1.7x），要动也只动那一个形状的 tiling。

**这一节存在的主要意义是记录一个测量方法**：在 eager 下逐 kernel 计时会得到
"`gemm_2d` 比 cuBLAS 慢 5.3 倍、只有 6.9 TFLOPS"的结论，那是**假的**——每次调用
约 90μs 的 CPU 开销远大于 15μs 的 GPU 工作，GPU 是饿死的，CUDA event 量到的是
CPU 停顿。同样地，8.4 表里 T=128 那行的 "forward 总 GPU 20.49ms" 也被 CPU 拖高了，
真实 GPU 工作量按 FLOPs 估只有几毫秒。

**要测 kernel 的真实 GPU 效率，必须先用 CUDA Graph 或 torch.compile 把 CPU 摘掉。**
本仓库任何"kernel 慢"的结论，如果不是这样测的，先怀疑测量方法。

### 8.6 一个测量陷阱

不要在同一个进程里先跑 compiled 再跑 eager 做对照。dynamo 的 frame hook 会拦截
同一个函数对象反复做 guard 检查和重编译尝试，eager 会被测成 6142 ms/token
（真实值 41ms，差 150 倍）。要干净的 eager 基线就开新进程，或用 `compile=False`
构造独立的 runner。

## 9. 剩余工作

**正确性里程碑已达成**：13 个 kernel + loader + runner 打通，49 项逐算子对拍全部在容差内，oracle prompt 上端到端 greedy 32 个 token 与 Hugging Face 逐个相同（第 7 节）。注意 token 全对是 prompt 相关的，逐算子对拍才是稳定判据，见 7.3 末尾。

`demo.py` 可以直接对话：`python demo.py "你的问题"`。

已完成的：全部 kernel、`gemm_2d` 真实尺寸测试、数值 oracle、权重加载器、单层对拍、24 层完整重算 runner。

**注意优先级已按第 8 节的实测重排。** 早期版本把"切增量 decode"排第一，那是错的：
当前根本不是 compute bound，forward 耗时在 T=19 到 257 之间几乎不变，92% 是调用开销。
在解决调用开销之前，任何减少 FLOPs 的优化都量不出收益。

剩下的按建议顺序：

1. **降低启动成本**，让默认开的 compile 真正划算。当前盈亏平衡在约 137 次 forward，
   短生成反而更慢。两个方向：
   - 收敛 kernel 的 `T_BUCKET` 桶边界，减少跨桶重编译（现在 1/16/17/64/65/128/129
     各是一次重编译，每次数十秒）；桶少一点、粗一点，代价是个别长度的 tuning 不最优。
   - 评估低层 `torch.library.Library` 注册（7.25μs vs `triton_op` 18.56μs，见 8.2）。
     这条路 eager 就能快 2.6 倍，不依赖 compile，适合短生成场景。
2. **切增量 decode**（最大的一块）。两重收益：GDN 从 O(T) 变 O(1)——它是唯一随 T
   显著增长的项，T=1024 时占 GPU 时间 57%（见 8.4）；同时 T 固定为 1 让 shape 静态化，
   cudagraph 变得可用，而 cudagraph 能把 CPU 开销完全摘掉（8.5 里 93μs → 15μs 就是
   这么来的）。两者是乘性的。

   需要三件，**必须齐了才能切**——24 层里 18 层是 GDN，缺 `conv4_decode` 这 18 层就
   得全序列重算；而每层输入依赖上一层所有位置的输出，所以只要有一层要全序列，
   那 6 层 attention 也拿不到"只有一个新 token"的输入：

   - `depthwise_causal_conv4_decode`：维护 `[6144,4]` conv state，移位/环形更新后做
     4-tap dot product。三件里最小，建议先做；
   - 带 KV cache 的 GQA decode kernel：只算新 query 对全部历史 K/V。**注意它本身不是
     提速手段**——full attention 只占 GPU 时间 2-3% 且不随 T 增长（8.4），做它纯粹
     因为它是 decode 模式的必需件；
   - GDN 侧直接接已有的 `gdn_recurrent_decode`（state cache 路径已验证）。

   齐了之后把 runner 拆成 prefill + decode 两条路径，用当前的完整重算版本做等价性
   对拍——**现在这个 runner 就是那个基准**（`compile=False` 那条路径逐算子对得上 HF）。
3. **重写 chunk-64 prefill**。注意是重写不是切换：当前实现比 sequential 慢约 20 倍
   （见 3.10），`prepare_wy` 里 64 步串行的前代法求逆是主因。要动就得把它换成分块回代
   或 Newton 迭代，并在 DK/DV 上分块。优先级低于第 2 项，因为 prefill 只跑一次。
4. **扩测试覆盖**：目前 oracle 只有一个 19 token 的中文 prompt。至少再加一个长 prompt
   （跨过 chunk-64 和 `T_BUCKET` 边界，比如 T=65/129）和一个英文 prompt。
5. **kernel 级性能收尾**：`gdn_qk_norm_gates` 的 head 分块、`chunk_state`/`chunk_output`
   的 BLOCK_V autotune。**不包括 `gemm_2d`**——它已经是 cuBLAS 的 1.1-1.2 倍，达标了（8.5）。
6. 可选：top-k/top-p/temperature sampling、batch/padding。

host 侧还缺的：`tokenize_text.py` 目前只实现了单轮 non-thinking chat 模板的子集，多轮对话和 thinking 模式要补。停止 token（`<|im_end|>`=248046、`<|endoftext|>`=248044）已经在 runner 里作为默认值。

## 10. 权重与运行环境

### 10.1 Checkpoint 张量名

本地 checkpoint 为单文件 `Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors`，索引在 `model.safetensors.index.json`。三个顶层命名空间：

```text
model.language_model.*   320 个张量  ← 只需要这些
model.visual.*           153 个张量  ← 跳过
mtp.*                     15 个张量  ← 跳过
                         488 合计
```

文本主干的完整张量名：

```text
model.language_model.embed_tokens.weight        [248320,1024] BF16，同时作为 LM head
model.language_model.norm.weight                [1024]

model.language_model.layers.{i}.input_layernorm.weight
model.language_model.layers.{i}.post_attention_layernorm.weight
model.language_model.layers.{i}.mlp.gate_proj.weight      [3584,1024]
model.language_model.layers.{i}.mlp.up_proj.weight        [3584,1024]
model.language_model.layers.{i}.mlp.down_proj.weight      [1024,3584]
```

GDN 层（i ∉ {3,7,11,15,19,23}）额外有：

```text
    linear_attn.in_proj_qkv.weight   [6144,1024]
    linear_attn.in_proj_z.weight     [2048,1024]
    linear_attn.in_proj_a.weight     [16,1024]
    linear_attn.in_proj_b.weight     [16,1024]
    linear_attn.conv1d.weight        [6144,1,4]
    linear_attn.out_proj.weight      [1024,2048]
    linear_attn.A_log                [16]   FP32
    linear_attn.dt_bias              [16]   BF16
    linear_attn.norm.weight          [128]  FP32
```

full-attention 层（i ∈ {3,7,11,15,19,23}）额外有：

```text
    self_attn.q_proj.weight   [4096,1024]   一半是 Q 一半是 gate，见第 4 节
    self_attn.k_proj.weight   [512,1024]
    self_attn.v_proj.weight   [512,1024]
    self_attn.o_proj.weight   [1024,2048]
    self_attn.q_norm.weight   [256]
    self_attn.k_norm.weight   [256]
```

所有线性层都没有 bias；名字里唯一带 bias 的是 GDN 的 `dt_bias`，它是 softplus 的偏置，不是 GEMM bias。以上形状和 dtype 已直接从 safetensors header 核对。

### 10.2 环境

已确认环境：

```text
GPU:    NVIDIA A100-SXM4-40GB
Torch:  2.12.1+cu128
Triton: 3.7.1
```

主环境**没有装 transformers**，引擎本体也不依赖它。参考实现隔离在项目内的 `.venv-oracle/`（`python -m venv --system-site-packages` 复用系统 torch，只额外装了 transformers 5.16.1），仅用于跑 `tools/dump_oracle.py` 生成 `oracle/` 下的参考张量。venv 和 oracle 产物都已 gitignore；重建方式：

```bash
python -m venv --system-site-packages .venv-oracle
.venv-oracle/bin/pip install transformers
.venv-oracle/bin/python tools/dump_oracle.py
```

全部 kernel 自测命令（2026-09-01 全部通过）：

```bash
python triton_kernels/gemm_2d.py
python triton_kernels/qwen_rmsnorm.py
python triton_kernels/gqa_attention_without_kvcache_casual.py
python triton_kernels/partial_rope.py
python triton_kernels/attention_gate_pack.py
python triton_kernels/residual_add.py
python triton_kernels/swiglu.py
python triton_kernels/depthwise_causal_conv4_prefill.py
python triton_kernels/gdn_qk_norm_gates.py
python triton_kernels/gdn_recurrent_prefill.py
python triton_kernels/gdn_gated_rmsnorm.py
python triton_kernels/embedding_gather.py
python triton_kernels/vocab_argmax.py
```

引擎和对拍：

```bash
python demo.py "李世民是谁？和朱棣有什么共同点？"   # 交互 demo，流式输出
python engine/loader.py                        # 权重加载 + 全部断言
python engine/runner.py                        # 端到端生成 32 token
python tests/test_gemm_model_shapes.py         # gemm_2d 真实尺寸，28 组
python tests/test_oracle_parity.py             # 逐算子 + 端到端 + compiled/eager，49 项
python tests/test_oracle_parity.py --no-compile  # 跳过 compiled 那一节，省几秒到两分钟
```

对拍脚本用 `compile=False` 构造 runner——compiled 路径与 eager 相对差约 1.1%，
不能用来做逐算子严格对拍（见 8.3）。

`gdn_recurrent_prefill.py` 跑得最久（chunk + sequential + decode 三组，含 autotune）。

修改 kernel 后至少执行：

```bash
python -m py_compile triton_kernels/<file>.py
python triton_kernels/<file>.py
```

## 11. 协作偏好

用户有 kernel 经验，希望沟通简洁：默认只说明输入、输出、运算逻辑和必要的布局/精度问题。

用户通常自己写 kernel；只有明确要求修改代码时才直接编辑。修改已有 kernel 时应尽量保留用户的主体结构和命名，只做必要改动；如果必须修正本体逻辑，应明确展示和解释改动前后差异，避免无关重构。
