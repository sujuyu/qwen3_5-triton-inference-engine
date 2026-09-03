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

关于 cache 的现状：GDN 两个 cache 的 decode kernel 都已就位——`gdn_recurrent_decode` 原地更新 `[16,128,128]` FP32 recurrent state（3.10b），`depthwise_causal_conv4_decode` 原地更新 `[4,6144]` BF16 conv state（3.8b），两者都验证过「prefill 前缀 + 逐 token decode」与整段 prefill 一致。full attention 的 KV cache decode kernel 也已完成（3.3b）。**三类 cache 至此齐备**，下一步是把 runner 拆成 prefill + decode 两条路径；在那之前当前仍按「每生成一个 token，对增长后的完整序列重新执行一次 forward」实现。

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

共 15 个文件，全部位于 `triton_kernels/`，全部已有 autotune（少数固定配置）、`torch.library.triton_op` 封装、`register_fake` 和 PyTorch reference，且自测通过。注册的 op 命名空间统一为 `wy_lib::`。

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

### 3.8b `depthwise_causal_conv4_decode.py` — `wy_lib::depthwise_causal_conv4_decode`

3.8 的 decode 版，维护 GDN 的 conv state。

```text
x:      [1,D] BF16      新 token 的 in_proj_qkv 输出，模型 D=6144
state:  [4,D] BF16      原地更新，必须 contiguous
weight: [4,D] BF16      必须 contiguous，用 conv_weight_for_decode() 从
                        checkpoint 的 [D,1,4] 转换
out:    [1,D] BF16      含 SiLU
```

**注意是 `[4,D]` 不是 `[D,4]`。** kernel 里一个线程负责一个 channel，取第 k 个 tap 时
`[D,4]` 下相邻线程相隔 4 个元素（32 字节的访存粒度里只用 4/16），`[4,D]` 下地址连续、
完全合并。A100 实测（CUDA Graph，D=6144）：

```text
布局                      BLOCK=256   BLOCK=512   BLOCK=1024
state[D,4] weight[D,4]     1.88us      2.48us      5.14us
state[4,D] weight[D,4]     1.68us      1.95us      3.27us
state[4,D] weight[4,D]     1.60us      1.69us      1.89us   ← 采用
```

除了更快，`[4,D]` 对 BLOCK_D 几乎不敏感，autotune 才有实际选择空间。

两个张量都必须是**真正 contiguous 的 `[4,D]`**，不能是 `[D,4]` 的转置 view——
转置 view 数值上正确但内存布局仍是 `[D,4]`，合并访问的好处全没了。wrapper 里有
assert 挡着，测试里也有一条专门验证它会被拒绝。

因此 decode 需要一份独立于 prefill 的权重副本（prefill 用 `[D,1,4]`），
在 loader 里转一次，18 层多占 0.86 MiB。

conv kernel size 是 4，算 `y[t]` 要用 `x[t-3..t]`，decode 时只有 `x[t]` 是新的，
前三个从 state 取。**注意这与 delta rule 的 `[16,128,128]` recurrent state 是两个
独立的 cache**，GDN 每层两个都要。state 存的是 **conv 的输入**（`in_proj_qkv` 的输出），
不是 conv 输出、也不是 SiLU 之后的值。18 层合计 0.84 MiB。

约定对齐 transformers 的 `causal_conv1d_update`（`state_len = conv_kernel_size = 4`）：
更新后 `state[:,c]` 恰好是 `x[t-3..t]`，点积不用再做下标偏移。

```text
state[:,c] = concat(state[1:,c], x[c])          # 左移一格，新值放末尾
acc  = sum_{r=0..3} weight[r,c] * state[r,c]    # FP32 累加
y[c] = acc * sigmoid(acc)
```

实现上不需要 concat：kernel 里"移位"就是变量重命名（读 state 的第 1/2/3 行，
写回第 0/1/2 行，第 3 行写新值），第 0 行干脆不读。`tl.cat` 在这里**用不了**——
`[BLOCK_D,3]` 的 3 不是 2 的幂，会报 `Shape element 1 must be a power of 2`。
depthwise 通道间不混合，**不要用 `tl.dot`**。

kernel body 对布局是无感的——指针算术全靠 stride 参数驱动，从 `[D,4]` 换到 `[4,D]`
时 kernel 一行没改，只是 wrapper 里 `stride_state_d`/`stride_state_k` 的来源
从 `stride(0)`/`stride(1)` 换成了 `stride(1)`/`stride(0)`。

和 `gdn_recurrent_decode` 一样声明了 `mutates_args=("state",)`，autotune 配
`restore_value=["state_ptr"]`——不加的话调优反复试 config 会把 state 推进多次。
**调用方要注意这个 op 会就地改写传入的 state。**

配套工具：`conv_state_from_prefill(x) -> [4,D]` 从 prefill 的 conv 输入建初始 state
（取最后 4 行，`T < 4` 时上方补零）；`conv_weight_for_decode(w) -> [4,D]` 从 checkpoint
布局转换权重。

测试分两段：先验证 PyTorch 参考实现与 prefill kernel 一致（不依赖 Triton kernel），
再验证 Triton kernel 与参考实现一致。判据都是「prefill 前 n 个 token → 拿 state →
逐 token decode 剩下的」必须等于整段 prefill。覆盖 `T=1,2,3,4,17,65`（前四个专门
覆盖 `T<4` 的左侧补零路径）和通用 `D=1000`，最大绝对误差全部为 0。

访存量只有 `读 9D + 写 5D` ≈ 172 KiB。注意 eager 下单次调用要 89.4μs（几乎全是
`torch.library` 分发，见 8.2），而 kernel 本身只有 1.6μs——**这个 kernel 的任何
进一步优化都量不出来**，除非先把调用开销解决掉。

### 3.3b `gqa_attention_decode.py` — `wy_lib::gqa_attention_decode`

3.3 的 decode 版，带整块 KV cache。三类 cache 至此齐备。

```text
q:       [H_q, D]        BF16   新 token 的 query，已过 q_norm 和 RoPE
k_new:   [H_kv, D]       BF16   新 token 的 key，已过 k_norm 和 RoPE
v_new:   [H_kv, D]       BF16
k_cache: [H_kv,T_max,D]  BF16   原地追加
v_cache: [H_kv,T_max,D]  BF16   原地追加
past_len: int                   追加位置
out:     [H_q, D]        BF16
```

与 prefill 版的三个差别：**不需要 causal mask**（cache 里每个位置都该被 attend，
唯一的 mask 是 `offset_t < seq_len` 的越界保护）；query 只有一行，没有 Q 方向分块；
K/V 来自 cache。

**cache 里存的必须是 RoPE 之后的 K**——参考实现是先 `apply_rotary_pos_emb`
再 `past_key_values.update`，历史 token 的 position 不会变。

布局选 `[H_kv, T_max, D]`：读时 D 维连续、每行 512B 用满，写时每个 head 一次连续写。
另两种布局的写入都是跨步的。KV cache 8K 上下文 96 MiB 远超 40MB L2，这里的合并访问
是实打实的 DRAM 带宽。显存 12 KiB/token，8K 上下文 96 MiB。

**不做 paging**：paging 解决的四个问题（多序列碎片、continuous batching、前缀共享、
beam search 分叉）当前一个都不存在。将来要加也便宜——T 方向本来就分块遍历，
插 paging 只是循环里多一次块表查询，工作量在 host 侧分配器，与 kernel 解耦。

追加放在 python wrapper 里而不是 kernel 里：kernel 内写完紧接着要读回同一位置，
跨 program 的写后读没有可见性保证。代价是每层多两次 PyTorch copy 的 launch。

**grid 是 `(H_kv,)` = 2，不是 `(H_q,)`。** 与 prefill 版同一个选择（KV 只读一遍）。
kernel 里 `offset_h = pid * GROUP + tl.arange(0, GROUP)`，用 `num_q_heads` 起 grid
会让 pid≥2 的 program 越界读 cache、越界写 `out`（写到第 31 行而 out 只有 8 行），
踩坏分配器里相邻的张量，表现为"数据相关的错值"，最后变成 illegal memory access。
这个 bug 实际发生过，见下面的调试记录。

测试：`T=1,17,65,129,257` 的 prefill+decode 等价性（T 特意取非 `BLOCK_T` 整数倍，
用来抓尾块没置 `-inf` 的 bug），最大绝对误差 ≤ `0.015625`；外加一条越界保护检查——
把 cache 中 `past_len` 之后填 NaN，结果必须完全不变。

**split-K 版已完成**，同文件里的 `gqa_attention_decode_split` + `gqa_attention_decode_combine`，
用 `call_gqa_attention_decode_split_triton()` 串起来，接口与不切 T 的版本完全一致。
不切 T 的版本保留作对拍基准。

CUDA Graph 下的纯 GPU 时间（已摘掉 `torch.library` 分发开销）：

```text
    T      不切T    split-K    加速     ×6层不切T   ×6层split
  512     24.6us    10.8us    2.3x      0.148ms    0.065ms
 2048     68.8us    12.4us    5.6x      0.413ms    0.074ms
 8192    244.7us    19.1us   12.8x      1.468ms    0.115ms
32768   1022.8us    68.4us   15.0x      6.137ms    0.410ms
```

不切 T 的版本随 T 线性增长（2 个 CTA 扫完整个序列），split-K 基本平坦。对照每个
decode step 必须读一遍 1.4 GiB 权重的 1.23 ms 地板：**不切 T 在 T=8192 时 attention
就和整个模型的权重读取一样贵，split-K 只占 9%**。

**注意别在 eager 下量这个对比**：那里 split-K 反而"更慢"（T=512 时 246 vs 138us），
因为它走两次 `torch.library` 分发、每次 50-90us（8.2），完全盖过 GPU 时间。
同样的陷阱见 8.5。

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

**这条路径已优化，现在比 sequential 快 4.1x**（T=2048，`H=16, DK=DV=128`）：

```text
                改前        改后      倍数
prepare_wy     4.146ms    0.338ms    12.3x
chunk_state    1.630ms    0.256ms     6.4x
chunk_output  32.402ms    0.105ms   308x
────────────────────────────────────────────
chunked       38.228ms    0.682ms    56x
sequential     2.801ms
```

改动是精度（8 个 `tl.dot` 的 `input_precision="ieee"` 绕开了 tensor core）和
`chunk_output` 的 `BLOCK_V` 硬编码（导致 attention 矩阵被 8 个 CTA 各算一遍），
**算法本身没动**。完整推导见 8.13——尤其是"先测再改"那一段：原本认定瓶颈是
`prepare_wy` 的逐行前代法，准备换成分块回代或 Newton 迭代，结果只改精度就够了，
那套重写是白工。

`runner._gdn` 按 `token_num >= 2048` 在两条路径间选。**阈值不是 kernel 级的交叉点
（T≈192），而是模型级的（T≈2048）**，原因见 8.14。

测试结果（A100 实测，`H=16, DK=DV=128`；长序列档是这次补的）：

```text
prepare_wy       T=1,3,63,64,65,129,512,1025   max_abs_w 4e-8   max_abs_u 1.6e-6   max_abs_g 1.9e-5
chunked prefill  T=1,3,63,64,65,129,512,1025   max_abs_out 1.2e-4   max_abs_state 1.9e-6
sequential       T=1,3,17,65                   max_abs_out 3.1e-5   max_abs_state 9e-8
prefill(17)+decode(48) 等价性                   max_abs_out 3.1e-5   max_abs_state 9e-8
```

chunked 和 sequential 都以逐 token FP32 PyTorch 实现为 reference。三条路径互相一致。
模型级上两条 prefill 路径的 hidden 差约 2.8%，但 greedy token 完全相同，
理由见 8.14。

**作者归属**：(a) sequential 和 (b) decode 由用户编写；(c) chunk-64 三段式最初由 Codex 生成，用户没有逐行审过，后由 Claude 做了上述性能改造（算法未动）。对拍时仍以 sequential 为基准。本仓库其余 kernel 均为用户编写。

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

### 8.6 decode 阶段的 CTA 并行度：为什么 split-K 是必须的

按 KV head 分块时 grid 只有 `(H_kv,)` = 2。实测流式读 96 MiB 的达成带宽：

```text
CTA 数     2 warp     4 warp     8 warp    16 warp
    2       4GB/s      8GB/s     21GB/s     46GB/s
    8      16GB/s     32GB/s     81GB/s    181GB/s
   32      62GB/s    122GB/s    308GB/s    628GB/s
  108     206GB/s    393GB/s    877GB/s   1186GB/s
  432     675GB/s    998GB/s   1171GB/s   1195GB/s   ← 饱和约 1170 GB/s
```

**2 个 CTA 即使开 16 warp 也只有饱和带宽的 4%**。加 warp 有用（4→46 GB/s，11 倍）
但补不回来——CTA 被限制在单个 SM 上，天花板就是 2/108。带宽大致正比于在飞的 warp
总数（约 1.4 GB/s per warp），要打满需要 ~100 个 CTA（8 warp）或 ~50 个（16 warp）。

折算到 6 层合计的每步 KV 读取耗时，对照每步必须读一遍 1.4 GiB 权重的 1.23 ms 地板：

```text
T       不切 T（2 CTA/8 warp）    split-K      占 decode 步
512            300 us              39 us        24% vs 3%
2048           1.2 ms              42 us        98% vs 3%
8192           4.8 ms              92 us       390% vs 7%
```

所以 split-K 不是"长上下文才需要的优化"，T=512 就已经吃掉四分之一。
`num_splits` 取 50~200（即 100~400 个 CTA）进饱和区。

顺带：prefill 版的 autotune `num_warps` 只到 `[2,4]`，那是因为它 CTA 数本来就多；
decode 是纯 memory-bound 且 CTA 极少，warp 数影响很大，应该覆盖到 8/16。

### 8.7 一次真实的 IMA 排查：grid 与 kernel 分块假设不一致

`gqa_attention_decode` 开发时踩到的，过程值得记下来。

**症状**：走完整 op 路径时，某些 `seq_len` 结果错（`err=1.48`）、某些出 NaN，
但 `K.fn` 直调所有配置全对；错误看起来"数据相关"且一度被误判为非确定性。

**根因**：wrapper 用 `grid=(num_q_heads,)=8` 启动，但 kernel 把 `pid` 当 KV head 用
（`offset_h = pid * GROUP + tl.arange(0, GROUP)`）。pid=2..7 的 program 越界读 cache、
**越界写 `out`**（写到第 31 行，out 只有 8 行），踩坏分配器里相邻的张量。

**排查中的两个教训**：

1. **"非确定性"是我自己的比较脚本造出来的**。判据写成 `if max(diffs) > 0`，
   而 `NaN > 0` 是 False，于是含 NaN 的结果被打印成"完全一致"。真实行为一直是
   确定的——同样输入三轮跑，错的位置完全相同。判断随机性时要先确认比较逻辑
   对 NaN 的处理。
2. **失败率不是 100% 时，单样本扫描没有意义**。第一轮"每个配置单跑一次都通过"
   完全是运气，后来每配置跑 20 次才看清。

**下次遇到类似症状的定位顺序**：先看是不是 IMA（`CUDA_LAUNCH_BLOCKING=1` 能把
异步报错拉到出错点），IMA 优先怀疑 grid/索引越界而不是数学；再对比"走完整 op 路径"
与"直调 `K.fn`"——两者结果不同就说明问题在 wrapper 而不在 kernel body。

### 8.8 CUDA Graph 对标量参数的约束（decode 路径已按此改造）

**CUDA Graph 在 capture 时会把标量 kernel 参数和切片下标烧进 launch 配置，
replay 时永远用 capture 那一刻的值。** decode 每步 `seq_len` 都在变，直接传标量
图就废了。实测对照：

```text
seq_len 作为标量参数：      capture 时 n=100，replay 传 300/777 得到的仍是 100
seq_len 放显存 + tl.load：  replay 得到 300/777，正确

cache[:, past_len, :] = v   （past_len 是 python int）  replay 三次都写第 0 行
cache.index_copy_(1, pos, v)（pos 是显存张量）          replay 三次写第 0/1/2 行
```

同一约束适用于**所有从 host 传入、且逐步变化的量**。做法是把它们统一收敛到一个
显存里的位置张量：

```python
pos = allocate_position()          # [1] INT64
seq_len   = tl.load(pos_ptr) + 1   # kernel 里读
chunk     = cdiv(seq_len, MAX_SPLITS)   # 也在 kernel 里算，host 不传
num_active = cdiv(seq_len, chunk)
```

`chunk` 和 `num_active` 刻意在 kernel 内重算而不是从 host 传：split 和 combine
必须用同一个来源，否则两边对"哪些 split 有效"的理解可能不一致，读到残留的局部量。

dtype 用 INT64 是因为 `index_copy_` 要求 index 为 long；kernel 里 load 之后立刻
`.to(tl.int32)`，避免后续 int64 运算。

**图内自增**：把 `pos.add_(1)` 也捕获进图，每次 replay 位置自动前进，host 侧一行
都不用碰。整个 decode step 塞进一张图之后，host 每步只剩「写输入槽 + `g.replay()`」。

**这让图变成有状态的**，有两个后果：

1. capture 过程本身（warmup + 正式捕获）会把 `pos` 推进好几格、也会污染 cache，
   **捕获完必须显式复位**；
2. 换 prompt、或 prefill 之后重新开始，同样要复位。真实 runner 应该有显式的
   `reset()` 而不是靠调用方记得。

**grid 必须在 capture 时固定**，这是 cudagraph 绕不过去的硬约束。`MAX_SPLITS` 当初
定成常数（而不是让 `num_splits` 随 seq_len 变）正是为此——否则连 grid 都得想办法。

autotune 的 key 只能是 host 侧标量，所以三个 op 都额外收一个 `seq_bucket: int`。
**它只影响选哪个 config，不参与任何计算**；graph 下 config 冻结在 capture 那一刻，
选错只是慢一点、不会算错。correctness 全部由显存里的 `pos` 决定。

改造范围只有 GQA 的三个 decode op——`conv4_decode` 和 `gdn_recurrent_decode`
都不含随步变化的标量，不用动。`gqa_attention_decode.py` 的第 5 段测试做了完整验证：
捕获 1 次、replay 32 次、`pos` 自动 8 -> 40，结果与整段 causal prefill 一致。

### 8.9 每个 bucket 一张图，以及为什么必须共享内存池

8.8 解决了标量和 grid 的冻结，但还剩一个：**config 也会被冻结**。

autotune 的 key 只能是 host 侧标量，所以三个 decode op 都额外收一个
`seq_bucket: int`。它不参与任何计算，只决定选哪个 config——但 config 里的
`BLOCK_T` 会随 bucket 变，capture 之后就固定了：

```text
seq_len=  64  BLOCK_T=16 warps=4 stages=2   12.53us
seq_len= 256  BLOCK_T=16 warps=4 stages=2   13.37us
seq_len=1024  BLOCK_T=16 warps=4 stages=1   13.43us
seq_len=4095  BLOCK_T=32 warps=4 stages=2   14.51us
```

拿短序列调出的 config 跑长序列只是慢一点、不会算错（correctness 全部由显存里的
`pos` 决定）。

**但上面这张表不能用来估算"每 bucket 一张图能赢多少"**，这里曾经犯过一次错。
它同时变了两件事：config 变了，序列也变长了。而序列变长的代价是物理的，
多少张图都省不掉。要隔离 config，得固定 seq_len、只改"冻结了哪个 config"——
也就是把短 prompt 时捕获的图拿去跑长序列（在图里测，否则 110μs 的 eager
分发开销会把 13μs 的差异整个盖住）：

```text
seq=  64 用自己的 config                    13.06us
seq=  64 用 4096 的 config                  13.38us     +2.4%
seq=4095 用自己的 config                    15.72us
seq=4095 用   64 的 config ← 短捕获长 replay  16.15us     +2.7%

序列 64 -> 4095 本身                                    +20.3%   ← 上表那 16% 是这个
```

**config 冻结的真实代价是 2.7%，不是 16%。** 而且还要再折一次：attention decode
只有 6 层，6 × 16μs ≈ 96μs，占 4.4ms 一步的 2.2%。所以每 bucket 一张图值
`2.7% × 2.2% ≈ 0.06%`。结论仍然是单张图够用，但理由是"这条路根本没什么可赢的"，
不是"跨度只有 16%"。要提速应该去看那 4.4ms 里剩下的 97.8%。

顺带一个副产品：中途误测了不分 split 的 `gqa_attention_decode`，seq=4095 要
170μs（grid 只有 2 个 CTA）。这反过来印证了 8.6 节坚持做 split-K 是对的——
split-K 把它从 170μs 压到 15.7μs。

要做得更好，方向是**每个 bucket 捕获一张图**，按 host 已知的 seq_len 选：

```python
graphs, pool = {}, None
for b in (64, 256, 1024, 4096):
    pos.fill_(b - 1); warmup()
    g = torch.cuda.CUDAGraph()
    with (torch.cuda.graph(g, pool=pool) if pool else torch.cuda.graph(g)):
        one_decode_step()
    pool = pool or g.pool()
    graphs[b] = g

graphs[_seq_bucket(seq_len)].replay()
```

**必须配共享内存池。** 先分清两类张量：

- **capture 之前**分配的（权重、KV cache、`pos`、输入输出槽）走普通 caching
  allocator，**不进图的池**。图只把它们的地址常量录进 kernel 参数，所以天然被
  所有图共享，与 `pool=` 无关。实测三次 capture 中它们的 `data_ptr` 始终不变。
- **capture 期间**分配的（op 里的 `out = torch.empty_like(q)` 等中间量）走图的
  **私有池**，默认每张图一个。

关键在于图里烧的是绝对地址，而图之后还要 replay 任意多次、每次都往那些地址写。
所以那批地址在图的整个生命周期内必须归它所有，**capture 结束不释放，池的生命周期
绑定在图对象上**：

```text
基线                        32.0 MiB
capture 图1（私有池）        66.0 MiB   +34
capture 图2（另一个私有池）   98.0 MiB   +34   ← 累加，不复用
删掉图1                     66.0 MiB   -34   ← 图一死才还回来
删掉图2                     32.0 MiB   -34

图3 存活时再普通分配 48 MiB，reserved 又涨 48 MiB   ← 池不与普通分配互通
```

所以「N 张图不共享 = N 套中间量同时驻留」**不是因为它们会并发执行**，而是每张图
永久持有自己 capture 时拿到的那批地址。哪怕永远串行 replay，只要图都活着，
地址就都得保留。实测（每次 capture 内 3 个 16 MiB 中间量，各配置独立进程）：

```text
图数      各自私有池     共享池
1 张       34.0 MiB     34.0 MiB
2 张       66.0 MiB     34.0 MiB
4 张      130.0 MiB     34.0 MiB
8 张      258.0 MiB     34.0 MiB      ← 私有 O(N)，共享 O(1)
```

共享池省的也不是「运行时错开使用」，而是 **capture 时就没有多分配**：第二次
capture 在同一个池的 free list 里找到上次回收的同样大小的块，原地发回去。
实测共享池时两张图的中间量地址完全相同，各开新池则不同。

**它并不「识别同一个变量」**。地址重合的条件是**同时存活的块的大小组合相同**——
allocator 按大小从 free list 找最佳匹配。多图方案里 N 张图跑的是同一段
`one_decode_step()`，只有 autotune config 不同，存活模式完全一致，才能拿到干净的
1/N。给结构不同的图（比如 prefill 图和 decode 图）共享池，只能「共用一片区域各取
所需」，不保证省这么多。

**代价：replay 之后必须先把结果拷到池外，才能 replay 同池的另一张图。**

```text
独立池：图A 的中间量活到「图A 下次被 replay」
共享池：图A 的中间量活到「同池任意一张图被 replay」
```

实测 replay 图1 得 1024、随后 replay 图2 得 2048，目标张量只剩最后一次的值。
所以 `one_decode_step()` 里那句 `o.copy_(...)` 是共享池方案的**必要条件**，
不是可有可无。「不能并发 replay」只是这条约束的特例（并发时窗口直接归零）。

`gqa_attention_decode.py` 里有一段 blog 风格的长注释完整讲了这套推理，
接 decode runner 前值得先读一遍。

### 8.10 上面这些在 PyTorch C++ 里的对应实现

上面 8.8 / 8.9 的结论都是黑盒实测出来的。回头对了一遍 PyTorch 源码
（`~/pytorch`，`a57db29aa6d`），实现与实测完全吻合，而且注释里写得比我们推断的更清楚。
记在这里，是因为「为什么 capture 的激活不会被内存压力回收」是个很自然的担心，
值得知道它是怎么被解决的。

**隔离靠 `PrivatePool`**（`c10/cuda/CUDACachingAllocator.cpp:1239`）：

```cpp
struct PrivatePool {
  MempoolId_t id{0, 0};
  int use_count{1};          // 有多少张活着的图在用这个池
  int cudaMalloc_count{0};
  BlockPool large_blocks;    // 自己的 free list
  BlockPool small_blocks;
};
```

注释里解释了为什么要独立容器而不是在全局池里加 pool id 判断：
「BlockComparator is performance-critical though, I'd rather not add more logic to it.」
所以隔离是"每个池自带一套 `std::set<Block*>`"——这正是我们实测到的
「池与普通分配互不通用」。

`beginAllocateToPool` / `endAllocateToPool`（3108 / 3124 行）在 capture 前后开关一个
allocation scope，把该 stream 上的分配路由到指定 `MempoolId_t`。
**`torch.cuda.graph(g, pool=...)` 不过是两次 capture 传了同一个 id。**

**生命周期靠 `use_count`**。`releasePool`（3217 行）的注释直接回答了"为什么不能
capture 完就释放"：

```text
We can't blindly delete and cudaFree the mempool its capture used, because
 1. other graph(s) might share the same pool
 2. the user might still hold references to output tensors allocated during capture.
```

`--use_count` 归零才把池挪进 `graph_pools_freeable`；而 `emptyCache` 走的
`release_cached_blocks`（4060 行）**只遍历 `graph_pools_freeable`**：

```cpp
for (auto it = graph_pools_freeable.begin(); it != graph_pools_freeable.end();) {
  TORCH_INTERNAL_ASSERT(it->second->use_count == 0);   // 只碰已归零的
  release_blocks(it->second->small_blocks, context);
  release_blocks(it->second->large_blocks, context);
```

于是链条是：图活着 → `use_count > 0` → 池不在 `graph_pools_freeable` 里 →
`release_cached_blocks` 根本不遍历它 → `emptyCache` 和内存压力都动不了。

**注意 `use_count` 是 `PrivatePool` 独有的字段，普通 BlockPool 没有。** 所以不能说
"即便走普通 allocator 也有 use_count 兜底"——保护本身就来自私有池。如果这些块在
普通池里，回收只看 tensor 引用计数：capture 结束时中间量的 Python 引用一失效，
块就回到普通 free list，随时可能被别的分配拿走、甚至整个 segment 被 `cudaFree`，
replay 就会写到别人的数据上或未映射的地址。

**两层保护缺一不可，且第 2 条依赖第 1 条**——没有私有池这个容器，就没有地方挂
`use_count`。

顺带：`CUDAGraph.cpp` 的 `retain_pool()` / `has_retained_pool()`（83-98 行）允许在图
销毁后继续抬着 `use_count`，对应上面注释里的第 2 种情形。所以池的释放条件精确说
不是"图被删除"，而是 **`use_count` 归零 且 块变成 unused**——单张图时两者时机重合，
共享池或 retain 时才分开。

### 8.11 一个测量陷阱

不要在同一个进程里先跑 compiled 再跑 eager 做对照。dynamo 的 frame hook 会拦截
同一个函数对象反复做 guard 检查和重编译尝试，eager 会被测成 6142 ms/token
（真实值 41ms，差 150 倍）。要干净的 eager 基线就开新进程，或用 `compile=False`
构造独立的 runner。

### 8.12 decode 接上 CUDA Graph：42.0 -> 4.4 ms/token

按 8.8 把 decode kernel 的标量挪到显存之后，整个 decode step 就能捕获成一张图。
`engine/runner.py` 的 `GraphedDecoder` 做的就是这件事：24 层前向 + argmax +
`pos.add_(1)` 全在图里。

```text
eager 逐步                    42.04 ms/token
graph replay                   4.41 ms/token    9.5x
graph replay + 每步 .item()    4.24 ms/token    9.9x
```

关键是**图内闭环**：`tok_slot.copy_(tok_out)` 把本步 argmax 的结果直接写回输入槽，
于是连续 replay 就自动逐 token 生成，host 每步只要 `graph.replay()`。位置也在图内
自增。

**一个预设被实测推翻了。** 原本担心每步 `.item()` 读回 token 判停止条件会打断
CPU/GPU 流水，打算"批量 replay N 步再统一检查"（代价是最多多生成 N-1 个 token）。
实测同步开销是 **0**（-0.16ms 在噪声内）——GPU 那 4.4ms 的工作足够长，同步完全被
掩盖。所以不需要那个取舍，每步照常判停即可。

**必须 reset。** capture 过程本身（warmup + 正式捕获，共 7 次 `one_step`）会把 pos
推进、把三类 cache 写脏，所以 `capture()` 之后、每个新 prompt 之前都要重新
`reset()` + `prefill()`。`generate_graphed()` 把这个约束封在里面了。

**容量守卫放在 host 侧。** `prompt_len + max_new_tokens > cache.max_len` 时
`index_copy_` 会在 device 上 assert，报错点离真正原因很远
（"CUDA error: device-side assert triggered"）。所以在 host 侧提前 assert 并给出
具体数字——这个坑我们踩过两次（一次在测 graph 内存池时，一次在这里）。

### 8.13 GDN chunk-64 prefill：38.2ms -> 0.68ms（56x）

这个三阶段 kernel 之前一直挂着"比 sequential 慢 20x，需要重写"的标签。实测下来
**不需要重写算法，问题全在两处实现细节上**，改完从慢 13.6x 变成快 4.1x。

T=2048、H=16、DK=DV=128 的逐阶段拆解：

```text
                改前        改后      倍数
prepare_wy     4.146ms    0.338ms    12.3x
chunk_state    1.630ms    0.256ms     6.4x
chunk_output  32.402ms    0.105ms   308x
────────────────────────────────────────────
chunked       38.228ms    0.682ms    56x
sequential     2.801ms    2.801ms
比值            13.6x 慢    4.1x 快
```

**问题一：8 个 `tl.dot` 全都带 `input_precision="ieee"`。**
sm80 上这会绕开 tensor core 走 FP32 软件模拟。算一下就知道有多离谱：T=2048 时
chunk_output 约 2.15 GFLOP / 32.4ms = **67 GFLOPS**，而 A100 的 BF16 峰值是
312 TFLOPS，就算是 FP32 非 tensor core 也有 19.5 TFLOPS。

改法要分操作数的实际 dtype 来定，不能一刀切：

| 点乘 | 操作数 | 处理 |
|---|---|---|
| `q @ kᵀ`（output）、`k @ kᵀ`（wy） | 两边显存里就是 BF16 | 直接用 BF16。**结果与 ieee 等价**——BF16 乘积需 16 位尾数，FP32 累加器（24 位）装得下，是精确的，只差累加顺序 |
| `attention @ delta`、`inverse @ ...`、`w @ state` | 至少一边是真 FP32 | `tf32x3`（三次 TF32 拼出接近 FP32） |
| `q @ state_in`（output） | q 是 BF16，state 是 FP32 | 裸 TF32 即可 |

最后一行是实测挑出来的，不是拍脑袋：先全用裸 TF32，整体相对误差从 1.7e-3 涨到
6.8e-3；单独把 `q @ state_in` 换成 tf32x3——**误差纹丝不动（还是 6.8e-3），
却多花了 2.9ms**；反过来只把 `attention @ delta` 换成 tf32x3，误差降回 1.0–3.4e-3
而耗时只从 0.103 涨到 0.105ms。**误差全部来自那一个点乘。** 教训是：
tf32x3 的代价与操作数尺寸强相关（这里 state_in 是 [128,128] 的 FP32 tile，
拆分开销远大于 [64,64] 的 attention），所以要逐个测，不要整片套用。

**问题二：`chunk_output` 的 `BLOCK_V=16` 是写死的。**
`grid` 的 z 维 = DV/BLOCK_V = 8，而 kernel 里 `q`、`k`、`qk = q @ kᵀ`、
`exp(gated_diff)` 和因果掩码**都与 `pid_v` 无关**——8 个 CTA 把同一份 [64,64]
的 attention 矩阵各算了一遍。改成 autotune（BLOCK_V ∈ {32,64,128}）后选到 128，
z 维变 1，这部分直接省掉 7/8。

**`chunk_state` 的 `BLOCK_V=16` 反而是对的，没动。** 它在 chunk 方向是串行扫描，
并行度只有 `H × DV/BLOCK_V = 16 × 8 = 128` 个 CTA；BLOCK_V 开到 128 的话
CTA 数掉到 16，喂不满 108 个 SM。同一个参数在两个 kernel 里的最优值相反，
因为一个的并行度来自 chunk 维、另一个只能来自 V 维。

**先测再改救了一次。** 一开始认定 `prepare_wy` 的瓶颈是那个 63 次迭代的前向替换
循环（每次对整个 [64,64] 做三遍跨 lane 归约），正准备换成 Neumann 倍增或分块
求逆。结果只改精度就从 4.146 降到 0.333ms——循环根本不是瓶颈，那套重写完全是
白工。

**测试覆盖补了长序列。** 原来的 `test_cases` 只到 T=129，即最多 3 个 chunk，
而误差是沿 chunk 方向累积的。加了 512 和 1025 之后 state 误差 1.9e-6（判据 5e-4）、
out 误差 1.2e-4（判据 1e-2），余量都很大。

另记一条：`k` 必须是 L2 归一化的（模型里 `gdn_qk_norm_gates` 保证了这点）。
用裸 `randn`（‖k‖≈√DK）造测试数据会让 WY 变换的三角系统 `(I + tril(diag(β)·KKᵀ))`
的元素到 O(DK)，前向替换直接发散成 Inf/NaN。这不是 kernel bug，但很容易误判成 bug。

### 8.14 接进 runner：kernel 级交叉点 ≠ 模型级交叉点

`_gdn` 里按 `token_num` 在两条路径间选。这里有个值得记住的坑：

```text
T       kernel 级（单层 GDN）        模型级（整个 prefill）
        seq       chunk             seq       chunk
 512   0.702ms   0.224ms   3.1x    43.1ms    49.4ms   0.87x
1536      —         —              48.2ms    49.2ms   0.98x
2048   2.801ms   0.682ms   4.1x    66.9ms    47.7ms   1.40x
3072      —         —             106.0ms    48.3ms   2.20x
4096      —         —             141.5ms    63.6ms   2.23x
```

**kernel 级交叉点在 T≈192，模型级在 T≈2048，差一个数量级。** 两个原因都不在
kernel 里：chunked 是三个 op 而 sequential 是一个，18 个 GDN 层就多 36 次分发；
而 prefill 在 T<2048 时本来就是 CPU 分发受限的（下面 8.15），GPU 省下的时间露不
出来，那 36 次分发却全额计入。所以 `GDN_CHUNKED_PREFILL_MIN_TOKENS = 2048`。

**两条路径的数值差异，以及为什么可以接受。** 24 层之后 hidden 相对差约 2.8%、
recurrent state 约 1.6%，数字看着不小。三条佐证说明这在噪声底以内：

1. 模型自身对 HF oracle 的 hidden 误差就有 1.8%（6.3 节），同一个量级；
2. **改精度之前的原版 chunked 路径同样偏离 2.3%**，所以不是 tensor core 化引入的，
   是 chunk 与 sequential 累加顺序不同在 18 层上累积的固有差异；
3. T=2048 和 3072 下，两条路径的 64 个 greedy token **逐个相同**。

oracle 无法裁决这件事——它的 prompt 只有 19 个 token，只够一个 chunk，而单
chunk 内两条路径本就等价（实测 final hidden 对 oracle 的误差两边都是 1.812e-2，
四位有效数字相同）。要真正验证长序列数值，需要一个长 prompt 的 oracle，
见第 9 节。

`tests/test_decode_parity.py` 第 6 段覆盖这条路径，判据用 greedy token 序列而不是
相对误差——理由如上。

### 8.15 prefill 的 43ms 地板，以及它没走 compile 路径

```text
prompt   prefill    每 token
    32    42.9ms    1342.1us
   128    45.4ms     354.7us
   512    43.3ms      84.6us
  2048    66.9ms      32.7us
```

T 从 32 涨到 512，时间几乎不变。这个 43ms 和 8.1 节 eager decode 的 42ms/token
是同一个数——**一次前向约 400 次 op 分发的 CPU 开销**。短 prompt 的 prefill 完全
是分发受限，GPU 基本闲着。

`compile=True` 对此**毫无作用**（43.6 vs 43.1ms）。原因：`prefill()` 直接调
`self._forward(...)`，而 compile 的分支在 `forward()` 里（`runner.py:399`），
prefill 从来没走过编译路径。这不是 bug，是当初接 cache 时没顾上——`_forward`
带 `caches=` 时有原地写入，要进 compile 区域需要另外处理。

顺带澄清一个用词：**prefill 不做"完整重算"**。"完整重算"特指不带 cache 的
`forward()`/`generate()`——每生成一个 token 就把整个序列重算一遍，每 token O(T)、
总共 O(T²)，只用于测试对拍。`prefill()` 复用的是同一份 `_forward` 代码（额外传
`caches=` 让每层顺手写 cache），但**只跑一遍**。两者差多少：

```text
prompt   prefill 一遍   若真用完整重算逐 token 生成同样长度
    32      42.9ms                ~0.7s
   512      43.3ms               ~11.1s
  2048      66.9ms               ~68.5s
```

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

   - ~~`depthwise_causal_conv4_decode`~~ **已完成**，见 3.8b；
   - ~~带 KV cache 的 GQA decode kernel~~ **已完成**（含 split-K），见 3.3b；
   - GDN 侧直接接已有的 `gdn_recurrent_decode`（state cache 路径已验证）。

   齐了之后把 runner 拆成 prefill + decode 两条路径，用当前的完整重算版本做等价性
   对拍——**现在这个 runner 就是那个基准**（`compile=False` 那条路径逐算子对得上 HF）。
3. ~~**重写 chunk-64 prefill**~~ **已完成**，见 8.13/8.14。结论与原计划不同：不需要
   重写算法，慢是 `input_precision="ieee"` 绕开 tensor core + `BLOCK_V` 硬编码
   造成的，改完 56x，现在比 sequential 快 4.1x 并已按 T≥2048 接进 `_gdn`。

4. **拆掉 prefill 的 43ms 分发地板**（见 8.15）。这是现在 prefill 的实际瓶颈——
   T<2048 时 GPU 基本闲着，8.13 那 56x 完全露不出来。两个方向：
   - 让 `prefill()` 走 compile 路径。现在它直接调 `_forward` 绕过了 `forward()` 里的
     compile 分支，而 `caches=` 的原地写入需要额外处理才能进编译区域。
   - 按 T 分桶 + padding 给 prefill 也捕获 CUDA Graph。decode 能进图是因为 shape 恒定，
     prefill 的 T 随 prompt 变，代价是每个桶一张图。
   做成之后 `GDN_CHUNKED_PREFILL_MIN_TOKENS` 应该同步下调（现在的 2048 是被 CPU
   地板顶上去的，kernel 级交叉点只有 192）。

5. **扩测试覆盖**：目前 oracle 只有一个 19 token 的中文 prompt。**这条现在有了具体
   的缺口**：8.14 里 chunked 与 sequential 在长序列上的 2.8% 差异无法被 oracle 裁决，
   因为 19 个 token 只够一个 chunk，两条路径在单 chunk 内本就等价。需要一个
   T≥2048 的长 prompt oracle（跑 `tools/dump_oracle.py`，在 `.venv-oracle` 里）。
   另外再加一个英文 prompt。

6. **kernel 级性能收尾**：`gdn_qk_norm_gates` 的 head 分块。
   **不包括 `gemm_2d`**（已是 cuBLAS 的 1.1-1.2 倍，8.5）、
   **也不包括 `chunk_state`/`chunk_output` 的 BLOCK_V**（8.13 已处理：chunk_output
   接了 autotune；chunk_state 的 16 经分析是对的，开大反而把 CTA 数从 128 降到 16）。

7. 可选：top-k/top-p/temperature sampling、batch/padding。注意当前
   `lm_head_argmax` 把 LM head 的 GEMV 和 argmax 融在一个 kernel 里，
   248320 维的 logits 从不物化——要做采样得改图的输出（出 logits 交给图外采样，
   或把采样也放进图里，后者需要一个 capture 安全的 RNG，和 `pos.add_(1)` 一个套路）。

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
python triton_kernels/gqa_attention_decode.py
python triton_kernels/partial_rope.py
python triton_kernels/attention_gate_pack.py
python triton_kernels/residual_add.py
python triton_kernels/swiglu.py
python triton_kernels/depthwise_causal_conv4_prefill.py
python triton_kernels/depthwise_causal_conv4_decode.py
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
