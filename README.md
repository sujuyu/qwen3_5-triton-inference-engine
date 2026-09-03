# Qwen3.5-0.8B Triton 推理引擎

用手写的 Triton kernel 把 Qwen3.5-0.8B 端到端跑起来。**推理路径上不依赖 transformers、
vLLM、FlashAttention，也不调用 cuBLAS/cuDNN**——从 embedding 查表到 LM head 的
argmax，24 层里的每一次计算都发生在这个仓库里的 Triton kernel 中。

PyTorch 只用来做四件事：分配显存、变换形状（`view`/`contiguous`）、拷贝
（`copy_`/`index_copy_`），以及调用 CUDA Graph 的 API。没有一个 `torch.matmul`、
`F.softmax` 或 `F.linear` 出现在前向路径上。

这是一个学习项目：目的是搞清楚一个真实的、非标准结构的 LLM 从权重到 token 的
每一步到底在算什么，以及在 GPU 上把它跑快需要面对哪些具体问题。

[English README](README-EN.md)

## 为什么是这个模型

Qwen3.5-0.8B 的文本主干不是常见的「24 层全注意力」，而是按
`3 个 Gated DeltaNet + 1 个全注意力` 重复 6 次：

- **18 层 Gated DeltaNet**（线性注意力）——每层含一个 kernel=4 的 depthwise causal
  Conv1D 和一个 delta rule 递推，状态是 `[16, 128, 128]` 的矩阵而不是 KV 序列；
- **6 层 GQA 全注意力**（层号 3/7/11/15/19/23）——8 个 Q head / 2 个 KV head，
  head_dim 256，**RoPE 只作用在前 64 维**（partial rotary）；
- 每层一个 1024 → 3584 → 1024 的 SwiGLU MLP。

两种层的缓存机制完全不同，这正是有意思的地方：GDN 的状态是定长的（与上下文长度
无关），全注意力的 KV cache 随长度线性增长。一个引擎里要同时管三类缓存。

文本主干 752,393,024 个参数，BF16 下 1.401 GiB。

## 效果

A100-SXM4-40GB 实测：

```
                    加载+prefill+捕获    稳态生成
CUDA Graph              2.7s           4.3 ms/token
eager（逐 op）           1.9s          44.8 ms/token
```

```console
$ python demo.py "李世民是谁？和朱棣有什么共同点？"
提问：李世民是谁？和朱棣有什么共同点？
prompt 23 tokens，最多生成 512 tokens
------------------------------------------------------------------------
李世民（唐太宗）是唐朝的开国皇帝，也是唐朝的第三位皇帝。他生于 601 年，
卒于 649 年，在位 48 年，是唐朝历史上最具影响力的君主之一。

### 李世民的主要成就：
1. **统一全国**：他通过一系列军事行动，成功平定了叛乱，统一了……
```

## 快速开始

### 1. 环境要求

- **NVIDIA GPU，计算能力 8.0 及以上**（Ampere/Hopper）。开发和测试都在 A100 上做的；
  kernel 里的一些精度选择（`tf32x3`、BF16 tensor core）是按 sm80 调的，
  在更老的卡上跑不了，在更新的卡上能跑但未必最优。
- 显存 ≥ 6 GiB（权重 1.4 GiB + 缓存 + 中间量）。
- Linux。Windows 未测试。

### 2. 装依赖

```bash
pip install torch triton safetensors tokenizers
```

开发时使用的版本（其他相近版本应该也可以）：

| 包 | 版本 |
|---|---|
| Python | 3.11.14 |
| torch | 2.12.1+cu128 |
| triton | 3.7.1 |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |
| CUDA | 12.8 |

注意 **`transformers` 不是运行依赖**。它只在生成数值对拍基准时需要，见
「正确性验证」一节。

### 3. 下载权重

从 Hugging Face 取 [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)，
放在仓库根目录下名为 `Qwen3.5-0.8B/` 的文件夹里：

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir Qwen3.5-0.8B
```

开发时锁定的 revision 是 `2fc06364715b967f1860aea9cf38778875588b17`。
loader 会断言参数量、层数分布和每个张量的形状与 dtype，checkpoint 一旦变了会直接
报错而不是静默出错。

目录里需要这些文件（其余的用不到）：

```
Qwen3.5-0.8B/
├── config.json
├── model.safetensors.index.json
├── model.safetensors-00001-of-00001.safetensors
└── tokenizer.json
```

### 4. 跑起来

```bash
python demo.py "用一句话解释什么是注意力机制"
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--max-tokens N` | 生成上限，同时决定 KV cache 分配多大。默认 512 |
| `--thinking` | 保留模型的 think 段（默认跳过） |
| `--no-graph` | 关掉 CUDA Graph，走逐 op 的 eager 路径做对照 |

**第一次运行会慢**（几十秒到一分钟），因为 Triton 要 JIT 编译并 autotune 所有
kernel。结果会缓存到 `~/.triton/cache`，之后启动时间就只剩加载 1.4 GiB 权重本身
（约 2 秒）。这个磁盘缓存由 `triton_kernels/__init__.py` 自动开启。

## 项目结构

```
triton_kernels/     15 个文件，21 个通过 torch.library 注册的算子
engine/
  loader.py         safetensors 权重加载 + 布局重排（见下）
  cache.py          三类缓存的分配与生命周期
  runner.py         24 层前向、prefill/decode 两条路径、CUDA Graph 封装
demo.py             命令行对话入口
tests/              数值对拍
tools/dump_oracle.py  生成参考张量（需要独立环境）
```

`loader.py` 做了两处不是原样搬运的事，都是为了让下游 kernel 能拿到连续内存：

1. `q_proj [4096,1024]` 拆成 Q 和 gate 各 `[2048,1024]`——checkpoint 里两者按 head
   交错存放，不拆的话切出来的 Q 是 strided view；
2. `conv1d` 存两份，prefill 用 `[6144,4]`、decode 用 `[4,6144]`——decode 时一个线程
   负责一个 channel，`[4,D]` 布局下相邻线程的地址连续，访存能合并。

### 算子清单

<details>
<summary>21 个算子（点击展开）</summary>

**通用**
`gemm_2d`、`qwen_rmsnorm`、`residual_add`、`swiglu`、`embedding_gather`、`lm_head_argmax`

**全注意力**
`gqa_attention_without_kvcache_casual`（prefill）、`partial_rope`、`attention_gate_pack`、
`gqa_attention_decode`、`gqa_attention_decode_split` + `gqa_attention_decode_combine`（flash-decoding 风格的 split-K）

**Gated DeltaNet**
`depthwise_causal_conv4_prefill`、`depthwise_causal_conv4_decode`、`gdn_qk_norm_gates`、
`gdn_gated_rmsnorm`、`gdn_recurrent_prefill_sequential`、`gdn_recurrent_decode`、
`gdn_chunk_prepare_wy` + `gdn_chunk_state` + `gdn_chunk_output`（chunk-64 并行 prefill）

</details>

## 正确性验证

三层判据，从局部到整体：

```bash
python tests/test_gemm_model_shapes.py   # GEMM 在模型真实尺寸下的正确性与效率
python tests/test_oracle_parity.py       # 49 项逐算子对拍 Hugging Face
python tests/test_decode_parity.py       # 增量 decode vs 完整重算 vs CUDA Graph
python triton_kernels/gdn_recurrent_prefill.py   # kernel 自带的单元测试
```

**后两个需要先生成参考张量**，而这一步（也只有这一步）需要 `transformers`。
参考张量约 13 MB，没有提交进仓库：

```bash
python -m venv .venv-oracle
.venv-oracle/bin/pip install torch transformers accelerate
.venv-oracle/bin/python tools/dump_oracle.py
```

脚本会把 HF 参考实现每一层、每个子模块的输入输出存进 `oracle/`。之后主环境不需要
再装 transformers。

逐算子对拍才是稳定判据——「生成的 token 和 HF 完全一样」是 prompt 相关的，
在 top-1 和 top-2 极接近时 BF16 的舍入差异就足以翻转 argmax。

## 局限性

这是一个学习项目，**不是能用于生产的推理服务**。已知的边界：

**功能上**

- **只支持单个请求。** batch 恒为 1，没有 padding、没有 continuous batching、
  没有请求队列。
- **没有连续对话能力。** 每次运行处理一轮独立的问答，不保留历史、不复用上一轮的
  KV cache。要多轮对话得自己把历史拼进 prompt 重新 prefill。
- **只有 greedy 采样。** 没有 temperature/top-k/top-p。这不是漏掉了，而是
  `lm_head_argmax` 刻意把 LM head 的 GEMV 和 argmax 融进了一个 kernel，
  248320 维的 logits 从来不物化——省掉了 1 MB 的写和读。要加采样得改这个 kernel
  的输出形式。
- **只做文本。** checkpoint 里的视觉塔（`model.visual.*`）和 MTP（`mtp.*`）
  完全不加载。这个模型本身是多模态的，图像输入这里不支持。
- **只有 BF16。** 没有量化（INT8/FP8/GPTQ/AWQ）。
- 上下文长度受启动时预分配的缓存限制（由 `--max-tokens` 决定），中途不会扩容。

**工程上**

- **只在 A100（sm80）上验证过。** 需要 sm80 及以上。
- 没有服务化：只有一个命令行 demo，没有 HTTP API、没有 OpenAI 兼容接口。
- prefill 目前受 CPU 侧的算子分发开销限制（短 prompt 约 43ms 是个地板，与序列长度
  几乎无关），只有 decode 走了 CUDA Graph。
- 数值对拍的 oracle 只覆盖一个 19 token 的中文 prompt，长序列和英文的覆盖还不够。

## 关于代码归属

绝大部分 kernel 由仓库作者手写。其中 Gated DeltaNet 的 chunk-64 三段式并行 prefill
（`gdn_chunk_*`）最初由 AI 助手生成，后续的性能改造（tensor core 化、分块修正，
56x）由 Claude 完成，算法本身未动。数值对拍时以作者手写的 sequential 递推版本
为基准。

## 许可

模型权重遵循 Qwen3.5-0.8B 自身的 Apache-2.0 许可，
见 [官方仓库](https://huggingface.co/Qwen/Qwen3.5-0.8B)。

本仓库代码的许可尚未确定（还没有 LICENSE 文件）。
