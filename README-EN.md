# Qwen3.5-0.8B Triton Inference Engine

With kernel agents taking off the way they are, I suspect there won't be many
occasions left to write kernels by hand. I had some spare time recently, so I wanted
to get Qwen3.5-0.8B running on nothing but hand-written Triton kernels — as a way to
deepen my understanding of GPUs and get a first real look at how an LLM works.

Running Qwen3.5-0.8B end to end on hand-written Triton kernels. **The inference path
does not depend on transformers, vLLM or FlashAttention, and never calls
cuBLAS/cuDNN.** From the embedding lookup to the LM head argmax, every computation
across all 24 layers happens in a Triton kernel in this repository.

PyTorch is used for exactly four things: allocating memory, reshaping
(`view`/`contiguous`), copying (`copy_`/`index_copy_`), and calling the CUDA Graph
API. Not a single `torch.matmul`, `F.softmax` or `F.linear` appears in the forward
path.

The kernels in this repository are essentially all written by hand by me. The
connective tissue — the runner, the loader, cache management, plus the tests and
documentation — was done with help from Codex and Claude Code. The one exception is
the chunk-64 parallel prefill for GDN, where the kernel itself was also improved with
their help; see "Attribution" at the end for details.

Writing this demo turned a number of things I had only read about into something
concrete. That in eager mode nine tenths of the time goes to operator dispatch rather
than arithmetic. That CUDA Graph bakes scalar arguments into the launch configuration,
so the sequence position has to live in device memory. That when a decode kernel
launches only two CTAs, most of the GPU's hundred-plus SMs sit idle, and split-K is
what brings the parallelism back. That the same `BLOCK_V` can have opposite optimal
values in two different kernels.

Only single-request inference is supported at the moment; this is still being worked
on.

[中文 README](README.md)

## Why this model

The text backbone of Qwen3.5-0.8B is not the usual "24 layers of full attention".
It repeats `3 Gated DeltaNet + 1 full attention` six times:

- **18 Gated DeltaNet layers** (linear attention) — each with a depthwise causal
  Conv1D of kernel size 4 and a delta-rule recurrence whose state is a
  `[16, 128, 128]` matrix rather than a KV sequence;
- **6 GQA full-attention layers** (indices 3/7/11/15/19/23) — 8 Q heads / 2 KV heads,
  head_dim 256, with **RoPE applied to only the first 64 dimensions** (partial rotary);
- a 1024 → 3584 → 1024 SwiGLU MLP in every layer.

The two layer types cache completely differently, which is what makes this
interesting: the GDN state is fixed-size (independent of context length) while the
attention KV cache grows linearly. One engine has to manage three kinds of cache
at once.

The text backbone has 752,393,024 parameters, 1.401 GiB in BF16.

## Results

Measured on an A100-SXM4-40GB:

```
                       load+prefill+capture    steady state
CUDA Graph                    2.7s            4.3 ms/token
eager (op by op)              1.9s           44.8 ms/token
```

```console
$ python demo.py "Explain attention in one sentence"
```

## Getting started

### 1. Requirements

- **NVIDIA GPU with compute capability 8.0+** (Ampere/Hopper). Everything was
  developed and tested on an A100; some precision choices in the kernels
  (`tf32x3`, BF16 tensor cores) are tuned for sm80. Older cards will not run;
  newer ones will run but may not be optimal.
- ≥ 6 GiB of VRAM (1.4 GiB of weights plus caches and intermediates).
- Linux. Windows is untested.

### 2. Install dependencies

```bash
pip install torch triton safetensors tokenizers
```

Versions used during development (nearby versions should be fine):

| Package | Version |
|---|---|
| Python | 3.11.14 |
| torch | 2.12.1+cu128 |
| triton | 3.7.1 |
| safetensors | 0.8.0 |
| tokenizers | 0.22.2 |
| CUDA | 12.8 |

Note that **`transformers` is not a runtime dependency**. It is only needed to
generate the numerical reference tensors — see "Verification" below.

### 3. Download the weights

Get [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) from Hugging Face
and place it in a directory named `Qwen3.5-0.8B/` at the repository root:

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir Qwen3.5-0.8B
```

Development pinned revision `2fc06364715b967f1860aea9cf38778875588b17`. The loader
asserts the parameter count, the layer-type distribution, and the shape and dtype of
every tensor, so a changed checkpoint fails loudly rather than silently.

Only these files are needed:

```
Qwen3.5-0.8B/
├── config.json
├── model.safetensors.index.json
├── model.safetensors-00001-of-00001.safetensors
└── tokenizer.json
```

### 4. Run it

```bash
python demo.py "Explain what an attention mechanism is, in one sentence"
```

| Flag | Meaning |
|---|---|
| `--max-tokens N` | Generation cap; also determines how much KV cache is allocated. Default 512 |
| `--thinking` | Keep the model's think section (skipped by default) |
| `--no-graph` | Disable CUDA Graph and run the eager op-by-op path for comparison |

**The first run is slow** (tens of seconds to a minute) because Triton has to JIT
compile and autotune every kernel. Results are cached in `~/.triton/cache`, after
which startup is just the 1.4 GiB weight load (about 2 seconds). This disk cache is
enabled automatically by `triton_kernels/__init__.py`.

## Layout

```
triton_kernels/     15 files, 21 ops registered through torch.library
engine/
  loader.py         safetensors loading + layout rearrangement (see below)
  cache.py          allocation and lifetime of the three cache types
  runner.py         24-layer forward, prefill/decode paths, CUDA Graph wrapper
demo.py             command-line entry point
tests/              numerical parity tests
tools/dump_oracle.py  generates reference tensors (needs a separate venv)
```

`loader.py` does two things that are not a straight copy, both so that downstream
kernels get contiguous memory:

1. `q_proj [4096,1024]` is split into Q and gate, `[2048,1024]` each — the checkpoint
   interleaves them per head, so slicing at runtime would give a strided view;
2. `conv1d` is stored twice, `[6144,4]` for prefill and `[4,6144]` for decode —
   during decode one thread owns one channel, and the `[4,D]` layout makes adjacent
   threads read adjacent addresses so the accesses coalesce.

### Kernel inventory

<details>
<summary>21 ops (click to expand)</summary>

**General**
`gemm_2d`, `qwen_rmsnorm`, `residual_add`, `swiglu`, `embedding_gather`, `lm_head_argmax`

**Full attention**
`gqa_attention_without_kvcache_casual` (prefill), `partial_rope`, `attention_gate_pack`,
`gqa_attention_decode`, `gqa_attention_decode_split` + `gqa_attention_decode_combine`
(flash-decoding style split-K)

**Gated DeltaNet**
`depthwise_causal_conv4_prefill`, `depthwise_causal_conv4_decode`, `gdn_qk_norm_gates`,
`gdn_gated_rmsnorm`, `gdn_recurrent_prefill_sequential`, `gdn_recurrent_decode`,
`gdn_chunk_prepare_wy` + `gdn_chunk_state` + `gdn_chunk_output` (chunk-64 parallel prefill)

</details>

## Verification

Three levels, local to global:

```bash
python tests/test_gemm_model_shapes.py   # GEMM correctness and efficiency at real model shapes
python tests/test_oracle_parity.py       # 49 per-operator checks against Hugging Face
python tests/test_decode_parity.py       # incremental decode vs full recompute vs CUDA Graph
python triton_kernels/gdn_recurrent_prefill.py   # the kernel's own unit tests
```

**The last two need reference tensors first**, and that step — only that step —
requires `transformers`. The tensors are about 13 MB and are not committed:

```bash
python -m venv .venv-oracle
.venv-oracle/bin/pip install torch transformers accelerate
.venv-oracle/bin/python tools/dump_oracle.py
```

The script records the inputs and outputs of every layer and submodule of the HF
reference implementation into `oracle/`. After that the main environment never needs
transformers again.

Per-operator parity is the stable criterion. "The generated tokens match HF exactly"
is prompt-dependent — when top-1 and top-2 are close, BF16 rounding alone is enough
to flip the argmax.

## Limitations

This is a learning project, **not a production inference server**. Known boundaries:

**Functional**

- **Single request only.** Batch size is always 1. No padding, no continuous
  batching, no request queue.
- **No multi-turn conversation.** Each run handles one independent prompt; history is
  not kept and the KV cache is not reused across turns. Multi-turn would mean
  concatenating the history into the prompt and re-running prefill yourself.
- **Greedy sampling only.** No temperature/top-k/top-p. This is deliberate rather than
  missing: `lm_head_argmax` deliberately fuses the LM head GEMV and the argmax into
  one kernel, so the 248320-dimensional logit vector is never materialized — saving a
  1 MB write and read. Adding sampling means changing what that kernel outputs.
- **Text only.** The vision tower (`model.visual.*`) and MTP (`mtp.*`) weights in the
  checkpoint are not loaded at all. The model itself is multimodal; image input is not
  supported here.
- **BF16 only.** No quantization (INT8/FP8/GPTQ/AWQ).
- Context length is bounded by the cache preallocated at startup (set by
  `--max-tokens`); it does not grow on demand.

**Engineering**

- **Only verified on an A100 (sm80).** sm80 or newer is required.
- No serving layer: a command-line demo only, no HTTP API, no OpenAI-compatible
  endpoint.
- Prefill is currently bound by CPU-side operator dispatch (a ~43 ms floor for short
  prompts, nearly independent of sequence length); only decode goes through CUDA Graph.
- The numerical oracle covers a single 19-token Chinese prompt; long-sequence and
  English coverage is still thin.

## Attribution

The exception mentioned at the top is the chunk-64 three-stage parallel prefill for
Gated DeltaNet (`gdn_chunk_*`). It was originally generated by Codex, and later
optimized by Claude: replacing the tensor-core-bypassing `input_precision="ieee"` on
a per-operand-dtype basis, and fixing the blocking in `chunk_output` that had 8 CTAs
each recomputing the same attention matrix. 56x in total, with the algorithm itself
unchanged.

For numerical comparison on this path, my hand-written sequential recurrence is the
baseline.

## License

The code is under the [MIT License](LICENSE).

The model weights follow Qwen3.5-0.8B's own Apache-2.0 license — see the
[official repository](https://huggingface.co/Qwen/Qwen3.5-0.8B).
