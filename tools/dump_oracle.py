#!/usr/bin/env python3
"""Dump Hugging Face reference tensors for the Qwen3.5-0.8B Triton engine.

这个脚本**不属于推理引擎本体**，只在独立的 .venv-oracle 环境里跑一次，
把参考实现的中间量存成 .pt，之后主环境不需要再装 transformers。

用法：
    .venv-oracle/bin/python tools/dump_oracle.py

产出（全部在 oracle/，已在 .gitignore 中排除）：
    oracle/meta.pt          input_ids、prompt、config 关键值、生成的 token
    oracle/hidden.pt        embedding 输出 + 24 层每层输出 + final norm，共 26 个 [1,T,1024]
    oracle/layer00_gdn.pt   第 0 层（Gated DeltaNet）所有子模块的输入输出
    oracle/layer03_attn.pt  第 3 层（full attention）所有子模块的输入输出
    oracle/logits.pt        最后一个位置的 logits [248320]，以及 greedy token

所有张量按 CPU 保存。逐层对拍时用 torch.load 读回即可。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "Qwen3.5-0.8B"
OUT_DIR = ROOT / "oracle"

PROMPT = "你好，请简单介绍一下自己。"
NUM_GENERATE = 32

# 与 tokenize_text.py 保持一致的单轮 chat 模板（non-thinking）。
RENDERED = f"<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def capture_module_io(module: torch.nn.Module, prefix: str, store: dict) -> list:
    """给 module 及其所有子模块挂 forward hook，记录输入输出张量。"""
    handles = []

    def make_hook(name: str):
        def hook(_mod, args, output):
            for i, a in enumerate(args):
                if isinstance(a, torch.Tensor):
                    store[f"{name}.in{i}"] = a.detach().float().cpu()
            if isinstance(output, torch.Tensor):
                store[f"{name}.out"] = output.detach().float().cpu()
            elif isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        store[f"{name}.out{i}"] = o.detach().float().cpu()

        return hook

    for name, sub in module.named_modules():
        label = prefix if name == "" else f"{prefix}.{name}"
        handles.append(sub.register_forward_hook(make_hook(label)))
    return handles


def patch_free_functions(store: dict) -> None:
    """conv、delta rule、RoPE 都是模块级自由函数，forward hook 抓不到。

    这里按调用顺序记录：conv 和 delta rule 每个 GDN 层各调一次（第 0 次 = layer 0），
    apply_rotary_pos_emb 每个 full-attention 层各调一次（第 0 次 = layer 3）。
    只保留第 0 次调用，正好对应我们要细看的那两层。
    """
    import transformers.models.qwen3_5.modeling_qwen3_5 as m

    counters = {"conv": 0, "delta": 0, "rope": 0}

    def wrap(name: str, fn, record):
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            if counters[name] == 0:
                record(args, kwargs, out)
            counters[name] += 1
            return out

        return wrapper

    def rec_conv(args, kwargs, out):
        store["conv.in_qkv_bct"] = args[0].detach().float().cpu()
        store["conv.weight"] = args[1].detach().float().cpu()
        store["conv.out_bct"] = out.detach().float().cpu()

    def rec_delta(args, kwargs, out):
        # 签名为 (query, key, value, g, beta, ...)，q/k 尚未做 l2norm 和 1/sqrt(D) scale
        for i, key in enumerate(("query", "key", "value")):
            store[f"delta.{key}_pre_l2norm"] = args[i].detach().float().cpu()
        for key in ("g", "beta"):
            if key in kwargs:
                store[f"delta.{key}"] = kwargs[key].detach().float().cpu()
        store["delta.core_attn_out"] = out[0].detach().float().cpu()
        if out[1] is not None:
            store["delta.final_state"] = out[1].detach().float().cpu()

    def rec_rope(args, kwargs, out):
        store["rope.q_in"] = args[0].detach().float().cpu()
        store["rope.k_in"] = args[1].detach().float().cpu()
        store["rope.cos"] = args[2].detach().float().cpu()
        store["rope.sin"] = args[3].detach().float().cpu()
        store["rope.q_out"] = out[0].detach().float().cpu()
        store["rope.k_out"] = out[1].detach().float().cpu()

    m.causal_conv1d_fn = wrap("conv", m.causal_conv1d_fn, rec_conv)
    m.torch_chunk_gated_delta_rule = wrap(
        "delta", m.torch_chunk_gated_delta_rule, rec_delta
    )
    m.apply_rotary_pos_emb = wrap("rope", m.apply_rotary_pos_emb, rec_rope)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    torch.manual_seed(0)

    config = AutoConfig.from_pretrained(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    # eager 实现：走本仓库对照过的那条数学路径，不落到 flash/sdpa 的融合分支。
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()

    input_ids = torch.tensor(
        [tokenizer.encode(RENDERED, add_special_tokens=False)],
        dtype=torch.long,
        device="cuda",
    )
    seq_len = input_ids.shape[1]
    print(f"token_count={seq_len}")

    text_model = model.model.language_model
    layers = text_model.layers
    print(f"num_layers={len(layers)}  layer_types={config.get_text_config().layer_types[:8]} ...")

    gdn_store: dict[str, torch.Tensor] = {}
    attn_store: dict[str, torch.Tensor] = {}
    handles = capture_module_io(layers[0], "layer00", gdn_store)
    handles += capture_module_io(layers[3], "layer03", attn_store)

    # 自由函数的中间量：conv/delta 归到 layer00，rope 归到 layer03。
    free_store: dict[str, torch.Tensor] = {}
    patch_free_functions(free_store)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )

    for h in handles:
        h.remove()

    for key, value in free_store.items():
        if key.startswith("rope."):
            attn_store[f"layer03.{key}"] = value
        else:
            gdn_store[f"layer00.{key}"] = value
    assert any(k.endswith("conv.out_bct") for k in gdn_store), (
        "conv 输出没抓到——transformers 可能改了 causal_conv1d_fn 的调用方式，"
        "需要重新确认 patch 点"
    )

    hidden = {
        f"hidden_{i:02d}": h.detach().float().cpu()
        for i, h in enumerate(out.hidden_states)
    }
    with torch.no_grad():
        hidden["final_norm"] = (
            text_model.norm(out.hidden_states[-1]).detach().float().cpu()
        )

    logits_last = out.logits[0, -1].detach().float().cpu()
    greedy_first = int(logits_last.argmax())

    # 连续 greedy 生成，作为端到端验收基准。
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=NUM_GENERATE,
            do_sample=False,
            num_beams=1,
        )
    new_tokens = generated[0, seq_len:].tolist()

    torch.save(
        {
            "prompt": PROMPT,
            "rendered": RENDERED,
            "input_ids": input_ids[0].cpu(),
            "seq_len": seq_len,
            "greedy_first_token": greedy_first,
            "generated_tokens": new_tokens,
            "generated_text": tokenizer.decode(new_tokens, skip_special_tokens=False),
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
        },
        OUT_DIR / "meta.pt",
    )
    torch.save(hidden, OUT_DIR / "hidden.pt")
    torch.save(gdn_store, OUT_DIR / "layer00_gdn.pt")
    torch.save(attn_store, OUT_DIR / "layer03_attn.pt")
    torch.save(
        {"logits_last": logits_last, "greedy_first_token": greedy_first},
        OUT_DIR / "logits.pt",
    )

    print(f"\nhidden tensors: {len(hidden)}")
    print(f"layer00 (GDN) captured: {len(gdn_store)}")
    print(f"layer03 (attn) captured: {len(attn_store)}")
    print(f"greedy_first_token={greedy_first}")
    print(f"generated_tokens={new_tokens}")
    print("generated_text:")
    print(tokenizer.decode(new_tokens, skip_special_tokens=False))

    index = {
        "meta.pt": ["prompt", "input_ids", "generated_tokens"],
        "hidden.pt": sorted(hidden.keys()),
        "layer00_gdn.pt": sorted(gdn_store.keys()),
        "layer03_attn.pt": sorted(attn_store.keys()),
        "logits.pt": ["logits_last", "greedy_first_token"],
    }
    (OUT_DIR / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
