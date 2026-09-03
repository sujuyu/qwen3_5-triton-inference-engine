#!/usr/bin/env python3
"""Qwen3.5-0.8B Triton 引擎的交互 demo。

    python demo.py "李世民是谁？和朱棣有什么共同点？"
    python demo.py "用一句话解释 RoPE" --max-tokens 200
    python demo.py "写一首关于秋天的诗" --thinking
全部计算走 triton_kernels/ 下手写的 15 个 kernel，不依赖 transformers。

走增量 decode + CUDA Graph：prefill 一次把三类 cache 填好（conv state、
recurrent state、KV cache），然后把整个 decode step（24 层前向 + argmax +
pos 自增）捕获成一张图，之后每步只 replay。遇到 EOS 提前停。

实测 42.0 -> 4.4 ms/token（9.5x）。省掉的是 eager 下每步约 400 次 op 调用的
torch.library 分发开销。`--no-graph` 可切回逐 op 的 eager decode 做对照。

`--max-tokens` 同时是生成上限和 KV cache 的分配依据，默认 512。

对拍基准是 runner 里的完整重算路径（`forward` / `generate`，每步对整个序列重算），
它在 tests/ 里用——`test_oracle_parity.py` 拿它对 Hugging Face，
`test_decode_parity.py` 拿它验证 cache 管理。两条路径的 greedy 结果一致。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.cache import allocate_caches
from engine.runner import DEFAULT_STOP_IDS, GraphedDecoder, build_runner
from triton_kernels.vocab_argmax import lm_head_argmax

MODEL_DIR = ROOT / "Qwen3.5-0.8B"


def render_chat(prompt: str, thinking: bool) -> str:
    """checkpoint 官方 chat 模板的单轮子集，与 tokenize_text.py 保持一致。"""
    prefix = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return prefix + ("<think>\n" if thinking else "<think>\n\n</think>\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("prompt", nargs="?", default="李世民是谁？和朱棣有什么共同点？")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true", help="保留 think 段")
    parser.add_argument(
        "--no-graph", action="store_true", help="关掉 CUDA Graph，逐 op eager 执行"
    )
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    rendered = render_chat(args.prompt, args.thinking)
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False).ids

    print(f"提问：{args.prompt}")
    print(f"prompt {len(prompt_ids)} tokens，最多生成 {args.max_tokens} tokens")
    print("-" * 72, flush=True)

    load_start = time.time()
    # decode 每步 shape 固定，torch.compile 的 dynamic shape 处理用不上，
    # 反而要付一次编译成本，所以这里不开。真正的提速手段是 CUDA Graph。
    runner = build_runner(str(MODEL_DIR), compile=False)
    caches = allocate_caches(runner.w, len(prompt_ids) + args.max_tokens + 8)
    prompt_t = torch.tensor(prompt_ids, dtype=torch.int32, device=runner.device)

    decoder = None
    if not args.no_graph:
        # 捕获前先 prefill，让 warmup 跑在合法的 cache 状态上；捕获本身会把
        # cache 写脏，所以下面还要再 prefill 一次。
        caches.reset()
        runner.prefill(prompt_t, caches)
        decoder = GraphedDecoder(runner, caches)
        decoder.capture()

    caches.reset()
    hidden = runner.prefill(prompt_t, caches)
    first_token = int(lm_head_argmax(hidden, runner.w.embed_tokens).item())
    load_elapsed = time.time() - load_start

    generated: list[int] = []
    slot = torch.empty(1, dtype=torch.int32, device=runner.device)
    shown = ""
    first_token_time = None
    start = time.time()

    step_times: list[float] = []
    pending = first_token
    for _ in range(args.max_tokens):
        step_start = time.time()
        if pending is not None:                       # prefill 已经出了第一个 token
            token, pending = pending, None
        elif decoder is not None:
            token = decoder.step(generated[-1])
        else:
            slot.fill_(generated[-1])
            hidden = runner.decode_step(slot, caches)
            # pos 的推进必须在 forward 之后：attention 要"写入位置 = 当前 pos"
            caches.pos.add_(1)
            token = int(lm_head_argmax(hidden, runner.w.embed_tokens).item())
        step_times.append(time.time() - step_start)
        if first_token_time is None:
            first_token_time = time.time() - start
        if token in DEFAULT_STOP_IDS:
            break
        generated.append(token)

        # 每步整体重解码再取增量：byte-level BPE 的一个字符可能横跨多个 token，
        # 逐 token 解码会在 UTF-8 边界上吐出替换字符 U+FFFD。
        # 末尾的替换字符说明还有字符没解完，先扣住不吐，等下一个 token 补齐——
        # 否则先打了替换字符，下一步 text 不再以 shown 开头，就会整段丢输出。
        text = tokenizer.decode(generated, skip_special_tokens=False)
        safe = text.rstrip("�")
        if len(safe) > len(shown):
            assert safe.startswith(shown), "解码增量不是前缀，tokenizer 行为异常"
            sys.stdout.write(safe[len(shown) :])
            sys.stdout.flush()
            shown = safe


    elapsed = time.time() - start
    n = len(generated)
    ordered = sorted(step_times)
    median = ordered[len(ordered) // 2] * 1e3
    # 明显偏离中位数的是预热停顿。两个来源：
    #   - Triton 侧：首次遇到某个 T_BUCKET 时 JIT 编译 + autotune 全部 13 个 kernel，
    #     关掉 torch.compile 也躲不掉；
    #   - decode kernel 首次被调用时同样要 JIT + autotune。
    stalls = [(i, t) for i, t in enumerate(step_times) if t > max(0.2, median * 20 / 1e3)]

    print("\n" + "-" * 72)
    mode = "eager" if args.no_graph else "CUDA Graph"
    print(f"[{mode}] 加载+prefill+捕获 {load_elapsed:.1f}s | "
          f"生成 {n} tokens，总用时 {elapsed:.1f}s")
    print(f"稳态 {median:.1f} ms/token（中位数）")
    if stalls:
        total_stall = sum(t for _, t in stalls)
        print(
            f"预热停顿 {len(stalls)} 次，合计 {total_stall:.1f}s"
            f"（占总时间 {total_stall / elapsed:.0%}）："
        )
        for i, t in stalls[:8]:
            print(f"    step {i:>3}  {t:6.1f}s")
        print("  详见 HANDOFF.md 8.3；减少这类停顿是当前性能工作的第一项。")


if __name__ == "__main__":
    main()
