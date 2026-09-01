#!/usr/bin/env python3
"""Qwen3.5-0.8B Triton 引擎的交互 demo。

    python demo.py "李世民是谁？和朱棣有什么共同点？"
    python demo.py "用一句话解释 RoPE" --max-tokens 200
    python demo.py "写一首关于秋天的诗" --thinking
    python demo.py "你好" --max-tokens 32 --no-compile     # 短生成用 eager 更快

全部计算走 triton_kernels/ 下手写的 13 个 kernel，不依赖 transformers。

当前是完整重算：每生成一个 token，对增长后的整个序列重跑一次 forward。所以
生成速度基本与序列长度无关（forward 耗时在 T=19..257 之间都差不多），
但也拿不到增量 decode 的加速。详见 HANDOFF.md 第 8、9 节。
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

from engine.runner import DEFAULT_STOP_IDS, build_runner

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
    parser.add_argument("--no-compile", action="store_true", help="关闭 torch.compile")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    rendered = render_chat(args.prompt, args.thinking)
    prompt_ids = tokenizer.encode(rendered, add_special_tokens=False).ids

    print(f"提问：{args.prompt}")
    print(f"prompt {len(prompt_ids)} tokens，最多生成 {args.max_tokens} tokens")
    print("-" * 72, flush=True)

    load_start = time.time()
    runner = build_runner(str(MODEL_DIR), compile=not args.no_compile)
    load_elapsed = time.time() - load_start

    ids = torch.tensor(prompt_ids, dtype=torch.int32, device=runner.device)
    generated: list[int] = []
    shown = ""
    first_token_time = None
    start = time.time()

    step_times: list[float] = []
    for _ in range(args.max_tokens):
        step_start = time.time()
        token = runner.next_token(ids)
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

        ids = torch.cat(
            [ids, torch.tensor([token], dtype=torch.int32, device=runner.device)]
        )

    elapsed = time.time() - start
    n = len(generated)
    ordered = sorted(step_times)
    median = ordered[len(ordered) // 2] * 1e3
    # 明显偏离中位数的是预热停顿。两个来源：
    #   - Triton 侧：首次遇到某个 T_BUCKET 时 JIT 编译 + autotune 全部 13 个 kernel，
    #     关掉 torch.compile 也躲不掉；
    #   - torch.compile 侧：首次编译、切 dynamic shape、以及跨过 T_BUCKET 边界
    #     （1/16/17/64/65/128/129 附近）时重新特化。
    stalls = [(i, t) for i, t in enumerate(step_times) if t > max(0.2, median * 20 / 1e3)]

    print("\n" + "-" * 72)
    print(f"加载 {load_elapsed:.1f}s | 生成 {n} tokens，总用时 {elapsed:.1f}s")
    print(f"稳态 {median:.1f} ms/token（中位数）")
    if stalls:
        total_stall = sum(t for _, t in stalls)
        print(
            f"预热停顿 {len(stalls)} 次，合计 {total_stall:.1f}s"
            f"（占总时间 {total_stall / elapsed:.0%}）："
        )
        for i, t in stalls[:8]:
            print(f"    step {i:>3}  T={len(prompt_ids) + i:>4}  {t:6.1f}s")
        print("  详见 HANDOFF.md 8.3；减少这类停顿是当前性能工作的第一项。")


if __name__ == "__main__":
    main()
