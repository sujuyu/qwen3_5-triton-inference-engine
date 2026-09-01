#!/usr/bin/env python3
"""Tokenize one text-only chat prompt with the local Qwen3.5-0.8B tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer


MODEL_DIR = Path(__file__).resolve().parent / "Qwen3.5-0.8B"


def render_single_user_chat(prompt: str, enable_thinking: bool) -> str:
    """Render the one-user-turn subset of the checkpoint's official chat template."""
    prefix = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    if enable_thinking:
        return prefix + "<think>\n"
    return prefix + "<think>\n\n</think>\n\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="你好，请简单介绍一下自己。")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    rendered_prompt = render_single_user_chat(args.prompt, args.thinking)
    encoded = tokenizer.encode(rendered_prompt, add_special_tokens=False)

    # These are ordinary Python integer lists and can be passed to the custom
    # runtime as int32 input_ids. A single unpadded request has an all-ones mask.
    input_ids: list[int] = encoded.ids
    attention_mask = [1] * len(input_ids)

    print(f"tokenizer={type(tokenizer).__name__}")
    print(f"tokenizer_vocab_size={tokenizer.get_vocab_size(with_added_tokens=True)}")
    print(f"token_count={len(input_ids)}")
    print(f"input_ids={input_ids}")
    print(f"attention_mask={attention_mask}")
    print("rendered_prompt:")
    print(tokenizer.decode(input_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
