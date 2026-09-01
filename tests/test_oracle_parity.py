"""把 Triton runner 的中间量逐个对上 Hugging Face oracle。

前置：先在 .venv-oracle 里跑过 tools/dump_oracle.py 生成 oracle/。
本测试跑在主环境，不需要 transformers。

    python tests/test_oracle_parity.py

参考实现有四处在 BF16 下计算而我们全程 FP32（见 HANDOFF.md 6.3），所以这里按
容差比而不是 bit-match。判定标准分两层：

- 逐层 hidden 的相对误差不随层数发散；
- greedy token 序列必须逐个相同——这是真正的验收线。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.runner import build_runner

ORACLE = ROOT / "oracle"


def rel_error(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """返回 (最大绝对误差, 相对于 expected 幅度的相对误差)。"""
    a = actual.detach().float().cpu().reshape(-1)
    e = expected.detach().float().cpu().reshape(-1)
    assert a.numel() == e.numel(), f"元素数不同: {a.numel()} vs {e.numel()}"
    max_abs = (a - e).abs().max().item()
    scale = e.abs().max().item()
    return max_abs, (max_abs / scale if scale > 0 else 0.0)


def check(name: str, actual, expected, tol: float, results: list) -> None:
    max_abs, rel = rel_error(actual, expected)
    ok = rel <= tol
    results.append((name, max_abs, rel, tol, ok))
    flag = "  " if ok else "!!"
    print(f"{flag} {name:44} max_abs={max_abs:10.6f}  rel={rel:8.4%}  (tol {tol:.1%})")


def main() -> None:
    assert ORACLE.exists(), (
        f"{ORACLE} 不存在，先跑 .venv-oracle/bin/python tools/dump_oracle.py"
    )
    meta = torch.load(ORACLE / "meta.pt", weights_only=False)
    hidden = torch.load(ORACLE / "hidden.pt")
    o_gdn = torch.load(ORACLE / "layer00_gdn.pt")
    o_attn = torch.load(ORACLE / "layer03_attn.pt")

    runner = build_runner()
    input_ids = meta["input_ids"].to(torch.int32).cuda()
    token_num = int(meta["seq_len"])
    heads, head_dim = runner.w.linear_num_heads, runner.w.linear_head_dim

    trace: dict[str, torch.Tensor] = {}
    runner.forward(input_ids, trace=trace, trace_layers={0, 3})

    results: list[tuple] = []

    # hidden_states 索引：hidden_00 = embedding 输出，hidden_{i+1} = 第 i 层输出（i≤22），
    # 而 hidden_24 已经是 final norm 之后。第 23 层的原始输出由单独的 hook 存成 layer23_out。
    print("\n===== 逐层 hidden（26 个锚点）=====")
    check("embed", trace["embed"], hidden["hidden_00"][0], 0.02, results)
    for i in range(runner.w.num_layers - 1):
        check(
            f"layer{i:02d}.out",
            trace[f"layer{i:02d}.out"],
            hidden[f"hidden_{i + 1:02d}"][0],
            0.05,
            results,
        )
    check("layer23.out", trace["layer23.out"], hidden["layer23_out"][0], 0.05, results)
    check("final_norm", trace["final_norm"], hidden["final_norm"][0], 0.05, results)

    print("\n===== layer00 Gated DeltaNet 内部 =====")
    check(
        "layer00.input_layernorm",
        trace["layer00.input_layernorm"],
        o_gdn["layer00.input_layernorm.out"][0],
        0.02,
        results,
    )
    check(
        "layer00.in_proj_qkv",
        trace["layer00.in_proj_qkv"],
        o_gdn["layer00.linear_attn.in_proj_qkv.out"][0],
        0.02,
        results,
    )
    check(
        "layer00.in_proj_z",
        trace["layer00.in_proj_z"],
        o_gdn["layer00.linear_attn.in_proj_z.out"][0],
        0.02,
        results,
    )
    # oracle 的 conv 是 [1,C,T]，我们是 [T,C]
    check(
        "layer00.conv(+silu)",
        trace["layer00.conv"],
        o_gdn["layer00.conv.out_bct"][0].transpose(0, 1),
        0.02,
        results,
    )
    for name, key in (("q", "query"), ("k", "key"), ("v", "value")):
        check(
            f"layer00.{name}_split",
            trace[f"layer00.{name}_pre_norm" if name != "v" else "layer00.v"],
            o_gdn[f"layer00.delta.{key}_pre_l2norm"][0],
            0.02,
            results,
        )
    check("layer00.beta", trace["layer00.beta"], o_gdn["layer00.delta.beta"][0], 0.02, results)
    check("layer00.g", trace["layer00.g"], o_gdn["layer00.delta.g"][0], 0.02, results)
    check(
        "layer00.core_attn_out (delta rule)",
        trace["layer00.core_attn_out"],
        o_gdn["layer00.delta.core_attn_out"][0],
        0.05,
        results,
    )
    check(
        "layer00.gated_rmsnorm",
        trace["layer00.gated_norm"].reshape(token_num * heads, head_dim),
        o_gdn["layer00.linear_attn.norm.out"],
        0.05,
        results,
    )
    check(
        "layer00.out_proj",
        trace["layer00.out_proj"],
        o_gdn["layer00.linear_attn.out_proj.out"][0],
        0.05,
        results,
    )

    print("\n===== layer03 full attention 内部 =====")
    check(
        "layer03.input_layernorm",
        trace["layer03.input_layernorm"],
        o_attn["layer03.input_layernorm.out"][0],
        0.02,
        results,
    )
    # oracle 的 q_proj 输出未拆分，这里按 head 切出 Q 和 gate 与我们的两个 GEMM 对
    q_proj_full = o_attn["layer03.self_attn.q_proj.out"][0].view(
        token_num, runner.w.num_attention_heads, runner.w.head_dim * 2
    )
    check(
        "layer03.q_proj (拆分后 Q)",
        trace["layer03.q_proj_q"].view(token_num, runner.w.num_attention_heads, -1),
        q_proj_full[..., : runner.w.head_dim],
        0.02,
        results,
    )
    check(
        "layer03.q_proj (拆分后 gate)",
        trace["layer03.q_proj_gate"].view(token_num, runner.w.num_attention_heads, -1),
        q_proj_full[..., runner.w.head_dim :],
        0.02,
        results,
    )
    check(
        "layer03.k_proj",
        trace["layer03.k_proj"],
        o_attn["layer03.self_attn.k_proj.out"][0],
        0.02,
        results,
    )
    check(
        "layer03.v_proj",
        trace["layer03.v_proj"],
        o_attn["layer03.self_attn.v_proj.out"][0],
        0.02,
        results,
    )
    check(
        "layer03.q_norm",
        trace["layer03.q_after_norm"],
        o_attn["layer03.self_attn.q_norm.out"][0],
        0.02,
        results,
    )
    check(
        "layer03.k_norm",
        trace["layer03.k_after_norm"],
        o_attn["layer03.self_attn.k_norm.out"][0],
        0.02,
        results,
    )
    check(
        "layer03.rope_q",
        trace["layer03.rope_q"][0],
        o_attn["layer03.rope.q_out"][0],
        0.02,
        results,
    )
    check(
        "layer03.rope_k",
        trace["layer03.rope_k"][0],
        o_attn["layer03.rope.k_out"][0],
        0.02,
        results,
    )
    check(
        "layer03.gated pack (o_proj 输入)",
        trace["layer03.packed"],
        o_attn["layer03.self_attn.o_proj.in0"][0],
        0.05,
        results,
    )
    check(
        "layer03.o_proj",
        trace["layer03.out_proj"],
        o_attn["layer03.self_attn.o_proj.out"][0],
        0.05,
        results,
    )

    print("\n===== 端到端 greedy =====")
    expected_tokens = meta["generated_tokens"]
    actual_tokens = runner.generate(input_ids, max_new_tokens=len(expected_tokens))
    match = actual_tokens == expected_tokens
    print(f"   oracle : {expected_tokens}")
    print(f"   triton : {actual_tokens}")
    if match:
        print(f"   32 个 token 逐个相同 ✓")
    else:
        first = next(
            i for i, (a, b) in enumerate(zip(actual_tokens, expected_tokens)) if a != b
        )
        print(f"   !! 第 {first} 个 token 开始分叉: {actual_tokens[first]} vs {expected_tokens[first]}")

    print("\n===== 汇总 =====")
    failed = [r for r in results if not r[4]]
    worst = max(results, key=lambda r: r[2])
    print(f"逐算子检查 {len(results)} 项，超容差 {len(failed)} 项")
    print(f"最大相对误差: {worst[0]} = {worst[2]:.4%}")
    layer_rels = [r[2] for r in results if r[0].endswith(".out") and r[0].startswith("layer")]
    print(
        f"逐层 hidden 相对误差: 首层 {layer_rels[0]:.4%} -> 末层 {layer_rels[-1]:.4%}"
        f"（最大 {max(layer_rels):.4%}）"
    )

    assert not failed, f"以下检查超容差: {[r[0] for r in failed]}"
    assert match, "greedy token 序列与 oracle 不一致"
    print("\n全部通过。")


if __name__ == "__main__":
    main()
