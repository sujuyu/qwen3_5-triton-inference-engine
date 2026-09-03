"""增量 decode 路径 vs 完整重算路径的等价性。

    python tests/test_decode_parity.py

完整重算那条路径已经对过 Hugging Face oracle（tests/test_oracle_parity.py，49 项
逐算子 + 端到端），所以这里拿它当基准，只验证"用 cache 续算"和"每步从头重算"
是否等价。这样把两件事分开：模型实现对不对，和 cache 管理对不对。

判据分三层，从局部到整体：

1. **逐层 hidden**：prefill 之后紧接一个 decode step，把它的每层输出与"对
   prompt+1 做完整重算"的对应行比。这一层能定位到具体是哪个 cache 出问题。
2. **单步 hidden**：连续 decode 若干步，每步与完整重算比最终 hidden。
3. **greedy token 序列**：整体验收。

三类 cache 各自的失效表现不一样，值得记住：

    conv state 错      -> 只影响 GDN 层，误差从第 1 个 decode token 就出现
    recurrent state 错 -> 同上，但因为状态会累积，误差随步数放大
    KV cache 错        -> 只影响 attention 层；如果是"没存 RoPE 后的 K"，
                          误差随距离增大而增大（位置越远旋转差得越多）
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.cache import allocate_caches
from engine.runner import build_runner
from triton_kernels.vocab_argmax import lm_head_argmax

ORACLE = ROOT / "oracle"


def rel(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    a, b = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    m = (a - b).abs().max().item()
    scale = b.abs().max().item()
    return m, (m / scale if scale > 0 else 0.0)


def main() -> None:
    # 增量路径与完整重算的差别只在 cache，本身不涉及 compile；关掉以免 5s 编译
    runner = build_runner(compile=False)

    meta = torch.load(ORACLE / "meta.pt", weights_only=False)
    prompt = meta["input_ids"].to(torch.int32).cuda()
    prompt_len = prompt.shape[0]
    n_steps = 24
    caches = allocate_caches(runner.w, prompt_len + n_steps + 8)

    # 先用完整重算跑出参考 token 序列（这条路径已对过 oracle）
    ref_tokens = runner.generate(prompt, max_new_tokens=n_steps, stop_ids=None)

    # ---- 1. prefill + 第一个 decode step 的逐层对拍 --------------------
    print("=== prefill + 首个 decode step 的逐层 hidden ===")
    caches.reset()
    runner.prefill(prompt, caches)
    assert int(caches.pos.item()) == prompt_len

    slot = torch.tensor([ref_tokens[0]], dtype=torch.int32, device=runner.device)
    dec_trace: dict[str, torch.Tensor] = {}
    dec_hidden = runner.decode_step(slot, caches, trace=dec_trace)

    full_ids = torch.cat([prompt, slot])
    full_trace: dict[str, torch.Tensor] = {}
    full_hidden = runner.forward(full_ids, trace=full_trace, trace_layers=set())

    # 逐层对：GDN 层暴露 conv/recurrent state 的问题，attention 层暴露 KV cache 的
    by_kind = {"GDN": [], "attn": []}
    for i in range(runner.w.num_layers):
        key = f"layer{i:02d}.out"
        m, r = rel(dec_trace[key][0], full_trace[key][-1])
        kind = "attn" if i in (3, 7, 11, 15, 19, 23) else "GDN"
        by_kind[kind].append(r)
        if i < 4 or i >= runner.w.num_layers - 2:
            print(f"  layer{i:02d} ({kind:<4}) max_abs={m:.6f}  rel={r:.4%}")
    for kind, rs in by_kind.items():
        print(f"  {kind:<4} 层 rel 误差: 首层 {rs[0]:.4%}  末层 {rs[-1]:.4%}  "
              f"最大 {max(rs):.4%}")
    m, r = rel(dec_hidden[0], full_hidden[-1])
    print(f"  final norm 后 max_abs={m:.6f}  rel={r:.4%}")
    assert r < 0.03, "首个 decode step 就与完整重算不一致"

    # ---- 2. 连续 decode，每步与完整重算比 hidden ------------------------
    print("\n=== 连续 decode，每步与完整重算比 final hidden ===")
    caches.reset()
    runner.prefill(prompt, caches)
    worst = 0.0
    ids = prompt.clone()
    for step, tok in enumerate(ref_tokens[:n_steps]):
        slot.fill_(tok)
        dec_hidden = runner.decode_step(slot, caches)
        caches.pos.add_(1)
        ids = torch.cat([ids, slot])
        full_hidden = runner.forward(ids)
        m, r = rel(dec_hidden[0], full_hidden[-1])
        worst = max(worst, r)
        if step < 3 or step == n_steps - 1:
            print(f"  step {step:>2}  pos={int(caches.pos.item()):>3}  "
                  f"max_abs={m:.6f}  rel={r:.4%}")
    print(f"  {n_steps} 步内最大相对误差 {worst:.4%}")
    assert worst < 0.03, f"增量 decode 与完整重算偏离 {worst:.2%}"

    # ---- 3. greedy token 序列 -------------------------------------------
    print("\n=== greedy token 序列 ===")
    cached_tokens = runner.generate_cached(
        prompt, max_new_tokens=n_steps, stop_ids=None, caches=caches
    )
    print(f"  完整重算: {ref_tokens}")
    print(f"  增量decode: {cached_tokens}")
    if cached_tokens == ref_tokens:
        print(f"  {len(ref_tokens)} 个 token 逐个相同 ✓")
    else:
        first = next(
            i for i, (a, b) in enumerate(zip(cached_tokens, ref_tokens)) if a != b
        )
        raise AssertionError(
            f"第 {first} 个 token 分叉: {cached_tokens[first]} vs {ref_tokens[first]}"
        )

    # ---- 4. reset 之后可以重跑 ------------------------------------------
    print("\n=== reset 后重跑 ===")
    again = runner.generate_cached(
        prompt, max_new_tokens=8, stop_ids=None, caches=caches
    )
    assert again == ref_tokens[:8], "reset 之后结果不可复现——cache 没清干净"
    print(f"  同一份 caches 复用，结果一致 ✓")

    print("\nAll decode parity tests passed.")


if __name__ == "__main__":
    main()
