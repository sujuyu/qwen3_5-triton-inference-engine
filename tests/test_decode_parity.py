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

    # ---- 5. CUDA Graph 化的 decode -------------------------------------
    # 图内闭环（argmax 结果直接写回输入槽、pos 自增），host 每步只 replay。
    print("\n=== CUDA Graph decode ===")
    from engine.runner import GraphedDecoder

    caches.reset()
    runner.prefill(prompt, caches)
    decoder = GraphedDecoder(runner, caches)
    decoder.capture()  # 会把 cache 写脏，下面 generate_graphed 里会重新 prefill

    graphed = runner.generate_graphed(
        prompt, max_new_tokens=n_steps, stop_ids=None, decoder=decoder
    )
    print(f"  graph : {graphed[:10]} ...")
    assert graphed == ref_tokens, "CUDA Graph 路径与完整重算不一致"
    print(f"  {len(graphed)} 个 token 与完整重算逐个相同 ✓")

    # 复用同一张图再跑一次，验证 reset 把状态清干净了
    again = runner.generate_graphed(
        prompt, max_new_tokens=n_steps, stop_ids=None, decoder=decoder
    )
    assert again == ref_tokens, "复用图后结果不可复现——cache 或 pos 没复位"
    print("  复用同一张图，结果一致 ✓")

    # 容量守卫：越界应该在 host 侧报清楚的错，而不是 device-side assert
    try:
        runner.generate_graphed(
            prompt, max_new_tokens=caches.max_len, stop_ids=None, decoder=decoder
        )
        raise SystemExit("容量越界没有被拦住")
    except AssertionError as exc:
        assert "超出 cache 容量" in str(exc)
        print("  超出 cache 容量时在 host 侧报错 ✓")

    # ---- 6. GDN prefill 的两条路径：sequential vs chunk-64 ---------------
    # 这一段是必要的，因为**上面所有测试都跑不到 chunked 路径**——oracle 的
    # prompt 只有 19 个 token，远低于 GDN_CHUNKED_PREFILL_MIN_TOKENS，而
    # chunked 与 sequential 在单个 chunk 内本来就等价，分辨不出任何东西。
    #
    # 判据用 **greedy token 序列**而不是 hidden 的相对误差。原因：两条路径在
    # 24 层之后的 hidden 相对差约 2.8e-2，看着很大，但模型自身对 HF oracle 的
    # 误差就有 1.8e-2——这个量级是 BF16 在 24 层上累积的噪声底，不是谁算错了。
    # （实测原版 ieee 精度的 chunked 路径同样偏离 2.3e-2，所以这不是后来做
    # tensor core 化引入的。）真正该问的是"输出会不会变"，答案是不会。
    print("\n=== GDN prefill：sequential vs chunk-64 ===")
    import engine.runner as runner_module

    long_prompt = prompt.repeat((2048 + prompt_len - 1) // prompt_len)[:2048].contiguous()
    long_caches = allocate_caches(runner.w, long_prompt.shape[0] + 80)
    saved_threshold = runner_module.GDN_CHUNKED_PREFILL_MIN_TOKENS
    try:
        variants = {}
        for name, threshold in (("sequential", 10**9), ("chunked", 0)):
            runner_module.GDN_CHUNKED_PREFILL_MIN_TOKENS = threshold
            long_caches.reset()
            hidden = runner.prefill(long_prompt, long_caches)
            states = [s.clone() for s in long_caches.recurrent_states if s is not None]
            variants[name] = (hidden.clone(), states)
            variants[name + "_tokens"] = runner.generate_cached(
                long_prompt, max_new_tokens=64, stop_ids=None, caches=long_caches
            )
    finally:
        runner_module.GDN_CHUNKED_PREFILL_MIN_TOKENS = saved_threshold

    h_seq, s_seq = variants["sequential"]
    h_chk, s_chk = variants["chunked"]
    _, h_rel = rel(h_chk, h_seq)
    s_rel = max(rel(a, b)[1] for a, b in zip(s_chk, s_seq))
    print(f"  T={long_prompt.shape[0]}  hidden rel={h_rel:.2%}  recurrent state rel={s_rel:.2%}")
    print(f"  （参考：整个模型对 HF oracle 的 hidden 误差就有 1.8%）")

    tok_seq = variants["sequential_tokens"]
    tok_chk = variants["chunked_tokens"]
    if tok_seq == tok_chk:
        print(f"  {len(tok_seq)} 个 greedy token 逐个相同 ✓")
    else:
        first = next(i for i, (a, b) in enumerate(zip(tok_seq, tok_chk)) if a != b)
        raise AssertionError(
            f"chunked 与 sequential 的 greedy 结果在第 {first} 个 token 分叉："
            f"{tok_chk[first]} vs {tok_seq[first]}"
        )

    print("\nAll decode parity tests passed.")


if __name__ == "__main__":
    main()
