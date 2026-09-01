"""gemm_2d 在 runner 实际用到的全部尺寸上的正确性测试。

kernel 自带的自测只覆盖 (M,K,N)=(1,128,128)，而 runner 每一层都在用它。这里按真实
投影尺寸补齐，独立成文件是为了不改动 triton_kernels/gemm_2d.py 本体。

    python tests/test_gemm_model_shapes.py

reference 用 FP32 matmul。BF16 输入 + FP32 累加的结果与 FP32 参考的差异应在 BF16
量级（相对误差 ~1%），这里按相对误差判定而非绝对值。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triton_kernels.gemm_2d import gemm_2d


# (K, N, 说明)——K 是 in_features，N 是 out_features，权重为 [N,K]。
MODEL_SHAPES = [
    (1024, 16, "GDN in_proj_a / in_proj_b"),
    (1024, 512, "attn k_proj / v_proj"),
    (1024, 2048, "attn q_proj_q / q_proj_gate, GDN in_proj_z"),
    (1024, 3584, "MLP gate_proj / up_proj"),
    (1024, 6144, "GDN in_proj_qkv"),
    (2048, 1024, "attn o_proj, GDN out_proj"),
    (3584, 1024, "MLP down_proj"),
]

# T=1 是 decode，19 是 oracle 的 prompt 长度，65/129 跨过常见 tile 边界。
TOKEN_COUNTS = [1, 19, 65, 129]

REL_TOL = 0.02


def main() -> None:
    torch.manual_seed(0)
    failures = []

    print(f"{'shape (M,K,N)':>22}  {'max_abs':>10}  {'rel':>9}   说明")
    for k, n, note in MODEL_SHAPES:
        assert k % 128 == 0, f"K={k} 不满足 gemm_2d 的 K%128==0"
        weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        for m in TOKEN_COUNTS:
            x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")

            actual = gemm_2d(x, weight)
            expected = (x.float() @ weight.float().T).to(torch.bfloat16)

            assert actual.shape == (m, n), f"输出形状 {tuple(actual.shape)} != {(m, n)}"
            assert actual.dtype == torch.bfloat16
            assert actual.is_contiguous()
            assert torch.isfinite(actual).all(), "输出含 NaN/Inf"

            diff = (actual.float() - expected.float()).abs().max().item()
            scale = expected.float().abs().max().item()
            rel = diff / scale if scale > 0 else 0.0
            ok = rel <= REL_TOL
            if not ok:
                failures.append((m, k, n, rel))
            print(
                f"{f'({m},{k},{n})':>22}  {diff:10.6f}  {rel:8.4%}  {'  ' if ok else '!!'} {note}"
            )

    assert not failures, f"超容差: {failures}"
    print(f"\n{len(MODEL_SHAPES) * len(TOKEN_COUNTS)} 组全部通过（相对容差 {REL_TOL:.0%}）。")


if __name__ == "__main__":
    main()
