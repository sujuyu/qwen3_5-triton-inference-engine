import torch

import triton
import triton.language as tl


autotune_configs = [
    triton.Config({"BLOCK_M": 1, "BLOCK_N": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 2, "BLOCK_N": 256}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 256}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=8, num_stages=1),
]


@triton.autotune(
    configs=autotune_configs,
    key=["N", "M_BUCKET"],
)
@triton.jit 
def _swiglu_triton(
    gate_ptr, 
    stride_gate_m, stride_gate_n,
    up_ptr,
    stride_up_m, stride_up_n,
    out_ptr,
    stride_out_m, stride_out_n,
    M, N: tl.constexpr,
    M_BUCKET: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offset_m < M
    mask_n = offset_n < N

    gate = tl.load(
        gate_ptr + offset_m[:, None] * stride_gate_m + offset_n[None, :] * stride_gate_n,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)
    up = tl.load(
        up_ptr + offset_m[:, None] * stride_up_m + offset_n[None, :] * stride_up_n,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    ).to(tl.float32)

    out = up * gate * tl.sigmoid(gate)

    tl.store(
        out_ptr + offset_m[:, None] * stride_out_m + offset_n[None, :] * stride_out_n,
        out.to(tl.bfloat16),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _row_bucket(num_rows: int) -> int:
    if num_rows == 1:
        return 1
    if num_rows <= 16:
        return 16
    if num_rows <= 128:
        return 128
    return 129


@torch.library.triton_op(
    "wy_lib::swiglu",
    mutates_args=(),
)
def swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    assert gate.ndim == 2 and up.ndim == 2
    assert gate.shape == up.shape
    assert gate.dtype == torch.bfloat16 and up.dtype == torch.bfloat16
    assert gate.device == up.device

    num_rows, num_cols = gate.shape
    assert num_rows > 0 and num_cols > 0
    out = torch.empty_like(gate)

    def grid(meta):
        return (
            triton.cdiv(num_rows, meta["BLOCK_M"]),
            triton.cdiv(num_cols, meta["BLOCK_N"]),
        )

    torch.library.wrap_triton(_swiglu_triton)[grid](
        gate_ptr=gate,
        stride_gate_m=gate.stride(0),
        stride_gate_n=gate.stride(1),
        up_ptr=up,
        stride_up_m=up.stride(0),
        stride_up_n=up.stride(1),
        out_ptr=out,
        stride_out_m=out.stride(0),
        stride_out_n=out.stride(1),
        M=num_rows,
        N=num_cols,
        M_BUCKET=_row_bucket(num_rows),
    )
    return out


@torch.library.register_fake("wy_lib::swiglu")
def _swiglu_fake(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(gate)


def call_swiglu_triton(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    return swiglu(gate, up)


def _torch_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    gate_fp32 = gate.float()
    up_fp32 = up.float()
    return (gate_fp32 * torch.sigmoid(gate_fp32) * up_fp32).to(gate.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)

    test_shapes = [
        (1, 3584),
        (3, 3584),
        (17, 3584),
        (65, 3584),
        (129, 3584),
        (7, 1000),
    ]

    for shape in test_shapes:
        gate = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        up = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

        actual = call_swiglu_triton(gate, up)
        expected = _torch_reference(gate, up)
        max_abs_error = (actual.float() - expected.float()).abs().max().item()

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print(
            f"shape={shape}, max_abs_error={max_abs_error:.8f}, "
            f"best_config={_swiglu_triton.best_config}"
        )

    print("All SwiGLU tests passed.")
