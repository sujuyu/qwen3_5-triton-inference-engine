import torch

import triton
import triton.language as tl


@triton.jit
def _residual_add_kernel(
    x_ptr,
    residual_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + residual, mask=mask)


@torch.library.triton_op(
    "wy_lib::residual_add",
    mutates_args=(),
)
def residual_add(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    assert x.shape == residual.shape
    assert x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
    assert x.device == residual.device
    assert x.is_contiguous() and residual.is_contiguous()
    assert x.numel() > 0

    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)

    torch.library.wrap_triton(_residual_add_kernel)[grid](
        x_ptr=x,
        residual_ptr=residual,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return out


@torch.library.register_fake("wy_lib::residual_add")
def _residual_add_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


def call_residual_add_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return residual_add(x, residual)


if __name__ == "__main__":
    torch.manual_seed(0)

    test_shapes = [
        (1, 1024),
        (3, 1024),
        (17, 1024),
        (65, 1024),
        (2, 3, 1024),
        (3, 1000),
    ]

    for shape in test_shapes:
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        residual = torch.randn_like(x)

        actual = call_residual_add_triton(x, residual)
        expected = x + residual
        max_abs_error = (actual.float() - expected.float()).abs().max().item()

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        print(f"shape={shape}, max_abs_error={max_abs_error:.8f}")

    print("All residual add tests passed.")
