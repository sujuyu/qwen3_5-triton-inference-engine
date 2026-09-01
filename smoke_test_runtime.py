#!/usr/bin/env python3
"""Compile and run one minimal Triton kernel on the active CUDA device."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def main() -> None:
    n_elements = 65_537
    x = torch.randn(n_elements, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    output = torch.empty_like(x)

    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    add_kernel[grid](x, y, output, n_elements=n_elements, BLOCK_SIZE=block_size)
    torch.testing.assert_close(output, x + y)

    print(f"torch={torch.__version__}")
    print(f"triton={triton.__version__}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print("triton_vector_add=PASS")


if __name__ == "__main__":
    main()
