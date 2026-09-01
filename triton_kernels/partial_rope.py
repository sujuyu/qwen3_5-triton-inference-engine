import torch

import triton
import triton.language as tl


# x:            [B, H, T, D] BF16
# position_ids: [B, T] I32/I64
# inv_freq:     [R / 2] FP32
# out:          [B, H, T, D] BF16
#
# The first R dimensions are rotated as two contiguous halves:
# [0, R/2) <-> [R/2, R). Dimensions [R, D) pass through unchanged.


autotune_configs = [
    triton.Config({"BLOCK_T": 1}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_T": 2}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_T": 4}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_T": 8}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_T": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_T": 16}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 32}, num_warps=4, num_stages=1),
]


@triton.autotune(
    configs=autotune_configs,
    # Avoid tuning once for every sequence length while keeping decode and
    # different prefill regimes independent.
    key=["D", "R", "T_BUCKET"],
)
@triton.jit
def _partial_rope_kernel(
    x_ptr,
    stride_x_b,
    stride_x_h,
    stride_x_t,
    stride_x_d,
    out_ptr,
    stride_out_b,
    stride_out_h,
    stride_out_t,
    stride_out_d,
    position_ids_ptr,
    stride_position_ids_b,
    stride_position_ids_t,
    inv_freq_ptr,
    token_num,
    D: tl.constexpr,
    R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    T_BUCKET: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_b = tl.program_id(2)

    offset_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offset_r = tl.arange(0, BLOCK_R)
    mask_t = offset_t < token_num
    mask_r = offset_r < R // 2

    positions = tl.load(
        position_ids_ptr
        + pid_b * stride_position_ids_b
        + offset_t * stride_position_ids_t,
        mask=mask_t,
        other=0,
    ).to(tl.float32)
    inv_freq = tl.load(inv_freq_ptr + offset_r, mask=mask_r, other=0.0)
    angles = positions[:, None] * inv_freq[None, :]
    cos = tl.cos(angles)
    sin = tl.sin(angles)

    x_base = x_ptr + pid_b * stride_x_b + pid_h * stride_x_h
    out_base = out_ptr + pid_b * stride_out_b + pid_h * stride_out_h
    rotate_mask = mask_t[:, None] & mask_r[None, :]

    first_offset = offset_r
    second_offset = offset_r + R // 2
    first = tl.load(
        x_base
        + offset_t[:, None] * stride_x_t
        + first_offset[None, :] * stride_x_d,
        mask=rotate_mask,
        other=0.0,
    ).to(tl.float32)
    second = tl.load(
        x_base
        + offset_t[:, None] * stride_x_t
        + second_offset[None, :] * stride_x_d,
        mask=rotate_mask,
        other=0.0,
    ).to(tl.float32)

    out_first = first * cos - second * sin
    out_second = first * sin + second * cos
    tl.store(
        out_base
        + offset_t[:, None] * stride_out_t
        + first_offset[None, :] * stride_out_d,
        out_first,
        mask=rotate_mask,
    )
    tl.store(
        out_base
        + offset_t[:, None] * stride_out_t
        + second_offset[None, :] * stride_out_d,
        out_second,
        mask=rotate_mask,
    )

    # Copy the non-rotary tail without requiring D or R/2 to be a power of 2.
    for start_d in range(R, D, BLOCK_R):
        offset_d = start_d + offset_r
        pass_mask = mask_t[:, None] & (offset_d[None, :] < D)
        value = tl.load(
            x_base
            + offset_t[:, None] * stride_x_t
            + offset_d[None, :] * stride_x_d,
            mask=pass_mask,
            other=0.0,
        )
        tl.store(
            out_base
            + offset_t[:, None] * stride_out_t
            + offset_d[None, :] * stride_out_d,
            value,
            mask=pass_mask,
        )


def _token_bucket(token_num: int) -> int:
    if token_num == 1:
        return 1
    if token_num <= 16:
        return 16
    if token_num <= 128:
        return 128
    return 129


@torch.library.triton_op(
    "wy_lib::partial_rope",
    mutates_args=(),
)
def partial_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    assert x.ndim == 4
    assert x.dtype == torch.bfloat16
    assert position_ids.ndim == 2
    assert position_ids.dtype in (torch.int32, torch.int64)
    assert inv_freq.ndim == 1 and inv_freq.dtype == torch.float32
    assert x.device == position_ids.device == inv_freq.device

    batch_size, num_heads, token_num, head_dim = x.shape
    assert position_ids.shape == (batch_size, token_num)
    assert rotary_dim > 0 and rotary_dim <= head_dim and rotary_dim % 2 == 0
    assert inv_freq.shape == (rotary_dim // 2,)

    out = torch.empty_like(x)
    block_r = triton.next_power_of_2(rotary_dim // 2)

    def grid(meta):
        return (
            triton.cdiv(token_num, meta["BLOCK_T"]),
            num_heads,
            batch_size,
        )

    torch.library.wrap_triton(_partial_rope_kernel)[grid](
        x_ptr=x,
        stride_x_b=x.stride(0),
        stride_x_h=x.stride(1),
        stride_x_t=x.stride(2),
        stride_x_d=x.stride(3),
        out_ptr=out,
        stride_out_b=out.stride(0),
        stride_out_h=out.stride(1),
        stride_out_t=out.stride(2),
        stride_out_d=out.stride(3),
        position_ids_ptr=position_ids,
        stride_position_ids_b=position_ids.stride(0),
        stride_position_ids_t=position_ids.stride(1),
        inv_freq_ptr=inv_freq,
        token_num=token_num,
        D=head_dim,
        R=rotary_dim,
        BLOCK_R=block_r,
        T_BUCKET=_token_bucket(token_num),
    )
    return out


@torch.library.register_fake("wy_lib::partial_rope")
def _partial_rope_fake(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    return torch.empty_like(x)


def call_partial_rope_triton(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    return partial_rope(x, position_ids, inv_freq, rotary_dim)


def _torch_reference(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    rotary_dim: int,
) -> torch.Tensor:
    half = rotary_dim // 2
    angles = position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, half)
    cos = torch.cos(angles).unsqueeze(1)
    sin = torch.sin(angles).unsqueeze(1)

    first = x[..., :half].float()
    second = x[..., half:rotary_dim].float()
    out = x.clone()
    out[..., :half] = (first * cos - second * sin).to(x.dtype)
    out[..., half:rotary_dim] = (first * sin + second * cos).to(x.dtype)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    theta = 10_000_000.0
    test_cases = [
        (1, 8, 1, 256, 64),
        (1, 2, 17, 256, 64),
        (2, 3, 65, 128, 64),
        # Exercise non-power-of-two rotary dimensions and a non-rotary tail.
        (1, 4, 33, 160, 96),
    ]

    for batch_size, num_heads, token_num, head_dim, rotary_dim in test_cases:
        x = torch.randn(
            (batch_size, num_heads, token_num, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        )
        position_ids = torch.arange(token_num, device="cuda", dtype=torch.int32)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, rotary_dim, 2, device="cuda", dtype=torch.float32)
                / rotary_dim
            )
        )

        actual = call_partial_rope_triton(x, position_ids, inv_freq, rotary_dim)
        expected = _torch_reference(x, position_ids, inv_freq, rotary_dim)
        max_abs_error = (actual.float() - expected.float()).abs().max().item()

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        assert torch.equal(actual[..., rotary_dim:], x[..., rotary_dim:])
        print(
            f"shape={tuple(x.shape)}, rotary_dim={rotary_dim}, "
            f"max_abs_error={max_abs_error:.8f}, "
            f"best_config={_partial_rope_kernel.best_config}"
        )

    print("All partial RoPE tests passed.")
