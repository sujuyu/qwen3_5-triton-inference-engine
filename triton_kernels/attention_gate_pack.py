import torch

import triton
import triton.language as tl


autotune_configs = [
    triton.Config({"BLOCK_T": 1}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_T": 2}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_T": 4}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_T": 8}, num_warps=4, num_stages=1),
]


@triton.autotune(
    configs=autotune_configs,
    key=["D", "T_BUCKET"],
)
@triton.jit 
def _attention_gate_pack_kernel(
    x_ptr, # [B, H, T, D] BF16
    stride_x_b, stride_x_h, stride_x_t, stride_x_d,
    gate_ptr, # [B, H, T, D] BF16
    stride_gate_b, stride_gate_h, stride_gate_t, stride_gate_d,
    out_ptr, # [B*T, H*D] BF16
    stride_out_m, stride_out_n,
    token_num: int, 
    D: tl.constexpr,
    T_BUCKET: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # batch head维度上单独起block
    # token维度上进行分块
    # 在这次的场景下 batch恒定是 1
    pid_b, pid_h = tl.program_id(2), tl.program_id(1)
    pid_t = tl.program_id(0)

    offset_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offset_t < token_num

    offset_d = tl.arange(0, D)

    x_base = x_ptr + pid_b * stride_x_b + pid_h * stride_x_h
    gate_base = gate_ptr + pid_b * stride_gate_b + pid_h * stride_gate_h

    gate = tl.load(
        gate_base + offset_t[:, None] * stride_gate_t + offset_d[None, :] * stride_gate_d,
        mask=mask_t[:, None],
        other=0.0,
    ).to(tl.float32)
    x = tl.load(
        x_base + offset_t[:, None] * stride_x_t + offset_d[None, :] * stride_x_d,
        mask=mask_t[:, None],
        other=0.0,
    )
    out = x * tl.sigmoid(gate)

    # 因为batch恒定是1 写回的时候不需要考虑pid_b
    offset_out_d = pid_h * D + offset_d
    tl.store(
        out_ptr + offset_t[:, None] * stride_out_m + offset_out_d[None, :] * stride_out_n,
        out,
        mask=mask_t[:, None],
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
    "wy_lib::attention_gate_pack",
    mutates_args=(),
)
def attention_gate_pack(
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    assert x.ndim == 4 and gate.ndim == 4
    assert x.shape == gate.shape
    assert x.dtype == torch.bfloat16 and gate.dtype == torch.bfloat16
    assert x.device == gate.device

    batch_size, num_heads, token_num, head_dim = x.shape
    assert batch_size == 1
    assert token_num > 0
    assert head_dim > 0 and triton.next_power_of_2(head_dim) == head_dim

    out = torch.empty(
        (token_num, num_heads * head_dim),
        dtype=x.dtype,
        device=x.device,
    )

    def grid(meta):
        return (
            triton.cdiv(token_num, meta["BLOCK_T"]),
            num_heads,
            batch_size,
        )

    torch.library.wrap_triton(_attention_gate_pack_kernel)[grid](
        x_ptr=x,
        stride_x_b=x.stride(0),
        stride_x_h=x.stride(1),
        stride_x_t=x.stride(2),
        stride_x_d=x.stride(3),
        gate_ptr=gate,
        stride_gate_b=gate.stride(0),
        stride_gate_h=gate.stride(1),
        stride_gate_t=gate.stride(2),
        stride_gate_d=gate.stride(3),
        out_ptr=out,
        stride_out_m=out.stride(0),
        stride_out_n=out.stride(1),
        token_num=token_num,
        D=head_dim,
        T_BUCKET=_token_bucket(token_num),
    )
    return out


@torch.library.register_fake("wy_lib::attention_gate_pack")
def _attention_gate_pack_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[2], x.shape[1] * x.shape[3]),
        dtype=x.dtype,
        device=x.device,
    )


def call_attention_gate_pack_triton(
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return attention_gate_pack(x, gate)


def _torch_reference(
    x: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_heads, token_num, head_dim = x.shape
    out = x.float() * torch.sigmoid(gate.float())
    return out.permute(0, 2, 1, 3).reshape(
        batch_size * token_num,
        num_heads * head_dim,
    ).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)

    for token_num in (1, 3, 17, 65, 129):
        # The permute creates the strided [B,H,T,D] layout produced by the
        # surrounding attention path.
        x = torch.randn(
            (1, token_num, 8, 256),
            dtype=torch.bfloat16,
            device="cuda",
        ).permute(0, 2, 1, 3)
        gate = torch.randn(
            (1, token_num, 8, 256),
            dtype=torch.bfloat16,
            device="cuda",
        ).permute(0, 2, 1, 3)

        actual = call_attention_gate_pack_triton(x, gate)
        expected = _torch_reference(x, gate)
        max_abs_error = (actual.float() - expected.float()).abs().max().item()

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        assert actual.is_contiguous()
        print(
            f"shape={tuple(x.shape)}, max_abs_error={max_abs_error:.8f}, "
            f"best_config={_attention_gate_pack_kernel.best_config}"
        )

    print("All attention gate pack tests passed.")
