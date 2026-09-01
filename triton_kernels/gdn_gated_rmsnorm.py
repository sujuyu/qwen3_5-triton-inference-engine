import torch

import triton
import triton.language as tl


autotune_configs = [
    triton.Config({"BLOCK_T": 1}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_T": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 2}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 4}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 8}, num_warps=4, num_stages=1),
]


@triton.autotune(
    configs=autotune_configs,
    key=["head_num", "d_model", "T_BUCKET"],
)
@triton.jit
def _gdn_gated_rmsnorm_triton(
    x_ptr, 
    stride_x_t: tl.constexpr, stride_x_h: tl.constexpr, stride_x_d: tl.constexpr, 
    z_ptr, 
    stride_z_t: tl.constexpr, stride_z_h: tl.constexpr, stride_z_d: tl.constexpr,
    weight_ptr,
    out_ptr, 
    stride_o_t: tl.constexpr, stride_o_h: tl.constexpr, stride_o_d: tl.constexpr,
    d_model: tl.constexpr, 
    head_num: tl.constexpr,
    token_num,
    T_BUCKET: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    # 在token维度上按照BLOCK_T上进行切分
    # 在head维度上 BLOCK_H=1
    # d_model维度上不做切分

    pid_t, pid_h = tl.program_id(1), tl.program_id(0)
    offset_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offset_d = tl.arange(0, d_model)

    x = tl.load(
        x_ptr + offset_t[:, None] * stride_x_t + pid_h * stride_x_h + offset_d[None, :] * stride_x_d, 
        mask = offset_t[:, None] < token_num, 
        other = 0.0
    ).to(tl.float32)
    z = tl.load(
        z_ptr + offset_t[:, None] * stride_z_t + pid_h * stride_z_h + offset_d[None, :] * stride_z_d,
        mask = offset_t[:, None] < token_num, 
        other = 0.0
    ).to(tl.float32)

    inv_rms = tl.rsqrt(tl.sum(x * x, axis = -1) / d_model + 1e-6)
    normalized = x * inv_rms[:, None]

    # gate = z / (1 + tl.exp(-z))

    exp_neg_abs = tl.exp(-tl.abs(z))
    sigmoid_z = tl.where(
        z >= 0, 
        1.0 / (1.0 + exp_neg_abs), 
        exp_neg_abs / (1.0 + exp_neg_abs)
    )
    gate = z * sigmoid_z

    w = tl.load(weight_ptr + offset_d)
    
    out =  normalized * w[None, :] * gate

    tl.store(
        out_ptr + offset_t[:, None] * stride_o_t + pid_h * stride_o_h + offset_d[None, :] * stride_o_d, 
        out,
        mask = offset_t[:, None] < token_num
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
    "wy_lib::gdn_gated_rmsnorm",
    mutates_args=(),
)
def gdn_gated_rmsnorm(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    assert x.ndim == 3 and z.ndim == 3
    assert x.shape == z.shape
    assert x.dtype == torch.bfloat16 and z.dtype == torch.bfloat16
    assert weight.ndim == 1 and weight.dtype == torch.float32
    assert x.device == z.device == weight.device

    token_num, head_num, d_model = x.shape
    assert token_num > 0
    assert weight.shape == (d_model,)
    assert triton.next_power_of_2(head_num) == head_num
    assert triton.next_power_of_2(d_model) == d_model

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)

    def grid(meta):
        return (
            head_num,
            triton.cdiv(token_num, meta["BLOCK_T"]),
        )

    torch.library.wrap_triton(_gdn_gated_rmsnorm_triton)[grid](
        x_ptr=x,
        stride_x_t=x.stride(0),
        stride_x_h=x.stride(1),
        stride_x_d=x.stride(2),
        z_ptr=z,
        stride_z_t=z.stride(0),
        stride_z_h=z.stride(1),
        stride_z_d=z.stride(2),
        weight_ptr=weight,
        out_ptr=out,
        stride_o_t=out.stride(0),
        stride_o_h=out.stride(1),
        stride_o_d=out.stride(2),
        d_model=d_model,
        head_num=head_num,
        token_num=token_num,
        T_BUCKET=_token_bucket(token_num),
    )
    return out


@torch.library.register_fake("wy_lib::gdn_gated_rmsnorm")
def _gdn_gated_rmsnorm_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


def call_gdn_gated_rmsnorm_triton(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return gdn_gated_rmsnorm(x, z, weight)


def _torch_reference(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    x_fp32 = x.float()
    z_fp32 = z.float()
    inv_rms = torch.rsqrt(
        torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True) + 1e-6
    )
    out = x_fp32 * inv_rms * weight.float() * torch.nn.functional.silu(z_fp32)
    return out.to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)

    test_cases = [
        (1, 16, 128),
        (3, 16, 128),
        (17, 16, 128),
        (65, 16, 128),
        (7, 8, 64),
    ]

    for token_num, head_num, d_model in test_cases:
        # Use strided views to verify all advertised input strides.
        x_storage = torch.randn(
            (token_num, head_num, d_model * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )
        z_storage = torch.randn_like(x_storage)
        x = x_storage[..., ::2]
        z = z_storage[..., ::2]
        weight = torch.randn(
            (d_model,),
            dtype=torch.float32,
            device="cuda",
        )

        # Exercise both tails of the stable SiLU implementation.
        z_storage[0, 0, 0] = 100.0
        if d_model > 1:
            z_storage[0, 0, 2] = -100.0

        actual = call_gdn_gated_rmsnorm_triton(x, z, weight)
        expected = _torch_reference(x, z, weight)
        max_abs_error = (actual.float() - expected.float()).abs().max().item()

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        assert actual.is_contiguous()
        assert torch.isfinite(actual).all()
        print(
            f"shape={(token_num, head_num, d_model)}, "
            f"max_abs_error={max_abs_error:.8f}, "
            f"best_config={_gdn_gated_rmsnorm_triton.best_config}"
        )

    print("All GDN gated RMSNorm tests passed.")
