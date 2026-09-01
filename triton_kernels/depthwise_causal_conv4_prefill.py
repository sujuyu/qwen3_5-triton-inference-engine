import torch

import triton
import triton.language as tl


autotune_configs = [
    triton.Config({"BLOCK_T": 1, "BLOCK_D": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 2, "BLOCK_D": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 4, "BLOCK_D": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 4, "BLOCK_D": 128}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_T": 8, "BLOCK_D": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 8, "BLOCK_D": 128}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_T": 16, "BLOCK_D": 64}, num_warps=8, num_stages=1),
]


@triton.autotune(
    configs=autotune_configs,
    key=["D", "T_BUCKET"],
)
@triton.jit 
def _depthwise_causal_conv4_prefill_kernel(
    x_ptr, # [T, 6144]  BF16
    stride_x_t, stride_x_d, 
    weight_ptr, # [6144, 4] BF16, 
    stride_weight_d: tl.constexpr, stride_weight_k: tl.constexpr,
    out_ptr, # [T, 6144] BF16, 
    stride_out_t, stride_out_d,
    T: int, 
    D: int, 
    T_BUCKET: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offset_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offset_k = tl.arange(0, 4)

    input_t = offset_t[None, :, None] + offset_k[:, None, None] - 3 
    x_offsets = input_t * stride_x_t + offset_d[None, None, :] * stride_x_d 
    x_mask = (
        (offset_t[None, :, None] < T) &
        (input_t >= 0) & 
        (input_t < T) & 
        (offset_d[None, None, :] < D)
    )
    x = tl.load(
        x_ptr + x_offsets,
        mask=x_mask,
        other=0.0,
    ).to(tl.float32)

    # 载入weight weight在第二维度上进行广播
    w = tl.load(
        weight_ptr + offset_d[None, None, :] * stride_weight_d + offset_k[:, None, None] * stride_weight_k, 
        mask=offset_d[None, None, :] < D,
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(x * w, axis = 0)
    out = acc * tl.sigmoid(acc)

    tl.store(
        out_ptr + offset_t[:, None] * stride_out_t + offset_d[None, :] * stride_out_d,
        out.to(out_ptr.dtype.element_ty),
        mask=(offset_t[:, None] < T) & (offset_d[None, :] < D),
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
    "wy_lib::depthwise_causal_conv4_prefill",
    mutates_args=(),
)
def depthwise_causal_conv4_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    assert x.ndim == 2
    assert weight.ndim in (2, 3)
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert x.device == weight.device

    token_num, hidden_dim = x.shape
    assert token_num > 0 and hidden_dim > 0
    if weight.ndim == 2:
        assert weight.shape == (hidden_dim, 4)
        stride_weight_d = weight.stride(0)
        stride_weight_k = weight.stride(1)
    else:
        assert weight.shape == (hidden_dim, 1, 4)
        stride_weight_d = weight.stride(0)
        stride_weight_k = weight.stride(2)

    out = torch.empty_like(x)

    def grid(meta):
        return (
            triton.cdiv(token_num, meta["BLOCK_T"]),
            triton.cdiv(hidden_dim, meta["BLOCK_D"]),
        )

    torch.library.wrap_triton(_depthwise_causal_conv4_prefill_kernel)[grid](
        x_ptr=x,
        stride_x_t=x.stride(0),
        stride_x_d=x.stride(1),
        weight_ptr=weight,
        stride_weight_d=stride_weight_d,
        stride_weight_k=stride_weight_k,
        out_ptr=out,
        stride_out_t=out.stride(0),
        stride_out_d=out.stride(1),
        T=token_num,
        D=hidden_dim,
        T_BUCKET=_token_bucket(token_num),
    )
    return out


@torch.library.register_fake("wy_lib::depthwise_causal_conv4_prefill")
def _depthwise_causal_conv4_prefill_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)


def call_depthwise_causal_conv4_prefill_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return depthwise_causal_conv4_prefill(x, weight)


def _torch_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    weight_3d = weight if weight.ndim == 3 else weight[:, None, :]
    token_num, hidden_dim = x.shape
    conv = torch.nn.functional.conv1d(
        x.transpose(0, 1).unsqueeze(0).float(),
        weight_3d.float(),
        padding=3,
        groups=hidden_dim,
    )[:, :, :token_num]
    return torch.nn.functional.silu(conv).squeeze(0).transpose(0, 1).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)

    test_cases = [
        (1, 6144, 3),
        (2, 6144, 3),
        (3, 6144, 3),
        (4, 6144, 3),
        (17, 6144, 3),
        (65, 6144, 3),
        (7, 1000, 2),
    ]

    for token_num, hidden_dim, weight_ndim in test_cases:
        x = torch.randn(
            (token_num, hidden_dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        weight_shape = (hidden_dim, 1, 4) if weight_ndim == 3 else (hidden_dim, 4)
        weight = torch.randn(weight_shape, dtype=torch.bfloat16, device="cuda")

        actual = call_depthwise_causal_conv4_prefill_triton(x, weight)
        expected = _torch_reference(x, weight)
        max_abs_error = (actual.float() - expected.float()).abs().max().item()

        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
        print(
            f"shape={tuple(x.shape)}, weight_shape={tuple(weight.shape)}, "
            f"max_abs_error={max_abs_error:.8f}, "
            f"best_config={_depthwise_causal_conv4_prefill_kernel.best_config}"
        )

    print("All depthwise causal Conv4 prefill tests passed.")
