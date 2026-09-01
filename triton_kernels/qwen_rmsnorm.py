import torch

import triton
import triton.language as tl


autotune_configs = [
    # 单行和小行数：尽量减少 mask 掉的无效计算。
    triton.Config({"BLOCK_M": 1}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_M": 1}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 2}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=1),
    # prefill：每个 program 同时处理多行。
    triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=1),
]


@triton.autotune(
    configs=autotune_configs,
    # 不使用精确 num_rows，避免序列长度每变化一次就重新 autotune。
    key=["d_model", "ROW_BUCKET"],
)
@triton.jit
def _qwen_rmsnorm_kernel(
    x_ptr,  # *BF16
    weight_ptr,  # *BF16
    o_ptr,  # *BF16
    num_rows,
    d_model: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_n: tl.constexpr,
    o_stride_m: tl.constexpr,
    o_stride_n: tl.constexpr,
    eps,
    ROW_BUCKET: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)

    offset_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = tl.arange(0, d_model)
    row_mask = offset_m[:, None] < num_rows

    x = tl.load(
        x_ptr
        + offset_m[:, None] * x_stride_m
        + offset_n[None, :] * x_stride_n,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)

    mean_square = tl.sum(x * x, axis=-1, keep_dims=True) / d_model
    rsigma = tl.rsqrt(mean_square + eps)

    weight = tl.load(weight_ptr + offset_n[None, :]).to(tl.float32)
    y = x * rsigma * (1.0 + weight)

    tl.store(
        o_ptr
        + offset_m[:, None] * o_stride_m
        + offset_n[None, :] * o_stride_n,
        y.to(o_ptr.dtype.element_ty),
        mask=row_mask,
    )


def _row_bucket(num_rows: int) -> int:
    if num_rows == 1:
        return 1
    if num_rows <= 16:
        return 16
    return 17


@torch.library.triton_op(
    "wy_lib::qwen_rmsnorm",
    mutates_args=(),
)
def qwen_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous()

    d_model = x.shape[-1]
    assert d_model in (256, 1024)
    assert weight.shape == (d_model,)

    num_rows = x.numel() // d_model
    out = torch.empty_like(x)

    def grid(meta):
        return (triton.cdiv(num_rows, meta["BLOCK_M"]),)

    torch.library.wrap_triton(_qwen_rmsnorm_kernel)[grid](
        x_ptr=x,
        weight_ptr=weight,
        o_ptr=out,
        num_rows=num_rows,
        d_model=d_model,
        x_stride_m=d_model,
        x_stride_n=1,
        o_stride_m=d_model,
        o_stride_n=1,
        eps=eps,
        ROW_BUCKET=_row_bucket(num_rows),
    )

    return out


@torch.library.register_fake("wy_lib::qwen_rmsnorm")
def _qwen_rmsnorm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return torch.empty_like(x)


def call_qwen_rmsnorm_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return qwen_rmsnorm(x, weight, eps)


def _torch_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_fp32 = x.float()
    mean_square = torch.mean(x_fp32 * x_fp32, dim=-1, keepdim=True)
    out = x_fp32 * torch.rsqrt(mean_square + eps)
    return (out * (1.0 + weight.float())).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    eps = 1e-6

    test_shapes = [
        (1, 256),  # Q/K norm 的小行数路径
        (2, 8, 17, 256),  # Q projection 按 head 展开后的 prefill
        (1, 1, 1024),  # decoder RMSNorm decode
        (2, 129, 1024),  # decoder RMSNorm prefill，包含非 tile 对齐行数
    ]

    for shape in test_shapes:
        x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(
            (shape[-1],),
            dtype=torch.bfloat16,
            device="cuda",
        )

        actual = call_qwen_rmsnorm_triton(x, weight, eps)
        expected = _torch_reference(x, weight, eps)
        max_abs_error = (actual.float() - expected.float()).abs().max().item()

        torch.testing.assert_close(
            actual,
            expected,
            rtol=2e-2,
            atol=2e-2,
        )
        print(
            f"shape={shape}, max_abs_error={max_abs_error:.8f}, "
            f"best_config={_qwen_rmsnorm_kernel.best_config}"
        )

    print("All Qwen RMSNorm tests passed.")
