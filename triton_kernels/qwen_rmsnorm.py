import torch

import triton
import triton.language as tl


autotune_configs = [
    # 单行和小行数: 尽量减少 mask 掉的无效计算.
    triton.Config({"BLOCK_M": 1}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_M": 1}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 2}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 2}, num_warps=4, num_stages=1),
    # prefill: 每个 program 同时处理多行.
    triton.Config({"BLOCK_M": 4}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 8}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_M": 16}, num_warps=8, num_stages=1),
]


@triton.autotune(
    configs=autotune_configs,
    # 不使用精确 num_rows, 避免序列长度每变化一次就重新 autotune.
    key=["d_model", "ROW_BUCKET"],
)
@triton.jit
def _qwen_rmsnorm_kernel(
    x_ptr,  # *BF16
    weight_ptr,  # *BF16
    o_ptr,  # *BF16
    num_rows,
    d_model: tl.constexpr,
    x_stride_b: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_n: tl.constexpr,
    o_stride_b: tl.constexpr,
    o_stride_m: tl.constexpr,
    o_stride_n: tl.constexpr,
    eps,
    ROW_BUCKET: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # grid = (cdiv(num_rows, BLOCK_M), B).
    #
    # 加这一维是为了接住 `[B, H, D]` 这种**两级均匀但一维表达不了**的布局:
    # 融合 GEMV 的输出切片 `fused[:, :2048].view(B, 8, 256)` stride 是
    # (5120, 256, 1), 行地址 = b*5120 + h*256. 同 batch 内隔 256, 跨 batch 隔 5120,
    # 用单个 x_stride_m 写不出来, 但拆成两维就是两次乘加.
    #
    # B=1 时 x_stride_b 传什么都无所谓(pid_b 恒为 0), 所以旧调用方一行不用改.
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)
    x_ptr = x_ptr + pid_b * x_stride_b
    o_ptr = o_ptr + pid_b * o_stride_b

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
    assert weight.is_contiguous()
    # 最后一维必须连续(kernel 里 offset_n 是按 x_stride_n 走的, 非 1 会退化成
    # 逐元素寻址). 其余维只要是均匀 stride 就行, 不要求整体 contiguous --
    # 这样融合 GEMV 切出来的 [B, H, D] 可以直接传进来, 省掉一次拷贝.
    assert x.stride(-1) == 1

    d_model = x.shape[-1]
    assert d_model in (256, 1024)
    assert weight.shape == (d_model,)

    # 三维时把第 0 维当 batch 交给 grid 的第二维; 否则退化成 B=1 的老行为.
    if x.ndim == 3:
        batch, num_rows = x.shape[0], x.shape[1]
        x_stride_b, x_stride_m = x.stride(0), x.stride(1)
    else:
        assert x.is_contiguous(), "二维输入按扁平行处理, 必须 contiguous"
        batch, num_rows = 1, x.numel() // d_model
        x_stride_b, x_stride_m = 0, d_model

    # 输出总是连续的: 下游(partial_rope, gdn_qk_norm_gates)不介意,
    # 而连续输出省掉一次潜在的拷贝.
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    o_stride_b = out.stride(0) if x.ndim == 3 else 0
    o_stride_m = out.stride(1) if x.ndim == 3 else d_model

    def grid(meta):
        return (triton.cdiv(num_rows, meta["BLOCK_M"]), batch)

    torch.library.wrap_triton(_qwen_rmsnorm_kernel)[grid](
        x_ptr=x,
        weight_ptr=weight,
        o_ptr=out,
        num_rows=num_rows,
        d_model=d_model,
        x_stride_b=x_stride_b,
        x_stride_m=x_stride_m,
        x_stride_n=1,
        o_stride_b=o_stride_b,
        o_stride_m=o_stride_m,
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
        (2, 129, 1024),  # decoder RMSNorm prefill, 包含非 tile 对齐行数
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
