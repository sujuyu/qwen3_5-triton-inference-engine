import torch

import triton
import triton.language as tl


#   x:      [M, K] BF16
#   weight: [N, K] BF16
#   y:      [M, N] BF16
#   y = x @ weight.T
# 需要assert K是128的整数倍


autotune_configs = [
    # M 很小时（特别是 decode）避免在 M 方向浪费太多线程。
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=2,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4,
        num_stages=3,
    ),
    # prefill 的中等 tile。
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64},
        num_warps=8,
        num_stages=3,
    ),
    # 长 prefill 与大 N 投影的大 tile。
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64},
        num_warps=8,
        num_stages=3,
    ),
]


@triton.autotune(
    configs=autotune_configs,
    # M 会随序列长度改变，不把它直接放入 key。M=1 的 decode
    # 单独调优，其余 prefill 长度共享同一组结果。
    key=["N", "K", "IS_DECODE"],
)
@triton.jit
def _gemm_2d_triton(
    x_ptr, weight_ptr, y_ptr,
    stride_x_m, stride_x_k,
    stride_weight_n, stride_weight_k,
    stride_y_m, stride_y_n,
    M, N,
    K: tl.constexpr,
    IS_DECODE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for k in range(0, K, BLOCK_K):
        offset_k = k + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + offset_m[:, None] * stride_x_m + offset_k[None, :] * stride_x_k, 
                mask = offset_m[:, None] < M, other = 0.0)
        w = tl.load(weight_ptr + offset_k[:, None] * stride_weight_k + offset_n[None, :] * stride_weight_n,
                mask = offset_n[None, :] < N, other = 0.0)
        acc += tl.dot(x, w)

    tl.store(
        y_ptr + offset_m[:, None] * stride_y_m + offset_n[None, :] * stride_y_n,
        acc.to(y_ptr.dtype.element_ty),
        mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
    )


@torch.library.triton_op(
    "wy_lib::gemm_2d",
    mutates_args=(),
)
def gemm_2d(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert x.shape[1] == weight.shape[1]
    assert x.shape[1] % 128 == 0

    M, K = x.shape
    N = weight.shape[0]

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    torch.library.wrap_triton(_gemm_2d_triton)[grid](
        x_ptr=x,
        weight_ptr=weight,
        y_ptr=out,
        stride_x_m=x.stride(0),
        stride_x_k=x.stride(1),
        stride_weight_n=weight.stride(0),
        stride_weight_k=weight.stride(1),
        stride_y_m=out.stride(0),
        stride_y_n=out.stride(1),
        M=M,
        N=N,
        K=K,
        IS_DECODE=M == 1,
    )

    return out


@torch.library.register_fake("wy_lib::gemm_2d")
def _gemm_2d_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], weight.shape[0]),
        dtype=x.dtype,
        device=x.device,
    )


def call_gemm_2d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return gemm_2d(x, weight)


if __name__ == "__main__":
    x = torch.randn((1, 128), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    out = call_gemm_2d_triton(x, weight)
    print(out.shape)
    # print(out)
    out_linear = torch.nn.functional.linear(x, weight)
    print(torch.max(torch.abs(out - out_linear)))