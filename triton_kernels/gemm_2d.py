import torch

import triton
import triton.language as tl


#   x:      [M, K] BF16
#   weight: [N, K] BF16
#   y:      [M, N] BF16
#   y = x @ weight.T
# 需要assert K是128的整数倍


# ============================ decode（M=1）的 config ============================
#
# decode 时 M 恒为 1，一步 forward 里 192 次调用无一例外。原来这些调用和 prefill
# 共用一份 config 列表，而那份列表里 **BLOCK_K 最大只有 64，且只和 BLOCK_N>=128
# 配对**——恰恰没有 M=1 需要的「窄 BLOCK_N + 大 BLOCK_K」。autotune 再怎么调，
# 也只能在给定的候选里选。
#
# 为什么 BLOCK_K 是主要矛盾
# ------------------------
# M=1 时算术强度约等于 1 FLOP/byte，tensor core 完全用不上，瓶颈是访存延迟能不能
# 被盖住。而 CTA 数少的时候盖不住，于是 K 循环的迭代次数直接乘在总时间上：
# K=1024 时 BLOCK_K=32 要循环 32 次，BLOCK_K=128 只要 8 次。
#
# 最直接的证据是 [16,1024]：它只有 1 个 CTA，改 BLOCK_N 影响不了 CTA 数，
# 但光把 BLOCK_K 从 32 提到 128 就快了 2.44 倍（10.63 -> 4.36us）。
#
# 各形状实测（A100，profiler 计时，权重轮换到超过 3 倍 L2 避免测成 L2 带宽）：
#
#     形状          次数/步   改前us   改后us   最优 BM/BN/BK/stages
#     [6144,1024]     18      15.76    12.02   2/32/128/3
#     [3584,1024]     48      13.65     8.85   4/64/128/4
#     [2048,1024]     30      12.64     6.66   2/32/128/4
#     [1024,3584]     24      37.60    12.64   4/16/128/4
#     [1024,2048]     24      22.09     8.34   2/16/128/4
#     [ 512,1024]     12      11.78     5.27   2/16/128/4
#     [  16,1024]     36      10.63     4.36   4/16/128/4
#
# BLOCK_M 为什么最小是 2 而不是 1
# ------------------------------
# Triton 3.7.1 的 tl.dot 支持 M<16，编译和数值都没问题。但 BLOCK_M=1 时 x 的载入
# 会退回同步 ld.global（cp.async 从 22 条掉到 14 条），软流水断掉，反而比 BLOCK_M=2
# 慢 2.4 倍。BLOCK_M 取 2/4 相比原来的 16 只值 10~15%，远不如 BLOCK_K 重要。
#
# BLOCK_N 的目标是把 CTA 数顶到 64~112（A100 有 108 个 SM），所以 N 窄的形状
# （1024/512/16）要 16，N 宽的（6144/3584/2048）要 32~64，再切碎就亏了。
#
# **BLOCK_K 不能超过 128**：kernel 的 K 循环没有对 offset_k 做 mask，靠 wrapper 里
# `K % 128 == 0` 的断言保证不越界。BLOCK_K=256 对当前这几个 K 恰好也能整除，但只要
# 出现 K=1152 这种就会读越界。
decode_autotune_configs = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 128}, num_warps=4, num_stages=st)
    for bm, bn, st in [
        (2, 16, 4), (4, 16, 4),
        (2, 32, 3), (2, 32, 4), (4, 32, 4),
        (2, 64, 4), (4, 64, 4),
    ]
]


# ============================ prefill（M 较大）的 config ========================
# 这一组维持原样：M 是 prompt 长度，几百到几千，BLOCK_M 开大才能摊薄权重读取。
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


def _prune_by_is_decode(configs, named_args, **kwargs):
    """按 IS_DECODE 把候选劈成两组，各自只试自己那半。

    `triton.autotune` 只接受一份 config 列表，所有 key 共用。直接把两组合并的话，
    decode 的每个 (N,K) 都要白跑一遍 prefill 的大 tile，prefill 反之亦然，
    首次启动的 autotune 时间几乎翻倍——这个项目之前就被 autotune 的启动开销
    坑过（见 triton_kernels/__init__.py）。用 early_config_prune 提前筛掉。

    IS_DECODE 是 constexpr，Triton 可能把它放在 kwargs 也可能放在 named_args，
    两边都查一下；都取不到就退回全集，宁可慢也不要选错。
    """
    is_decode = kwargs.get("IS_DECODE", named_args.get("IS_DECODE"))
    if is_decode is None:
        return decode_autotune_configs + autotune_configs
    return decode_autotune_configs if is_decode else autotune_configs


@triton.autotune(
    configs=decode_autotune_configs + autotune_configs,
    # M 会随序列长度改变，不把它直接放入 key。M=1 的 decode
    # 单独调优，其余 prefill 长度共享同一组结果。
    key=["N", "K", "IS_DECODE"],
    prune_configs_by={"early_config_prune": _prune_by_is_decode},
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