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
    seq_start_ptr, # [T] INT32, 每个 token 所属序列的起始下标; VARLEN=False 时不读
    T: int,
    D: int,
    T_BUCKET: tl.constexpr,
    VARLEN: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offset_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offset_k = tl.arange(0, 4)

    input_t = offset_t[None, :, None] + offset_k[:, None, None] - 3

    # 变长打包(packing)下唯一要改的地方: 回看的下界.
    #
    # conv 的窗口是固定的 4 个 tap, `input_t` 落在 [offset_t-3, offset_t] 里 --
    # **只往回看, 不往前看**. 所以上界不用管(input_t <= offset_t, 而 offset_t
    # 本来就在自己序列内), 只有下界要从「全局 0」换成「本序列的起点」,
    # 否则一条序列开头的前 3 个位置会读到上一条序列的尾巴.
    #
    # 这也是它比 attention 好改的原因: kernel 需要多少「序列感知」, 取决于它
    # 往回够多远. conv 的作用域是固定的 4, 所以一个下界钳位就够, grid 一点不用动,
    # 也就没有空转 CTA; attention 的作用域是整条序列, 才需要 grid 加 batch 维、
    # 早退、per-序列的循环边界那一整套.
    if VARLEN:
        lower = tl.load(
            seq_start_ptr + offset_t, mask=offset_t < T, other=0
        )[None, :, None]
    else:
        lower = 0

    x_offsets = input_t * stride_x_t + offset_d[None, None, :] * stride_x_d
    x_mask = (
        (offset_t[None, :, None] < T) &
        (input_t >= lower) &
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
    seq_start: torch.Tensor | None = None,
) -> torch.Tensor:
    """depthwise causal Conv4 + SiLU.

    `seq_start` 传 `[T]` INT32 时进入变长打包模式: 第 t 个 token 回看时不会越过
    `seq_start[t]`. 传 None 时是原来的单序列行为(下界为 0), 两者数值完全一致
    -- 单序列下 seq_start 恒为 0.

    host 侧构造:

        seq_start = torch.repeat_interleave(cu_seqlens[:-1], lengths)
        # lengths=[3,6,2] -> [0,0,0, 3,3,3,3,3,3, 9,9]
    """
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

    varlen = seq_start is not None
    if varlen:
        assert seq_start.dtype == torch.int32 and seq_start.shape == (token_num,), (
            f"seq_start 应为 [T]={(token_num,)} 的 INT32, 实际 "
            f"{tuple(seq_start.shape)} {seq_start.dtype}"
        )
        assert seq_start.device == x.device
    else:
        # kernel 里 VARLEN=False 那条分支不会读它, 但 Triton 仍要一个合法指针
        seq_start = x

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
        seq_start_ptr=seq_start,
        T=token_num,
        D=hidden_dim,
        T_BUCKET=_token_bucket(token_num),
        VARLEN=varlen,
    )
    return out


@torch.library.register_fake("wy_lib::depthwise_causal_conv4_prefill")
def _depthwise_causal_conv4_prefill_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    seq_start: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(x)


def call_depthwise_causal_conv4_prefill_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    seq_start: torch.Tensor | None = None,
) -> torch.Tensor:
    return depthwise_causal_conv4_prefill(x, weight, seq_start)


def build_seq_start(
    lengths: list[int], *, device: torch.device | str = "cuda"
) -> torch.Tensor:
    """长度列表 -> `[total_tokens]` INT32, 每个 token 所属序列的起始下标."""
    starts, acc = [], 0
    for n in lengths:
        starts.append(acc)
        acc += n
    return torch.repeat_interleave(
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(lengths, device=device),
    )


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

    # ---- 变长打包 --------------------------------------------------------
    # 判据: 打包起来跑一次, 逐条切出来必须和「逐条单独跑」逐字节相同.
    # 这直接检验边界钳位对不对 -- 如果下界还是全局 0, 每条序列开头的前 3 个位置
    # 会读到上一条序列的尾巴, 结果就对不上.
    print("\n=== 变长打包 ===")
    hidden_dim = 6144
    for lengths in ([5], [3, 6, 2], [1, 1, 1, 1], [4, 4], [1, 17, 3, 64], [2, 1, 3]):
        total = sum(lengths)
        x = torch.randn((total, hidden_dim), dtype=torch.bfloat16, device="cuda")
        weight = torch.randn((hidden_dim, 4), dtype=torch.bfloat16, device="cuda")

        seq_start = build_seq_start(lengths)
        packed = call_depthwise_causal_conv4_prefill_triton(x, weight, seq_start)

        parts, offset = [], 0
        for n in lengths:
            parts.append(
                call_depthwise_causal_conv4_prefill_triton(x[offset : offset + n], weight)
            )
            offset += n
        one_by_one = torch.cat(parts, dim=0)

        same = torch.equal(packed, one_by_one)
        print(f"  lengths={str(lengths):<20} total={total:<5} 逐字节相同={same}")
        assert same, "打包结果与逐条单独跑不一致, 边界钳位有问题"

    # 不传 seq_start 时必须和原来完全一样
    x = torch.randn((17, hidden_dim), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((hidden_dim, 4), dtype=torch.bfloat16, device="cuda")
    assert torch.equal(
        call_depthwise_causal_conv4_prefill_triton(x, weight),
        call_depthwise_causal_conv4_prefill_triton(x, weight, build_seq_start([17])),
    ), "单序列下传不传 seq_start 应当完全一致"
    print("  单序列: 传与不传 seq_start 结果相同")

    print("\nAll depthwise causal Conv4 prefill tests passed.")
