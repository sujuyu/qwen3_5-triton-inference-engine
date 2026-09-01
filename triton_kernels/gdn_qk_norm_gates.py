import torch

import triton
import triton.language as tl


autotune_configs = [
    triton.Config({"BLOCK_T": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 1}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK_T": 2}, num_warps=8, num_stages=1),
]


@triton.autotune(
    configs=autotune_configs,
    key=["H", "D", "T_BUCKET"],
)
@triton.jit
def _gdn_qk_norm_gates_kernel(
    q_ptr, # [T, 16, 128] BF16
    k_ptr, # [T, 16, 128] BF16
    a, # [T, 16] BF16 
    b, # [T, 16] BF16
    A_log, # [16] FP32
    dt_bias, # [16] BF16 

    q_norm_ptr, # [T,16,128] BF16
    k_norm_ptr, # [T,16,128] BF16
    beta_ptr, # [T,16] FP32
    g_ptr, #  [T,16] FP32

    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_a_t, stride_a_h,
    stride_b_t, stride_b_h,
    stride_q_norm_t, stride_q_norm_h, stride_q_norm_d,
    stride_k_norm_t, stride_k_norm_h, stride_k_norm_d,
    stride_beta_t, stride_beta_h,
    stride_g_t, stride_g_h,

    T: int, # token_num
    H: tl.constexpr, # head_num = 16
    D: tl.constexpr, # head_dim = 128
    Q_SCALE: tl.constexpr,
    T_BUCKET: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # 在token维度上进行切分block
    # head维度上直接一次性16全部算完
    pid_t = tl.program_id(0)

    offset_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offset_t < T
    offset_h = tl.arange(0, H)
    offset_d = tl.arange(0, D)

    q = tl.load(
        q_ptr + offset_t[:, None, None] * stride_q_t + offset_h[None, :, None] * stride_q_h + offset_d[None, None, :] * stride_q_d,
        mask=mask_t[:, None, None], 
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_ptr + offset_t[:, None, None] * stride_k_t + offset_h[None, :, None] * stride_k_h + offset_d[None, None, :] * stride_k_d,
        mask=mask_t[:, None, None],
        other=0.0,
    ).to(tl.float32)

    q_norm = q * tl.expand_dims(tl.rsqrt(tl.sum((q * q), axis=-1) + 1e-6), axis=-1) * Q_SCALE
    k_norm = k * tl.expand_dims(tl.rsqrt(tl.sum((k * k), axis=-1) + 1e-6), axis=-1)

    b = tl.load(
        b + offset_t[:, None] * stride_b_t + offset_h * stride_b_h, 
        mask = offset_t[:, None] < T, 
        other = 0.0
    ).to(tl.float32)
    beta = tl.sigmoid(b)

    a_log = tl.load(A_log + offset_h).to(tl.float32)
    dt_bias = tl.load(dt_bias + offset_h).to(tl.float32)
    a = tl.load(
        a + offset_t[:, None] * stride_a_t + offset_h[None, :] * stride_a_h, 
        mask = offset_t[:, None] < T, 
        other = 0.0
    ).to(tl.float32)
    softplus_input = a + dt_bias[None, :]
    softplus = tl.maximum(softplus_input, 0.0) + tl.log(
        1.0 + tl.exp(-tl.abs(softplus_input))
    )
    g = -tl.exp(a_log)[None, :] * softplus

    tl.store(
        q_norm_ptr + offset_t[:, None, None] * stride_q_norm_t +\
              offset_h[None, :, None] * stride_q_norm_h +\
                  offset_d[None, None, :] * stride_q_norm_d, 
        q_norm, 
        mask=mask_t[:, None, None]
    )
    tl.store(
        k_norm_ptr + offset_t[:, None, None] * stride_k_norm_t +\
              offset_h[None, :, None] * stride_k_norm_h +\
                  offset_d[None, None, :] * stride_k_norm_d,
        k_norm, 
        mask = mask_t[:, None, None],
    )
    tl.store(
        beta_ptr + offset_t[:, None] * stride_beta_t + offset_h * stride_beta_h, 
        beta, 
        mask = offset_t[:, None] < T
    )
    tl.store(
        g_ptr + offset_t[:, None] * stride_g_t + offset_h[None, :] * stride_g_h,
        g, 
        mask = offset_t[:, None] < T
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
    "wy_lib::gdn_qk_norm_gates",
    mutates_args=(),
)
def gdn_qk_norm_gates(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert q.ndim == 3 and k.ndim == 3
    assert q.shape == k.shape
    assert a.ndim == 2 and b.ndim == 2 and a.shape == b.shape
    assert a_log.ndim == 1 and dt_bias.ndim == 1
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    assert a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
    assert a_log.dtype == torch.float32 and dt_bias.dtype == torch.bfloat16
    assert q.device == k.device == a.device == b.device == a_log.device == dt_bias.device

    token_num, num_heads, head_dim = q.shape
    assert token_num > 0
    assert a.shape == (token_num, num_heads)
    assert a_log.shape == (num_heads,) and dt_bias.shape == (num_heads,)
    assert triton.next_power_of_2(num_heads) == num_heads
    assert triton.next_power_of_2(head_dim) == head_dim

    q_norm = torch.empty_like(q)
    k_norm = torch.empty_like(k)
    beta = torch.empty((token_num, num_heads), dtype=torch.float32, device=q.device)
    g = torch.empty_like(beta)

    def grid(meta):
        return (triton.cdiv(token_num, meta["BLOCK_T"]),)

    torch.library.wrap_triton(_gdn_qk_norm_gates_kernel)[grid](
        q_ptr=q,
        k_ptr=k,
        a=a,
        b=b,
        A_log=a_log,
        dt_bias=dt_bias,
        q_norm_ptr=q_norm,
        k_norm_ptr=k_norm,
        beta_ptr=beta,
        g_ptr=g,
        stride_q_t=q.stride(0),
        stride_q_h=q.stride(1),
        stride_q_d=q.stride(2),
        stride_k_t=k.stride(0),
        stride_k_h=k.stride(1),
        stride_k_d=k.stride(2),
        stride_a_t=a.stride(0),
        stride_a_h=a.stride(1),
        stride_b_t=b.stride(0),
        stride_b_h=b.stride(1),
        stride_q_norm_t=q_norm.stride(0),
        stride_q_norm_h=q_norm.stride(1),
        stride_q_norm_d=q_norm.stride(2),
        stride_k_norm_t=k_norm.stride(0),
        stride_k_norm_h=k_norm.stride(1),
        stride_k_norm_d=k_norm.stride(2),
        stride_beta_t=beta.stride(0),
        stride_beta_h=beta.stride(1),
        stride_g_t=g.stride(0),
        stride_g_h=g.stride(1),
        T=token_num,
        H=num_heads,
        D=head_dim,
        Q_SCALE=head_dim**-0.5,
        T_BUCKET=_token_bucket(token_num),
    )
    return q_norm, k_norm, beta, g


@torch.library.register_fake("wy_lib::gdn_qk_norm_gates")
def _gdn_qk_norm_gates_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_num, num_heads, _ = q.shape
    q_norm = torch.empty_like(q)
    k_norm = torch.empty_like(k)
    beta = torch.empty((token_num, num_heads), dtype=torch.float32, device=q.device)
    g = torch.empty_like(beta)
    return q_norm, k_norm, beta, g


def call_gdn_qk_norm_gates_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return gdn_qk_norm_gates(q, k, a, b, a_log, dt_bias)


def _torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_fp32 = q.float()
    k_fp32 = k.float()
    q_norm = q_fp32 * torch.rsqrt(torch.sum(q_fp32 * q_fp32, dim=-1, keepdim=True) + 1e-6)
    q_norm = (q_norm * (q.shape[-1] ** -0.5)).to(q.dtype)
    k_norm = (
        k_fp32 * torch.rsqrt(torch.sum(k_fp32 * k_fp32, dim=-1, keepdim=True) + 1e-6)
    ).to(k.dtype)
    beta = torch.sigmoid(b.float())
    g = -torch.exp(a_log.float()).unsqueeze(0) * torch.nn.functional.softplus(
        a.float() + dt_bias.float().unsqueeze(0)
    )
    return q_norm, k_norm, beta, g


if __name__ == "__main__":
    torch.manual_seed(0)

    test_cases = [
        (1, 16, 128),
        (3, 16, 128),
        (17, 16, 128),
        (65, 16, 128),
        (7, 8, 64),
    ]

    for token_num, num_heads, head_dim in test_cases:
        # Q/K are strided views matching slices of the Conv4 QKV output.
        mixed_qkv = torch.randn(
            (token_num, num_heads * head_dim * 3),
            dtype=torch.bfloat16,
            device="cuda",
        )
        q = mixed_qkv[:, : num_heads * head_dim].view(token_num, num_heads, head_dim)
        k = mixed_qkv[:, num_heads * head_dim : 2 * num_heads * head_dim].view(
            token_num, num_heads, head_dim
        )
        a = torch.randn((token_num, num_heads), dtype=torch.bfloat16, device="cuda")
        b = torch.randn_like(a)
        a_log = torch.randn((num_heads,), dtype=torch.float32, device="cuda")
        dt_bias = torch.randn((num_heads,), dtype=torch.bfloat16, device="cuda")
        a[0, 0] = 100.0
        if num_heads > 1:
            a[0, 1] = -100.0

        actual = call_gdn_qk_norm_gates_triton(q, k, a, b, a_log, dt_bias)
        expected = _torch_reference(q, k, a, b, a_log, dt_bias)
        errors = [
            (actual_tensor.float() - expected_tensor.float()).abs().max().item()
            for actual_tensor, expected_tensor in zip(actual, expected)
        ]

        torch.testing.assert_close(actual[0], expected[0], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual[1], expected[1], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual[2], expected[2], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual[3], expected[3], rtol=2e-5, atol=2e-6)
        assert all(torch.isfinite(tensor).all() for tensor in actual)
        print(
            f"shape={(token_num, num_heads, head_dim)}, "
            f"max_abs_errors={errors}, "
            f"best_config={_gdn_qk_norm_gates_kernel.best_config}"
        )

    print("All GDN Q/K norm and gates tests passed.")
