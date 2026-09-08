import torch

import triton
import triton.language as tl


@triton.jit
def _lm_head_argmax_stage1_triton(
    hidden_ptr,  # [T, HIDDEN_SIZE] BF16
    stride_hidden_t: tl.constexpr,
    weight_ptr,  # [VOCAB_SIZE, HIDDEN_SIZE] BF16
    stride_w_vocab: tl.constexpr,
    stride_w_hidden: tl.constexpr,
    partial_values_ptr,  # [B, ceil_div(VOCAB_SIZE, GROUP_V)] FP32
    stride_pv_b: tl.constexpr,
    partial_indices_ptr,  # [B, ceil_div(VOCAB_SIZE, GROUP_V)] INT32
    stride_pi_b: tl.constexpr,
    row_stride,  # 每个 program 负责的 hidden 行间隔；见下
    VOCAB_SIZE: tl.constexpr,  # 248320
    HIDDEN_SIZE: tl.constexpr,  # 1024
    GROUP_V: tl.constexpr,  # 每个 program 负责的词表范围，建议 512
    TILE_V: tl.constexpr,  # program 内每次计算的词表子块，建议 8/16/32
    TILE_K: tl.constexpr,  # GEMV reduction 子块，建议 128/256
):
    # 248320 = 485 * 512。每个 program 计算 GROUP_V 个 logits，
    # 输出一个局部最大值及其全局 token index。
    #
    # grid = (B, num_partials)。两种调用方式共用这个 kernel：
    #   prefill  B=1，row_stride = (T-1)*stride_hidden_t，只取最后一个位置的 logits
    #   decode   B 条序列各一个 token，row_stride = stride_hidden_t，第 b 行对应第 b 条
    # 把「读哪一行」抽成 row_stride 而不是写死 (token_num-1)，两种情形就统一了。
    pid_b, pid = tl.program_id(0), tl.program_id(1)
    start_offset_v = pid * GROUP_V
    hidden_ptr = hidden_ptr + pid_b * row_stride
    partial_values_ptr = partial_values_ptr + pid_b * stride_pv_b
    partial_indices_ptr = partial_indices_ptr + pid_b * stride_pi_b

    local_index = 0
    local_max = -float('inf')

    for start_v in tl.range(0, GROUP_V, TILE_V):
        offset_v = start_offset_v + start_v + tl.arange(0, TILE_V)
        valid_v = offset_v < VOCAB_SIZE

        result = tl.zeros([TILE_V], dtype = tl.float32)

        for start_k in tl.range(0, HIDDEN_SIZE, TILE_K):
            offset_k = start_k + tl.arange(0, TILE_K) # HIDDEN_SIZE一定是TILE_K的整数倍 这里无需mask
            x = tl.load(hidden_ptr + offset_k[None, :])  # [1, TILE_K]
            w = tl.load(
                weight_ptr + offset_v[:, None] *  stride_w_vocab + offset_k[None, :] * stride_w_hidden, 
                mask = valid_v[:, None], 
                other = 0.0
            )
            result += tl.sum(x.to(tl.float32) * w.to(tl.float32), axis = -1)
        
        result = tl.where(valid_v, result, -float('inf'))
        
        tile_max = tl.max(result, axis = -1)
        tile_index = tl.argmax(result, axis = -1) + start_offset_v + start_v
        take_tile = tile_max > local_max
        local_max = tl.where(take_tile, tile_max, local_max)
        local_index = tl.where(take_tile, tile_index, local_index)

    tl.store(partial_values_ptr + pid, local_max)
    tl.store(partial_indices_ptr + pid, local_index)


@triton.jit
def _lm_head_argmax_stage2_triton(
    partial_values_ptr,  # [B, num_partials] FP32
    stride_pv_b: tl.constexpr,
    partial_indices_ptr,  # [B, num_partials] INT32
    stride_pi_b: tl.constexpr,
    token_id_ptr,  # [B] INT64
    num_partials,  # 当前模型为 485
    BLOCK_PARTIAL: tl.constexpr,  # next_power_of_2(num_partials)，当前为 512
):
    # 对 stage 1 的局部结果做最终 argmax；相同最大值选择较小 index。grid = (B,)
    pid_b = tl.program_id(0)
    partial_values_ptr = partial_values_ptr + pid_b * stride_pv_b
    partial_indices_ptr = partial_indices_ptr + pid_b * stride_pi_b
    offset = tl.arange(0, BLOCK_PARTIAL)
    value = tl.load(
        partial_values_ptr + offset, 
        mask = offset < num_partials, 
        other = -float('inf')
    )
    local_index = tl.argmax(value, axis = 0)
    token_id = tl.load(
        partial_indices_ptr + local_index
    )
    tl.store(token_id_ptr + pid_b, token_id)


GROUP_V = 512
TILE_V = 16
TILE_K = 128


@torch.library.triton_op(
    "wy_lib::lm_head_argmax",
    mutates_args=(),
)
def lm_head_argmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    last_only: bool = True,
) -> torch.Tensor:
    """融合 LM head 的 GEMV 与 argmax，**248320 维的 logits 从不物化**。

    两种用法：

        last_only=True （默认）  hidden [T,1024] -> 标量。prefill 用：
                                只要最后一个位置的 logits，前面 T-1 行不算。
        last_only=False          hidden [B,1024] -> [B]。batch decode 用：
                                每条序列各出一个 token。

    不物化 logits 省掉的是 248320*4B = 1 MiB 的一写一读。代价是**做不了采样**
    （temperature/top-k/top-p 都需要完整分布）。要加采样得改这个 kernel 的输出形式。
    """
    assert hidden.ndim == 2 and weight.ndim == 2
    assert hidden.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert hidden.device == weight.device
    assert hidden.is_contiguous() and weight.is_contiguous()

    token_num, hidden_size = hidden.shape
    vocab_size, weight_hidden_size = weight.shape
    assert token_num > 0 and vocab_size > 0
    assert weight_hidden_size == hidden_size
    assert hidden_size % TILE_K == 0
    assert GROUP_V % TILE_V == 0

    # last_only 时只算最后一行：直接传一个指向那一行的视图，
    # kernel 里就统一成「第 pid_b 行」，不用再区分两种基址。
    rows_view = hidden[token_num - 1 :] if last_only else hidden
    rows = rows_view.shape[0]

    num_partials = triton.cdiv(vocab_size, GROUP_V)
    block_partial = triton.next_power_of_2(num_partials)
    partial_values = torch.empty(
        (rows, num_partials), dtype=torch.float32, device=hidden.device
    )
    partial_indices = torch.empty(
        (rows, num_partials), dtype=torch.int32, device=hidden.device
    )
    token_id = torch.empty((rows,), dtype=torch.int64, device=hidden.device)

    torch.library.wrap_triton(_lm_head_argmax_stage1_triton)[(rows, num_partials)](
        hidden_ptr=rows_view,
        stride_hidden_t=rows_view.stride(0),
        weight_ptr=weight,
        stride_w_vocab=weight.stride(0),
        stride_w_hidden=weight.stride(1),
        partial_values_ptr=partial_values,
        stride_pv_b=partial_values.stride(0),
        partial_indices_ptr=partial_indices,
        stride_pi_b=partial_indices.stride(0),
        row_stride=rows_view.stride(0),
        VOCAB_SIZE=vocab_size,
        HIDDEN_SIZE=hidden_size,
        GROUP_V=GROUP_V,
        TILE_V=TILE_V,
        TILE_K=TILE_K,
        num_warps=4,
        num_stages=1,
    )
    torch.library.wrap_triton(_lm_head_argmax_stage2_triton)[(rows,)](
        partial_values_ptr=partial_values,
        stride_pv_b=partial_values.stride(0),
        partial_indices_ptr=partial_indices,
        stride_pi_b=partial_indices.stride(0),
        token_id_ptr=token_id,
        num_partials=num_partials,
        BLOCK_PARTIAL=block_partial,
        num_warps=4,
        num_stages=1,
    )
    # last_only 时保持原来的标量返回，调用方不用改
    return token_id[0] if last_only else token_id


@torch.library.register_fake("wy_lib::lm_head_argmax")
def _lm_head_argmax_fake(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    last_only: bool = True,
) -> torch.Tensor:
    shape = () if last_only else (hidden.shape[0],)
    return torch.empty(shape, dtype=torch.int64, device=hidden.device)


def call_lm_head_argmax_triton(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return lm_head_argmax(hidden, weight)


def _torch_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    logits = torch.mv(weight.float(), hidden[-1].float())
    return torch.argmax(logits)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")

    # Covers multiple partial groups and a non-full final GROUP_V.
    hidden = torch.randn((3, 128), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((1025, 128), dtype=torch.bfloat16, device="cuda")
    actual = call_lm_head_argmax_triton(hidden, weight)
    expected = _torch_reference(hidden, weight)
    assert actual.item() == expected.item()
    print(
        f"shape={tuple(hidden.shape)}x{tuple(weight.shape)}, "
        f"token_id={actual.item()}, reference={expected.item()}"
    )

    # All logits tie at zero; argmax must select the smallest vocab index.
    hidden.zero_()
    actual = call_lm_head_argmax_triton(hidden, weight)
    assert actual.item() == 0
    print("all-zero tie test: token_id=0")

    # Target checkpoint dimensions. Plant a clear winner so this test checks
    # the complete 485-group indexing path without depending on a tiny top-2 gap.
    vocab_size = 248320
    hidden_size = 1024
    winner = vocab_size - 17
    hidden = torch.randn((2, hidden_size), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(
        (vocab_size, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    weight[winner].copy_(hidden[-1] * 8.0)

    actual = call_lm_head_argmax_triton(hidden, weight)
    expected = _torch_reference(hidden, weight)
    assert actual.item() == expected.item() == winner
    print(
        f"target_shape={(vocab_size, hidden_size)}, "
        f"token_id={actual.item()}, reference={expected.item()}"
    )

    print("All LM-head argmax tests passed.")
