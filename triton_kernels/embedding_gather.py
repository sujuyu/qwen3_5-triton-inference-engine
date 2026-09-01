import torch

import triton
import triton.language as tl


autotune_configs = [
    triton.Config({"BLOCK_T": 1}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 2}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 4}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_T": 8}, num_warps=4, num_stages=1),
]


@triton.autotune(
    configs=autotune_configs,
    key=["d_model", "T_BUCKET"],
)
@triton.jit
def _embedding_gather_triton(
    input_ids_ptr, # [T] 
    weight_ptr, # [248320, 1024] BF16
    stride_w: tl.constexpr, 
    output_ptr, # [T, 1024], 
    stride_o_t: tl.constexpr, stride_o_d: tl.constexpr,
    d_model: tl.constexpr, # 1024 
    token_num,
    T_BUCKET: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    pid = tl.program_id(0)
    offset_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offset_d = tl.arange(0, d_model)

    input_ids = tl.load(
        input_ids_ptr + offset_t, 
        mask = offset_t < token_num, 
        other = 0
    )

    w = tl.load(
        weight_ptr + input_ids[:, None] * stride_w + offset_d[None, :], 
        mask = offset_t[:, None] < token_num)

    tl.store(
        output_ptr + offset_t[:, None] * stride_o_t + offset_d[None, :], 
        w, 
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
    "wy_lib::embedding_gather",
    mutates_args=(),
)
def embedding_gather(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    assert input_ids.ndim == 1
    assert input_ids.dtype in (torch.int32, torch.int64)
    assert input_ids.is_contiguous()
    assert weight.ndim == 2 and weight.dtype == torch.bfloat16
    assert weight.is_contiguous()
    assert input_ids.device == weight.device

    token_num = input_ids.shape[0]
    d_model = weight.shape[1]
    assert token_num > 0 and weight.shape[0] > 0
    assert triton.next_power_of_2(d_model) == d_model

    output = torch.empty(
        (token_num, d_model),
        dtype=weight.dtype,
        device=weight.device,
    )

    def grid(meta):
        return (triton.cdiv(token_num, meta["BLOCK_T"]),)

    torch.library.wrap_triton(_embedding_gather_triton)[grid](
        input_ids_ptr=input_ids,
        weight_ptr=weight,
        stride_w=weight.stride(0),
        output_ptr=output,
        stride_o_t=output.stride(0),
        stride_o_d=output.stride(1),
        d_model=d_model,
        token_num=token_num,
        T_BUCKET=_token_bucket(token_num),
    )
    return output


@torch.library.register_fake("wy_lib::embedding_gather")
def _embedding_gather_fake(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        (input_ids.shape[0], weight.shape[1]),
        dtype=weight.dtype,
        device=weight.device,
    )


def call_embedding_gather_triton(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return embedding_gather(input_ids, weight)


def _torch_reference(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.embedding(input_ids.long(), weight)


if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 257
    d_model = 1024
    weight = torch.randn(
        (vocab_size, d_model),
        dtype=torch.bfloat16,
        device="cuda",
    )

    for input_dtype in (torch.int32, torch.int64):
        for token_num in (1, 3, 17, 65):
            input_ids = torch.randint(
                0,
                vocab_size,
                (token_num,),
                dtype=input_dtype,
                device="cuda",
            )
            input_ids[0] = 0
            if token_num > 1:
                input_ids[1] = 0
                input_ids[-1] = vocab_size - 1

            actual = call_embedding_gather_triton(input_ids, weight)
            expected = _torch_reference(input_ids, weight)

            assert torch.equal(actual, expected)
            assert actual.is_contiguous()
            print(
                f"dtype={input_dtype}, token_num={token_num}, "
                f"max_abs_error=0.0, "
                f"best_config={_embedding_gather_triton.best_config}"
            )

    print("All embedding gather tests passed.")
