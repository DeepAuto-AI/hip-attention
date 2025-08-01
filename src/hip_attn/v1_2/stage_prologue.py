import torch
import triton

from .attention_metadata import HiPAttentionArgs, Stage
from .utils import capture


@capture
@torch.compile(dynamic=True)
def stage_prologue(
    q: torch.Tensor,
    indices_left: torch.Tensor,
    indices_right: torch.Tensor,
    out_scores: torch.Tensor,
    stage_k: int,
    stage_chunk_size: int,
    chunk_size: int,
    stage_info: Stage,
    args: HiPAttentionArgs,
    TDST,
    BDST,
    STAGE_STRIDE,
    BLOCK_SIZE_Q,
):
    assert (stage_k % chunk_size) == 0, f"{stage_k} % {chunk_size}"
    indices_left = indices_left[..., : stage_k // chunk_size]
    require_align = stage_info.require_realign_index
    if require_align:
        indices_left = (
            indices_left - args.sink_token_size
        ) // chunk_size * chunk_size + args.sink_token_size
        indices_right = indices_left + chunk_size
    else:
        indices_right = indices_right[..., : stage_k // chunk_size]
    out_scores = out_scores[..., : stage_k // chunk_size]
    # NOTE: revert this
    if stage_info.require_reset_score:
        out_scores.fill_(-32000.0)

    require_sort = BDST > 1
    if require_sort:
        indices_left, t_indices = indices_left.sort(dim=-1)
        indices_right = indices_right.gather(dim=-1, index=t_indices)
        out_scores = out_scores.gather(dim=-1, index=t_indices)

    if BLOCK_SIZE_Q != stage_info.stage_block_size_q:
        assert stage_info.stage_block_size_q > 0
        assert BLOCK_SIZE_Q > stage_info.stage_block_size_q
        assert (BLOCK_SIZE_Q % stage_info.stage_block_size_q) == 0

        num_split = BLOCK_SIZE_Q // stage_info.stage_block_size_q
        BLOCK_SIZE_Q = stage_info.stage_block_size_q
        BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)

        indices_left = indices_left.repeat_interleave(num_split, 1)[
            :, -BDST:
        ].contiguous()
        indices_right = indices_right.repeat_interleave(num_split, 1)[
            :, -BDST:
        ].contiguous()
        out_scores = out_scores.repeat_interleave(num_split, 1)[:, -BDST:].contiguous()

    if STAGE_STRIDE != stage_info.stage_stride:
        assert stage_info.stage_stride < STAGE_STRIDE
        assert STAGE_STRIDE > 0
        indices_left = indices_left.repeat_interleave(
            STAGE_STRIDE // stage_info.stage_stride, 1
        )[:, -BDST:].contiguous()
        indices_right = indices_right.repeat_interleave(
            STAGE_STRIDE // stage_info.stage_stride, 1
        )[:, -BDST:].contiguous()
        out_scores = out_scores.repeat_interleave(
            STAGE_STRIDE // stage_info.stage_stride, 1
        )[:, -BDST:].contiguous()
        STAGE_STRIDE = stage_info.stage_stride

    assert (chunk_size % stage_chunk_size) == 0
    splits = chunk_size // stage_chunk_size
    chunk_sizes = ((indices_right - indices_left).float() / splits).clamp_min_(0)
    indices_left = (
        indices_left[..., None]
        + (
            torch.arange(0, splits, device=q.device)[None, None, None, None, :]
            * chunk_sizes[..., None]
        )
        .floor()
        .long()
    )
    indices_left = indices_left.flatten(-2, -1)
    indices_right = (
        indices_right[..., None]
        - (
            (
                (splits - 1)
                - torch.arange(0, splits, device=q.device)[None, None, None, None, :]
            )
            * chunk_sizes[..., None]
        )
        .floor()
        .long()
    )
    indices_right = indices_right.flatten(-2, -1)
    out_scores = out_scores.repeat_interleave(splits, -1)

    return indices_left, indices_right, out_scores, BLOCK_SIZE_Q, BDST, STAGE_STRIDE
