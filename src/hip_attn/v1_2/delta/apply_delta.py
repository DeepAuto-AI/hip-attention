import torch
import triton
import triton.language as tl

from ..attention_metadata import safe_stride
from ..utils import capture


@triton.jit
def _fused_apply_delta(
    DENSE,
    stride_dense_bsz,
    stride_dense_tdelta,
    stride_dense_head,
    stride_dense_hid,
    SPARSE,
    stride_sparse_bsz,
    stride_sparse_t,
    stride_sparse_head,
    stride_sparse_hid,
    OUT,
    stride_out_bsz,
    stride_out_t,
    stride_out_head,
    stride_out_hid,
    IDS,
    stride_ids_t,
    args_w: int,
    T: int,
    N_DELTA: int,
    HEAD: int,
    ARGS_SMOOTH: tl.constexpr,
    HID: tl.constexpr,
    BLOCK_DELTA: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    idx_head = pid % HEAD
    pid = pid // HEAD
    t = tl.cdiv(N_DELTA, BLOCK_DELTA)
    idx_tdelta = tl.arange(0, BLOCK_DELTA) + (pid % t) * BLOCK_DELTA
    mask_tdelta = idx_tdelta < N_DELTA
    idx_bsz = pid // t

    idx_hid = tl.arange(0, HID)

    dense = tl.load(
        DENSE
        + idx_bsz * stride_dense_bsz
        + idx_tdelta[:, None] * stride_dense_tdelta
        + idx_head * stride_dense_head
        + idx_hid[None, :] * stride_dense_hid,
        mask=mask_tdelta[:, None],
    )

    ids = tl.load(IDS + idx_tdelta * stride_ids_t, mask=mask_tdelta)

    tl.store(
        OUT
        + idx_bsz * stride_out_bsz
        + ids[:, None] * stride_out_t
        + idx_head * stride_out_head
        + idx_hid[None, :] * stride_out_hid,
        mask=mask_tdelta[:, None],
        value=dense,
    )

    sparse_sample = tl.load(
        SPARSE
        + idx_bsz * stride_sparse_bsz
        + ids[:, None] * stride_sparse_t
        + idx_head * stride_sparse_head
        + idx_hid[None, :] * stride_sparse_hid,
        mask=mask_tdelta[:, None],
    )

    delta = dense - sparse_sample

    if ARGS_SMOOTH:
        idx_tdelta_next = tl.minimum(idx_tdelta + 1, N_DELTA - 1)
        dense_next = tl.load(
            DENSE
            + idx_bsz * stride_dense_bsz
            + idx_tdelta_next[:, None] * stride_dense_tdelta
            + idx_head * stride_dense_head
            + idx_hid[None, :] * stride_dense_hid,
            mask=mask_tdelta[:, None],
        )

        ids_next = tl.load(IDS + idx_tdelta_next * stride_ids_t, mask=mask_tdelta)

        sparse_sample_next = tl.load(
            SPARSE
            + idx_bsz * stride_sparse_bsz
            + ids_next[:, None] * stride_sparse_t
            + idx_head * stride_sparse_head
            + idx_hid[None, :] * stride_sparse_hid,
            mask=mask_tdelta[:, None],
        )

        delta_next = dense_next - sparse_sample_next

    for i in range(1, args_w):
        ids_off = ids + i
        mask_ids_off = (ids_off < T) & mask_tdelta
        sparse_other = tl.load(
            SPARSE
            + idx_bsz * stride_sparse_bsz
            + ids_off[:, None] * stride_sparse_t
            + idx_head * stride_sparse_head
            + idx_hid[None, :] * stride_sparse_hid,
            mask=mask_ids_off[:, None],
        )

        if ARGS_SMOOTH:
            delta_now = delta * ((args_w - i) / args_w) + delta_next * (i / args_w)
        else:
            delta_now = delta

        sparse_corr = sparse_other + delta_now

        tl.store(
            OUT
            + idx_bsz * stride_out_bsz
            + ids_off[:, None] * stride_out_t
            + idx_head * stride_out_head
            + idx_hid[None, :] * stride_out_hid,
            mask=mask_ids_off[:, None],
            value=sparse_corr,
        )


@capture
def apply_delta(
    context_dense: torch.Tensor,
    context_sparse: torch.Tensor,
    idx: torch.Tensor,
    num_last_dense: int,
    args_w: int,
    args_smooth: bool,
):
    __fused = True
    if __fused:
        context_dense, last_context_dense = (
            context_dense[:, :-num_last_dense],
            context_dense[:, -num_last_dense:],
        )

        idx, last_idx = (
            idx[:-num_last_dense],
            idx[-num_last_dense:],
        )

        BSZ, N_DELTA, HEAD, HID = context_dense.shape
        assert idx.shape[0] == N_DELTA
        _, T, _, _ = context_sparse.shape
        assert context_sparse.shape == (BSZ, T, HEAD, HID)
        assert (T % args_w) == 0

        context = torch.empty(
            (BSZ, T + num_last_dense, HEAD, HID),
            dtype=context_sparse.dtype,
            device=context_sparse.device,
        )

        BLOCK_DELTA = 32

        grid = (HEAD * BSZ * triton.cdiv(N_DELTA, BLOCK_DELTA),)

        _fused_apply_delta[grid](
            context_dense,
            *safe_stride(context_dense, 4),
            context_sparse,
            *safe_stride(context_sparse, 4),
            context,
            *safe_stride(context, 4),
            idx,
            *safe_stride(idx, 1),
            args_w,
            T,
            N_DELTA,
            HEAD,
            args_smooth,
            HID,
            BLOCK_DELTA,
        )

        if num_last_dense > 0:
            context.index_copy_(dim=1, index=last_idx, source=last_context_dense)
    else:
        context_sparse_for_diff = context_sparse[:, idx[:-num_last_dense]]

        context_dense_concat = context_dense
        context_dense, last_context_dense = (
            context_dense[:, :-num_last_dense],
            context_dense[:, -num_last_dense:],
        )

        # context_sparse_for_diff_norm = context_sparse_for_diff.float().square().sum(dim=-1, keepdim=True).sqrt()
        # context_dense_norm = context_dense.float().square().sum(dim=-1, keepdim=True).sqrt()
        # scale = context_dense_norm / context_sparse_for_diff_norm

        # take difference
        context_diff = context_dense - context_sparse_for_diff  # * scale

        context_diff = context_diff.repeat_interleave(args_w, dim=1)

        if args_smooth:
            # (exp) linear interpolate diff
            context_diff_shift = torch.roll(context_diff, -args_w, 1)
            context_diff_shift[:, -args_w:] = context_diff[:, -1:]

            offset = torch.arange(0, context_diff.shape[1], device=context_diff.device)
            offset = (offset % args_w).float() / args_w
            context_diff = (
                context_diff
                + (context_diff_shift - context_diff) * offset[None, :, None, None]
            )

        # context_sparse_norm = context_sparse.float().square().sum(dim=-1, keepdim=True).sqrt()
        # scale = context_dense_norm.repeat_interleave(delta_attention_args_w, dim=1) / context_sparse_norm

        # context = context_sparse * scale + context_diff
        context = context_sparse + context_diff
        context = torch.cat([context, last_context_dense], dim=1).to(
            context_sparse.dtype
        )
        context[:, idx] = context_dense_concat

        # if get_local_rank() == 0:
        #     print(
        #         'hit', layer_id,
        #         context_diff.shape,
        #         context_sparse.shape,
        #         context_diff.abs().mean().item(),
        #         context_sparse.abs().mean().item()
        #     )

    return context
