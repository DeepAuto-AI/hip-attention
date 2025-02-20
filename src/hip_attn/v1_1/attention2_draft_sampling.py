import math
import os
from typing import Optional

import cv2
import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor

from hip_attn.v1_1.attention2_draft_prefetch import (
    HiPAttentionArgs,
    block_sparse_attention,
    masking_iteration_draft,
    to_dense,
)


@triton.jit
def sampling_mask_cuda(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head_kv,
    stride_k_hid,
    OUT_INDICES,
    stride_out_indices_bsz,
    stride_out_indices_bdst,
    stride_out_indices_head,
    stride_out_indices_chunk,
    OUT_SCORES,
    stride_out_scores_bsz,
    stride_out_scores_bdst,
    stride_out_scores_head,
    stride_out_scores_chunk,
    CHUNK_COUNT: int,
    TSRC: int,
    TDST: int,
    HEAD: int,
    sliding_window_size: int,
    num_sinks: int,
    BLOCK_HID: tl.constexpr = 128,
    BLOCK_SIZE_Q: tl.constexpr = 32,
    STRIDE_Q: tl.constexpr = 1,
    BLOCK_CHUNK: tl.constexpr = 32,
    HEAD_GROUP: tl.constexpr = 4,
    CAUSAL_CHUNKING: tl.constexpr = True,
    REDUCE: tl.constexpr = "mean",
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)
    BCHUNK = tl.cdiv(CHUNK_COUNT, BLOCK_CHUNK)

    pid = tl.program_id(0).to(tl.int64)

    idx_head = pid % HEAD
    pid = pid // HEAD
    idx_bdst = pid % BDST
    pid = pid // BDST
    idx_bchunk = pid % BCHUNK
    pid = pid // BCHUNK
    idx_bsz = pid

    # idx_bsz = tl.program_id(3).to(tl.int64)
    # idx_bchunk = tl.program_id(2).to(tl.int64)
    # idx_bdst = tl.program_id(1).to(tl.int64)
    # idx_head = tl.program_id(0).to(tl.int64)

    # if ((idx_bchunk + 1) * BLOCK_CHUNK * (TSRC / CHUNK_COUNT)) > (idx_bdst * BLOCK_SIZE_Q + TSRC - TDST):
    #     return

    pos_tdst_min = (
        idx_bdst * BLOCK_SIZE_Q + TSRC - TDST - sliding_window_size - num_sinks
    ).to(tl.float32)
    pos_tdst_min = tl.maximum(pos_tdst_min, 0)

    idx_chunk = idx_bchunk * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
    mask_chunk = idx_chunk < CHUNK_COUNT

    if CAUSAL_CHUNKING:
        idx_tsrc_start = (
            tl.floor(idx_chunk.to(tl.float32) * (pos_tdst_min / CHUNK_COUNT)).to(
                tl.int64
            )
            + num_sinks
        )
        idx_tsrc_end = (
            tl.where(
                (idx_chunk + 1) == CHUNK_COUNT,
                pos_tdst_min,
                tl.floor(
                    (idx_chunk + 1).to(tl.float32) * (pos_tdst_min / CHUNK_COUNT)
                ).to(tl.int64),
            ).to(tl.int64)
            + num_sinks
        )
    else:
        idx_tsrc_start = (
            tl.floor(idx_chunk.to(tl.float64) * ((TDST - num_sinks) / CHUNK_COUNT)).to(
                tl.int64
            )
            + num_sinks
        )
        idx_tsrc_end = (
            tl.where(
                (idx_chunk + 1) == CHUNK_COUNT,
                pos_tdst_min,
                tl.floor(
                    (idx_chunk + 1).to(tl.float64) * ((TDST - num_sinks) / CHUNK_COUNT)
                ).to(tl.int64),
            ).to(tl.int64)
            + num_sinks
        )

    max_chunk_size = tl.ceil(TSRC / CHUNK_COUNT).to(tl.float32)

    idx_tdst = (
        idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // STRIDE_Q) * STRIDE_Q
    )
    mask_tdst = idx_tdst < TDST

    idx_hid = tl.arange(0, BLOCK_HID)

    queries = tl.load(
        Q
        + idx_bsz * stride_q_bsz
        + idx_tdst[:, None] * stride_q_tdst
        + idx_head * stride_q_head
        + idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0,
    )

    idx_tsrc_left = idx_tsrc_start
    idx_tsrc_right = idx_tsrc_end
    scores = tl.zeros((BLOCK_CHUNK,), dtype=tl.float32) - 32000.0

    while max_chunk_size > 1:
        max_chunk_size /= 2.0
        mask_tsrc_active = (
            mask_chunk
            & (idx_tsrc_left < idx_tsrc_right)
            & (idx_tsrc_left < (pos_tdst_min))
        )
        idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2
        # idx_tsrc_left = (idx_tsrc_left + idx_tsrc_center) // 2
        # idx_tsrc_right = (idx_tsrc_center + idx_tsrc_right) // 2

        keys_left = tl.load(
            K
            + idx_bsz * stride_k_bsz
            + ((idx_tsrc_left + idx_tsrc_center) // 2)[None, :] * stride_k_tsrc
            + (idx_head // HEAD_GROUP) * stride_k_head_kv
            + idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc_active[None, :],
            other=0,
        )
        scores_left = tl.dot(
            (queries / 8).to(tl.float16),
            (keys_left / 8).to(tl.float16),
            out_dtype=tl.float16,
        )
        if REDUCE == "max":
            scores_left = tl.where(mask_tdst[:, None], scores_left, float("-inf")).to(
                scores_left.dtype
            )
            scores_left = tl.max(scores_left, axis=0).to(scores_left.dtype)
        elif REDUCE == "mean":
            scores_left = tl.where(mask_tdst[:, None], scores_left, 0.0).to(
                scores_left.dtype
            )
            scores_left = tl.sum(scores_left, axis=0).to(scores_left.dtype)
            scores_left = (scores_left / tl.sum(mask_tdst.to(tl.float32))).to(
                scores_left.dtype
            )
        else:
            raise Exception()
        scores_left = tl.where(mask_tsrc_active, scores_left, float("-inf")).to(
            scores_left.dtype
        )

        keys_right = tl.load(
            K
            + idx_bsz * stride_k_bsz
            + ((idx_tsrc_center + idx_tsrc_right) // 2)[None, :] * stride_k_tsrc
            + (idx_head // HEAD_GROUP) * stride_k_head_kv
            + idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc_active[None, :],
            other=0,
        )
        scores_right = tl.dot(
            (queries / 8).to(tl.float16),
            (keys_right / 8).to(tl.float16),
            out_dtype=tl.float16,
        )
        # scores_right = tl.where(mask_tdst[:, None], scores_right, float('-inf')).to(scores_right.dtype)
        # scores_right = tl.max(scores_right, axis=0).to(scores_right.dtype)
        if REDUCE == "max":
            scores_right = tl.where(mask_tdst[:, None], scores_right, float("-inf")).to(
                scores_right.dtype
            )
            scores_right = tl.max(scores_right, axis=0).to(scores_right.dtype)
        elif REDUCE == "mean":
            scores_right = tl.where(mask_tdst[:, None], scores_right, 0.0).to(
                scores_right.dtype
            )
            scores_right = tl.sum(scores_right, axis=0).to(scores_right.dtype)
            scores_right = (scores_right / tl.sum(mask_tdst.to(tl.float32))).to(
                scores_right.dtype
            )
        else:
            raise Exception()
        scores_right = tl.where(mask_tsrc_active, scores_right, float("-inf")).to(
            scores_right.dtype
        )

        mask_left_win = scores_left > scores_right
        idx_tsrc_left = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                idx_tsrc_left,
                idx_tsrc_center,
            ),
            idx_tsrc_left,
        )

        idx_tsrc_right = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                idx_tsrc_center,
                idx_tsrc_right,
            ),
            idx_tsrc_right,
        )

        scores = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                scores_left,
                scores_right,
            ),
            scores,
        )

    tl.store(
        OUT_INDICES
        + idx_bsz * stride_out_indices_bsz
        + idx_bdst * stride_out_indices_bdst
        + idx_head * stride_out_indices_head
        + (tl.arange(0, BLOCK_CHUNK) + idx_bchunk * BLOCK_CHUNK)
        * stride_out_indices_chunk,
        value=(idx_tsrc_left + idx_tsrc_right) // 2,
        mask=mask_chunk,
    )

    tl.store(
        OUT_SCORES
        + idx_bsz * stride_out_scores_bsz
        + idx_bdst * stride_out_scores_bdst
        + idx_head * stride_out_scores_head
        + (tl.arange(0, BLOCK_CHUNK) + idx_bchunk * BLOCK_CHUNK)
        * stride_out_scores_chunk,
        value=scores,
        mask=mask_chunk,
    )


DEBUG = os.getenv("HIP_DEBUG", "0") == "1"


def dual_stage_sub_quadratic_hip_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    first_stage_args: HiPAttentionArgs,
    second_stage_init_k: int,
    second_stage_init_chunk: int,
    second_stage_args: HiPAttentionArgs,
):
    global DEBUG

    BSZ, TDST, HEAD, HID = q.shape
    BSZ, TSRC, HEAD_KV, HID = k.shape
    assert v.shape == k.shape

    chunk_count = first_stage_args.mask_k
    chunk_size = TSRC // chunk_count
    # assert (chunk_size * chunk_count) == TSRC

    MAX_TDST = TDST
    MAX_TSRC = TSRC

    if (first_stage_args.position_ids is not None) and (
        second_stage_args.position_ids is not None
    ):
        assert (
            first_stage_args.position_ids.data_ptr()
            == second_stage_args.position_ids.data_ptr()
        )
        position_ids = first_stage_args.position_ids
    elif first_stage_args.position_ids is not None:
        position_ids = first_stage_args.position_ids
    elif second_stage_args.position_ids is not None:
        position_ids = second_stage_args.position_ids
    else:
        position_ids = (torch.arange(0, TDST, device=q.device) + (TSRC - TDST))[
            None, :
        ].expand(BSZ, TDST)

    assert first_stage_args.block_size_q == second_stage_args.block_size_q

    BLOCK_CHUNK = first_stage_args.block_size_k
    BLOCK_SIZE_Q = first_stage_args.block_size_q

    out_indices = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count),
        device=q.device,
        dtype=torch.int64,
    )
    out_scores = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count),
        device=q.device,
        dtype=torch.float16,
    )

    sampling_mask_cuda[
        (
            BSZ
            * triton.cdiv(chunk_count, BLOCK_CHUNK)
            * triton.cdiv(TDST, BLOCK_SIZE_Q)
            * HEAD,
        )
    ](
        q,
        *q.stride(),
        k,
        *k.stride(),
        out_indices,
        *out_indices.stride(),
        out_scores,
        *out_scores.stride(),
        chunk_count,
        TSRC,
        TDST,
        HEAD,
        first_stage_args.sliding_window_size - max(BLOCK_SIZE_Q, BLOCK_CHUNK),
        first_stage_args.sink_token_size,
        BLOCK_HID=q.shape[-1],
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        STRIDE_Q=first_stage_args.block_stride_q,
        BLOCK_CHUNK=BLOCK_CHUNK,
        HEAD_GROUP=HEAD // HEAD_KV,
    )

    _, indices = out_scores.sort(dim=-1, descending=True, stable=False)
    out_indices = out_indices.gather(dim=-1, index=indices)

    if DEBUG:
        out_indices_cpu = out_indices.cpu()
        debug = np.zeros(
            (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK))
        )
        for i in range(out_indices_cpu.shape[1]):
            t_chunk_size = triton.cdiv(
                ((i + 1) * BLOCK_SIZE_Q), chunk_count * BLOCK_CHUNK
            )
            # print(i, t_chunk_size)
            for j in range(
                max(
                    second_stage_init_chunk,
                    math.ceil(
                        out_indices_cpu.shape[-1]
                        * (second_stage_init_k / ((i + 1) * BLOCK_SIZE_Q))
                    ),
                )
            ):
                if j >= out_indices_cpu.shape[-1]:
                    continue
                t = out_indices_cpu[0, i, -3, j] // BLOCK_CHUNK
                t = t // t_chunk_size * t_chunk_size
                debug[i, t : t + t_chunk_size] = 1
        cv2.imwrite("dummy_sampled.png", debug * 255)

    idx_tdst = torch.arange(0, TDST, device=q.device)
    seq_lens = (
        position_ids.gather(dim=1, index=idx_tdst.unsqueeze(0).expand(BSZ, -1)) + 1
    )

    repeated_seq_lens = seq_lens[:, ::BLOCK_SIZE_Q].repeat_interleave(HEAD, 0)
    ks_seed = torch.clamp(
        torch.ceil(
            out_indices.shape[-1]
            * (second_stage_init_k / (repeated_seq_lens + BLOCK_SIZE_Q))
        ),
        second_stage_init_chunk,
        out_indices.shape[-1],
    )

    mask_block_k = second_stage_args.mask_k // second_stage_args.block_size_k

    chunk_sizes = torch.ceil(repeated_seq_lens / (chunk_count * BLOCK_CHUNK)).to(
        torch.int32
    )
    chunk_sizes = (
        chunk_sizes // second_stage_args.block_size_k * second_stage_args.block_size_k
    )

    indices_seed = out_indices.permute(0, 2, 1, 3).flatten(0, 1)
    indices_seed = indices_seed // chunk_sizes.unsqueeze(-1) * chunk_sizes.unsqueeze(-1)

    active_indices_mask = torch.arange(0, indices_seed.shape[-1], device=q.device)[
        None, None, :
    ] < ks_seed.unsqueeze(-1)
    indices_seed = torch.where(
        active_indices_mask,
        indices_seed,
        MAX_TSRC,
    )
    indices_seed, _ = indices_seed.sort(dim=-1)

    group_size_seed = torch.where(
        active_indices_mask,
        chunk_sizes.unsqueeze(-1).expand_as(active_indices_mask),
        0,
    )

    assert indices_seed.shape[-1] <= mask_block_k

    indices_seed_padded = torch.full(
        (indices_seed.shape[0], indices_seed.shape[1], mask_block_k),
        fill_value=MAX_TSRC,
        device=q.device,
        dtype=torch.int32,
    )
    indices_seed_padded[:, :, : indices_seed.shape[-1]] = indices_seed
    indices_seed = indices_seed_padded

    group_sizes_padded = torch.full(
        (group_size_seed.shape[0], group_size_seed.shape[1], mask_block_k),
        fill_value=MAX_TSRC,
        device=q.device,
        dtype=torch.int32,
    )
    group_sizes_padded[:, :, : group_size_seed.shape[-1]] = group_size_seed
    group_size_seed = group_sizes_padded // second_stage_args.block_size_k

    (
        indices,
        ks,
        ks_count,
        ks_start_end,
        scores,
        group_sizes,
        key_access_log,
        key_access_count,
        block_access_log,
        block_access_score,
        block_access_count,
    ) = masking_iteration_draft(
        q=q,
        k=k,
        position_ids=seq_lens,
        indices_tdst=idx_tdst,
        args=second_stage_args,
        indices_seed=indices_seed,
        ks_seed=ks_seed,
        # group_size_seed=group_size_seed,
        max_group_size_seed=chunk_size // second_stage_args.block_size_k,
        scores_seed=None,
    )

    HIP_DEBUG_MASK_UNIQUE = os.getenv("HIP_DEBUG_MASK_UNIQUE", "0") == "1"
    if HIP_DEBUG_MASK_UNIQUE:
        # indices = indices.sort(dim=-1).values
        unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
        indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
        indices = indices.sort(dim=-1).values
        # active_mask = unique_mask
        active_mask = indices < (
            position_ids[:, :: second_stage_args.block_size_q, None].repeat_interleave(
                HEAD, 0
            )
            + second_stage_args.block_size_q
        )
        ks = active_mask.int().sum(-1)
        ks_count = ks.unsqueeze(-1)
        ks_start_end[:, :, -1] = ks

    # if DEBUG:
    #     max_query = 1024
    #     B, TDST, H, HID = q.shape
    #     TDST = min(TDST, max_query)
    #     if k is not None:
    #         _, TSRC, H_KV, _ = k.shape
    #     else:
    #         TSRC = torch.max(second_stage_args.cache_seq_lens).item()
    #     N = B * H
    #     def render_mask():
    #         debug_mask = to_dense(
    #             indices[:, -triton.cdiv(TDST, second_stage_args.block_size_q):].cpu().numpy(),
    #             ks[:, -triton.cdiv(TDST, second_stage_args.block_size_q):].cpu().numpy(),
    #             None,
    #             triton.cdiv(N, second_stage_args.topk_head_group_size),
    #             TDST,
    #             TSRC * second_stage_args.topk_head_group_size,
    #             second_stage_args.block_size_q,
    #             second_stage_args.block_size_k * second_stage_args.block_size_k_group,
    #         )[0]
    #         if second_stage_args.group_size_q > 1:
    #             debug_mask = debug_mask.repeat(axis=0, repeats=second_stage_args.group_size_q)

    #         cv2.imwrite('dummy_raw.png', debug_mask.astype(np.uint8) * 255)
    #         print('saved dummy_raw.png', indices.shape, ks.shape, debug_mask.shape, q.shape, TSRC)
    #     render_mask()

    second_stage_args = second_stage_args.clone()
    second_stage_args.sliding_window_size += second_stage_args.block_size_q

    context = block_sparse_attention(
        q=q,
        k=k,
        v=v,
        seq_lens=position_ids + 1,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=second_stage_args,
    )

    return context


def sampling_only_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    args: HiPAttentionArgs,
):
    global DEBUG

    BSZ, TDST, HEAD, HID = q.shape
    BSZ, TSRC, HEAD_KV, HID = k.shape
    assert v.shape == k.shape
    assert TDST == TSRC

    chunk_count = TSRC // args.block_size_k
    chunk_size = TSRC // chunk_count
    # assert (chunk_size * chunk_count) == TSRC

    MAX_TDST = TDST
    MAX_TSRC = TSRC

    if args.position_ids is not None:
        position_ids = args.position_ids
    else:
        position_ids = (torch.arange(0, TDST, device=q.device) + (TSRC - TDST))[
            None, :
        ].expand(BSZ, TDST)

    BLOCK_CHUNK = args.block_size_k
    BLOCK_SIZE_Q = args.block_size_q

    out_indices = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count),
        device=q.device,
        dtype=torch.int64,
    )
    out_scores = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count),
        device=q.device,
        dtype=torch.float16,
    )

    sampling_mask_cuda[
        (
            BSZ
            * triton.cdiv(chunk_count, BLOCK_CHUNK)
            * triton.cdiv(TDST, BLOCK_SIZE_Q)
            * HEAD,
        )
    ](
        q,
        *q.stride(),
        k,
        *k.stride(),
        out_indices,
        *out_indices.stride(),
        out_scores,
        *out_scores.stride(),
        chunk_count,
        TSRC,
        TDST,
        HEAD,
        args.sliding_window_size - max(BLOCK_SIZE_Q, BLOCK_CHUNK),
        args.sink_token_size,
        BLOCK_HID=q.shape[-1],
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        STRIDE_Q=args.block_stride_q,
        BLOCK_CHUNK=BLOCK_CHUNK,
        HEAD_GROUP=HEAD // HEAD_KV,
    )

    _, indices = out_scores.sort(dim=-1, descending=True, stable=False)
    out_indices = out_indices.gather(dim=-1, index=indices)

    if DEBUG:
        out_indices_cpu = out_indices.cpu()
        debug = np.zeros(
            (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK))
        )
        for i in range(out_indices_cpu.shape[1]):
            t_chunk_size = triton.cdiv(
                ((i + 1) * BLOCK_SIZE_Q), chunk_count * BLOCK_CHUNK
            )
            # print(i, t_chunk_size)
            for j in range(
                max(
                    0,
                    math.ceil(
                        out_indices_cpu.shape[-1]
                        * (args.mask_k / ((i + 1) * BLOCK_SIZE_Q))
                    ),
                )
            ):
                if j >= out_indices_cpu.shape[-1]:
                    continue
                t = out_indices_cpu[0, i, -3, j] // BLOCK_CHUNK
                t = t // t_chunk_size * t_chunk_size
                debug[i, t : t + t_chunk_size] = 1
        cv2.imwrite("dummy_sampled.png", debug * 255)

    idx_tdst = torch.arange(0, TDST, device=q.device)
    seq_lens = (
        position_ids.gather(dim=1, index=idx_tdst.unsqueeze(0).expand(BSZ, -1)) + 1
    )

    repeated_seq_lens = seq_lens[:, ::BLOCK_SIZE_Q].repeat_interleave(HEAD, 0)
    ks_seed = torch.clamp(
        torch.ceil(
            out_indices.shape[-1] * (args.mask_k / (repeated_seq_lens + BLOCK_SIZE_Q))
        ),
        0,
        out_indices.shape[-1],
    )

    indices = out_indices.permute(0, 2, 1, 3).flatten(0, 1)

    active_indices_mask = torch.arange(0, indices.shape[-1], device=q.device)[
        None, None, :
    ] < ks_seed.unsqueeze(-1)
    indices = torch.where(
        active_indices_mask,
        indices,
        MAX_TSRC,
    )
    indices, _ = indices.sort(dim=-1)
    indices = indices // args.block_size_k * args.block_size_k

    # indices = indices.sort(dim=-1).values
    unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
    indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
    indices = indices.sort(dim=-1).values
    # active_mask = unique_mask
    active_mask = indices < (
        position_ids[:, :: args.block_size_q, None].repeat_interleave(HEAD, 0)
        + args.block_size_q
    )
    ks = active_mask.int().sum(-1)
    ks_count = ks.unsqueeze(-1)
    ks_start_end = torch.zeros(
        (ks.shape[0], ks.shape[1], 2), dtype=torch.int32, device=q.device
    )
    ks_start_end[:, :, -1] = ks

    if DEBUG:
        max_query = 1024
        B, TDST, H, HID = q.shape
        TDST = min(TDST, max_query)
        if k is not None:
            _, TSRC, H_KV, _ = k.shape
        else:
            TSRC = torch.max(args.cache_seq_lens).item()
        N = B * H

        def render_mask():
            debug_mask = to_dense(
                indices[:, -triton.cdiv(TDST, args.block_size_q) :].cpu().numpy(),
                ks[:, -triton.cdiv(TDST, args.block_size_q) :].cpu().numpy(),
                None,
                triton.cdiv(N, args.topk_head_group_size),
                TDST,
                TSRC * args.topk_head_group_size,
                args.block_size_q,
                args.block_size_k * args.block_size_k_group,
            )[0]
            if args.group_size_q > 1:
                debug_mask = debug_mask.repeat(axis=0, repeats=args.group_size_q)

            cv2.imwrite("dummy_raw.png", debug_mask.astype(np.uint8) * 255)
            print(
                "saved dummy_raw.png",
                indices.shape,
                ks.shape,
                debug_mask.shape,
                q.shape,
                TSRC,
            )

        render_mask()

    args = args.clone()
    args.sliding_window_size *= 2

    context = block_sparse_attention(
        q=q,
        k=k,
        v=v,
        seq_lens=position_ids + 1,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args,
    )

    return context


@triton.jit
def chunk_controllable_sampling_mask_cuda(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head_kv,
    stride_k_hid,
    INDICES_LEFT,
    stride_indices_left_bsz,
    stride_indices_left_bdst,
    stride_indices_left_head,
    stride_indices_left_chunk,
    INDICES_RIGHT,
    stride_indices_right_bsz,
    stride_indices_right_bdst,
    stride_indices_right_head,
    stride_indices_right_chunk,
    OUT_SCORES,
    stride_out_scores_bsz,
    stride_out_scores_bdst,
    stride_out_scores_head,
    stride_out_scores_chunk,
    CHUNK_COUNT: int,
    TSRC: int,
    TDST: int,
    HEAD: int,
    sliding_window_size: int,
    num_sinks: int,
    BLOCK_HID: tl.constexpr = 128,
    BLOCK_SIZE_Q: tl.constexpr = 32,
    STRIDE_Q: tl.constexpr = 1,
    BLOCK_CHUNK: tl.constexpr = 32,
    HEAD_GROUP: tl.constexpr = 4,
    REDUCE: tl.constexpr = "mean",
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)
    BCHUNK = tl.cdiv(CHUNK_COUNT, BLOCK_CHUNK)

    pid = tl.program_id(0).to(tl.int64)

    idx_head = pid % HEAD
    pid = pid // HEAD
    idx_bdst = pid % BDST
    pid = pid // BDST
    idx_bchunk = pid % BCHUNK
    pid = pid // BCHUNK
    idx_bsz = pid

    # if ((idx_bchunk + 1) * BLOCK_CHUNK * (TSRC / CHUNK_COUNT)) > (idx_bdst * BLOCK_SIZE_Q + TSRC - TDST):
    #     return

    pos_tdst_min = (
        idx_bdst * BLOCK_SIZE_Q + TSRC - TDST - sliding_window_size - num_sinks
    ).to(tl.float32)
    pos_tdst_min = tl.maximum(pos_tdst_min, 0)

    idx_chunk = idx_bchunk * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
    mask_chunk = idx_chunk < CHUNK_COUNT

    idx_tsrc_left = tl.load(
        INDICES_LEFT
        + idx_bsz * stride_indices_left_bsz
        + idx_bdst * stride_indices_left_bdst
        + idx_head * stride_indices_left_head
        + idx_chunk * stride_indices_left_chunk,
        mask=mask_chunk,
        other=TSRC,
    ).to(tl.int32)

    idx_tsrc_right = tl.load(
        INDICES_RIGHT
        + idx_bsz * stride_indices_right_bsz
        + idx_bdst * stride_indices_right_bdst
        + idx_head * stride_indices_right_head
        + idx_chunk * stride_indices_right_chunk,
        mask=mask_chunk,
        other=TSRC,
    ).to(tl.int32)

    max_chunk_size = tl.ceil(TSRC / CHUNK_COUNT).to(tl.float32)

    idx_tdst = (
        idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // STRIDE_Q) * STRIDE_Q
    )
    mask_tdst = idx_tdst < TDST

    idx_hid = tl.arange(0, BLOCK_HID)

    queries = tl.load(
        Q
        + idx_bsz * stride_q_bsz
        + idx_tdst[:, None] * stride_q_tdst
        + idx_head * stride_q_head
        + idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0,
    )

    scores = tl.zeros((BLOCK_CHUNK,), dtype=tl.float16) - 32000.0

    while max_chunk_size > 1:
        max_chunk_size /= 2.0
        mask_tsrc_active = (
            mask_chunk
            & (idx_tsrc_left < idx_tsrc_right)
            & (idx_tsrc_left <= pos_tdst_min)
        )
        idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2

        keys_left = tl.load(
            K
            + idx_bsz * stride_k_bsz
            + ((idx_tsrc_left + idx_tsrc_center) // 2)[None, :] * stride_k_tsrc
            + (idx_head // HEAD_GROUP) * stride_k_head_kv
            + idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc_active[None, :],
            other=0,
        )
        scores_left = tl.dot(
            queries,
            keys_left.to(queries.dtype),
            out_dtype=tl.float32,
        ).to(queries.dtype)
        if REDUCE == "max":
            scores_left = tl.where(mask_tdst[:, None], scores_left, float("-inf"))
            scores_left = tl.max(scores_left, axis=0).to(scores_left.dtype)
        elif REDUCE == "mean":
            scores_left = tl.where(mask_tdst[:, None], scores_left, float("0"))
            scores_left = tl.sum(scores_left, axis=0).to(scores_left.dtype)
            scores_left = (scores_left / tl.sum(mask_tdst.to(tl.float32))).to(
                scores_left.dtype
            )
        else:
            raise Exception()
        scores_left = tl.where(mask_tsrc_active, scores_left, float("-inf")).to(
            scores_left.dtype
        )

        keys_right = tl.load(
            K
            + idx_bsz * stride_k_bsz
            + ((idx_tsrc_center + idx_tsrc_right) // 2)[None, :] * stride_k_tsrc
            + (idx_head // HEAD_GROUP) * stride_k_head_kv
            + idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc_active[None, :],
            other=0,
        )
        scores_right = tl.dot(
            queries,
            keys_right.to(queries.dtype),
            out_dtype=tl.float32,
        ).to(queries.dtype)
        # scores_right = tl.where(mask_tdst[:, None], scores_right, float('-inf'))
        # scores_right = tl.max(scores_right, axis=0).to(scores_right.dtype)
        if REDUCE == "max":
            scores_right = tl.where(mask_tdst[:, None], scores_right, float("-inf"))
            scores_right = tl.max(scores_right, axis=0).to(scores_right.dtype)
        elif REDUCE == "mean":
            scores_right = tl.where(mask_tdst[:, None], scores_right, float("0"))
            scores_right = tl.sum(scores_right, axis=0).to(scores_right.dtype)
            scores_right = (scores_right / tl.sum(mask_tdst.to(tl.float32))).to(
                scores_right.dtype
            )
        else:
            raise Exception()
        scores_right = tl.where(mask_tsrc_active, scores_right, float("-inf")).to(
            scores_right.dtype
        )

        mask_left_win = scores_left > scores_right
        idx_tsrc_left = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                idx_tsrc_left,
                idx_tsrc_center,
            ),
            idx_tsrc_left,
        )

        idx_tsrc_right = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                idx_tsrc_center,
                idx_tsrc_right,
            ),
            idx_tsrc_right,
        )

        scores = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                scores_left,
                scores_right,
            ),
            scores,
        )

    tl.store(
        INDICES_LEFT
        + idx_bsz * stride_indices_left_bsz
        + idx_bdst * stride_indices_left_bdst
        + idx_head * stride_indices_left_head
        + idx_chunk * stride_indices_left_chunk,
        value=idx_tsrc_left,
        mask=mask_chunk,
    )

    tl.store(
        INDICES_RIGHT
        + idx_bsz * stride_indices_right_bsz
        + idx_bdst * stride_indices_right_bdst
        + idx_head * stride_indices_right_head
        + idx_chunk * stride_indices_right_chunk,
        value=idx_tsrc_right,
        mask=mask_chunk,
    )

    tl.store(
        OUT_SCORES
        + idx_bsz * stride_out_scores_bsz
        + idx_bdst * stride_out_scores_bdst
        + idx_head * stride_out_scores_head
        + idx_chunk * stride_out_scores_chunk,
        value=scores,
        mask=mask_chunk,
    )


def dual_stage_quadratic_hip_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    args: HiPAttentionArgs,
    second_stage_k: int = 1024,
    stages=[
        (256, 8192),
        (128, 4096),
        (64, 2048),
    ],
    mask_only=False,
):
    DEBUG_HEAD = -5
    global DEBUG

    BSZ, TDST, HEAD, HID = q.shape
    BSZ, TSRC, HEAD_KV, HID = k.shape
    assert v.shape == k.shape
    # assert TDST == TSRC

    chunk_size = args.mask_k
    chunk_count = triton.cdiv(
        TSRC - args.sink_token_size - args.sliding_window_size, chunk_size
    )
    # assert (chunk_size * chunk_count) == TSRC

    MAX_TDST = TDST
    MAX_TSRC = TSRC

    args = args.clone()
    args.sliding_window_size = max(0, args.sliding_window_size - args.mask_k)

    if args.position_ids is not None:
        position_ids = args.position_ids
    else:
        position_ids = (torch.arange(0, TDST, device=q.device) + (TSRC - TDST))[
            None, :
        ].expand(BSZ, TDST)
    assert position_ids.shape == (BSZ, TDST), position_ids.shape

    BLOCK_CHUNK = args.block_size_k
    BLOCK_SIZE_Q = args.block_size_q
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)

    indices_left = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count),
        device=q.device,
        dtype=torch.int64,
    )
    indices_right = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count),
        device=q.device,
        dtype=torch.int64,
    )

    indices_left[:, :, :, :] = (
        torch.floor(
            torch.arange(0, chunk_count, device=q.device, dtype=torch.float64)
            * chunk_size
            + args.sink_token_size
        ).to(torch.int64)
    )[None, None, None, :]
    indices_right[:, :, :, :] = indices_left + chunk_size
    indices_right.clamp_max_(TSRC - args.sliding_window_size)

    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    out_scores = torch.full(
        (BSZ, BDST, HEAD, triton.next_power_of_2(chunk_count)),
        device=q.device,
        dtype=q.dtype,
        fill_value=-32000.0,
    )

    # print(indices_left[0, -1, 0].sort().values)
    # print(indices_right[0, -1, 0].sort().values)

    VANILLA = True

    grid = (
        BSZ
        * triton.cdiv(chunk_count, BLOCK_CHUNK)
        * triton.cdiv(TDST, BLOCK_SIZE_Q)
        * HEAD,
    )
    chunk_controllable_sampling_mask_cuda[grid](
        q,
        *q.stride(),
        k,
        *k.stride(),
        indices_left,
        *indices_left.stride(),
        indices_right,
        *indices_right.stride(),
        out_scores,
        *out_scores.stride(),
        chunk_count,
        TSRC,
        TDST,
        HEAD,
        args.sliding_window_size - max(BLOCK_SIZE_Q, BLOCK_CHUNK),
        args.sink_token_size,
        BLOCK_HID=q.shape[-1],
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        STRIDE_Q=args.block_stride_q,
        BLOCK_CHUNK=BLOCK_CHUNK,
        HEAD_GROUP=HEAD // HEAD_KV,
    )

    # local_chunk_size = 4
    # local_drop_chunk_count = 1
    # out_scores = out_scores.view(out_scores.shape[0], out_scores.shape[1], out_scores.shape[2], out_scores.shape[3] // local_chunk_size, local_chunk_size)
    # _, t_indices = out_scores.topk(dim=-1, k=local_drop_chunk_count, sorted=False, largest=False)
    # out_scores.scatter_(dim=-1, index=t_indices, value=-32000.0)
    # out_scores = out_scores.flatten(-2, -1)

    pre_conv = out_scores.clone()
    strength = 1.0
    after_conv = torch.nn.functional.conv2d(
        out_scores.view(BSZ * BDST * HEAD, 1, 1, -1),
        weight=torch.tensor(
            [[[[-strength, 2 * strength, -strength]]]],
            device=out_scores.device,
            dtype=out_scores.dtype,
        ).expand(1, 1, 1, 3),
        bias=None,
        stride=1,
        padding="same",
    ).view(BSZ, BDST, HEAD, -1)
    after_conv[..., indices_left.shape[-1] :].fill_(-32000.0)
    after_conv = torch.where(pre_conv < -30000.0, pre_conv, after_conv)

    # print(out_scores[0, -1, DEBUG_HEAD])
    # print(after_conv[0, -1, DEBUG_HEAD])

    if not VANILLA:
        t_indices = after_conv.topk(dim=-1, k=second_stage_k // chunk_size // 8).indices
        out_scores.scatter_(dim=-1, index=t_indices, value=32000.0)
    # t_indices = pre_conv.topk(dim=-1, k=second_stage_k // chunk_size // 2).indices
    # out_scores.scatter_(dim=-1, index=t_indices, value=32000.0)

    _, t_indices = out_scores.sort(dim=-1, descending=True, stable=False)
    indices_left = indices_left.gather(
        dim=-1, index=t_indices[..., : indices_left.shape[-1]]
    )
    indices_right = indices_right.gather(
        dim=-1, index=t_indices[..., : indices_right.shape[-1]]
    )

    if DEBUG:
        out_indices_cpu = indices_left.cpu()
        debug = np.zeros(
            (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK))
        )
        for i in range(out_indices_cpu.shape[1]):
            t_chunk_size = triton.cdiv(TDST, chunk_count * BLOCK_CHUNK)
            # print(i, t_chunk_size)
            for j in range(
                max(0, math.ceil(out_indices_cpu.shape[-1] * (stages[0][1] / TDST)))
            ):
                if j >= out_indices_cpu.shape[-1]:
                    continue
                t = (
                    out_indices_cpu[0, i, DEBUG_HEAD, j] - args.sink_token_size
                ) // BLOCK_CHUNK + args.sink_token_size // BLOCK_CHUNK
                t = t // t_chunk_size * t_chunk_size
                debug[i, t : t + t_chunk_size] = 1
        cv2.imwrite("dummy_sampled.png", debug * 255)
        print("saved dummy_sampled.png")

        debug = np.zeros(
            (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK))
        )
        for i in range(out_indices_cpu.shape[1]):
            t_chunk_size = triton.cdiv(TDST, chunk_count * BLOCK_CHUNK)
            # print(i, t_chunk_size)
            for j in range(
                max(0, math.ceil(out_indices_cpu.shape[-1] * (second_stage_k / TDST)))
            ):
                if j >= out_indices_cpu.shape[-1]:
                    continue
                t = (
                    out_indices_cpu[0, i, DEBUG_HEAD, j] - args.sink_token_size
                ) // BLOCK_CHUNK + args.sink_token_size // BLOCK_CHUNK
                t = t // t_chunk_size * t_chunk_size
                debug[i, t : t + t_chunk_size] = 1
        cv2.imwrite("dummy_sampled_cut.png", debug * 255)
        print("saved dummy_sampled_cut.png")

    for i_stage, (stage_chunk_size, stage_k) in enumerate(stages):
        if stage_chunk_size > chunk_size:
            continue
        if stage_k > TSRC:
            continue

        assert (stage_k % chunk_size) == 0
        indices_left = indices_left[..., : stage_k // chunk_size]
        indices_left = (
            indices_left - args.sink_token_size
        ) // chunk_size * chunk_size + args.sink_token_size  # .clamp_(args.sink_token_size // 2, TSRC)
        indices_right = (
            indices_left + chunk_size
        )  # .clamp_(args.sink_token_size // 2, TSRC)
        # indices_right = indices_right[..., :stage_k // chunk_size]
        # out_scores.fill_(0.0)
        out_scores = out_scores[..., : stage_k // chunk_size]
        # out_scores.fill_(-32000.0)

        indices_left, t_indices = indices_left.sort(dim=-1)
        indices_right = indices_right.gather(dim=-1, index=t_indices)
        out_scores = out_scores.gather(dim=-1, index=t_indices)

        # print(indices_left[0, -1, 0].sort().values)
        # print(indices_right[0, -1, 0].sort().values)

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
                    - torch.arange(0, splits, device=q.device)[
                        None, None, None, None, :
                    ]
                )
                * chunk_sizes[..., None]
            )
            .floor()
            .long()
        )
        indices_right = indices_right.flatten(-2, -1)
        out_scores = out_scores.repeat_interleave(splits, -1)

        # print(indices_left[0, -1, 0].sort().values, splits)
        # print(indices_right[0, -1, 0].sort().values)

        chunk_size = stage_chunk_size
        chunk_count = indices_left.shape[-1]
        BLOCK_CHUNK = triton.next_power_of_2(min(chunk_count, BLOCK_CHUNK))

        grid = (
            BSZ
            * triton.cdiv(chunk_count, BLOCK_CHUNK)
            * triton.cdiv(TDST, BLOCK_SIZE_Q)
            * HEAD,
        )
        chunk_controllable_sampling_mask_cuda[grid](
            q,
            *q.stride(),
            k,
            *k.stride(),
            indices_left,
            *indices_left.stride(),
            indices_right,
            *indices_right.stride(),
            out_scores,
            *out_scores.stride(),
            chunk_count,
            TSRC,
            TDST,
            HEAD,
            args.sliding_window_size - max(BLOCK_SIZE_Q, BLOCK_CHUNK),
            args.sink_token_size,
            BLOCK_HID=q.shape[-1],
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            STRIDE_Q=args.block_stride_q,
            BLOCK_CHUNK=BLOCK_CHUNK,
            HEAD_GROUP=HEAD // HEAD_KV,
        )

        # head_reduced_out_scores = out_scores.sum(dim=-2, keepdim=True)
        # _, t_indices = head_reduced_out_scores.topk(dim=-1, k=second_stage_k // chunk_size // 8)

        # local_chunk_size = 8
        # local_drop_chunk_count = 4
        # out_scores = out_scores.view(out_scores.shape[0], out_scores.shape[1], out_scores.shape[2], out_scores.shape[3] // local_chunk_size, local_chunk_size)
        # _, t_indices = out_scores.topk(dim=-1, k=local_drop_chunk_count, sorted=False, largest=False)
        # out_scores.scatter_(dim=-1, index=t_indices, value=-32000.0)
        # out_scores = out_scores.flatten(-2, -1)

        pre_conv = out_scores.clone()
        strength = 1.0
        after_conv = (
            torch.nn.functional.conv2d(
                (indices_left.to(torch.float32) / TSRC)
                .to(torch.bfloat16)
                .view(BSZ * BDST * HEAD, 1, 1, -1),
                weight=torch.tensor(
                    [
                        [
                            [
                                [
                                    -3 * strength,
                                    -2 * strength,
                                    -strength,
                                    12 * strength,
                                    -strength,
                                    -2 * strength,
                                    -3 * strength,
                                ]
                            ]
                        ]
                    ],
                    device=out_scores.device,
                    dtype=torch.bfloat16,
                ).expand(1, 1, 1, -1),
                bias=None,
                stride=1,
                padding="same",
            )
            .view(BSZ, BDST, HEAD, -1)
            .abs()
        )

        # print(indices_left[0, -1, DEBUG_HEAD])
        # print(after_conv[0, -1, DEBUG_HEAD])

        after_conv[..., indices_left.shape[-1] :].fill_(-32000.0)
        after_conv = torch.where(pre_conv < -30000.0, pre_conv, after_conv)

        if not VANILLA:
            _, t_indices = after_conv.topk(dim=-1, k=second_stage_k // chunk_size // 8)
            out_scores.scatter_add_(
                dim=-1,
                index=t_indices,
                src=torch.tensor(
                    32000.0, device=q.device, dtype=out_scores.dtype
                ).broadcast_to(t_indices.shape),
            )
        # _, t_indices = pre_conv.topk(dim=-1, k=second_stage_k // chunk_size // 2)
        # out_scores.scatter_add_(dim=-1, index=t_indices, src=torch.tensor(32000.0, device=q.device, dtype=out_scores.dtype).broadcast_to(t_indices.shape))

        # out_scores.scatter_add_(dim=-1, index=t_indices, src=torch.tensor(64000.0, device=q.device, dtype=out_scores.dtype).broadcast_to(t_indices.shape))

        _, t_indices = out_scores.sort(dim=-1, descending=True, stable=False)
        indices_left = indices_left.gather(dim=-1, index=t_indices)
        indices_right = indices_right.gather(dim=-1, index=t_indices)

        # print(indices_left[0, -1, 0], t_indices[0, -1, 0])

        if DEBUG:
            out_indices_cpu = indices_left.cpu()
            debug = np.zeros(
                (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q))
            )
            for i in range(out_indices_cpu.shape[1]):
                for j in range(math.ceil(stage_k / chunk_size)):
                    if j >= out_indices_cpu.shape[-1]:
                        continue
                    t = out_indices_cpu[0, i, 7, j] // BLOCK_SIZE_Q
                    debug[i, t : t + triton.cdiv(chunk_size, BLOCK_SIZE_Q)] = 1
            cv2.imwrite(f"dummy_sampled_stage_{i_stage}.png", debug * 255)
            print(f"saved dummy_sampled_stage_{i_stage}.png")

    assert (second_stage_k % chunk_size) == 0
    indices = (
        indices_left[..., : second_stage_k // chunk_size] // chunk_size * chunk_size
    )

    # print(indices[0, -1, 0])

    if DEBUG:
        out_indices_cpu = indices.cpu()
        debug = np.zeros(
            (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q))
        )
        for i in range(out_indices_cpu.shape[1]):
            for j in range(indices.shape[-1]):
                if j >= out_indices_cpu.shape[-1]:
                    continue
                t = out_indices_cpu[0, i, DEBUG_HEAD, j] // BLOCK_SIZE_Q
                debug[i, t : t + 1] = 1
        cv2.imwrite("dummy_sampled_final.png", debug * 255)
        print("saved dummy_sampled_final.png")
        input(">>>")

    args = args.clone()
    args.sliding_window_size += args.mask_k
    # args.sink_token_size += BLOCK_SIZE_Q
    args.block_size_k = chunk_size
    args.mask_k = second_stage_k

    # idx_tdst = torch.arange(0, TDST, device=q.device)
    # seq_lens = position_ids\
    #     .gather(dim=1, index=idx_tdst.unsqueeze(0).expand(BSZ, -1)) + 1

    # repeated_seq_lens = seq_lens[:, ::BLOCK_SIZE_Q].repeat_interleave(HEAD, 0)
    # ks_seed = indices

    # print(ks_seed[0, -1])

    indices = indices.permute(0, 2, 1, 3).flatten(0, 1)

    # active_indices_mask = torch.arange(0, indices.shape[-1], device=q.device)[None, None, :] < ks_seed.unsqueeze(-1)
    # indices = torch.where(
    #     active_indices_mask,
    #     indices,
    #     MAX_TSRC,
    # )
    indices, _ = indices.sort(dim=-1)
    indices = indices // args.block_size_k * args.block_size_k

    # indices = indices.sort(dim=-1).values
    unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
    indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
    indices = indices.sort(dim=-1).values
    # active_mask = unique_mask
    active_mask = indices < (
        position_ids[:, :: args.block_size_q, None].repeat_interleave(HEAD, 0)
        + args.block_size_q
    )
    ks = active_mask.int().sum(-1)
    ks_count = ks.unsqueeze(-1)
    ks_start_end = torch.zeros(
        (ks.shape[0], ks.shape[1], 2), dtype=torch.int32, device=q.device
    )
    ks_start_end[:, :, -1] = ks

    # if DEBUG:
    #     if k is not None:
    #         _, TSRC, H_KV, _ = k.shape
    #     else:
    #         TSRC = torch.max(args.cache_seq_lens).item()
    #     max_query = 8192 * 8192 // TSRC
    #     B, TDST, H, HID = q.shape
    #     TDST = min(TDST, max_query)
    #     N = B * H
    #     def render_mask():
    #         debug_mask = to_dense(
    #             indices[:, -triton.cdiv(TDST, args.block_size_q):].cpu().numpy(),
    #             ks[:, -triton.cdiv(TDST, args.block_size_q):].cpu().numpy(),
    #             None,
    #             triton.cdiv(N, args.topk_head_group_size),
    #             TDST,
    #             TSRC * args.topk_head_group_size,
    #             args.block_size_q,
    #             args.block_size_k * args.block_size_k_group,
    #         )[0]
    #         if args.group_size_q > 1:
    #             debug_mask = debug_mask.repeat(axis=0, repeats=args.group_size_q)

    #         cv2.imwrite('dummy_raw.png', debug_mask.astype(np.uint8) * 255)
    #         print('saved dummy_raw.png', indices.shape, ks.shape, debug_mask.shape, q.shape, TSRC)
    #     render_mask()

    # print(indices[0, -1], ks[0, -1])

    if mask_only:
        return
    else:
        context = block_sparse_attention(
            q=q,
            k=k,
            v=v,
            seq_lens=position_ids + 1,
            indices=indices,
            ks=ks,
            ks_count=ks_count,
            ks_start_end=ks_start_end,
            args=args,
        )

        return context


def dual_stage_quadratic_scan_hip_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    scan_chunk_size: int,
    scan_k: int,
    args: HiPAttentionArgs,
    scan_block_k: int = 64,
    # scan_block_q: int = 64,
    scan_block_stride_q: int = 4,
):
    DEBUG_HEAD = -11
    global DEBUG

    BSZ, TDST, HEAD, HID = q.shape
    BSZ, TSRC, HEAD_KV, HID = k.shape
    assert v.shape == k.shape
    # assert TDST == TSRC

    args = args.clone()
    args.sliding_window_size = max(0, args.sliding_window_size - scan_chunk_size)

    chunk_size = scan_chunk_size
    chunk_count = triton.cdiv(
        TSRC - args.sink_token_size - args.sliding_window_size, chunk_size
    )
    # assert (chunk_size * chunk_count) == TSRC

    MAX_TDST = TDST
    MAX_TSRC = TSRC

    if args.position_ids is not None:
        position_ids = args.position_ids
    else:
        position_ids = (torch.arange(0, TDST, device=q.device) + (TSRC - TDST))[
            None, :
        ].expand(BSZ, TDST)
    assert position_ids.shape == (BSZ, TDST)

    BLOCK_CHUNK = scan_block_k
    BLOCK_SIZE_Q = args.block_size_q
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)

    indices_left = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count),
        device=q.device,
        dtype=torch.int64,
    )
    indices_right = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count),
        device=q.device,
        dtype=torch.int64,
    )

    indices_left[:, :, :, :] = (
        torch.floor(
            torch.arange(0, chunk_count, device=q.device, dtype=torch.float64)
            * chunk_size
            + args.sink_token_size
        ).to(torch.int64)
    )[None, None, None, :]
    indices_right[:, :, :, :] = indices_left + chunk_size
    indices_right.clamp_max_(TSRC - args.sliding_window_size)

    out_scores = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count),
        device=q.device,
        dtype=torch.float16,
    ).fill_(-32000.0)

    # print(indices_left[0, -1, 0, :])
    # print('a', indices_right[0, -1, 0, :])

    grid = (
        BSZ
        * triton.cdiv(chunk_count, BLOCK_CHUNK)
        * triton.cdiv(TDST, BLOCK_SIZE_Q)
        * HEAD,
    )
    chunk_controllable_sampling_mask_cuda[grid](
        q,
        *q.stride(),
        k,
        *k.stride(),
        indices_left,
        *indices_left.stride(),
        indices_right,
        *indices_right.stride(),
        out_scores,
        *out_scores.stride(),
        chunk_count,
        TSRC,
        TDST,
        HEAD,
        args.sliding_window_size - max(BLOCK_SIZE_Q, BLOCK_CHUNK),
        args.sink_token_size,
        BLOCK_HID=q.shape[-1],
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        STRIDE_Q=scan_block_stride_q,
        BLOCK_CHUNK=BLOCK_CHUNK,
        HEAD_GROUP=HEAD // HEAD_KV,
    )

    # print(indices_left[0, -1, 0, :])
    # print('a', indices_right[0, -1, 0, :])

    _, t_indices = out_scores.sort(dim=-1, descending=True, stable=False)
    indices_left = indices_left.gather(dim=-1, index=t_indices)
    # indices_right = indices_right.gather(dim=-1, index=t_indices)
    indices_left = (
        indices_left[:, :, :, : triton.cdiv(scan_k, chunk_size)] - args.sink_token_size
    ) // chunk_size * chunk_size + args.sink_token_size
    indices_right = indices_left + chunk_size

    # print(indices_left.shape)

    if DEBUG:
        out_indices_left_cpu = indices_left.cpu()
        out_indices_right_cpu = indices_right.cpu()
        debug = np.zeros(
            (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK))
        )
        for i in range(out_indices_left_cpu.shape[1]):
            for j in range(out_indices_left_cpu.shape[-1]):
                debug[
                    i,
                    out_indices_left_cpu[0, i, DEBUG_HEAD, j]
                    // BLOCK_SIZE_Q : out_indices_right_cpu[0, i, DEBUG_HEAD, j]
                    // BLOCK_SIZE_Q,
                ] = 1
        cv2.imwrite("dummy_sampled.png", debug * 255)
        print("saved dummy_sampled.png")

    idx_tdst = torch.arange(0, TDST, device=q.device)
    seq_lens = (
        position_ids.gather(dim=1, index=idx_tdst.unsqueeze(0).expand(BSZ, -1)) + 1
    )

    indices_seed = indices_left.permute(0, 2, 1, 3).flatten(0, 1)
    group_size_seed = (indices_right - indices_left).permute(0, 2, 1, 3).flatten(0, 1)

    BK = args.mask_k // args.block_size_k
    indices_seed_padded = torch.full(
        (indices_seed.shape[0], indices_seed.shape[1], BK),
        dtype=indices_seed.dtype,
        device=q.device,
        fill_value=MAX_TSRC * args.block_size_k + 1,
    )
    indices_seed_padded[:, :, : indices_seed.shape[-1]] = indices_seed
    indices_seed, indices_permute = indices_seed_padded.sort(dim=-1)

    group_size_seed_padded = torch.zeros(
        (group_size_seed.shape[0], group_size_seed.shape[1], BK),
        dtype=group_size_seed.dtype,
        device=q.device,
    )
    group_size_seed_padded[:, :, : group_size_seed.shape[-1]] = group_size_seed
    group_size_seed = group_size_seed_padded.gather(dim=-1, index=indices_permute)

    ks_seed = torch.full(
        (indices_seed.shape[0], indices_seed.shape[1]),
        dtype=torch.int32,
        device=q.device,
        fill_value=indices_seed.shape[-1],
    )

    (
        indices,
        ks,
        ks_count,
        ks_start_end,
        scores,
        group_sizes,
        key_access_log,
        key_access_count,
        block_access_log,
        block_access_score,
        block_access_count,
    ) = masking_iteration_draft(
        q=q,
        k=k,
        position_ids=seq_lens,
        indices_tdst=idx_tdst,
        args=args,
        indices_seed=indices_seed,
        ks_seed=ks_seed,
        group_size_seed=group_size_seed // args.block_size_k,
        max_group_size_seed=chunk_size // args.block_size_k,
        scores_seed=None,
    )

    # print(
    #     q.shape, k.shape,
    #     TDST,
    #     TSRC,
    #     seq_lens[0, -1],
    #     idx_tdst[-1],
    #     # indices_seed[DEBUG_HEAD, -1],
    #     indices[DEBUG_HEAD, -1],
    #     ks[DEBUG_HEAD, -1]
    # )

    HIP_DEBUG_MASK_UNIQUE = os.getenv("HIP_DEBUG_MASK_UNIQUE", "0") == "1"
    if HIP_DEBUG_MASK_UNIQUE:
        # indices = indices.sort(dim=-1).values
        unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
        indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
        indices = indices.sort(dim=-1).values
        # active_mask = unique_mask
        active_mask = indices < (
            position_ids[:, :: args.block_size_q, None].repeat_interleave(HEAD, 0)
            + args.block_size_q
        )
        ks = active_mask.int().sum(-1)
        ks_count = ks.unsqueeze(-1)
        ks_start_end[:, :, -1] = ks

    if DEBUG:
        out_indices_cpu = indices.cpu()
        debug = np.zeros(
            (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK))
        )
        for i in range(out_indices_cpu.shape[1]):
            for j in range(out_indices_cpu.shape[-1]):
                debug[
                    i,
                    out_indices_cpu[DEBUG_HEAD, i, j]
                    // BLOCK_SIZE_Q : out_indices_cpu[DEBUG_HEAD, i, j]
                    // BLOCK_SIZE_Q
                    + 1,
                ] = 1
        cv2.imwrite("dummy_sampled_after.png", debug * 128 + 128)
        print("saved dummy_sampled_after.png")

    args.sliding_window_size += scan_chunk_size

    context = block_sparse_attention(
        q=q,
        k=k,
        v=v,
        seq_lens=position_ids + 1,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args,
    )

    return context
