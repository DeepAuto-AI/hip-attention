import math
import os
from typing import List, Optional, Union

import torch
import triton
import triton.language as tl
from torch import Tensor

from hip_attn.v1_0.attention1_block_gpu_kernel.paged_cache_vllm_compat import (
    PagedKeyCacheVllmCompat,
)

if hasattr(tl.math, "round"):
    tl_device_round = tl.math.round
else:
    tl_device_round = tl.extra.cuda.libdevice.round


def next_multiple_of(x: int, multiple_by: int = 16):
    return triton.next_power_of_2(max(x, multiple_by))


@triton.jit
def _triton_kth_ascending(
    scores: tl.tensor,
    k: tl.tensor,
    BLOCK_SCORES: tl.constexpr,
    METHOD: tl.constexpr = "sort",
) -> tl.tensor:
    if METHOD == "sort":
        sorted_score = tl.sort(scores)
        sorted_score_mask = tl.arange(0, BLOCK_SCORES) < k
        kth_ascending_value = tl.max(
            tl.where(sorted_score_mask, sorted_score, -32000.0)
        )
    elif METHOD == "search":
        kth_ascending_value = tl.min(scores)
        step_scale = tl.abs(kth_ascending_value)
        step_size = 0.5
        for i in range(5):
            smaller_count = tl.sum((scores < kth_ascending_value).to(tl.int32))
            if smaller_count > k:
                kth_ascending_value -= step_scale * step_size
            else:
                kth_ascending_value += step_scale * step_size
            step_size *= 0.8
        tl.debug_barrier()
    else:
        raise Exception()
    return kth_ascending_value


@triton.jit
def _masking_iteration_topk(
    # buffers
    QUERIES,
    stride_queries_n,
    stride_queries_tdst,
    stride_queries_hid,
    QUERIES_GROUPED_ROPE,
    KEYS,
    stride_keys_n,
    stride_keys_tsrc,
    stride_keys_hid,
    MASK,
    stride_mask_n,
    stride_mask_bdst,
    stride_mask_src_grid,
    stride_mask_k,
    TMASK,
    stride_tmask_n,
    stride_tmask_bdst,
    stride_tmask_src_grid,
    stride_tmask_k,
    ATTEN_MASK,
    stride_atten_mask_n,
    stride_atten_mask_tsrc,
    SPARQ_INDICES,
    stride_sparq_indices_n,
    stride_sparq_indices_bdst,
    stride_sparq_indices_hid,
    BLOCK_TABLES,
    stride_block_tables_num_seqs,
    stride_block_tables_max_num_blocks_per_seq,
    SCORES,
    stride_scores_n,
    stride_scores_bdst,
    stride_scores_k,
    CONTEXT_LENGTH,
    # local tensors
    idx_n,
    idx_bdst,
    idx_src_grid,
    idx_iteration,
    idx_block_q,
    mask_w,
    mask_block_q,
    k_old_mask,
    k_new,
    w_old,
    w_new,
    t_src,
    context_length,
    loc_idx_start_vec,
    loc_idx_start_origin,
    num_pixels_vec,
    num_pixels_scalar,
    dup_pixels_vec,
    dup_pixels_first,
    # block constant
    IS_CAUSAL,
    USING_SCORE_CACHE: tl.constexpr,
    N_ITERATION,
    T_DST,
    T_SRC,
    KEY_CACHE_METHOD,
    KV_REPEAT_INTERLEAVE,
    REDUCE_METHOD,
    SAMPLING_METHOD,
    GRID_SRC_STRIDE,
    GRID_K_STRIDE,
    USING_SLIDING_WINDOW,
    SLIDING_WINDOW_SIZE,
    HID,
    SPARQ,
    SPARQ_HID,
    BLOCK_MAX_DUP,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_Q_PADDED,
    BLOCK_SIZE_K,
    BLOCK_MASK_K,
    BLOCK_MASK_K_PADDED,
    BLOCK_TMASK_K,
    BLOCK_TMASK_K_PADDED,
    BLOCK_HID,
    # vllm compat
    VLLM_NUM_KV_HEADS,
    VLLM_BLOCK_SIZE,
    VLLM_X,
    stride_keys_vllm_num_blocks,
    stride_keys_vllm_num_kv_heads,
    stride_keys_vllm_head_size_x,
    stride_keys_vllm_block_size,
    stride_keys_vllm_x,
    # rope support
    ROPE_METHOD,
    ROPE_COS,
    stride_rope_cos_idx,
    stride_rope_cos_hid,
    ROPE_SIN,
    stride_rope_sin_idx,
    stride_rope_sin_hid,
    POSITION_IDS,
    stride_position_ids_n,
    stride_position_ids_tdst,
    SELF_EXTEND_SCALE,
    SELF_EXTEND_WINDOW,
):
    """
    # need top_k, so compute scores
    vec_q = queries[i, j, :]
    for k in range(num_pixels):
        # NOTE: nearest
        loc = t_mask[i, j, k]
        vec_k = keys[i, int(loc * t_src), :]

        score = torch.dot(vec_q, vec_k)
        scores[i, j, k] = -score # NOTE: store negative store
    """

    for _idx in tl.static_range(BLOCK_MAX_DUP):
        # _idx = BLOCK_MAX_DUP - _idx - 1
        b_old_fp = tl.cdiv(w_old, BLOCK_SIZE_K).to(tl.float64)
        b_new_fp = tl.cdiv(w_new, BLOCK_SIZE_K).to(tl.float64)
        _value = (loc_idx_start_vec + _idx).to(tl.float64) / b_new_fp
        # _value = loc_idx_start_vec
        if USING_SCORE_CACHE:
            if (idx_iteration > 0) and (
                (idx_iteration < (N_ITERATION - 1)) and (_idx == 0)
            ):
                _mask = (loc_idx_start_origin.to(tl.float64) / b_old_fp) - _value
                _mask = tl.math.abs(_mask)
                _mask = _mask <= ((1.0 / N_ITERATION) / b_new_fp)
                # _value = tl.minimum(1.0, tl.maximum(0.0, _value))
                # tl.device_print('aa', tl.sum(_mask.to(tl.int32)))
                _value *= tl.where(_mask, -1.0, 1.0)

                score_cached = tl.load(
                    SCORES
                    + idx_n * stride_scores_n
                    + idx_bdst * stride_scores_bdst
                    + tl.arange(0, BLOCK_MASK_K) * stride_scores_k,
                )
                tl.store(
                    SCORES
                    + idx_n * stride_scores_n
                    + idx_bdst * stride_scores_bdst
                    + tl.maximum(0, tl.cumsum(_mask.to(tl.int32)) - 1)
                    * stride_scores_k,
                    mask=_mask,
                    value=score_cached,
                )

        idx_tmask = ((num_pixels_vec - dup_pixels_first) + _idx).to(tl.int64)
        tl.store(
            TMASK
            + idx_n * stride_tmask_n
            + idx_bdst * stride_tmask_bdst
            + idx_src_grid * stride_tmask_src_grid
            + idx_tmask * stride_tmask_k,
            mask=mask_w & k_old_mask & (_idx < dup_pixels_vec),
            value=_value,
        )
    tl.debug_barrier()

    assert REDUCE_METHOD == "max"
    scores = tl.full((BLOCK_TMASK_K_PADDED,), float("inf"), dtype=tl.float32)

    idx_tdst = (idx_bdst * BLOCK_SIZE_Q + idx_block_q).to(tl.int64)
    mask_tdst = mask_w & mask_block_q & (idx_tdst < T_DST)

    if ATTEN_MASK is not None:
        query_mask = tl.load(
            ATTEN_MASK
            + idx_n * stride_atten_mask_n
            + (idx_tdst + T_SRC - T_DST) * stride_atten_mask_tsrc,
            mask=mask_tdst,
            other=False,
        ).to(tl.int1)

    num_pixels_range = tl.arange(0, BLOCK_TMASK_K_PADDED).to(tl.int64)
    num_pixels_mask = (
        mask_w
        & (num_pixels_range < num_pixels_scalar)
        & (num_pixels_range < BLOCK_TMASK_K)
        & True
    )
    idx_tsrc_block = tl.load(
        TMASK
        + idx_n * stride_tmask_n
        + idx_bdst * stride_tmask_bdst
        + idx_src_grid * stride_tmask_src_grid
        + num_pixels_range * stride_tmask_k,
        mask=num_pixels_mask,
        other=0,
    )
    # NOTE: random key selection with in the block
    mask_tsrc_block = num_pixels_mask

    if USING_SCORE_CACHE:
        # scores_cached = tl.load(
        #     SCORES +\
        #         idx_n * stride_scores_n +\
        #         idx_bdst * stride_scores_bdst +\
        #         tl.arange(0, BLOCK_MASK) * stride_scores_k,
        #     other=32000.0,
        # ).to(scores.dtype)

        mask_tsrc_block_reuse = idx_tsrc_block < 0
        if (idx_iteration > 0) and (idx_iteration < (N_ITERATION - 1)):
            # mask_tsrc_block = (~mask_tsrc_block_reuse) & mask_tsrc_block
            pass
        idx_tsrc_block = tl.math.abs(idx_tsrc_block)

    idx_tsrc_block = idx_tsrc_block.to(tl.float64)
    if SAMPLING_METHOD == "random":
        # if ((idx_iteration > 0) and (idx_iteration < (N_ITERATION - 1))):
        if (idx_iteration > 0) and (idx_iteration == (N_ITERATION // 2)):
            idx_tsrc_block += tl.random.rand(idx_bdst, idx_tsrc_block) * (
                (0.5 / (idx_iteration + 1)) / (tl.cdiv(w_new, BLOCK_SIZE_K) + 1.0)
            )
    idx_tsrc_block = (idx_tsrc_block * t_src.to(tl.float64)).to(tl.int64)
    idx_tsrc_block = tl.maximum(0, tl.minimum(t_src - 1, idx_tsrc_block))
    idx_tsrc_block = (idx_tsrc_block // BLOCK_SIZE_K) * BLOCK_SIZE_K

    mask_strided_block_q = True  # (idx_block_q % REDUCE_STRDIE) == 0
    if SPARQ:
        hidden_size = SPARQ_HID
        vec_q = tl.zeros(
            (BLOCK_SIZE_Q_PADDED, SPARQ_HID), dtype=QUERIES.dtype.element_ty
        )
        if ROPE_METHOD == "self_extend":
            vec_q_grouped = tl.zeros(
                (BLOCK_SIZE_Q_PADDED, SPARQ_HID), dtype=QUERIES.dtype.element_ty
            )
        need_reload_vec_q = False
    else:
        hidden_size = HID
        vec_q = tl.zeros(
            (BLOCK_SIZE_Q_PADDED, BLOCK_HID), dtype=QUERIES.dtype.element_ty
        )
        if ROPE_METHOD == "self_extend":
            vec_q_grouped = tl.zeros(
                (BLOCK_SIZE_Q_PADDED, BLOCK_HID), dtype=QUERIES.dtype.element_ty
            )
        need_reload_vec_q = hidden_size != BLOCK_HID

    # reuse q
    if not need_reload_vec_q:
        pid_hid = 0
        if SPARQ:
            idx_hid = (tl.arange(0, SPARQ_HID) + pid_hid * SPARQ_HID).to(tl.int64)
            mask_hid = mask_w & (idx_hid < hidden_size)
            idx_hid = tl.load(
                SPARQ_INDICES
                + idx_n * stride_sparq_indices_n
                + idx_bdst * stride_sparq_indices_bdst
                + idx_hid * stride_sparq_indices_hid,
                mask=mask_hid,
                other=HID,
            )
        else:
            idx_hid = (tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID).to(tl.int64)
        # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
        mask_vec_q = mask_tdst & mask_block_q & mask_strided_block_q & True
        if ATTEN_MASK is not None:
            mask_vec_q = mask_vec_q & query_mask
        mask_hid = mask_w & (idx_hid < HID)
        vec_q = tl.load(
            QUERIES
            + idx_n * stride_queries_n
            + idx_tdst[:, None] * stride_queries_tdst
            + idx_hid[None, :] * stride_queries_hid,
            mask=mask_w & mask_vec_q[:, None] & mask_hid[None, :],
            other=0,
        )
        if ROPE_METHOD == "self_extend":
            vec_q_grouped = tl.load(
                QUERIES_GROUPED_ROPE
                + idx_n * stride_queries_n
                + idx_tdst[:, None] * stride_queries_tdst
                + idx_hid[None, :] * stride_queries_hid,
                mask=mask_w & mask_vec_q[:, None] & mask_hid[None, :],
                other=0,
            )

    for _idx_block_k in range(0, BLOCK_SIZE_K):
        scores_partial = tl.zeros(
            (BLOCK_SIZE_Q_PADDED, BLOCK_TMASK_K_PADDED), dtype=tl.float32
        )

        # [BLOCK_TMASK_K_PADDED, ]
        idx_tsrc = (idx_tsrc_block + _idx_block_k).to(tl.int64)
        mask_tsrc = (idx_tsrc < T_SRC) & (_idx_block_k < BLOCK_SIZE_K) & mask_tsrc_block

        # if CONTEXT_LENGTH is not None:
        #     mask_tsrc = mask_tsrc & (idx_tsrc < context_length)

        # [BLOCK_TMASK_K_PADDED, ]
        if ATTEN_MASK is not None:
            key_mask = tl.load(
                ATTEN_MASK
                + idx_n * stride_atten_mask_n
                + idx_tsrc * stride_atten_mask_tsrc,
                mask=mask_tsrc,
                other=False,
            ).to(tl.int1)
        # mask_tsrc = mask_tsrc & key_mask

        for pid_hid in range(
            tl.cdiv(hidden_size, BLOCK_HID if not SPARQ else SPARQ_HID)
        ):
            if SPARQ:
                idx_hid = (tl.arange(0, SPARQ_HID) + pid_hid * SPARQ_HID).to(tl.int64)
                mask_hid = mask_w & (idx_hid < hidden_size)
            else:
                idx_hid = (tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID).to(tl.int64)
                mask_hid = mask_w & (idx_hid < hidden_size)

            if SPARQ:
                idx_hid = tl.load(
                    SPARQ_INDICES
                    + idx_n * stride_sparq_indices_n
                    + idx_bdst * stride_sparq_indices_bdst
                    + idx_hid * stride_sparq_indices_hid,
                    mask=mask_hid,
                    other=HID,
                )
            # mask_hid = idx_hid < hidden_size

            # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
            if need_reload_vec_q:
                # COMPILER BUG!
                assert ROPE_METHOD == "none"

                mask_vec_q = (
                    # mask_hid[None, :] &
                    mask_tdst
                    & mask_block_q
                    & mask_strided_block_q
                    & True
                )
                if ATTEN_MASK is not None:
                    mask_vec_q = mask_vec_q & query_mask
                vec_q = tl.load(
                    QUERIES
                    + idx_n * stride_queries_n
                    + idx_tdst[:, None] * stride_queries_tdst
                    + idx_hid[None, :] * stride_queries_hid,
                    mask=mask_w & mask_vec_q[:, None] & mask_hid[None, :],
                    other=0,
                )
                if ROPE_METHOD == "self_extend":
                    vec_q_grouped = tl.load(
                        QUERIES_GROUPED_ROPE
                        + idx_n * stride_queries_n
                        + idx_tdst[:, None] * stride_queries_tdst
                        + idx_hid[None, :] * stride_queries_hid,
                        mask=mask_w & mask_vec_q[:, None] & mask_hid[None, :],
                        other=0,
                    )

            # [BLOCK_HID: hid, BLOCK_TMASK_K_PADDED: tsrc]
            vec_k_mask = (
                num_pixels_mask
                &
                # mask_hid[:, None] &
                mask_tsrc
                &
                # key_mask[None, :] &
                True
            )
            if CONTEXT_LENGTH is not None:
                vec_k_mask &= idx_tsrc < context_length

            if ROPE_METHOD == "self_extend":
                mask_tsrc_neighbor = tl.zeros((BLOCK_TMASK_K_PADDED,), dtype=tl.int1)

            if KEY_CACHE_METHOD == "cont":
                # [BLOCK_HID: hid, BLOCK_TMASK_K_PADDED: tsrc]
                vec_k = tl.load(
                    KEYS
                    + (idx_n // KV_REPEAT_INTERLEAVE) * stride_keys_n
                    + idx_tsrc[None, :] * stride_keys_tsrc
                    + idx_hid[:, None] * stride_keys_hid,
                    mask=mask_w & vec_k_mask[None, :] & mask_hid[:, None],
                    other=0,
                )

                if ROPE_METHOD == "none":
                    pass
                elif ROPE_METHOD == "self_extend":
                    assert ROPE_SIN is not None
                    assert ROPE_COS is not None
                    assert POSITION_IDS is not None

                    idx_hid_rot = (idx_hid + HID // 2) % HID
                    mask_hid_rot = mask_w & (idx_hid_rot < HID) & mask_hid
                    vec_k_rot = tl.load(
                        KEYS
                        + (idx_n // KV_REPEAT_INTERLEAVE) * stride_keys_n
                        + idx_tsrc[None, :] * stride_keys_tsrc
                        + idx_hid_rot[:, None] * stride_keys_hid,
                        mask=mask_w & vec_k_mask[None, :] & mask_hid_rot[:, None],
                        other=0,
                    )
                    vec_k_rot = tl.where(
                        idx_hid[:, None] < HID // 2, -vec_k_rot, vec_k_rot
                    )

                    idx_last_tdst = idx_bdst * BLOCK_SIZE_Q + T_SRC - T_DST
                    mask_tsrc_neighbor = idx_tsrc >= (
                        idx_last_tdst - SELF_EXTEND_WINDOW
                    )

                    idx_rope = tl.where(
                        mask_tsrc_neighbor, idx_tsrc, idx_tsrc // SELF_EXTEND_SCALE
                    )

                    cos_k = tl.load(
                        ROPE_COS
                        + idx_rope[None, :] * stride_rope_cos_idx
                        + idx_hid[:, None] * stride_rope_cos_hid,
                        mask=mask_w & vec_k_mask[None, :] & mask_hid[:, None],
                        other=0,
                    )
                    sin_k = tl.load(
                        ROPE_SIN
                        + idx_rope[None, :] * stride_rope_sin_idx
                        + idx_hid[:, None] * stride_rope_sin_hid,
                        mask=mask_w & vec_k_mask[None, :] & mask_hid[:, None],
                        other=0,
                    )

                    vec_k = (
                        (vec_k.to(tl.float32) * cos_k)
                        + (vec_k_rot.to(tl.float32) * sin_k)
                    ).to(vec_k.dtype)
                else:
                    raise Exception()
            elif KEY_CACHE_METHOD == "vllm":
                """
                idx_block = block_tables[idx_batch, idx_tsrc // block_size]
                offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
                key = key_cache[idx_block, idx_head, :, offset_block, :].reshape(-1)
                """
                idx_batch = ((idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS).to(
                    tl.int64
                )
                idx_head = ((idx_n // KV_REPEAT_INTERLEAVE) % VLLM_NUM_KV_HEADS).to(
                    tl.int64
                )
                idx_block = tl.load(
                    BLOCK_TABLES
                    + idx_batch * stride_block_tables_num_seqs
                    + (idx_tsrc // VLLM_BLOCK_SIZE)
                    * stride_block_tables_max_num_blocks_per_seq,
                    mask=mask_w & mask_tsrc,
                ).to(tl.int64)
                offset_block = (
                    idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)
                ).to(tl.int64)

                # [BLOCK_HID: hid, BLOCK_TMASK_K_PADDED: tsrc]
                vec_k = tl.load(
                    KEYS
                    + idx_block[None, :] * stride_keys_vllm_num_blocks
                    + idx_head * stride_keys_vllm_num_kv_heads
                    + (idx_hid[:, None] // VLLM_X) * stride_keys_vllm_head_size_x
                    + offset_block[None, :] * stride_keys_vllm_block_size
                    + (idx_hid[:, None] % VLLM_X) * stride_keys_vllm_x,
                    mask=mask_w & vec_k_mask[None, :] & mask_hid[:, None],
                    other=0,
                )

                if vec_k.dtype == tl.uint8:
                    vec_k = vec_k.to(tl.float8e5, bitcast=True).to(vec_q.dtype)

                if ROPE_METHOD == "none":
                    pass
                elif ROPE_METHOD == "self_extend":
                    assert ROPE_SIN is not None
                    assert ROPE_COS is not None
                    assert POSITION_IDS is not None

                    idx_hid_rot = (idx_hid + HID // 2) % HID
                    mask_hid_rot = mask_w & (idx_hid_rot < HID) & mask_hid
                    vec_k_rot = tl.load(
                        KEYS
                        + idx_block[None, :] * stride_keys_vllm_num_blocks
                        + idx_head * stride_keys_vllm_num_kv_heads
                        + (idx_hid_rot[:, None] // VLLM_X)
                        * stride_keys_vllm_head_size_x
                        + offset_block[None, :] * stride_keys_vllm_block_size
                        + (idx_hid_rot[:, None] % VLLM_X) * stride_keys_vllm_x,
                        mask=mask_w & vec_k_mask[None, :] & mask_hid_rot[:, None],
                        other=0,
                    )

                    if vec_k_rot.dtype == tl.uint8:
                        vec_k_rot = vec_k_rot.to(tl.float8e5, bitcast=True).to(
                            vec_q.dtype
                        )

                    vec_k_rot = tl.where(
                        idx_hid[:, None] < HID // 2, -vec_k_rot, vec_k_rot
                    )

                    idx_last_tdst = idx_bdst * BLOCK_SIZE_Q + context_length - T_DST
                    mask_tsrc_neighbor = idx_tsrc >= (
                        idx_last_tdst - SELF_EXTEND_WINDOW
                    )

                    idx_rope = tl.where(
                        mask_tsrc_neighbor, idx_tsrc, idx_tsrc // SELF_EXTEND_SCALE
                    )

                    cos_k = tl.load(
                        ROPE_COS
                        + idx_rope[None, :] * stride_rope_cos_idx
                        + idx_hid[:, None] * stride_rope_cos_hid,
                        mask=mask_w & vec_k_mask[None, :] & mask_hid[:, None],
                        other=0,
                    )
                    sin_k = tl.load(
                        ROPE_SIN
                        + idx_rope[None, :] * stride_rope_sin_idx
                        + idx_hid[:, None] * stride_rope_sin_hid,
                        mask=mask_w & vec_k_mask[None, :] & mask_hid[:, None],
                        other=0,
                    )

                    vec_k = (
                        (vec_k.to(tl.float32) * cos_k)
                        + (vec_k_rot.to(tl.float32) * sin_k)
                    ).to(vec_k.dtype)
                else:
                    raise Exception()
            else:
                raise Exception()

            # [BLOCK_SIZE_PADDED: tdst, BLOCK_TMASK_K_PADDED: tsrc]
            if ROPE_METHOD == "self_extend":
                scores_micro = -tl.where(
                    mask_tsrc_neighbor[None, :],
                    tl.dot(vec_q, vec_k),
                    tl.dot(vec_q_grouped, vec_k),
                )
            elif ROPE_METHOD == "none":
                vec_q_max = tl.maximum(1.0, tl.max(tl.abs(vec_q))).to(tl.float32)
                vec_k_max = tl.maximum(1.0, tl.max(tl.abs(vec_k))).to(tl.float32)
                vec_q_scale = 1.0 / vec_q_max
                vec_k_scale = 1.0 / vec_k_max
                scores_micro = -tl.dot(
                    # always use fp16 to save compuational cost. we do not care accuracy in here
                    (vec_q * vec_q_scale).to(tl.float16),
                    (vec_k * vec_k_scale).to(tl.float16),
                    allow_tf32=True,
                ).to(scores_partial.dtype) * (vec_q_max * vec_k_max)
            else:
                raise Exception()
            scores_partial += scores_micro.to(scores_partial.dtype)

        # [BLOCK_SIZE_PADDED: tdst, BLOCK_TMASK_K_PADDED: tsrc]
        scores_partial_ignore_mask = (
            (~num_pixels_mask[None, :])
            | (~mask_tdst[:, None])
            | (~mask_tsrc[None, :])
            | (~mask_block_q[:, None])
            | (~mask_strided_block_q[:, None])
            |
            # (scores_partial == 0) |
            False
        )

        if IS_CAUSAL:
            if not USING_SLIDING_WINDOW:
                scores_partial_ignore_mask |= (
                    (idx_tdst[:, None] + T_SRC - T_DST) < idx_tsrc[None, :]
                ) | False
            else:
                scores_partial_ignore_mask |= (
                    (idx_tdst[:, None] + T_SRC - T_DST)
                    < (
                        idx_tsrc[None, :]
                        + tl.maximum(
                            0, SLIDING_WINDOW_SIZE - BLOCK_SIZE_Q - BLOCK_SIZE_K
                        )
                    )
                ) | False

        if ATTEN_MASK is not None:
            scores_partial_ignore_mask |= (
                (~key_mask[None, :]) | (~query_mask[:, None]) | False
            )

        if CONTEXT_LENGTH is not None:
            scores_partial_ignore_mask |= idx_tsrc[None, :] >= context_length

        # NOTE: owo powerful dark magic. select first / last block always. testing sink attention.
        # scores_partial_force_mask = (
        #     (
        #         (idx_tsrc[None, :] == 0) |
        #         (num_pixels_range[None, :] >= (num_pixels_scalar - 1)) |
        #         # ((idx_tdst[:, None]) <= idx_tsrc[None, :]) |
        #         False
        #     ) &
        #     ((idx_tdst[:, None] + T_SRC - T_DST) >= idx_tsrc[None, :]) &
        #     (mask_tsrc[None, :] & mask_tdst[:, None]) &
        #     (scores_partial != 0) &
        #     True
        # )
        # scores_partial_force_mask = False
        # scores_partial_ignore_mask = scores_partial_ignore_mask & (~scores_partial_force_mask)

        # NOTE: reduce
        scores_partial = scores_partial + scores_partial_ignore_mask * 32000.0
        # scores_partial = scores_partial + scores_partial_force_mask * (-32000.0)
        scores_partial = tl.min(scores_partial, axis=0)
        scores = tl.minimum(scores, scores_partial)

    if USING_SCORE_CACHE:
        if (idx_iteration > 0) and (idx_iteration < (N_ITERATION - 1)):
            idx_cache_score = tl.maximum(0, tl.cumsum(mask_tsrc_block_reuse) - 1).to(
                tl.int64
            )
            tl.debug_barrier()
            scores_ignored = (scores > 10000.0) & mask_tsrc_block_reuse
            scores_ignored = mask_tsrc_block_reuse
            scores_cached = tl.load(
                SCORES
                + idx_n * stride_scores_n
                + idx_bdst * stride_scores_bdst
                + idx_cache_score * stride_scores_k,
                mask=scores_ignored,
                other=32000.0,
            ).to(scores.dtype)
            scores = tl.minimum(scores_cached, scores)
            tl.debug_barrier()

    # done compute reduced scores

    """
    _, topk_indices = torch.topk(scores[i, j, :num_pixels], k=k_new, largest=False)
    for k in range(k_new):
        mask[i, j, k] = t_mask[i, j, topk_indices[k]]
    """

    # tl.device_print("", scores)

    # select min-k from negative scores -> select top-k
    masked_scores = scores

    kth = k_new
    scores_kth_large = _triton_kth_ascending(masked_scores, kth, BLOCK_TMASK_K_PADDED)
    # scores_avg = tl.sum(masked_scores * (masked_scores < 1.0)) / num_pixels_scalar
    # scores_min = tl.min(masked_scores)
    # scores_kth_large = scores_avg # - (scores_min * 0.1)
    topk_mask = masked_scores <= scores_kth_large

    topk_mask_cumsum = tl.cumsum(topk_mask.to(tl.int64))
    topk_range = tl.maximum(
        tl.minimum((topk_mask_cumsum - 1) * topk_mask, kth - 1), 0
    ).to(tl.int64)

    temp_range = tl.arange(0, BLOCK_TMASK_K_PADDED).to(tl.int64)
    temp_mask = mask_w & (temp_range < num_pixels_scalar) & (temp_range < BLOCK_TMASK_K)
    temp = tl.load(
        TMASK
        + idx_n * stride_tmask_n
        + idx_bdst * stride_tmask_bdst
        + idx_src_grid * stride_tmask_src_grid
        + temp_range * stride_tmask_k,
        mask=temp_mask,
        other=0,
    )
    tl.store(
        MASK
        + idx_n * stride_mask_n
        + idx_bdst * stride_mask_bdst
        + idx_src_grid * stride_mask_src_grid
        + topk_range * stride_mask_k,
        mask=topk_mask & temp_mask,
        value=temp,
        # value=0.1,
    )
    if USING_SCORE_CACHE:
        tl.store(
            SCORES
            + idx_n * stride_scores_n
            + idx_bdst * stride_scores_bdst
            + tl.arange(0, BLOCK_MASK_K_PADDED) * stride_scores_k,
            mask=mask_w,
            value=32000.0,
        )
        tl.store(
            SCORES
            + idx_n * stride_scores_n
            + idx_bdst * stride_scores_bdst
            + topk_range * stride_scores_k,
            # mask=mask_w & topk_mask & temp_mask,
            mask=mask_w & topk_mask & (~mask_tsrc_block_reuse),
            value=scores,
        )
    # tl.debug_barrier()


# @triton.autotune(
#     configs=[
#         triton.Config(kwargs={}, num_warps=16),
#         triton.Config(kwargs={}, num_warps=8, num_stages=1),
#         triton.Config(kwargs={}, num_warps=4, num_stages=1),
#         triton.Config(kwargs={}, num_warps=2, num_stages=1),
#     ],
#     key=['BLOCK_MASK_K'],
#     warmup=2,
#     rep=20,
# )
@triton.jit
def _masking_iteration_compute(
    # input matrices
    QUERIES,
    stride_queries_n,
    stride_queries_tdst,
    stride_queries_hid,
    QUERIES_GROUPED_ROPE,
    KEYS,
    stride_keys_n,
    stride_keys_tsrc,
    stride_keys_hid,
    ATTEN_MASK,
    stride_atten_mask_n,
    stride_atten_mask_tsrc,
    SPARQ_INDICES,
    stride_sparq_indices_n,
    stride_sparq_indices_bdst,
    stride_sparq_indices_hid,
    # input / temp metrices (blocked)
    MASK,
    stride_mask_n,
    stride_mask_bdst,
    stride_mask_src_grid,
    stride_mask_k,
    TMASK,
    stride_tmask_n,
    stride_tmask_bdst,
    stride_tmask_src_grid,
    stride_tmask_k,
    # temp vectors (blocked)
    WS,
    stride_ws_n,
    stride_ws_bdst,
    KS,
    stride_ks_n,
    stride_ks_bdst,
    WS_OUT,
    stride_ws_out_n,
    stride_ws_out_bdst,
    KS_OUT,
    stride_ks_out_n,
    stride_ks_out_bdst,
    stride_ks_out_src_grid,
    TSRCS,
    stride_tsrcs_n,
    stride_tsrcs_bdst,
    SCORES,
    stride_scores_n,
    stride_scores_bdst,
    stride_scores_k,
    # operation variables (blocked)
    SCALE_UP: tl.constexpr,
    N_PATCHES: tl.constexpr,
    MASK_K: tl.constexpr,
    TMASK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    # input variables
    KV_REPEAT_INTERLEAVE: int,
    N: int,
    T_DST: int,
    T_SRC: int,
    B_DST: int,
    B_SRC: int,
    HID: tl.constexpr,
    SPARQ_HID: tl.constexpr,
    SPARQ_HID_HALF: tl.constexpr,
    N_COMPLETED: int,
    N_ITERATION: int,
    # vLLM compat inputs
    stride_keys_vllm_num_blcoks,
    stride_keys_vllm_num_kv_heads,
    stride_keys_vllm_head_size_x,
    stride_keys_vllm_block_size,
    stride_keys_vllm_x,
    VLLM_NUM_BLOCKS: int,
    VLLM_NUM_KV_HEADS: int,
    VLLM_HEAD_SIZE_X: int,
    VLLM_BLOCK_SIZE: tl.constexpr,
    VLLM_X: int,
    VLLM_HEAD_SIZE: int,
    BLOCK_TABLES,
    stride_block_tables_num_seqs,
    stride_block_tables_max_num_blocks_per_seq,
    CONTEXT_LENGTH,
    stride_context_length_num_seqs,
    # rope methods
    ROPE_METHOD: tl.constexpr,
    ROPE_COS,
    stride_rope_cos_idx,
    stride_rope_cos_hid,
    ROPE_SIN,
    stride_rope_sin_idx,
    stride_rope_sin_hid,
    POSITION_IDS,
    stride_position_ids_n,
    stride_position_ids_tdst,
    SELF_EXTEND_SCALE,
    SELF_EXTEND_WINDOW,
    # dynamic k per query
    MAX_KS,
    stride_max_ks_n,
    stride_max_ks_bdst,
    SELECTED_MAX_KS: tl.constexpr,
    # block constant
    USING_SCORE_CACHE: tl.constexpr,
    KEY_CACHE_METHOD: tl.constexpr,
    SPARQ: tl.constexpr,
    REDUCE_METHOD: tl.constexpr,
    BLOCK_MASK_K: tl.constexpr,
    BLOCK_MASK_K_PADDED: tl.constexpr,
    BLOCK_TMASK_K: tl.constexpr,
    BLOCK_TMASK_K_PADDED: tl.constexpr,
    BLOCK_MASK_K_HALF: tl.constexpr,
    BLOCK_MASK_K_HALF_PADDED: tl.constexpr,
    BLOCK_TMASK_K_HALF: tl.constexpr,
    BLOCK_TMASK_K_HALF_PADDED: tl.constexpr,
    BLOCK_MAX_DUP: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    REDUCE_STRDIE: tl.constexpr,
    SAMPLING_METHOD: tl.constexpr,
    GRID_SRC_STRIDE: tl.constexpr,
    GRID_K_STRIDE: tl.constexpr,
    USING_SLIDING_WINDOW: tl.constexpr,
    SLIDING_WINDOW_SIZE: tl.constexpr,
):
    idx_n = tl.program_id(2).to(tl.int64)

    idx_bdst = tl.program_id(1).to(tl.int64) + N_COMPLETED

    idx_src_grid = tl.program_id(0).to(tl.int64)

    """ non blocked
    # for each query
    w_old = ws[i, j, 0]
    t_src = t_srcs[i, j, 0]
    w_new = min(torch.round(w_old * scale_up), t_src)
    """

    if CONTEXT_LENGTH is not None:
        context_length = tl.load(
            CONTEXT_LENGTH
            + ((idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS)
            * stride_context_length_num_seqs,
        ).to(tl.int64)
    else:
        context_length = None

    w_old = tl.load(
        WS + idx_n * stride_ws_n + idx_bdst * stride_ws_bdst,
    )

    t_src = tl.load(
        TSRCS + idx_n * stride_tsrcs_n + idx_bdst * stride_tsrcs_bdst,
    )
    if CONTEXT_LENGTH is not None:
        t_src = tl.minimum(context_length, t_src)

    if MAX_KS is not None:
        max_k = tl.load(
            MAX_KS + idx_n * stride_max_ks_n + idx_bdst * stride_max_ks_bdst,
        ).to(tl.int64)

        if max_k != SELECTED_MAX_KS:
            return

    k_old = tl.load(
        KS + idx_n * stride_ks_n + idx_bdst * stride_ks_bdst,
    ).to(tl.int64)

    for idx_iteration in range(N_ITERATION):
        tl.debug_barrier()
        # tl.device_print("dd", idx_bdst)

        w_new = tl.minimum(
            tl_device_round(w_old.to(tl.float64) * SCALE_UP).to(tl.float64), t_src
        ).to(tl.int64)

        """
        if w_old != w_new:
        """
        tl.debug_barrier()
        mask_w = w_old != w_new
        tl.debug_barrier()

        if mask_w:
            """
            k_old = ks[i, j, 0]
            k_new = max(n_patches, int(min(mask_k * BLOCK_SIZE / t_src, 1.0) * w_new) c/ BLOCK_SIZE)
            k_new = min(t_src c/ BLOCK_SIZE, max(n_patches, k_new))
            """

            # """
            k_new = tl.maximum(
                N_PATCHES,
                (
                    tl.minimum(
                        MASK_K / tl.cdiv(t_src, BLOCK_SIZE_K).to(tl.float64), 1.0
                    )
                    * tl.cdiv(w_new, BLOCK_SIZE_K)
                ).to(tl.int64),
            )
            # """
            # k_new = tl.maximum(
            #     N_PATCHES,
            #     tl.cdiv(
            #         (tl.minimum((MASK_K * BLOCK_SIZE).to(tl.float32) / t_src.to(tl.float32), 1.0) * w_new.to(tl.float32)).to(tl.int64),
            #         BLOCK_SIZE
            #     ),
            # )
            # tl.device_print("before", t_src)
            k_new = tl.minimum(
                tl.cdiv(t_src, BLOCK_SIZE_K * GRID_SRC_STRIDE),
                k_new,
            )
            if MAX_KS is not None:
                if idx_iteration > 0:
                    k_new = tl.minimum(k_new, max_k)

            """
            # mask -> t_mask
            num_pixels = 0
            for k in range(k_old):
                loc = mask[i, j, k]
                loc_idx_start = int(loc * w_old)
                loc_idx_end = loc_idx_start + 1
                loc_idx_start = int(loc_idx_start / w_old * w_new)
                loc_idx_end = int(loc_idx_end / w_old * w_new)
                dup_pixels = loc_idx_end - loc_idx_start
                for l in range(dup_pixels):
                    t_mask[i, j, num_pixels + l] = (loc_idx_start + l) / w_new
                num_pixels += dup_pixels
            """

            k_old_range = tl.arange(0, BLOCK_MASK_K_PADDED).to(tl.int64)
            k_old_mask = (k_old_range < k_old) & (k_old_range < BLOCK_MASK_K) & True
            # tl.debug_barrier()
            loc_vec = tl.load(
                MASK
                + idx_n * stride_mask_n
                + idx_bdst * stride_mask_bdst
                + idx_src_grid * stride_mask_src_grid
                + k_old_range * stride_mask_k,
                mask=mask_w & k_old_mask,
                other=0,
            )
            k_old_mask = k_old_mask & (loc_vec < 1.0)
            tl.debug_barrier()

            # w_old_fp = w_old.to(tl.float32)
            # w_new_fp = w_new.to(tl.float32)
            b_old_fp = tl.cdiv(w_old, BLOCK_SIZE_K).to(tl.float64)
            b_new_fp = tl.cdiv(w_new, BLOCK_SIZE_K).to(tl.float64)
            loc_idx_start_vec = (loc_vec * b_old_fp).to(tl.int64)
            # tl.device_print('aa', loc_vec)
            # tl.device_print('aa', (loc_vec * b_old_fp))
            loc_idx_start_origin = loc_idx_start_vec
            loc_idx_end_vec = loc_idx_start_vec + 1
            loc_idx_start_vec = (
                loc_idx_start_vec.to(tl.float64) * (b_new_fp / b_old_fp)
            ).to(tl.int64)
            loc_idx_end_vec = (
                loc_idx_end_vec.to(tl.float64) * (b_new_fp / b_old_fp)
            ).to(tl.int64)

            dup_pixels_vec = loc_idx_end_vec - loc_idx_start_vec
            dup_pixels_vec = dup_pixels_vec * k_old_mask
            num_pixels_vec = tl.cumsum(dup_pixels_vec)
            dup_pixels_first = tl.min(num_pixels_vec)
            num_pixels_scalar = tl.max(num_pixels_vec)

            # num_pixels_scalar_exceed = tl.maximum(num_pixels_scalar - tl.cdiv(TMASK_K, grid_kstride), 0)
            # num_pixels_vec = tl.maximum(0, num_pixels_vec - num_pixels_scalar_exceed)
            dup_pixels_first = tl.min(num_pixels_vec)
            num_pixels_scalar = tl.max(num_pixels_vec)

            # NOTE: hey?
            # loc_idx_start_vec = loc_vec * b_new_fp
            # loc_idx_start_vec = BLOCK_SIZE_K * b_new_fp

            # NOTE: compiler bug?

            """
            dup_pixels_range = tl.arange(0, BLOCK_MAX_DUP)
            dup_pixels_mask = (dup_pixels_range[None, :] <= dup_pixels_vec[:, None]) & k_old_mask[:, None]

            tl.store(
                TMASK + \
                    idx_n * stride_tmask_n +\
                    idx_bdst * stride_tmask_bdst +\
                    ((num_pixels_vec - dup_pixels_first)[:, None] + dup_pixels_range[None, :]) * stride_tmask_k,
                mask=dup_pixels_mask,
                value=(
                    (loc_idx_start_vec[:, None] + tl.arange(0, BLOCK_MAX_DUP)[None, :]).to(tl.float32) / w_new.to(tl.float32)
                )
                # value = num_pixels_scalar=
            )
            """

            # interp_loc_vec_padded = (loc_idx_start_vec[:, None] + tl.arange(0, BLOCK_MAX_DUP)[None, :]).to(tl.float32) / w_new.to(tl.float32)
            # mask_interp_loc_vec_padded = tl.arange(0, BLOCK_MAX_DUP)[None, :] < dup_pixels_vec[:, None]
            # interp_loc_vec_padded = tl.reshape(interp_loc_vec_padded, BLOCK_MASK_K * BLOCK_MAX_DUP)

            # idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
            # mask_block_k = idx_block_k < BLOCK_SIZE_K
            idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED).to(tl.int64) * REDUCE_STRDIE
            if BLOCK_SIZE_Q_PADDED == BLOCK_SIZE_Q:
                mask_block_q = True
            else:
                mask_block_q = idx_block_q < BLOCK_SIZE_Q

            """
            # t_mask -> mask (using scores)
            if k_new < num_pixels:
            """
            if ((k_new < num_pixels_scalar) or False) or (REDUCE_STRDIE > 1):
                # if (idx_iteration == 0) or (idx_iteration == (N_ITERATION - 1)):
                if idx_iteration == 0:
                    # first iteration should use
                    # - full block_tmask_k
                    # - half sparq hid
                    _masking_iteration_topk(
                        # buffers
                        QUERIES,
                        stride_queries_n,
                        stride_queries_tdst,
                        stride_queries_hid,
                        QUERIES_GROUPED_ROPE,
                        KEYS,
                        stride_keys_n,
                        stride_keys_tsrc,
                        stride_keys_hid,
                        MASK,
                        stride_mask_n,
                        stride_mask_bdst,
                        stride_mask_src_grid,
                        stride_mask_k,
                        TMASK,
                        stride_tmask_n,
                        stride_tmask_bdst,
                        stride_tmask_src_grid,
                        stride_tmask_k,
                        ATTEN_MASK,
                        stride_atten_mask_n,
                        stride_atten_mask_tsrc,
                        SPARQ_INDICES,
                        stride_sparq_indices_n,
                        stride_sparq_indices_bdst,
                        stride_sparq_indices_hid,
                        BLOCK_TABLES,
                        stride_block_tables_num_seqs,
                        stride_block_tables_max_num_blocks_per_seq,
                        SCORES,
                        stride_scores_n,
                        stride_scores_bdst,
                        stride_scores_k,
                        CONTEXT_LENGTH,
                        # local tensors
                        idx_n,
                        idx_bdst,
                        idx_src_grid,
                        idx_iteration,
                        idx_block_q,
                        mask_w,
                        mask_block_q,
                        k_old_mask,
                        k_new,
                        w_old,
                        w_new,
                        t_src,
                        context_length,
                        loc_idx_start_vec,
                        loc_idx_start_origin,
                        num_pixels_vec,
                        num_pixels_scalar,
                        dup_pixels_vec,
                        dup_pixels_first,
                        # block constant
                        IS_CAUSAL,
                        USING_SCORE_CACHE,
                        N_ITERATION,
                        T_DST,
                        T_SRC,
                        KEY_CACHE_METHOD,
                        KV_REPEAT_INTERLEAVE,
                        REDUCE_METHOD,
                        SAMPLING_METHOD,
                        GRID_SRC_STRIDE,
                        GRID_K_STRIDE,
                        USING_SLIDING_WINDOW,
                        SLIDING_WINDOW_SIZE,
                        HID,
                        SPARQ,
                        SPARQ_HID_HALF,  # NOTE: this hurt accuracy little
                        BLOCK_MAX_DUP,
                        BLOCK_SIZE_Q,
                        BLOCK_SIZE_Q_PADDED,
                        BLOCK_SIZE_K,
                        BLOCK_MASK_K,
                        BLOCK_MASK_K_PADDED,
                        BLOCK_TMASK_K,
                        BLOCK_TMASK_K_PADDED,
                        BLOCK_HID,
                        VLLM_NUM_KV_HEADS,
                        VLLM_BLOCK_SIZE,
                        VLLM_X,
                        stride_keys_vllm_num_blcoks,
                        stride_keys_vllm_num_kv_heads,
                        stride_keys_vllm_head_size_x,
                        stride_keys_vllm_block_size,
                        stride_keys_vllm_x,
                        ROPE_METHOD,
                        ROPE_COS,
                        stride_rope_cos_idx,
                        stride_rope_cos_hid,
                        ROPE_SIN,
                        stride_rope_sin_idx,
                        stride_rope_sin_hid,
                        POSITION_IDS,
                        stride_position_ids_n,
                        stride_position_ids_tdst,
                        SELF_EXTEND_SCALE,
                        SELF_EXTEND_WINDOW,
                    )
                else:
                    # otherwise
                    # - use half block_tmask_k
                    # - use full sparq_hid
                    _masking_iteration_topk(
                        # buffers
                        QUERIES,
                        stride_queries_n,
                        stride_queries_tdst,
                        stride_queries_hid,
                        QUERIES_GROUPED_ROPE,
                        KEYS,
                        stride_keys_n,
                        stride_keys_tsrc,
                        stride_keys_hid,
                        MASK,
                        stride_mask_n,
                        stride_mask_bdst,
                        stride_mask_src_grid,
                        stride_mask_k,
                        TMASK,
                        stride_tmask_n,
                        stride_tmask_bdst,
                        stride_tmask_src_grid,
                        stride_tmask_k,
                        ATTEN_MASK,
                        stride_atten_mask_n,
                        stride_atten_mask_tsrc,
                        SPARQ_INDICES,
                        stride_sparq_indices_n,
                        stride_sparq_indices_bdst,
                        stride_sparq_indices_hid,
                        BLOCK_TABLES,
                        stride_block_tables_num_seqs,
                        stride_block_tables_max_num_blocks_per_seq,
                        SCORES,
                        stride_scores_n,
                        stride_scores_bdst,
                        stride_scores_k,
                        CONTEXT_LENGTH,
                        # local tensors
                        idx_n,
                        idx_bdst,
                        idx_src_grid,
                        idx_iteration,
                        idx_block_q,
                        mask_w,
                        mask_block_q,
                        k_old_mask,
                        k_new,
                        w_old,
                        w_new,
                        t_src,
                        context_length,
                        loc_idx_start_vec,
                        loc_idx_start_origin,
                        num_pixels_vec,
                        num_pixels_scalar,
                        dup_pixels_vec,
                        dup_pixels_first,
                        # block constant
                        IS_CAUSAL,
                        USING_SCORE_CACHE,
                        N_ITERATION,
                        T_DST,
                        T_SRC,
                        KEY_CACHE_METHOD,
                        KV_REPEAT_INTERLEAVE,
                        REDUCE_METHOD,
                        SAMPLING_METHOD,
                        GRID_SRC_STRIDE,
                        GRID_K_STRIDE,
                        USING_SLIDING_WINDOW,
                        SLIDING_WINDOW_SIZE,
                        HID,
                        SPARQ,
                        SPARQ_HID,
                        BLOCK_MAX_DUP,
                        BLOCK_SIZE_Q,
                        BLOCK_SIZE_Q_PADDED,
                        BLOCK_SIZE_K,
                        BLOCK_MASK_K,
                        BLOCK_MASK_K_PADDED,
                        BLOCK_TMASK_K_HALF,
                        BLOCK_TMASK_K_HALF_PADDED,
                        BLOCK_HID,
                        VLLM_NUM_KV_HEADS,
                        VLLM_BLOCK_SIZE,
                        VLLM_X,
                        stride_keys_vllm_num_blcoks,
                        stride_keys_vllm_num_kv_heads,
                        stride_keys_vllm_head_size_x,
                        stride_keys_vllm_block_size,
                        stride_keys_vllm_x,
                        ROPE_METHOD,
                        ROPE_COS,
                        stride_rope_cos_idx,
                        stride_rope_cos_hid,
                        ROPE_SIN,
                        stride_rope_sin_idx,
                        stride_rope_sin_hid,
                        POSITION_IDS,
                        stride_position_ids_n,
                        stride_position_ids_tdst,
                        SELF_EXTEND_SCALE,
                        SELF_EXTEND_WINDOW,
                    )
            else:
                """
                else:
                    mask[i, j, :num_pixels] = t_mask[i, j, :num_pixels]
                """
                for _idx in range(BLOCK_MAX_DUP):
                    idx_mask_out = ((num_pixels_vec - dup_pixels_first) + _idx).to(
                        tl.int64
                    )
                    mask_mask_out = (
                        (idx_mask_out < BLOCK_MASK_K)
                        & (idx_mask_out < num_pixels_scalar)
                        & (_idx <= dup_pixels_vec)
                        & k_old_mask
                    )
                    value_mask_out = (loc_idx_start_vec + _idx).to(tl.float64)
                    value_mask_out = value_mask_out / tl.cdiv(w_new, BLOCK_SIZE_K).to(
                        tl.float64
                    )

                    tl.store(
                        MASK
                        + idx_n * stride_mask_n
                        + idx_bdst * stride_mask_bdst
                        + idx_src_grid * stride_mask_src_grid
                        + idx_mask_out * stride_mask_k,
                        mask=mask_w & mask_mask_out,
                        value=value_mask_out,
                    )
                tl.debug_barrier()

            """
            ws[i, j, 0] = w_new
            ks[i, j, 0] = min(k_new, num_pixels)
            """
            w_old = w_new
            k_old = tl.minimum(k_new, num_pixels_scalar)

            t_w_new = tl.minimum(
                tl_device_round(w_old.to(tl.float64) * SCALE_UP).to(tl.float64), t_src
            ).to(tl.int64)

            if t_w_new == w_old:
                if idx_src_grid == 0:
                    tl.store(
                        WS_OUT
                        + idx_n * stride_ws_out_n
                        + idx_bdst * stride_ws_out_bdst,
                        # mask = mask_w,
                        value=w_old,
                    )
                tl.store(
                    KS_OUT
                    + idx_n * stride_ks_out_n
                    + idx_bdst * stride_ks_out_bdst
                    + idx_src_grid * stride_ks_out_src_grid,
                    # mask = mask_w,
                    value=k_old,
                )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos  # [seq_len, dim]
    sin = sin  # [seq_len, dim]
    assert cos.ndim == 2
    cos = cos[position_ids]
    sin = sin[position_ids]  # [bs, 1, seq_len, dim]
    assert position_ids.ndim == 2
    assert cos.ndim == 3
    q_embed = (q * cos) + (rotate_half(q) * sin)

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


def masking_iteration(
    # input matrices
    queries: Tensor,
    keys: Union[Tensor, "PagedKeyCacheVllmCompat"],
    attention_mask: Tensor,
    # input metrices (blocked)
    mask: Tensor,
    t_mask: Tensor,
    sparq_indices,
    sparq_indices_strides,
    # temp vectors (blocked)
    ws: Tensor,
    ks: Tensor,
    t_srcs: Tensor,
    # operator variables
    scale_up: float,
    n_patches: int,
    mask_k: int,
    is_causal: bool,
    # iteration controls
    i_iteration: int,
    n_iteration: int,
    # rope_config
    ROPE_METHOD: str,
    ROPE_COS: Optional[Tensor],
    ROPE_SIN: Optional[Tensor],
    POSITION_IDS: Optional[Tensor],
    SELF_EXTEND_SCALE: int,
    SELF_EXTEND_WINDOW: int,
    # dynamic k per query
    maximum_ks: Optional[Tensor],
    maximum_ks_config: Optional[List[int]],
    # input constant
    KV_REPEAT_INTERLEAVE: int,
    N: int,
    T_DST: int,
    T_SRC: int,
    B_DST: int,
    B_SRC: int,
    HID: int,
    SPARQ: bool,
    SPARQ_HID: int,
    N_COMPLETED: int,
    # kernel constant
    BLOCK_SIZE_Q: int,
    BLOCK_SIZE_K: int,
    REDUCE_METHOD: str,
    REDUCE_STRIDE: int,
    SAMPLING_METHOD: str,
    GRID_SRC_STRIDE: int,
    GRID_K_STRIDE: int,
    USING_SLIDING_WINDOW: bool,
    SLIDING_WINDOW_SIZE: int,
    DEBUG: bool = False,
):
    if DEBUG:
        # print(ws)
        # print(ks[0, 10])
        # print(mask[0, 10])
        # print(t_srcs)
        print(
            "masking_iteration",
            queries.shape,
            queries.data_ptr(),
            keys.shape,
            keys.data_ptr(),
            mask.shape,
            mask.data_ptr(),
            t_mask.shape,
            t_mask.data_ptr(),
            ws.shape,
            ws.data_ptr(),
            ks.shape,
            ks.data_ptr(),
            t_srcs.shape,
            t_srcs.data_ptr(),
            N,
            T_DST,
            T_SRC,
            B_DST,
            B_SRC,
            HID,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
            REDUCE_METHOD,
            GRID_SRC_STRIDE,
            GRID_K_STRIDE,
        )
        K = mask.shape[-1]
        assert t_srcs.min() > 0
        assert t_srcs.max() <= T_SRC
        assert ks.min() >= 0
        assert ks.max() <= K
        assert keys.shape[1] == T_SRC
        assert queries.shape[1] == T_DST
        assert mask.min() >= 0
        # assert mask.max() < 1
        assert t_mask.min() >= 0
        # assert t_mask.max() < 1

    if ROPE_METHOD == "self_extend":
        q_scale = 1 / math.sqrt(HID)

        queries_neighbor = (
            apply_rotary_pos_emb(
                queries / q_scale, None, ROPE_COS, ROPE_SIN, POSITION_IDS
            )[0]
            * q_scale
        )
        queries_grouped = (
            apply_rotary_pos_emb(
                queries / q_scale,
                None,
                ROPE_COS,
                ROPE_SIN,
                POSITION_IDS // SELF_EXTEND_SCALE
                + SELF_EXTEND_WINDOW
                - SELF_EXTEND_WINDOW // SELF_EXTEND_SCALE,
            )[0]
            * q_scale
        )
        queries = queries_neighbor
        # queries_grouped = queries_neighbor
        assert queries.stride() == queries_grouped.stride()
    else:
        queries_grouped = None

    BLOCK_MASK_K = triton.next_power_of_2(mask.shape[-1])
    BLOCK_TMASK_K = triton.next_power_of_2(t_mask.shape[-1])
    # print(BLOCK_MASK_K, BLOCK_TMASK_K)

    # if i_iteration == 0 or i_iteration == (n_iteration - 1):
    #     pass
    # else:
    #     if i_iteration > 1:
    #         BLOCK_MASK_K = BLOCK_MASK_K // scale_up
    #     BLOCK_TMASK_K = BLOCK_TMASK_K // scale_up

    BLOCK_HID = triton.next_power_of_2(HID)
    if SPARQ:
        BLOCK_HID = triton.next_power_of_2(max(16, SPARQ_HID))
    if BLOCK_TMASK_K >= 2048:
        BLOCK_HID = min(BLOCK_HID, 16)
    elif BLOCK_TMASK_K >= 512:
        BLOCK_HID = min(BLOCK_HID, 32)
    elif BLOCK_TMASK_K >= 256:
        BLOCK_HID = min(BLOCK_HID, 64)
    elif BLOCK_TMASK_K >= 128:
        BLOCK_HID = min(BLOCK_HID, 128)
    # print(BLOCK_HID, BLOCK_TMASK_K, SPARQ, SPARQ_HID)

    if isinstance(keys, Tensor):
        KEY_CACHE_METHOD = "cont"
        stride_keys_vllm = (0, 0, 0, 0, 0)
        VLLM_NUM_BLOCKS = 0
        VLLM_NUM_KV_HEADS = 0
        VLLM_HEAD_SIZE_X = 0
        VLLM_BLOCK_SIZE = 0
        VLLM_X = 0
        VLLM_HEAD_SIZE = 0
        block_tables = keys
        block_tables_stride = (0, 0)
        context_length = None
        context_length_stride = (0,)
    elif isinstance(keys, PagedKeyCacheVllmCompat):
        """
        vLLM compatible paged attention

        q: [num_seqs, num_heads, head_size]
        k: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        v: [num_blocks, num_kv_heads, head_size, block_size]
        block_tables: [num_seqs, max_num_blocks_per_seq]
        context_lens: [num_seqs]
        """
        KEY_CACHE_METHOD = "vllm"
        stride_keys_vllm = keys.key_cache.stride()
        (
            VLLM_NUM_BLOCKS,
            VLLM_NUM_KV_HEADS,
            VLLM_HEAD_SIZE_X,
            VLLM_BLOCK_SIZE,
            VLLM_X,
        ) = keys.key_cache.shape
        VLLM_HEAD_SIZE = VLLM_HEAD_SIZE_X * VLLM_X
        block_tables = keys.block_table
        block_tables_stride = block_tables.stride()
        assert len(block_tables_stride) == 2

        context_length = keys.context_length
        context_length_stride = context_length.stride()
        assert len(context_length_stride) == 1

        # context_length = keys.context_length
        # context_length = context_length.unsqueeze(-1).repeat_interleave(VLLM_NUM_KV_HEADS, dim=0)
        # assert t_srcs.shape == context_length.shape, f"{t_srcs.shape} == {context_length.shape}"
        # t_srcs = context_length
    else:
        raise Exception()

    # NOTE: may improve latency, but hurt performance too much
    USING_SCORE_CACHE = False
    if USING_SCORE_CACHE:
        scores = torch.full_like(mask, 32000.0, dtype=torch.float16)
    else:
        scores = None

    # prepare for output
    ws_out = torch.empty_like(ws)
    ks_out = torch.empty(
        (N, B_DST, GRID_SRC_STRIDE),
        dtype=torch.int64,
        device=queries.device,
    )

    assert ROPE_METHOD in ["none", "self_extend"]
    if ROPE_METHOD in ["self_extend"]:
        assert ROPE_SIN is not None
        assert POSITION_IDS is not None
        assert ROPE_COS.ndim == 2
        assert ROPE_SIN.ndim == 2
        assert POSITION_IDS.ndim == 2
        assert POSITION_IDS.shape == (
            N,
            T_DST,
        ), f"{POSITION_IDS.shape} == {(N, T_DST)}, did you forget to repeat interleave?"
        rope_cos_stride = ROPE_COS.stride()
        rope_sin_stride = ROPE_SIN.stride()
        position_ids_stride = POSITION_IDS.stride()
    else:
        rope_cos_stride = (0, 0)
        rope_sin_stride = (0, 0)
        position_ids_stride = (0, 0)

    grid = (GRID_SRC_STRIDE, B_DST - N_COMPLETED, N)

    # HID cannot be chunked if use reduce
    # if REDUCE_METHOD in ['max', 'sum']:
    #     assert HID <= BLOCK_HID
    assert REDUCE_METHOD in ["max", "sum", "first"]

    assert queries.ndim == 3
    assert keys.ndim == 3
    if attention_mask is not None:
        assert attention_mask.ndim == 2
    assert mask.ndim == 4
    assert t_mask.ndim == 4
    assert ws.ndim == 2
    assert ws_out.ndim == 2
    assert ks.ndim == 2
    assert ks_out.ndim == 3
    assert t_srcs.ndim == 2

    if maximum_ks is not None:
        assert isinstance(maximum_ks, Tensor)
        assert maximum_ks.shape == (N, B_DST), f"{maximum_ks.shape} == {(N, B_DST)}"
        assert maximum_ks.dtype in [torch.int16, torch.int32, torch.int64, torch.long]
        maximum_ks = torch.ceil(maximum_ks / (BLOCK_SIZE_K * GRID_SRC_STRIDE)).to(
            maximum_ks.dtype
        )
        maximum_ks_stride = maximum_ks.stride()
        assert maximum_ks_config is not None
        maximum_ks_config = list(
            [math.ceil(x / (BLOCK_SIZE_K * GRID_SRC_STRIDE)) for x in maximum_ks_config]
        )
        # print(maximum_ks)
    else:
        maximum_ks_stride = (0, 0)

    # print('mask', mask[0, -1])
    # print('ks', ks, mask_k, BLOCK_SIZE_K, mask.shape, t_mask.shape, BLOCK_MASK_K)

    orig_device = torch.cuda.current_device()
    torch.cuda.set_device(queries.device)
    if maximum_ks is not None:
        calculated_maximum_ks_config = []
        for max_k in maximum_ks_config:
            calculated_maximum_ks_config.append(
                (
                    max_k,
                    max(maximum_ks_config) // max_k,
                )
            )

        for selected_max_k, scale in calculated_maximum_ks_config:
            # scale = 1
            _BLOCK_MASK_K = BLOCK_MASK_K // scale
            _BLOCK_TMASK_K = BLOCK_TMASK_K // scale

            _masking_iteration_compute[grid](
                # input matrices
                queries,
                *queries.stride(),
                queries_grouped,
                keys,
                *keys.stride(),
                attention_mask,
                *(attention_mask.stride() if attention_mask is not None else (0, 0)),
                sparq_indices,
                *sparq_indices_strides,
                # input matrices (blocked)
                mask,
                *mask.stride(),
                t_mask,
                *t_mask.stride(),
                # temp vectors (blocked)
                ws,
                *ws.stride(),
                ks,
                *ks.stride(),
                ws_out,
                *ws_out.stride(),
                ks_out,
                *ks_out.stride(),
                t_srcs,
                *t_srcs.stride(),
                scores,
                *(scores.stride() if scores is not None else (0, 0, 0)),
                # operation variables
                float(scale_up),
                int(triton.cdiv(n_patches, GRID_K_STRIDE)) // scale,
                int(mask.shape[-1]) // scale,
                int(t_mask.shape[-1]) // scale,
                is_causal,
                # input variables
                KV_REPEAT_INTERLEAVE,
                N,
                T_DST,
                T_SRC,
                int(B_DST),
                int(B_SRC),
                HID,
                SPARQ_HID,
                SPARQ_HID // 2 if SPARQ_HID > 16 else SPARQ_HID,
                N_COMPLETED,
                min(n_iteration, int(os.getenv("HIP_DEBUG_LIMIT_N_ITER", "99999999"))),
                # vLLM compat inputs
                *stride_keys_vllm,
                VLLM_NUM_BLOCKS,
                VLLM_NUM_KV_HEADS,
                VLLM_HEAD_SIZE_X,
                VLLM_BLOCK_SIZE,
                VLLM_X,
                VLLM_HEAD_SIZE,
                block_tables,
                *block_tables_stride,
                context_length,
                *context_length_stride,
                # rope methods
                ROPE_METHOD,
                ROPE_COS,
                *rope_cos_stride,
                ROPE_SIN,
                *rope_sin_stride,
                POSITION_IDS,
                *position_ids_stride,
                SELF_EXTEND_SCALE,
                SELF_EXTEND_WINDOW,
                # dynamic k per query,
                maximum_ks,
                *maximum_ks_stride,
                selected_max_k,
                # block constant
                USING_SCORE_CACHE,
                KEY_CACHE_METHOD,
                SPARQ,
                REDUCE_METHOD,
                _BLOCK_MASK_K,
                next_multiple_of(_BLOCK_MASK_K),
                _BLOCK_TMASK_K,
                next_multiple_of(_BLOCK_TMASK_K),
                _BLOCK_MASK_K // 2,
                next_multiple_of(_BLOCK_MASK_K // 2),
                _BLOCK_TMASK_K // 2,
                next_multiple_of(_BLOCK_TMASK_K // 2),
                triton.next_power_of_2(math.ceil(scale_up)),
                int(BLOCK_HID),
                int(BLOCK_SIZE_Q),
                next_multiple_of(triton.cdiv(BLOCK_SIZE_Q, REDUCE_STRIDE), 16),
                int(BLOCK_SIZE_K),
                next_multiple_of(BLOCK_SIZE_K, 1),
                REDUCE_STRIDE,
                SAMPLING_METHOD,
                GRID_SRC_STRIDE,
                GRID_K_STRIDE,
                USING_SLIDING_WINDOW,
                SLIDING_WINDOW_SIZE,
                # num_warps=max(2, (min(8, max(BLOCK_TMASK_K//32, 1)) if SPARQ else 4) // GRID_KSTRIDE),
                # num_warps=1,
                num_warps=8,
                num_stages=2,
                # enable_warp_specialization=False,
            )
    else:
        _masking_iteration_compute[grid](
            # input matrices
            queries,
            *queries.stride(),
            queries_grouped,
            keys,
            *keys.stride(),
            attention_mask,
            *(attention_mask.stride() if attention_mask is not None else (0, 0)),
            sparq_indices,
            *sparq_indices_strides,
            # input matrices (blocked)
            mask,
            *mask.stride(),
            t_mask,
            *t_mask.stride(),
            # temp vectors (blocked)
            ws,
            *ws.stride(),
            ks,
            *ks.stride(),
            ws_out,
            *ws_out.stride(),
            ks_out,
            *ks_out.stride(),
            t_srcs,
            *t_srcs.stride(),
            scores,
            *(scores.stride() if scores is not None else (0, 0, 0)),
            # operation variables
            float(scale_up),
            int(triton.cdiv(n_patches, GRID_K_STRIDE)),
            int(mask.shape[-1]),
            int(t_mask.shape[-1]),
            is_causal,
            # input variables
            KV_REPEAT_INTERLEAVE,
            N,
            T_DST,
            T_SRC,
            int(B_DST),
            int(B_SRC),
            HID,
            SPARQ_HID,
            SPARQ_HID // 2 if SPARQ_HID > 16 else SPARQ_HID,
            N_COMPLETED,
            min(n_iteration, int(os.getenv("HIP_DEBUG_LIMIT_N_ITER", "99999999"))),
            # vLLM compat inputs
            *stride_keys_vllm,
            VLLM_NUM_BLOCKS,
            VLLM_NUM_KV_HEADS,
            VLLM_HEAD_SIZE_X,
            VLLM_BLOCK_SIZE,
            VLLM_X,
            VLLM_HEAD_SIZE,
            block_tables,
            *block_tables_stride,
            context_length,
            *context_length_stride,
            # rope methods
            ROPE_METHOD,
            ROPE_COS,
            *rope_cos_stride,
            ROPE_SIN,
            *rope_sin_stride,
            POSITION_IDS,
            *position_ids_stride,
            SELF_EXTEND_SCALE,
            SELF_EXTEND_WINDOW,
            # dynamic k per query,
            maximum_ks,
            *maximum_ks_stride,
            0,
            # block constant
            USING_SCORE_CACHE,
            KEY_CACHE_METHOD,
            SPARQ,
            REDUCE_METHOD,
            BLOCK_MASK_K,
            next_multiple_of(BLOCK_MASK_K),
            BLOCK_TMASK_K,
            next_multiple_of(BLOCK_TMASK_K),
            BLOCK_MASK_K // 2,
            next_multiple_of(BLOCK_MASK_K // 2),
            BLOCK_TMASK_K // 2,
            next_multiple_of(BLOCK_TMASK_K // 2),
            triton.next_power_of_2(math.ceil(scale_up)),
            int(BLOCK_HID),
            int(BLOCK_SIZE_Q),
            next_multiple_of(triton.cdiv(BLOCK_SIZE_Q, REDUCE_STRIDE), 16),
            int(BLOCK_SIZE_K),
            next_multiple_of(BLOCK_SIZE_K, 1),
            REDUCE_STRIDE,
            SAMPLING_METHOD,
            GRID_SRC_STRIDE,
            GRID_K_STRIDE,
            USING_SLIDING_WINDOW,
            SLIDING_WINDOW_SIZE,
            # num_warps=max(2, (min(8, max(BLOCK_TMASK_K//32, 1)) if SPARQ else 4) // GRID_KSTRIDE),
            # num_warps=1,
            num_warps=8,
            num_stages=2,
            # enable_warp_specialization=False,
        )
    torch.cuda.set_device(orig_device)

    ks_out = ks_out.sum(-1)
    # print('ksout', ks_out, ws_out, N_COMPLETED, BLOCK_SIZE_Q, mask_k)

    # print('t_mask', t_mask[0, -1])

    if GRID_SRC_STRIDE > 1:
        # mask = mask.transpose(-1, -2).flatten(-2, -1)
        mask = mask.flatten(-2, -1)
        mask = mask.sort(dim=-1).values
    else:
        mask = mask.flatten(-2, -1)
    # print('mask', mask[0, -1])

    return mask, ws_out, ks_out
