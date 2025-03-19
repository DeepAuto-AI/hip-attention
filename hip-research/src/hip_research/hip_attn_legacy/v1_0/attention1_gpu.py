"""
- Need to stop expansion when reach #patch
> multiple = 4, #patch:p = 16, k = 64, w = 8192
| w    | z    | z'   | k'   | keep?|
|------|------|------|------|------|
| 64   | 64   | 1    | 16   | True |
| 256  | 64   | 2    | 16   | True |
| 1024 | 64   | 8    | 16   | True |
| 4096 | 64   | 32   | 32   | done |
| 8192 | done | done | done | done |

- When approximator interation stops?
w / T * k >= p

if p and k is constant
w = (p/k)T
approximator is logN, but sparse attention is linear

if p=T/C
w = T^2/(kC) -- log w = 2log T - log kC
approximator is quadratic, but sparse attention is linear

if k=T/C
w = pC
approximator is linear, but sparse attention is quadratic

if p=T/C and k=T/C
w = T
approximator is log N, but sparse attention is quadratic
"""

import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

from hip_attn.utils.benchmarking import get_bench

timer = lambda x: get_bench().region(x)


@triton.jit
def _triton_kth_large(
    scores: tl.tensor,
    k: tl.tensor,
    BLOCK_SCORES: tl.constexpr,
) -> tl.tensor:
    sorted_score = tl.sort(scores)
    # tl.debug_barrier()
    sorted_score_mask = tl.arange(0, BLOCK_SCORES) < k
    return tl.max(sorted_score * sorted_score_mask + (-32000.0) * (~sorted_score_mask))


@triton.jit
def _masking_iteration_compute(
    # input matrices
    queries,
    stride_queries_n,
    stride_queries_tdst,
    stride_queries_hid,
    keys,
    stride_keys_n,
    stride_keys_tsrc,
    stride_keys_hid,
    mask,
    stride_mask_n,
    stride_mask_tdst,
    stride_mask_k,
    tmask,
    stride_tmask_n,
    stride_tmask_tdst,
    stride_tmask_k,
    scores_out,
    stride_scores_out_n,
    stride_scores_out_tdst,
    stride_scores_out_k,
    # temp vectors
    ws,
    stride_ws_n,
    stride_ws_tdst,
    ks,
    stride_ks_n,
    stride_ks_tdst,
    tsrcs,
    stride_tsrcs_n,
    stride_tsrcs_tdst,
    # operation variables
    scale_up: float,
    n_patches: int,
    mask_k: int,
    # input variables
    N,
    T_DST,
    T_SRC,
    HID,
    # block constant
    GROUP_N,
    GROUP_TDST,
    BLOCK_MASK_K: tl.constexpr,
    BLOCK_TMASK_K: tl.constexpr,
    BLOCK_MAX_DUP: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    # TODO: we should make block across batch dim

    pid_n = tl.program_id(0)
    for _idx_n in range(GROUP_N):
        idx_n = _idx_n + GROUP_N * pid_n
        if idx_n < N:
            pid_tdst = tl.program_id(1)
            for _idx_tdst in range(GROUP_TDST):
                idx_tdst = pid_tdst * GROUP_TDST + _idx_tdst
                if idx_tdst < T_DST:
                    """
                    # for each query
                    w_old = ws[i, j, 0]
                    t_src = t_srcs[i, j, 0]
                    w_new = min(torch.round(w_old * scale_up), t_src)
                    """

                    w_old = tl.load(
                        ws + idx_n * stride_ws_n + idx_tdst * stride_ws_tdst,
                    )

                    t_src = tl.load(
                        tsrcs + idx_n * stride_tsrcs_n + idx_tdst * stride_tsrcs_tdst,
                    )

                    w_new = tl.minimum(
                        tl.math.round(
                            w_old.to(tl.float32) * scale_up.to(tl.float32)
                        ).to(tl.float32),
                        t_src,
                    ).to(tl.int64)

                    """
                    if w_old != w_new:
                    """
                    if w_old != w_new:
                        # return

                        """
                        k_old = ks[i, j, 0]
                        k_new = max(n_patches, int(min(mask_k / t_src, 1.0) * w_new))
                        k_new = min(t_src, max(n_patches, k_new))
                        """

                        k_old = tl.load(
                            ks + idx_n * stride_ks_n + idx_tdst * stride_ks_tdst,
                        ).to(tl.int64)
                        k_new = tl.maximum(
                            n_patches,
                            (
                                tl.minimum(
                                    mask_k.to(tl.float32) / t_src.to(tl.float32), 1.0
                                )
                                * w_new.to(tl.float32)
                            ).to(tl.int64),
                        )
                        k_new = tl.minimum(t_src, tl.maximum(n_patches, k_new))

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

                        k_old_range = tl.arange(0, BLOCK_MASK_K)
                        k_old_mask = k_old_range < k_old
                        loc_vec = tl.load(
                            mask
                            + idx_n * stride_mask_n
                            + idx_tdst * stride_mask_tdst
                            + k_old_range * stride_mask_k,
                            mask=k_old_mask,
                            other=0,
                        )

                        loc_idx_start_vec = (loc_vec * w_old).to(tl.int64)
                        loc_idx_end_vec = loc_idx_start_vec + 1
                        loc_idx_start_vec = (
                            loc_idx_start_vec.to(tl.float32)
                            / w_old.to(tl.float32)
                            * w_new.to(tl.float32)
                        ).to(tl.int64)
                        loc_idx_end_vec = (
                            loc_idx_end_vec.to(tl.float32)
                            / w_old.to(tl.float32)
                            * w_new.to(tl.float32)
                        ).to(tl.int64)

                        dup_pixels_vec = loc_idx_end_vec - loc_idx_start_vec
                        dup_pixels_vec = dup_pixels_vec * k_old_mask
                        num_pixels_vec = tl.cumsum(dup_pixels_vec)
                        dup_pixels_first = tl.min(num_pixels_vec)
                        num_pixels_scalar = tl.max(num_pixels_vec)

                        dup_pixels_range = tl.arange(0, BLOCK_MAX_DUP)
                        dup_pixels_mask = (
                            dup_pixels_range[None, :] <= dup_pixels_vec[:, None]
                        ) & k_old_mask[:, None]
                        # tl.debug_barrier()
                        tl.store(
                            tmask
                            + idx_n * stride_tmask_n
                            + idx_tdst * stride_tmask_tdst
                            + (
                                (num_pixels_vec - dup_pixels_first)[:, None]
                                + dup_pixels_range[None, :]
                            )
                            * stride_tmask_k,
                            mask=dup_pixels_mask,
                            value=(
                                (
                                    loc_idx_start_vec[:, None]
                                    + tl.arange(0, BLOCK_MAX_DUP)[None, :]
                                ).to(tl.float32)
                                / w_new.to(tl.float32)
                            ),
                            # value = num_pixels_scalar=
                        )
                        # tl.debug_barrier()

                        """
                        # t_mask -> mask (using scores)
                        if k_new < num_pixels:
                        """
                        if k_new < num_pixels_scalar and True:
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
                            scores = tl.zeros((BLOCK_TMASK_K,), dtype=tl.float32)
                            for _idx_hid in range(tl.cdiv(HID, BLOCK_HID)):
                                hid_range = (
                                    tl.arange(0, BLOCK_HID) + _idx_hid * BLOCK_HID
                                )
                                hid_mask = hid_range < HID
                                vec_q = tl.load(
                                    queries
                                    + idx_n * stride_queries_n
                                    + idx_tdst * stride_queries_tdst
                                    + (hid_range[None, :] + tl.arange(0, 16)[:, None])
                                    * stride_queries_hid,
                                    mask=(
                                        hid_mask[None, :]
                                        & (tl.arange(0, 16)[:, None] < 1)
                                    ),
                                    other=0,
                                )
                                # tl.debug_barrier()

                                num_pixels_range = tl.arange(0, BLOCK_TMASK_K)
                                num_pixels_mask = num_pixels_range < num_pixels_scalar
                                loc_k_vec = tl.load(
                                    tmask
                                    + idx_n * stride_tmask_n
                                    + idx_tdst * stride_tmask_tdst
                                    + num_pixels_range * stride_tmask_k,
                                    mask=num_pixels_mask,
                                    other=0,
                                )
                                # tl.debug_barrier()
                                # NOTE: random key selection with in the block
                                # loc_k_vec = loc_k_vec.to(tl.float32) + tl.rand(idx_n * idx_tdst, w_old, 10) * (1.0 / w_old)
                                loc_k_vec = (
                                    loc_k_vec.to(tl.float32) * t_src.to(tl.float32)
                                ).to(tl.int64)
                                vec_k_mask = (
                                    num_pixels_mask[None, :] & hid_mask[:, None]
                                )
                                vec_k = tl.load(
                                    keys
                                    + idx_n * stride_keys_n
                                    + loc_k_vec[None, :] * stride_keys_tsrc
                                    + hid_range[:, None] * stride_keys_hid,
                                    mask=vec_k_mask,
                                    other=0,
                                )
                                # tl.debug_barrier()

                                # TODO: support tensorCore
                                # scores = -tl.dot(vec_q, vec_k) # NOTE: negative scores
                                # 1x128 @ 128x512 512x128 @ 128x1
                                # scores = -tl.sum(
                                #     vec_q * vec_k,
                                #     axis=0,
                                # )
                                scores_partial = -tl.dot(vec_q, vec_k, allow_tf32=True)
                                scores_partial = tl.sum(scores_partial, axis=0)
                                scores_partial = (
                                    scores_partial + (~num_pixels_mask) * 32000.0
                                )
                                scores += scores_partial.to(scores.dtype)
                            # tl.debug_barrier()
                            # scores = tl.zeros((BLOCK_TMASK_K,), dtype=tl.float32)

                            """
                            _, topk_indices = torch.topk(scores[i, j, :num_pixels], k=k_new, largest=False)
                            for k in range(k_new):
                                mask[i, j, k] = t_mask[i, j, topk_indices[k]]
                            """

                            # select min-k from negative scores -> select top-k
                            # masked_scores = scores + (~num_pixels_mask) * 32000.0
                            masked_scores = scores
                            # tl.debug_barrier()
                            scores_kth_large = _triton_kth_large(
                                masked_scores, k_new, BLOCK_TMASK_K
                            )
                            # tl.debug_barrier()
                            topk_mask = masked_scores <= scores_kth_large
                            topk_mask_cumsum = tl.cumsum(topk_mask.to(tl.int64))
                            # tl.debug_barrier()
                            topk_range = tl.minimum(
                                (topk_mask_cumsum - 1) * topk_mask, k_new - 1
                            )
                            # tl.debug_barrier()

                            temp_range = tl.arange(0, BLOCK_TMASK_K)
                            temp_mask = temp_range < num_pixels_scalar
                            temp = tl.load(
                                tmask
                                + idx_n * stride_tmask_n
                                + idx_tdst * stride_tmask_tdst
                                + temp_range * stride_tmask_k,
                                mask=temp_mask,
                                other=0,
                            )
                            # tl.debug_barrier()
                            tl.store(
                                mask
                                + idx_n * stride_mask_n
                                + idx_tdst * stride_mask_tdst
                                + topk_range * stride_mask_k,
                                mask=topk_mask & temp_mask,
                                value=temp,
                                # value=0.1,
                            )
                            # tl.debug_barrier()
                        else:
                            """
                            else:
                                mask[i, j, :num_pixels] = t_mask[i, j, :num_pixels]
                            """
                            temp1_range = tl.arange(0, BLOCK_MASK_K)
                            temp1_mask = temp1_range < num_pixels_scalar
                            # tl.debug_barrier()
                            temp1 = tl.load(
                                tmask
                                + idx_n * stride_tmask_n
                                + idx_tdst * stride_tmask_tdst
                                + temp1_range * stride_tmask_k,
                                mask=temp1_mask,
                            )

                            # tl.debug_barrier()
                            tl.store(
                                mask
                                + idx_n * stride_mask_n
                                + idx_tdst * stride_mask_tdst
                                + temp1_range * stride_mask_k,
                                mask=temp1_mask,
                                value=temp1,
                            )
                            # tl.debug_barrier()
                            # del temp1, temp1_range, temp1_mask

                        """
                        ws[i, j, 0] = w_new
                        ks[i, j, 0] = min(k_new, num_pixels)
                        """
                        # tl.debug_barrier()
                        tl.store(
                            ws + idx_n * stride_ws_n + idx_tdst * stride_ws_tdst,
                            value=w_new,
                        )
                        # tl.debug_barrier()
                        tl.store(
                            ks + idx_n * stride_ks_n + idx_tdst * stride_ks_tdst,
                            value=tl.minimum(k_new, num_pixels_scalar),
                        )
                        # tl.debug_barrier()


def masking_iteration(
    # input matrices
    queries: Tensor,
    keys: Tensor,
    mask: Tensor,
    t_mask: Tensor,
    scores: Tensor,
    # temp vectors
    ws: Tensor,
    ks: Tensor,
    t_srcs: Tensor,
    # operator variables
    scale_up: float,
    n_patches: int,
    mask_k: int,
    # input constant
    N: int,
    T_DST: int,
    T_SRC: int,
    HID: int,
):
    global DEBUG
    if DEBUG:
        K = mask.shape[-1]
        assert t_srcs.min() > 0
        assert t_srcs.max() <= T_SRC
        assert ks.min() >= 0
        assert ks.max() <= K
        assert keys.shape[1] == T_SRC
        assert queries.shape[1] == T_DST
        assert mask.min() >= 0
        assert mask.max() < 1
        assert t_mask.min() >= 0
        assert t_mask.max() < 1

    GROUP_N = 1
    GROUP_TDST = 4
    BLOCK_HID = 16
    grid = (triton.cdiv(N, GROUP_N), triton.cdiv(T_DST, GROUP_TDST))

    _masking_iteration_compute[grid](
        # input matrices
        queries,
        queries.stride(0),
        queries.stride(1),
        queries.stride(2),
        keys,
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        mask,
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        t_mask,
        t_mask.stride(0),
        t_mask.stride(1),
        t_mask.stride(2),
        scores,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        # temp vectors
        ws,
        ws.stride(0),
        ws.stride(1),
        ks,
        ks.stride(0),
        ks.stride(1),
        t_srcs,
        t_srcs.stride(0),
        t_srcs.stride(1),
        # operation variables
        float(scale_up),
        int(n_patches),
        int(mask_k),
        # input variables
        N,
        T_DST,
        T_SRC,
        HID,
        # block constant
        GROUP_N,
        GROUP_TDST,
        triton.next_power_of_2(mask.shape[-1]),
        triton.next_power_of_2(t_mask.shape[-1]),
        triton.next_power_of_2(math.ceil(scale_up)),
        BLOCK_HID,
        num_warps=4,
        num_stages=1,
        enable_warp_specialization=True,
    )


@triton.jit
def _calc_score_compute(
    # matrices
    queries,
    stride_queries_n,
    stride_queries_tdst,
    stride_queries_hid,
    keys,
    stride_keys_n,
    stride_keys_tsrc,
    stride_keys_hid,
    indices,
    stride_indices_n,
    stride_indices_tdst,
    stride_indices_k,
    ks,
    stride_ks_n,
    stride_ks_tdst,
    scores_out,
    stride_scores_out_n,
    stride_scores_out_tdst,
    stride_scores_out_k,
    # input variables
    N,
    TDST,
    TSRC,
    HID,
    K,
    # kernel constant
    BLOCK_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    """
    q = [1, BLOCK_HID]
    k = [BLOCK_HID, BLOCK_K]
    """

    idx_n = tl.program_id(0)
    idx_tdst = tl.program_id(1)

    pid_k = tl.program_id(2)
    idx_k = tl.arange(0, BLOCK_K) + pid_k * BLOCK_K
    mask_k = idx_k < K

    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID

    # query: [BLOCK_HID]
    query = tl.load(
        queries
        + idx_n * stride_queries_n
        + idx_tdst * stride_queries_tdst
        + idx_hid * stride_queries_hid,
        mask=mask_hid,
        other=0,
    )

    # ks: [1,]
    n_k = tl.load(
        ks + idx_n * stride_ks_n + idx_tdst * stride_ks_tdst,
    )
    mask_n_k = idx_k < n_k

    # idx_keys: [BLOCK_K, ]
    idx_keys = tl.load(
        indices
        + idx_n * stride_indices_n
        + idx_tdst * stride_indices_tdst
        + idx_k * stride_indices_k,
        mask=mask_k,
        other=0,
    )
    mask_idx_keys = mask_n_k & ((idx_keys < TSRC) & (idx_keys >= 0))

    # tl.debug_barrier()
    # tl.device_print("", idx_keys)
    # tl.device_print("", tl.max(idx_keys))

    # key: [BLOCK_HID, BLOCK_K]
    key = tl.load(
        keys
        + idx_n * stride_keys_n
        + idx_keys[None, :] * stride_keys_tsrc
        + idx_hid[:, None] * stride_keys_hid,
        mask=(mask_k & mask_idx_keys)[None, :] & mask_hid[:, None],
        other=0,
    )

    scores = query[:, None] * key
    scores = tl.sum(scores, axis=0)

    # tl.debug_barrier()
    # tl.device_print("", tl.max(scores))

    tl.store(
        scores_out
        + idx_n * stride_scores_out_n
        + idx_tdst * stride_scores_out_tdst
        + idx_k * stride_scores_out_k,
        mask=mask_k & mask_idx_keys,
        value=scores,
    )


@triton.jit
def _calc_score_compute_bwd_queries(
    # input matrices
    ks,
    stride_ks_n,
    stride_ks_tdst,
    indices,
    stride_indices_n,
    stride_indices_tdst,
    stride_indices_k,
    keys,
    stride_keys_n,
    stride_keys_tsrc,
    stride_keys_hid,
    # grad output (read)
    grad_scores,
    stride_grad_scores_n,
    stride_grad_scores_tdst,
    stride_grad_scores_k,
    # grad input (write)
    grad_queries,
    stride_grad_queries_n,
    stride_grad_queries_tdst,
    stride_grad_queries_hid,
    # input variables
    N,
    TDST,
    TSRC,
    HID,
    K,
    # block constant
    BLOCK_HID: tl.constexpr,
):
    """
    ks: int[N, TDST]
    indices: int[N, TDST, K]
    keys: fp[N, TSRC, HID]
    grad_scores: fp[N, TDST, K]
    grad_queries: fp[N, TDST, HID]
    -----
    foreach n in [..N]
    foreach tdst in [..TDST]

    scalar_ks = ks[n, tdst]

    acc = zeros(HID)
    for k in [..K]:
        idx_tsrc = indices[n, tdst, k]
        mask_tsrc = idx_tsrc < T_SRC & k < scalar_ks
        acc += grad_scores[n, tdst, k] * keys[n, idx_tsrc, :]
    grad_queries[n, tdst, :] = acc
    """

    idx_n = tl.program_id(0)
    idx_tdst = tl.program_id(1)

    scalar_ks = tl.load(ks + idx_n * stride_ks_n + idx_tdst * stride_ks_tdst)

    accumulator = tl.zeros((BLOCK_HID,), dtype=tl.float32)
    for idx_k in range(K):
        idx_tsrc = tl.load(
            indices
            + idx_n * stride_indices_n
            + idx_tdst * stride_indices_tdst
            + idx_k * stride_indices_k,
        )
        mask_tsrc = (idx_tsrc < TSRC) & (idx_k < scalar_ks)

        idx_hid = tl.arange(0, BLOCK_HID)
        mask_hid = idx_hid < HID
        grad_score = tl.load(
            grad_scores
            + idx_n * stride_grad_scores_n
            + idx_tdst * stride_grad_scores_tdst
            + idx_k * stride_grad_scores_k,
            mask=mask_tsrc,
            other=0,
        )
        key = tl.load(
            keys
            + idx_n * stride_keys_n
            + idx_tsrc * stride_keys_tsrc
            + idx_hid * stride_keys_hid,
            mask=mask_hid[:] & mask_tsrc[None],
            other=0,
        )
        accumulator += grad_score * key

    tl.store(
        grad_queries
        + idx_n * stride_grad_queries_n
        + idx_tdst * stride_grad_queries_tdst
        + tl.arange(0, BLOCK_HID) * stride_grad_queries_hid,
        mask=tl.arange(0, BLOCK_HID) < HID,
        value=accumulator,
    )


@triton.jit
def _calc_score_compute_bwd_keys(
    # input matrices
    ks,
    stride_ks_n,
    stride_ks_tdst,
    indices,
    stride_indices_n,
    stride_indices_tdst,
    stride_indices_k,
    queries,
    stride_queries_n,
    stride_queries_tdst,
    stride_queries_hid,
    # grad output (read)
    grad_scores,
    stride_grad_scores_n,
    stride_grad_scores_tdst,
    stride_grad_scores_k,
    # grad input (write)
    grad_keys,
    stride_grad_keys_n,
    stride_grad_keys_tsrc,
    stride_grad_keys_hid,
    # input variables
    N,
    TDST,
    TSRC,
    HID,
    K,
    # block constant
    BLOCK_HID: tl.constexpr,
):
    """
    indices: int[N, TDST, K]
    ks: int[N, TDST, K]
    queries: int[N, TDST, HID]
    grad_scores: fp[N, TDST, K]
    grad_keys: fp[N, TSRC, HID]
    -----
    foreach n in [..N]
    foreach tdst in [..TDST]
    foreach k in [..K]

    scalar_ks = ks[n, tdst]
    if k >= scalar_ks: return

    grad_keys[n, indices[n, tdst, k], hid] +=(atomic)
        grad_scores[n, tdst, k] * queries[n, tdst, :]
    """
    idx_n = tl.program_id(0)
    idx_tdst = tl.program_id(1)
    idx_k = tl.program_id(2)

    scalar_ks = tl.load(
        ks + idx_n * stride_ks_n + idx_tdst * stride_ks_tdst,
    )
    mask_job = idx_k < scalar_ks
    # if idx_k >= scalar_ks: return

    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = (idx_hid < HID) & mask_job

    grad_score = tl.load(
        grad_scores
        + idx_n * stride_grad_scores_n
        + idx_tdst * stride_grad_scores_tdst
        + idx_k * stride_grad_scores_k,
        mask=mask_job,
    )
    query = tl.load(
        queries
        + idx_n * stride_queries_n
        + idx_tdst * stride_queries_tdst
        + idx_hid * stride_queries_hid,
        mask=mask_hid,
        other=0,
    )
    scores = grad_score * query

    idx_tsrc = tl.load(
        indices
        + idx_n * stride_indices_n
        + idx_tdst * stride_indices_tdst
        + idx_k * stride_indices_k,
        mask=mask_job,
    )
    tl.atomic_add(
        grad_keys
        + idx_n * stride_grad_keys_n
        + idx_tsrc * stride_grad_keys_tsrc
        + idx_hid * stride_grad_keys_hid,
        val=scores,
        mask=mask_hid,
    )


# NOTE: you have to perform softmax after this
class CalcScoreAutoGradFn(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx,
        # matrices
        queries: Tensor,
        keys: Tensor,
        # indices matrices
        indices: Tensor,
        ks: Tensor,
        # output scores
        scores: Tensor,
    ):
        ctx.save_for_backward(queries, keys, indices, ks)

        N, TDST, HID = queries.shape
        _, TSRC, ___ = keys.shape
        assert indices.shape == scores.shape
        _, _, K = indices.shape

        BLOCK_K = 32
        BLOCK_HID = triton.next_power_of_2(HID)

        grid = (
            N,
            TDST,
            triton.cdiv(K, BLOCK_K),
        )

        scores.fill_(torch.finfo(scores.dtype).min)

        assert indices.dtype in [torch.int64, torch.int32], indices.dtype

        assert queries.is_contiguous()
        assert keys.is_contiguous()
        assert indices.is_contiguous()
        assert scores.is_contiguous()

        assert queries.ndim == 3
        assert keys.ndim == 3
        assert indices.ndim == 3
        assert ks.ndim == 2
        assert scores.ndim == 3
        _calc_score_compute[grid](
            # matrices
            queries,
            queries.stride(0),
            queries.stride(1),
            queries.stride(2),
            keys,
            keys.stride(0),
            keys.stride(1),
            keys.stride(2),
            indices,
            indices.stride(0),
            indices.stride(1),
            indices.stride(2),
            ks,
            ks.stride(0),
            ks.stride(1),
            scores,
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            # variables
            N,
            TDST,
            TSRC,
            HID,
            K,
            # constants
            BLOCK_K,
            BLOCK_HID,
        )

        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        queries, keys, indices, ks = ctx.saved_tensors
        grad_queries = grad_keys = grad_indices = grad_ks = None

        N, T_DST, HID = queries.shape
        _, T_SRC, _HID = keys.shape
        assert HID == _HID
        _, _, K = indices.shape

        # for queries
        if ctx.needs_input_grad[0]:
            grid = (N, T_DST)
            BLOCK_HID = triton.next_power_of_2(HID)

            grad_queries = torch.zeros_like(queries)

            _calc_score_compute_bwd_queries[grid](
                ks,
                ks.stride(0),
                ks.stride(1),
                indices,
                indices.stride(0),
                indices.stride(1),
                indices.stride(2),
                keys,
                keys.stride(0),
                keys.stride(1),
                keys.stride(2),
                grad_scores,
                grad_scores.stride(0),
                grad_scores.stride(1),
                grad_scores.stride(2),
                grad_queries,
                grad_queries.stride(0),
                grad_queries.stride(1),
                grad_queries.stride(2),
                N,
                T_DST,
                T_SRC,
                HID,
                K,
                BLOCK_HID,
            )

        # for keys
        if ctx.needs_input_grad[1]:
            grid = (N, T_DST, K)
            BLOCK_HID = triton.next_power_of_2(HID)

            grad_keys = torch.zeros_like(keys)

            _calc_score_compute_bwd_keys[grid](
                ks,
                ks.stride(0),
                ks.stride(1),
                indices,
                indices.stride(0),
                indices.stride(1),
                indices.stride(2),
                queries,
                queries.stride(0),
                queries.stride(1),
                queries.stride(2),
                grad_scores,
                grad_scores.stride(0),
                grad_scores.stride(1),
                grad_scores.stride(2),
                grad_keys,
                grad_keys.stride(0),
                grad_keys.stride(1),
                grad_keys.stride(2),
                N,
                T_DST,
                T_SRC,
                HID,
                K,
                BLOCK_HID,
            )

        return (grad_queries, grad_keys, grad_indices, grad_ks, None)


def calc_score_return_prob(
    # matrices
    queries: Tensor,
    keys: Tensor,
    # indices matrices
    indices: Tensor,
    ks: Tensor,
    # output scores
    scores: Tensor,
):
    scores = CalcScoreAutoGradFn.apply(queries, keys, indices, ks, scores)
    probs = scores.softmax(dim=-1)
    return probs


def to_dense(
    indices,
    ks,
    value,
    N: int,
    T_DST: int,
    T_SRC: int,
):
    # print('convert to dense')
    dense = np.zeros((N, T_DST, T_SRC))
    for i in range(1):
        for j in range(T_DST):
            nonzero_k = ks[i, j].item()
            for k in range(nonzero_k):
                if value is None:
                    dense[i, j, indices[i, j, k]] = 1
                else:
                    dense[i, j, indices[i, j, k]] = value[i, j, k]
    return dense


DEBUG = False


def attention_matrix(
    queries: Tensor,
    keys: Tensor,
    w_start: int,
    n_patches: int,
    mask_k: int,
    scale_up: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    global DEBUG

    if DEBUG:
        os.makedirs("saves/models/hip_attention/", exist_ok=True)

    dtype = queries.dtype
    device = queries.device
    assert queries.device == keys.device

    N, T_DST, HID = queries.shape
    _, T_SRC, _ = keys.shape
    assert T_DST <= T_SRC

    # NOTE: width of last query
    w_curr = round(w_start / scale_up)

    with timer("attention_matrix.prepare"):
        # vectors
        tsrcs = (
            torch.arange(
                T_SRC - T_DST + 1,
                T_SRC + 1,
                1,
                dtype=torch.int64,
                device=device,
            )
            .view(1, T_DST)
            .expand(N, T_DST)
        )
        ws = torch.clamp(tsrcs, 0, w_curr)
        ks = ws.clone()

        # matrices
        # NOTE: float16 -> int32 seems not possible
        mask = (
            torch.arange(mask_k, device=device).view(1, 1, mask_k) / ks.unsqueeze(-1)
        ).to(torch.float32)
        tmask = torch.zeros(
            (N, T_DST, mask_k * math.ceil(scale_up)), dtype=mask.dtype, device=device
        )
        scores = torch.ones_like(mask, dtype=dtype)

    # NOTE: Calc. Mask
    while w_curr < T_SRC:
        with timer(f"iteration_{w_curr}_zerofill"):
            tmask.fill_(0)
            mask.clamp_(0, (w_curr - 1) / w_curr)
        with timer(f"iteration_{w_curr}"):
            masking_iteration(
                # input matrices
                queries,
                keys,
                mask,
                tmask,
                scores,
                # temp vectors
                ws,
                ks,
                tsrcs,
                # operator variables
                scale_up,
                n_patches,
                mask_k,
                # input constant
                N,
                T_DST,
                T_SRC,
                HID,
            )
        w_curr = round(w_curr * scale_up)

        if DEBUG:
            indices = torch.round(mask * ws.unsqueeze(-1)).to(torch.int32)
            indices = torch.clamp(indices, 0, T_SRC - 1)
            x = to_dense(
                indices.cpu().numpy(),
                ks.cpu().unsqueeze(-1).numpy(),
                None,
                N,
                T_DST,
                T_SRC,
            )[0]
            x = skimage.measure.block_reduce(x, (4, 4), np.max) ** 0.1
            plt.imshow(x)
            path = f"saves/models/hip_attention/hello_{w_curr}.png"
            print("saved", path)
            plt.savefig(path, dpi=200, bbox_inches="tight")

    with timer("attention_matrix.indices"):
        # NOTE: Calc. Prob.
        indices = torch.round(mask * ws.unsqueeze(-1)).to(torch.int32)
        indices.clamp_(0, T_SRC - 1)

    with timer("calc_score_return_prob"):
        # NOTE: are you sure this function is the only thing can differentiate?
        probs = calc_score_return_prob(
            queries=queries,
            keys=keys,
            indices=indices,
            ks=ks,
            scores=scores,
        )

    if DEBUG:
        x = to_dense(
            indices.cpu().numpy(),
            ks.cpu().unsqueeze(-1).numpy(),
            probs.cpu().numpy(),
            N,
            T_DST,
            T_SRC,
        )[0]
        x = skimage.measure.block_reduce(x, (4, 4), np.max) ** 0.1
        plt.imshow(x)
        path = "saves/models/hip_attention/hello_est.png"
        print("saved", path)
        plt.savefig(path, dpi=200, bbox_inches="tight")

        x = np.matmul(
            queries[0].cpu().numpy(), keys[0].cpu().numpy().transpose((-1, -2))
        )
        x = x + (1 - np.tri(*x.shape)) * (-32000)
        x = np.exp(x - x.max(-1, keepdims=True))
        x = x / x.sum(-1, keepdims=True)
        x = skimage.measure.block_reduce(x, (4, 4), np.max) ** 0.1
        plt.imshow(x)
        path = "saves/models/hip_attention/hello_truth.png"
        print("saved", path)
        plt.savefig(path, dpi=200, bbox_inches="tight")
        print(ks)

    return indices, ks, probs


@triton.jit
def _sdbmm_compute(
    # inputs
    indices,
    stride_indices_n,
    stride_indices_tdst,
    stride_indices_k,
    ks,
    stride_ks_n,
    stride_ks_tdst,
    probs,
    stride_probs_n,
    stride_probs_tdst,
    stride_probs_k,
    values,
    stride_values_n,
    stride_values_tsrc,
    stride_values_hid,
    # output
    context,
    stride_context_n,
    stride_context_tdst,
    stride_context_hid,
    # variables
    N,
    TSRC,
    TDST,
    HID,
    K,
    # kernel blocks
    GROUP_N,
    BLOCK_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    pid_n = tl.program_id(0)

    for _idx_n in range(GROUP_N):
        idx_n = _idx_n + pid_n * GROUP_N
        if idx_n < N:
            idx_tdst = tl.program_id(1)
            tl.device_assert(idx_n < N)
            tl.device_assert(idx_tdst < TDST)

            idx_k = tl.arange(0, BLOCK_K)

            n_k = tl.load(
                ks + idx_n * stride_ks_n + idx_tdst * stride_ks_tdst,
            )
            mask_k = (idx_k < K) & (idx_k < n_k)

            pid_hid = tl.program_id(2)
            idx_hid = tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID
            mask_hid = idx_hid < HID

            # atten_indices: [BLOCK_K]
            atten_indices = tl.load(
                indices
                + idx_n * stride_indices_n
                + idx_tdst * stride_indices_tdst
                + idx_k * stride_indices_k,
                mask=mask_k,
                other=0,
            )
            tl.device_assert(tl.max(atten_indices) < TSRC, "should be index < TSRC")
            tl.device_assert(tl.min(atten_indices) >= 0, "should be index >= 0")

            # atten_probs: [BLOCK_K]
            atten_probs = tl.load(
                probs
                + idx_n * stride_probs_n
                + idx_tdst * stride_probs_tdst
                + (idx_k[None, :] + tl.arange(0, 16)[:, None]) * stride_probs_k,
                mask=mask_k[None, :] & (tl.arange(0, 16)[:, None] < 1),
                other=0,
            )

            # value: [BLOCK_K, BLOCK_HID]
            value = tl.load(
                values
                + idx_n * stride_values_n
                + atten_indices[:, None] * stride_values_tsrc
                + idx_hid[None, :] * stride_values_hid,
                mask=mask_k[:, None] & mask_hid[None, :],
                other=0,
            )

            # output: [BLOCK_HID] <- atten_probs[1, BLOCK_K] @ value[BLOCK_K, BLOCK_HID]
            # output = tl.sum(atten_probs[None, :] * value, axis=1)
            output = tl.dot(atten_probs.to(value.dtype), value, allow_tf32=True)
            output = tl.sum(output, axis=0)

            tl.store(
                context
                + idx_n * stride_context_n
                + idx_tdst * stride_context_tdst
                + idx_hid * stride_context_hid,
                mask=mask_hid,
                value=output,
            )


@triton.jit
def _sdbmm_compute_bwd_values(
    # input matrices
    probs,
    stride_probs_n,
    stride_probs_tdst,
    stride_probs_k,
    indices,
    stride_indices_n,
    stride_indices_tdst,
    stride_indices_k,
    # grad output (read)
    grad_context,
    stride_grad_context_n,
    stride_grad_context_tdst,
    stride_grad_context_hid,
    # grad input (write)
    grad_values,
    stride_grad_values_n,
    stride_grad_values_tsrc,
    stride_grad_values_hid,
    # input variables
    N,
    TDST,
    TSRC,
    HID,
    K,
    # block constant
    BLOCK_HID: tl.constexpr,
):
    """
    probs: fp[N, TDST, K]
    indices: int[N, TDST, K]

    grad_context: fp[N, TDST, HID]
    grad_values: fp[N, TSRC, HID]
    ----
    foreach n in range(N)
    foreach tdst in range(TDST)
    foreach k in range(K)

    grad_values[n, indices[n, tdst, k], :] +=(atmoic) probs[n, tdst, k] * grad_context[n, tdst, :]
    """

    idx_n = tl.program_id(0)
    idx_tdst = tl.program_id(1)
    idx_k = tl.program_id(2)

    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID

    idx_src = tl.load(
        indices
        + idx_n * stride_indices_n
        + idx_tdst * stride_indices_tdst
        + idx_k * stride_indices_k
    )

    prob = tl.load(
        probs
        + idx_n * stride_probs_n
        + idx_tdst * stride_probs_tdst
        + idx_k * stride_probs_k,
    )
    grad = tl.load(
        grad_context
        + idx_n * stride_grad_context_n
        + idx_tdst * stride_grad_context_tdst
        + idx_hid * stride_grad_context_hid,
        mask=mask_hid,
    )
    output = prob * grad

    tl.atomic_add(
        grad_values
        + idx_n * stride_grad_values_n
        + idx_src * stride_grad_values_tsrc
        + idx_hid * stride_grad_values_hid,
        val=output,
        mask=mask_hid,
    )


@triton.jit
def _sdbmm_compute_bwd_probs(
    # input indices
    indices,
    stride_indices_n,
    stride_indices_tdst,
    stride_indices_k,
    values,
    stride_values_n,
    stride_values_trsc,
    stride_values_hid,
    # grad output (read)
    grad_context,
    stride_grad_context_n,
    stride_grad_context_tdst,
    stride_grad_context_hid,
    # grad input (write)
    grad_probs,
    stride_grad_probs_n,
    stride_grad_probs_tdst,
    stride_grad_probs_k,
    # input variables
    N,
    TDST,
    TSRC,
    HID,
    K,
    # blcok constant
    BLOCK_HID: tl.constexpr,
):
    """
    indices: fp[N, TDST, K]
    values: fp[N, TSRC, HID]
    grad_context: fp[N, TDST, HID]
    grad_probs: fp[N, TDST, K]
    -----
    foreach n in [..N]
    foreach tdst in [..TDST]
    foreach k in [..K]

    grad_probs[n, tdst, k] = sum(
        values[n, indices[n, tdst, k], :] * grad_context[n, tdst, :]
    )
    """

    idx_n = tl.program_id(0)
    idx_tdst = tl.program_id(1)
    idx_k = tl.program_id(2)

    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID

    idx_tsrc = tl.load(
        indices
        + idx_n * stride_indices_n
        + idx_tdst * stride_indices_tdst
        + idx_k * stride_indices_k,
    )

    value = tl.load(
        values
        + idx_n * stride_values_n
        + idx_tsrc * stride_values_trsc
        + idx_hid * stride_values_hid,
        mask=mask_hid,
        other=0,
    )
    vec_grad_context = tl.load(
        grad_context
        + idx_n * stride_grad_context_n
        + idx_tdst * stride_grad_context_tdst
        + idx_hid * stride_grad_context_hid,
        mask=mask_hid,
        other=0,
    )
    scores = value * vec_grad_context
    score = tl.sum(scores)

    tl.store(
        grad_probs
        + idx_n * stride_grad_probs_n
        + idx_tdst * stride_grad_probs_tdst
        + idx_k * stride_grad_probs_k,
        value=score,
    )


class SparseAttentionAutoGradFn(Function):
    @staticmethod
    def forward(
        ctx,
        # attention values
        values: Tensor,
        # attention matrix
        indices: Tensor,
        ks: Tensor,
        probs: Tensor,
    ):
        global DEBUG

        N, T_SRC, HID = values.shape
        _N, T_DST, K = indices.shape
        assert N == _N
        assert ks.shape == (N, T_DST)
        assert probs.shape == indices.shape

        ctx.save_for_backward(values, indices, ks, probs)

        context = torch.zeros((N, T_DST, HID), dtype=values.dtype, device=values.device)

        GROUP_N = 1
        BLOCK_K = triton.next_power_of_2(K)
        # BLOCK_HID = triton.next_power_of_2(HID)
        BLOCK_HID = 64
        grid = (triton.cdiv(N, GROUP_N), T_DST, triton.cdiv(HID, BLOCK_HID))

        # NOTE: I have no idea what this sprase matrix format LOL, but for temporary
        if DEBUG:
            # print('sdbmm', grid, BLOCK_K, BLOCK_HID)
            assert indices.max() < T_SRC
            assert indices.min() >= 0
            assert indices.is_contiguous()
            assert ks.is_contiguous()
            assert probs.is_contiguous()
            assert values.is_contiguous()
            assert context.is_contiguous()
            torch.cuda.synchronize()

        _sdbmm_compute[grid](
            # inputs
            indices,
            indices.stride(0),
            indices.stride(1),
            indices.stride(2),
            ks,
            ks.stride(0),
            ks.stride(1),
            probs,
            probs.stride(0),
            probs.stride(1),
            probs.stride(2),
            values,
            values.stride(0),
            values.stride(1),
            values.stride(2),
            # output
            context,
            context.stride(0),
            context.stride(1),
            context.stride(2),
            # input variables
            N,
            T_SRC,
            T_DST,
            HID,
            K,
            # blocks
            GROUP_N,
            BLOCK_K,
            BLOCK_HID,
            num_warps=4,
        )

        return context

    @staticmethod
    def backward(ctx, grad_context):
        values, indices, ks, probs = ctx.saved_tensors
        grad_values = grad_indices = grad_ks = grad_probs = None

        N, T_SRC, HID = values.shape
        _, T_DST, K = indices.shape
        assert ks.shape == (N, T_DST)
        assert probs.shape == indices.shape

        # for values
        if ctx.needs_input_grad[0]:
            grid = (N, T_DST, K)
            BLOCK_HID = triton.next_power_of_2(HID)

            grad_values = torch.zeros(
                (N, T_SRC, HID),
                device=values.device,
                dtype=values.dtype,
            )

            _sdbmm_compute_bwd_values[grid](
                probs,
                probs.stride(0),
                probs.stride(1),
                probs.stride(2),
                indices,
                indices.stride(0),
                indices.stride(1),
                indices.stride(2),
                grad_context,
                grad_context.stride(0),
                grad_context.stride(1),
                grad_context.stride(2),
                grad_values,
                grad_values.stride(0),
                grad_values.stride(1),
                grad_values.stride(2),
                N,
                T_DST,
                T_SRC,
                HID,
                K,
                BLOCK_HID,
            )

            # print(grad_values.abs().sum())

        # for probs
        if ctx.needs_input_grad[3]:
            grid = (N, T_DST, K)
            BLOCK_HID = triton.next_power_of_2(HID)

            grad_probs = torch.zeros(
                (N, T_DST, K),
                device=probs.device,
                dtype=probs.dtype,
            )

            _sdbmm_compute_bwd_probs[grid](
                indices,
                indices.stride(0),
                indices.stride(1),
                indices.stride(2),
                values,
                values.stride(0),
                values.stride(1),
                values.stride(2),
                grad_context,
                grad_context.stride(0),
                grad_context.stride(1),
                grad_context.stride(2),
                grad_probs,
                grad_probs.stride(0),
                grad_probs.stride(1),
                grad_probs.stride(2),
                N,
                T_DST,
                T_SRC,
                HID,
                K,
                BLOCK_HID,
            )

        return grad_values, grad_indices, grad_ks, grad_probs


def sparse_attention(
    # attention values
    values: Tensor,
    # attention matrix
    indices: Tensor,
    ks: Tensor,
    probs: Tensor,
) -> Tensor:
    return SparseAttentionAutoGradFn.apply(
        values,
        indices,
        ks,
        probs,
    )


def hip_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    w_start: int = None,
    n_patches: int = None,
    mask_k: int = 256,
    scale_up: float = 2,
    # heuristics: mask_k == n_patches * scale_up
    # heuristics: mask_k == w_start * scale_up
):
    if w_start is None:
        w_start = math.ceil(mask_k * scale_up)
        # w_start = mask_k
    if n_patches is None:
        n_patches = math.ceil(mask_k / scale_up)

    assert q.ndim == 3
    assert k.ndim == 3
    assert v.ndim == 3
    N, T_DST, HID = q.shape
    _N, T_SRC, _HID = k.shape
    assert k.shape[:-1] == v.shape[:-1]
    assert N == _N
    assert HID == _HID

    assert q.dtype == k.dtype
    assert q.dtype == v.dtype

    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not v.is_contiguous():
        v = v.contiguous()

    with timer("hip_attention"):
        with timer("attention_matrix"):
            indices, ks, probs = attention_matrix(
                q,
                k,
                w_start,
                n_patches,
                mask_k,
                scale_up,
            )

        with timer("sparse_attention"):
            context = sparse_attention(
                v,
                indices,
                ks,
                probs,
            )

    # context_avg = v.cumsum(1) / torch.arange(0, v.shape[1], device=v.device)[None, :, None]
    # context_avg = context_avg[:, T_SRC-T_DST:, :]

    # # context = context * 0.975 + context_avg * 0.025
    # min_probs, _ = torch.topk(probs, k=3, dim=-1, largest=False)
    # t_srcs = torch.arange(T_SRC-T_DST, T_SRC, device=v.device) + 1
    # min_probs = torch.clamp(min_probs.mean(-1, keepdim=True) * (t_srcs[None, :, None] - mask_k) * 0.2, 0, 0.2)
    # # print(min_probs)
    # context = context * (1 - min_probs) + context_avg * min_probs

    return context, (indices, ks, probs)
