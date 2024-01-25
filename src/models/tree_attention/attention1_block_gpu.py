"""
block version of attention1
score = reduce_fn(score[block_ptrs])

k = 256 (16 block)
scale_up = 2

# infer by heuristics
n_patches = 128 (8 block)
w_start = 512 (32 block)

> example of block scale
1024: 64 block
2048: 128 block
4096: 256 block
"""

from matplotlib import pyplot as plt
import numpy as np
import skimage.measure
import skimage
import torch
from torch import Tensor
import tqdm
import triton
import triton.language as tl
from typing import Tuple, List
import os
import math
from torch.autograd import Function

from src.utils import get_bench, seed
from src.models.tree_attention.common import load_checkouts

timer = lambda x: get_bench().region(x)

@triton.jit
def _triton_kth_large(
    scores: tl.tensor, k: tl.tensor,
    BLOCK_SCORES: tl.constexpr,
) -> tl.tensor:
    sorted_score = tl.sort(scores)
    # tl.debug_barrier()
    sorted_score_mask = tl.arange(0, BLOCK_SCORES) < k
    return tl.max(sorted_score * sorted_score_mask + (-32000.0) * (~sorted_score_mask))

@triton.jit
def _masking_iteration_compute(
    # input matrices
    QUERIES, stride_queries_n, stride_queries_tdst, stride_queries_hid,
    KEYS, stride_keys_n, stride_keys_tsrc, stride_keys_hid,
    
    # input metrices (blocked)
    MASK, stride_mask_n, stride_mask_bdst, stride_mask_k,
    TMASK, stride_tmask_n, stride_tmask_bdst, stride_tmask_k,
    
    # temp vectors (blocked)
    WS, stride_ws_n, stride_ws_bdst,
    KS, stride_ks_n, stride_ks_bdst,
    TSRCS, stride_tsrcs_n, stride_tsrcs_bdst,
    
    # operation variables (blocked)
    SCALE_UP: float, N_PATCHES: int, MASK_K: int,
    
    # input variables
    N: int, T_DST: int, T_SRC: int, B_DST: int, B_SRC: int, HID: int,
    
    # block constant
    REDUCE_METHOD: tl.constexpr,
    BLOCK_MASK_K: tl.constexpr, 
    BLOCK_TMASK_K: tl.constexpr, 
    BLOCK_MAX_DUP: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_PADDED: tl.constexpr,
):
    idx_n = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    """ non blocked
    # for each query
    w_old = ws[i, j, 0]
    t_src = t_srcs[i, j, 0]
    w_new = min(torch.round(w_old * scale_up), t_src)
    """
    # tl.device_print("dd", idx_bdst)
    w_old = tl.load(
        WS + \
            idx_n * stride_ws_n + \
            idx_bdst * stride_ws_bdst,
    )
    
    t_src = tl.load(
        TSRCS + \
            idx_n * stride_tsrcs_n + \
            idx_bdst * stride_tsrcs_bdst,
    )
    
    w_new = tl.minimum(
        tl.math.round(w_old.to(tl.float32) * SCALE_UP.to(tl.float32)).to(tl.float32), 
        t_src
    ).to(tl.int64)
    
    """
    if w_old != w_new:
    """
    if w_old == w_new:
        return

    """
    k_old = ks[i, j, 0]
    k_new = max(n_patches, int(min(mask_k * BLOCK_SIZE / t_src, 1.0) * w_new) c/ BLOCK_SIZE)
    k_new = min(t_src c/ BLOCK_SIZE, max(n_patches, k_new))
    """
    
    k_old = tl.load(
        KS + \
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst,
    ).to(tl.int64)
    k_new = tl.maximum(
        N_PATCHES,
        (
            tl.minimum(
                MASK_K.to(tl.float32) / tl.cdiv(t_src, BLOCK_SIZE).to(tl.float32),
                1.0
            ) * tl.cdiv(w_new, BLOCK_SIZE)
        ).to(tl.int64),
    )
    # tl.device_print("before", t_src)
    k_new = tl.minimum(tl.cdiv(t_src, BLOCK_SIZE), tl.maximum(N_PATCHES, k_new))
    
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
    k_old_mask = tl.arange(0, BLOCK_MASK_K) < k_old
    # tl.debug_barrier()
    loc_vec = tl.load(
        MASK +\
            idx_n * stride_mask_n +\
            idx_bdst * stride_mask_bdst +\
            k_old_range * stride_mask_k,
        mask = k_old_mask,
        other = 0
    )
    
    b_old_fp = tl.cdiv(w_old, BLOCK_SIZE).to(tl.float32)
    b_new_fp = tl.cdiv(w_new, BLOCK_SIZE).to(tl.float32)
    loc_idx_start_vec = (loc_vec * b_old_fp).to(tl.int64)
    loc_idx_end_vec = loc_idx_start_vec + 1
    loc_idx_start_vec = (loc_idx_start_vec.to(tl.float32) / b_old_fp * b_new_fp).to(tl.int64)
    loc_idx_end_vec = (loc_idx_end_vec.to(tl.float32) / b_old_fp * b_new_fp).to(tl.int64)
    
    dup_pixels_vec = loc_idx_end_vec - loc_idx_start_vec
    dup_pixels_vec = dup_pixels_vec * k_old_mask
    num_pixels_vec = tl.cumsum(dup_pixels_vec)
    dup_pixels_first = tl.min(num_pixels_vec)
    num_pixels_scalar = tl.max(num_pixels_vec)
    
    # NOTE: compiler bug?
    
    """
    dup_pixels_range = tl.arange(0, BLOCK_MAX_DUP)
    dup_pixels_mask = (dup_pixels_range[None, :] <= dup_pixels_vec[:, None]) & k_old_mask[:, None]
    
    tl.store(
        TMASK + \
            idx_n * stride_tmask_n +\
            idx_bdst * stride_tmask_bdst +\
            ((num_pixels_vec - dup_pixels_first)[:, None] + dup_pixels_range[None, :]) * stride_tmask_k,
        # mask=dup_pixels_mask,
        value=(
            (loc_idx_start_vec[:, None] + tl.arange(0, BLOCK_MAX_DUP)[None, :]).to(tl.float32) / w_new.to(tl.float32)
        )
        # value = num_pixels_scalar=
    )
    """
    
    for _idx in range(BLOCK_MAX_DUP):
        tl.store(
            TMASK + \
                idx_n * stride_tmask_n +\
                idx_bdst * stride_tmask_bdst +\
                ((num_pixels_vec - dup_pixels_first) + _idx) * stride_tmask_k,
            mask=(_idx <= dup_pixels_vec) & k_old_mask,
            value=(
                (loc_idx_start_vec + _idx).to(tl.float32) / tl.cdiv(w_new, BLOCK_SIZE).to(tl.float32)
            )
        )
    
    """
    # t_mask -> mask (using scores)
    if k_new < num_pixels:
    """
    if k_new < num_pixels_scalar and True:
    # if False:
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
        
        if REDUCE_METHOD == 'first':
            for _idx_hid in range(tl.cdiv(HID, BLOCK_HID)):
                hid_range = tl.arange(0, BLOCK_HID) + _idx_hid * BLOCK_HID
                hid_mask = hid_range < HID
                vec_q = tl.load(
                    QUERIES +\
                        idx_n * stride_queries_n +\
                        (idx_bdst * BLOCK_SIZE) * stride_queries_tdst +\
                        (hid_range[None, :] + tl.arange(0, 16)[:, None]) * stride_queries_hid,
                    mask = (hid_mask[None, :] & (tl.arange(0, 16)[:, None] < 1)),
                    other = 0,
                )
                # tl.debug_barrier()
                
                num_pixels_range = tl.arange(0, BLOCK_TMASK_K)
                num_pixels_mask = num_pixels_range < num_pixels_scalar
                loc_k_vec = tl.load(
                    TMASK +\
                        idx_n * stride_tmask_n +\
                        idx_bdst * stride_tmask_bdst +\
                        num_pixels_range * stride_tmask_k,
                    mask = num_pixels_mask,
                    other = 0,
                )
                # tl.debug_barrier()
                # NOTE: random key selection with in the block
                # loc_k_vec = loc_k_vec.to(tl.float32) + tl.rand(idx_n * idx_tdst, w_old, 10) * (1.0 / w_old)
                loc_k_vec = (loc_k_vec.to(tl.float32) * t_src.to(tl.float32)).to(tl.int64)
                vec_k_mask = num_pixels_mask[None, :] & hid_mask[:, None]
                vec_k = tl.load(
                    KEYS +\
                        idx_n * stride_keys_n +\
                        loc_k_vec[None, :] * stride_keys_tsrc + \
                        hid_range[:, None] * stride_keys_hid,
                    mask = vec_k_mask,
                    other = 0,
                )
                # tl.debug_barrier()
                
                # TODO: support tensorCore
                # scores = -tl.dot(vec_q, vec_k) # NOTE: negative scores
                # 1x128 @ 128x512 512x128 @ 128x1
                # scores = -tl.sum(
                #     vec_q * vec_k, 
                #     axis=0,
                # )
                scores_partial = -tl.dot(vec_q, vec_k)
                scores_partial = tl.sum(scores_partial, axis=0)
                scores_partial = scores_partial + (~num_pixels_mask) * 32000.0
                scores_partial = scores_partial +\
                    ((idx_bdst * BLOCK_SIZE) < loc_k_vec) * 32000.0
                
                scores += scores_partial.to(scores.dtype)
        elif REDUCE_METHOD == 'max' or REDUCE_METHOD == 'sum':
            # NOTE: init scores
            if REDUCE_METHOD == 'max':
                scores += 32000.0
            elif REDUCE_METHOD == 'sum':
                scores *= 0.0
            
            for _idx_block in range(BLOCK_SIZE):
                hid_range = tl.arange(0, BLOCK_HID)
                hid_mask = hid_range < HID
                # [BLOCK_SIZE_PADDED, BLOCK_HID]
                vec_q = tl.load(
                    QUERIES +\
                        idx_n * stride_queries_n +\
                        (idx_bdst * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_PADDED)[:, None]) * stride_queries_tdst +\
                        hid_range[None, :] * stride_queries_hid,
                    mask = (hid_mask[None, :] & (tl.arange(0, BLOCK_SIZE_PADDED)[:, None] < BLOCK_SIZE)),
                    other = 0,
                )
                # tl.debug_barrier()
                
                num_pixels_range = tl.arange(0, BLOCK_TMASK_K)
                num_pixels_mask = num_pixels_range < num_pixels_scalar
                loc_k_vec = tl.load(
                    TMASK +\
                        idx_n * stride_tmask_n +\
                        idx_bdst * stride_tmask_bdst +\
                        num_pixels_range * stride_tmask_k,
                    mask = num_pixels_mask,
                    other = 0,
                )
                # tl.debug_barrier()
                # NOTE: random key selection with in the block
                # loc_k_vec = loc_k_vec.to(tl.float32) + tl.rand(idx_n * idx_tdst, w_old, 10) * (1.0 / w_old)
                loc_k_vec = (loc_k_vec.to(tl.float32) * t_src.to(tl.float32)).to(tl.int64)
                vec_k_mask = num_pixels_mask[None, :] & hid_mask[:, None]
                
                # [BLOCK_HID, BLOCK_TMASK_K]
                vec_k = tl.load(
                    KEYS +\
                        idx_n * stride_keys_n +\
                        (loc_k_vec[None, :] + _idx_block) * stride_keys_tsrc + \
                        hid_range[:, None] * stride_keys_hid,
                    mask = vec_k_mask,
                    other = 0,
                )
                
                # [BLOCK_SIZE_PADDED, BLOCK_TMASK_K]
                scores_partial = -tl.dot(vec_q, vec_k)
                scores_partial = scores_partial + (~num_pixels_mask) * 32000.0
                scores_partial = scores_partial + (
                    (idx_bdst * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_PADDED)[:, None]) <\
                    (loc_k_vec[None, :] + _idx_block)
                ) * 32000.0
                
                # NOTE: reduce
                if REDUCE_METHOD == 'max':
                    scores_partial = tl.min(scores_partial, axis=0)
                    scores = tl.minimum(scores, scores_partial)
                elif REDUCE_METHOD == 'sum':
                    scores_partial = tl.sum(scores_partial, axis=0)
                    scores = scores + scores_partial
        else:
            raise Exception()
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
        scores_kth_large = _triton_kth_large(masked_scores, k_new, BLOCK_TMASK_K)
        # tl.debug_barrier()
        topk_mask = masked_scores <= scores_kth_large
        topk_mask_cumsum = tl.cumsum(topk_mask.to(tl.int64))
        # tl.debug_barrier()
        topk_range = tl.minimum((topk_mask_cumsum - 1) * topk_mask, k_new - 1)
        # tl.debug_barrier()
        
        temp_range = tl.arange(0, BLOCK_TMASK_K)
        temp_mask = temp_range < num_pixels_scalar
        temp = tl.load(
            TMASK +\
                idx_n * stride_tmask_n +\
                idx_bdst * stride_tmask_bdst +\
                temp_range * stride_tmask_k,
            mask=temp_mask,
            other=0
        )
        # tl.debug_barrier()
        tl.store(
            MASK +\
                idx_n * stride_mask_n +\
                idx_bdst * stride_mask_bdst +\
                topk_range * stride_mask_k,
            mask=topk_mask & temp_mask,
            value=temp,
            # value=0.1,
        )
        # tl.debug_barrier()
        pass
    else:
        """
        else:
            mask[i, j, :num_pixels] = t_mask[i, j, :num_pixels]
        """
        temp1_range = tl.arange(0, BLOCK_MASK_K)
        temp1_mask = temp1_range < num_pixels_scalar
        # tl.debug_barrier()
        temp1 = tl.load(
            TMASK +\
                idx_n * stride_tmask_n +\
                idx_bdst * stride_tmask_bdst +\
                temp1_range * stride_tmask_k,
            mask=temp1_mask,
        )
        
        # tl.debug_barrier()
        tl.store(
            MASK +\
                idx_n * stride_mask_n +\
                idx_bdst * stride_mask_bdst +\
                temp1_range * stride_mask_k,
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
        WS +\
            idx_n * stride_ws_n +\
            idx_bdst * stride_ws_bdst,
        value = w_new
    )
    # tl.debug_barrier()
    tl.store(
        KS +\
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst,
        value = tl.minimum(k_new, num_pixels_scalar)
        # value = k_new,
        # value = num_pixels_scalar,
    )
    # tl.debug_barrier()

DEBUG = False

def next_multiple_of(x: int, multiple_by: int = 16):
    if (x % multiple_by) == 0:
        return x
    return x + multiple_by - (x % multiple_by)

def masking_iteration(
    # input matrices
    queries: Tensor, keys: Tensor,
    # input metrices (blocked) 
    mask: Tensor, t_mask: Tensor, 
    # temp vectors (blocked)
    ws: Tensor, ks: Tensor, t_srcs: Tensor, 
    # operator variables
    scale_up: float, n_patches: int, mask_k: int, 
    # input constant
    N: int, T_DST: int, T_SRC: int, B_DST: int, B_SRC: int, HID: int,
    # kernel constant
    BLOCK_SIZE: int, 
    REDUCE_METHOD: str = 'max',
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
    GROUP_BDST = 1
    BLOCK_HID = triton.next_power_of_2(HID)
    grid = (triton.cdiv(N, GROUP_N), triton.cdiv(B_DST, GROUP_BDST))
    
    assert GROUP_N == 1
    assert GROUP_BDST == 1
    
    # HID cannot be chunked if use reduce
    if REDUCE_METHOD in ['max', 'sum']:
        assert HID <= BLOCK_HID
    assert REDUCE_METHOD in ['max', 'sum', 'first']
    
    _masking_iteration_compute[grid](
        # input matrices
        queries, queries.stride(0), queries.stride(1), queries.stride(2),
        keys, keys.stride(0), keys.stride(1), keys.stride(2),
        
        # input matrices (blocked)
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        t_mask, t_mask.stride(0), t_mask.stride(1), t_mask.stride(2),
        
        # temp vectors (blocked)
        ws, ws.stride(0), ws.stride(1),
        ks, ks.stride(0), ks.stride(1),
        t_srcs, t_srcs.stride(0), t_srcs.stride(1),
        
        # operation variables
        float(scale_up), int(n_patches), int(mask_k),
        
        # input variables
        N, T_DST, T_SRC, int(B_DST), int(B_SRC), HID,
        
        # block constant
        REDUCE_METHOD,
        triton.next_power_of_2(mask.shape[-1]),
        triton.next_power_of_2(t_mask.shape[-1]),
        triton.next_power_of_2(math.ceil(scale_up)),
        int(BLOCK_HID),
        int(BLOCK_SIZE),
        next_multiple_of(BLOCK_SIZE, 16),
        
        num_warps=4,
        num_stages=2,
        enable_warp_specialization=False,
    )

def torch_cdiv(a, b):
    return torch.floor(a / b) + torch.ceil((a / b) - torch.floor(a / b))

@triton.jit
def _safe_indices_compute(
    INDICES, stride_indices_n, stride_indices_k,
    N, K,
    BLOCK_N: tl.constexpr,
): 
    pid_n = tl.program_id(0)
    idx_n = tl.arange(0, BLOCK_N) + pid_n * BLOCK_N
    mask_n = idx_n < N
    
    last_col = tl.zeros((BLOCK_N, ), dtype=tl.int64) - 1
    for idx_k in range(K):
        col = tl.load(
            INDICES +\
                idx_n * stride_indices_n +\
                idx_k * stride_indices_k,
            mask = mask_n,
            other = 0,
        )
        col = tl.maximum(last_col + 1, col)
        tl.store(
            INDICES +\
                idx_n * stride_indices_n +\
                idx_k * stride_indices_k,
            value = col,
            mask = mask_n
        )
        last_col = col

def safe_indices(indices):
    N, TDST, K = indices.shape
    indices = indices.reshape(N*TDST, K).clone()
    
    BLOCK_BATCH = 32
    grid = (triton.cdiv(N*TDST, BLOCK_BATCH), )
    
    assert indices.ndim == 2
    _safe_indices_compute[grid](
        indices, *indices.stride(),
        N*TDST, K,
        BLOCK_BATCH,
        num_warps=1,
    )
    
    indices = indices.reshape(N, TDST, K)
    
    return indices

@triton.jit
def _calc_score_compute(
    # input matrix
    QUERIES, stride_queries_n, stride_queries_tdst, stride_queries_hid,
    KEYS, stride_keys_n, stride_keys_tsrc, stride_keys_hid,
    
    # block indices
    INDICES, stride_indices_n, stride_indices_bdst, stride_indices_k,
    KS, stride_ks_n, stride_ks_bdst,
    
    # out matrix
    SCORES, stride_scores_n, stride_scores_tdst, stride_scores_k,
    
    # input variables
    N, TDST, TSRC, HID, K, BDST, BSRC,
    
    # kernel constatnts
    BLOCK_SIZE,
    BLOCK_SIZE_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    idx_n = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_k = tl.program_id(2)
    
    ks = tl.load(
        KS +\
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst,
    )
    
    if idx_k >= ks:
        return
    
    idx_tsrc = tl.load(
        INDICES +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_k * stride_indices_k,
    )
    
    idx_tsrc = idx_tsrc + tl.arange(0, BLOCK_SIZE_PADDED)
    mask_tsrc = idx_tsrc < TSRC
    
    idx_tdst = idx_bdst * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_PADDED)
    mask_tdst = idx_tdst < TDST
    
    scores = tl.zeros((BLOCK_SIZE_PADDED, BLOCK_SIZE_PADDED), dtype=tl.float32)
    for pid_hid in range(tl.cdiv(HID, BLOCK_HID)):
        idx_hid = tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID
        mask_hid = idx_hid < HID
        
        # [BLOCK_SIZE_PADDED, BLOCK_HID]
        queries = tl.load(
            QUERIES +\
                idx_n * stride_queries_n +\
                idx_tdst[:, None] * stride_queries_tdst +\
                idx_hid[None, :] * stride_queries_hid,
            mask = mask_tdst[:, None] & mask_hid[None, :],
            other = 0
        )
        
        # [BLOCK_HID, BLOCK_SIZE_PADDED]
        keys = tl.load(
            KEYS +\
                idx_n * stride_keys_n +\
                idx_tsrc[None, :] * stride_keys_tsrc +\
                idx_hid[:, None] * stride_queries_hid,
            mask = mask_tsrc[None, :] & mask_hid[:, None],
            other = 0
        )
        
        scores_mini = tl.dot(queries, keys)
        scores += scores_mini.to(scores.dtype)
    
    idx_scorek = (idx_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_PADDED))
    mask_scorek = idx_scorek < (K * BLOCK_SIZE)
    
    tl.store(
        SCORES +\
            idx_n * stride_scores_n +\
            idx_tdst[:, None] * stride_scores_tdst +\
            idx_scorek[None, :] * stride_scores_k,
        mask = mask_tdst[:, None] & mask_scorek[None, :] & (idx_tdst[:, None] >= idx_tsrc[None, :]),
        value = scores,
    )

@triton.jit
def _calc_score_compute_bwd_queries(
    # input matrices
    ks, stride_ks_n, stride_ks_bdst,
    indices, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    keys, stride_keys_n, stride_keys_tsrc, stride_keys_hid,
    # grad output (read)
    grad_scores, stride_grad_scores_n, stride_grad_scores_tdst, stride_grad_scores_k,
    # grad input (write)
    grad_queries, stride_grad_queries_n, stride_grad_queries_tdst, stride_grad_queries_hid,
    # input variables
    N, TDST, TSRC, HID, BK, K,
    # block constant
    BLOCK_SIZE,
    BLOCK_SIZE_PADDED: tl.constexpr,
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
    idx_bdst = tl.program_id(1)
    
    scalar_ks = tl.load(
        ks +\
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst
    )
    
    mask_block = tl.arange(0, BLOCK_SIZE_PADDED) < BLOCK_SIZE
    
    idx_tdst = (idx_bdst * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_PADDED))
    mask_tdst = (idx_tdst < TDST) & mask_block
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    
    accumulator = tl.zeros((BLOCK_SIZE_PADDED, BLOCK_HID,), dtype=tl.float32)
    for idx_bk in range(BK):
        idx_tsrc = tl.load(
            indices + \
                idx_n * stride_indices_n + \
                idx_bdst * stride_indices_bdst + \
                idx_bk * stride_indices_bk,
        )
        
        idx_tsrc = idx_tsrc + tl.arange(0, BLOCK_SIZE_PADDED)
        mask_tsrc = (idx_tsrc < TSRC) & mask_block
        
        idx_k = idx_bk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_PADDED)
        mask_k = (idx_k < K) & mask_block
        
        # [BLOCK_SIZE_PADDED: tdst, BLOCK_SIZE_PADDED: score]
        grad_score = tl.load(
            grad_scores +\
                idx_n * stride_grad_scores_n +\
                idx_tdst[:, None] * stride_grad_scores_tdst + \
                idx_k[None, :] * stride_grad_scores_k,
            mask = mask_tdst[:, None] & (mask_tsrc & mask_k)[None, :],
            other = 0,
        )
        
        # [BLOCK_SIZE_PADDED: score, BLOCK_HID: hid]
        key = tl.load(
            keys +\
                idx_n * stride_keys_n +\
                idx_tsrc[:, None] * stride_keys_tsrc +\
                idx_hid[None, :] * stride_keys_hid,
            mask = mask_hid[None, :] & (mask_tsrc & mask_k)[:, None],
            other = 0
        )
        
        # tl.device_print("", idx_tsrc)
        accumulator += tl.dot(grad_score, key).to(accumulator.dtype)
    
    tl.store(
        grad_queries +\
            idx_n * stride_grad_queries_n +\
            idx_tdst[:, None] * stride_grad_queries_tdst +\
            idx_hid[None, :] * stride_grad_queries_hid,
        mask = mask_hid[None, :] & mask_tdst[:, None],
        value = accumulator
    )

@triton.jit
def _calc_score_compute_bwd_keys(
    # input matrices
    ks, stride_ks_n, stride_ks_bdst,
    indices, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    queries, stride_queries_n, stride_queries_tdst, stride_queries_hid,
    # grad output (read)
    grad_scores, stride_grad_scores_n, stride_grad_scores_tdst, stride_grad_scores_k,
    # grad input (write)
    grad_keys, stride_grad_keys_n, stride_grad_keys_tsrc, stride_grad_keys_hid,
    # input variables
    N, TDST, TSRC, HID, BK, K,
    # block constant
    BLOCK_SIZE,
    BLOCK_SIZE_PADDED: tl.constexpr,
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
    idx_bdst = tl.program_id(1)
    idx_bk = tl.program_id(2)
    
    scalar_ks = tl.load(
        ks +\
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst,
    )
    # mask_job = idx_bk < scalar_ks
    if idx_bk >= scalar_ks: return
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = (idx_hid < HID)
    
    idx_block = tl.arange(0, BLOCK_SIZE_PADDED)
    mask_block = idx_block < BLOCK_SIZE
    
    idx_tdst = idx_bdst * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_PADDED)
    mask_tdst = (idx_tdst < TDST) & mask_block
    
    idx_k = idx_bk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_PADDED)
    mask_k = (idx_k < K) & mask_block
    
    # [BLOCK_SIZE_PADDED: tsrc, BLOCK_SIZE_PADDED: tdst]
    grad_score = tl.load(
        grad_scores +\
            idx_n * stride_grad_scores_n +\
            idx_tdst[None, :] * stride_grad_scores_tdst +\
            idx_k[:, None] * stride_grad_scores_k,
        mask = mask_tdst[None, :] & mask_k[:, None],
        other = 0
    )
    # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
    query = tl.load(
        queries +\
            idx_n * stride_queries_n +\
            idx_tdst[:, None] * stride_queries_tdst +\
            idx_hid[None, :] * stride_queries_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        other = 0,
    )
    # [BLOCK_SIZE_PADDED: tsrc, BLOCK_HID: hid]
    scores = tl.dot(grad_score, query)
    
    idx_tsrc = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
    )
    idx_tsrc = idx_tsrc + idx_block
    mask_tsrc = (idx_tsrc < TSRC) & mask_block
    tl.atomic_add(
        grad_keys +\
            idx_n * stride_grad_keys_n +\
            idx_tsrc[:, None] * stride_grad_keys_tsrc +\
            idx_hid[None, :] * stride_grad_keys_hid,
        val = scores,
        mask = mask_tsrc[:, None] & mask_hid[None, :]
    )

# NOTE: you have to perform softmax after this
class CalcScoreAutoGradFn(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx, 
        # matrices
        queries: Tensor, keys: Tensor,
        # indices matrices
        indices: Tensor, ks: Tensor,
        # block constant
        BLOCK_SIZE: int,
    ):
        ctx.save_for_backward(queries, keys, indices, ks)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        
        N, TDST, HID = queries.shape
        _, TSRC, _ = keys.shape
        _, _, K = indices.shape
        
        BDST = triton.cdiv(TDST, BLOCK_SIZE)
        BSRC = triton.cdiv(TSRC, BLOCK_SIZE)
        
        assert keys.shape == (N, TSRC, HID)
        assert indices.shape == (N, BDST, K)
        assert ks.shape == (N, BDST)
        
        scores = torch.full(
            (N, TDST, K * BLOCK_SIZE), 
            torch.finfo(queries.dtype).min,
            device=queries.device, 
            dtype=queries.dtype
        )
        
        BLOCK_HID = 32
        BLOCK_SIZE_PADDED = next_multiple_of(BLOCK_SIZE, 16)
        grid = (N, BDST, K)
        
        assert queries.ndim == 3
        assert keys.ndim == 3
        assert indices.ndim == 3
        assert ks.ndim == 2
        assert scores.ndim == 3
        _calc_score_compute[grid](
            queries, *queries.stride(),
            keys, *keys.stride(),
            
            indices, *indices.stride(),
            ks, *ks.stride(),
            
            scores, *scores.stride(),
            
            N, TDST, TSRC, HID, K, BDST, BSRC,
            
            BLOCK_SIZE,
            BLOCK_SIZE_PADDED,
            BLOCK_HID
        )
        
        # print(scores[0, 300, :])
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        queries, keys, indices, ks = ctx.saved_tensors
        BLOCK_SIZE = ctx.BLOCK_SIZE
        grad_queries = grad_keys = None
        
        N, T_DST, HID = queries.shape
        _, T_SRC, _HID = keys.shape
        assert HID == _HID
        _, _, BK = indices.shape
        _, _, K = grad_scores.shape

        # for queries
        if ctx.needs_input_grad[0]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE))
            BLOCK_HID = triton.next_power_of_2(HID)
            
            grad_queries = torch.zeros_like(queries)
            
            _calc_score_compute_bwd_queries[grid](
                ks, ks.stride(0), ks.stride(1),
                indices, indices.stride(0), indices.stride(1), indices.stride(2), 
                keys, keys.stride(0), keys.stride(1), keys.stride(2),
                
                grad_scores, grad_scores.stride(0), grad_scores.stride(1), grad_scores.stride(2),
                
                grad_queries, grad_queries.stride(0), grad_queries.stride(1), grad_queries.stride(2),
                
                N, T_DST, T_SRC, HID, BK, K,
                
                BLOCK_SIZE,
                next_multiple_of(BLOCK_SIZE, 16),
                BLOCK_HID,
            )
        
        # for keys
        if ctx.needs_input_grad[1]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE), BK)
            BLOCK_HID = triton.next_power_of_2(HID)
            
            grad_keys = torch.zeros_like(keys)
            
            _calc_score_compute_bwd_keys[grid](
                ks, ks.stride(0), ks.stride(1),
                indices, indices.stride(0), indices.stride(1), indices.stride(2), 
                queries, queries.stride(0), queries.stride(1), queries.stride(2),
                
                grad_scores, grad_scores.stride(0), grad_scores.stride(1), grad_scores.stride(2),
                
                grad_keys, grad_keys.stride(0), grad_keys.stride(1), grad_keys.stride(2),
                
                N, T_DST, T_SRC, HID, BK, K,
                
                BLOCK_SIZE,
                next_multiple_of(BLOCK_SIZE, 16),
                BLOCK_HID,
            )
        
        return (
            grad_queries, 
            grad_keys, 
            None, 
            None, 
            None,
            None,
        )

def calc_score_return_prob(
    queries: Tensor, keys: Tensor, 
    indices: Tensor, ks: Tensor,
    
    BLOCK_SIZE: int
):
    scores = CalcScoreAutoGradFn.apply(
        queries, keys,
        indices, ks,
        
        BLOCK_SIZE,
    )
    
    probs = scores.softmax(-1)    
    return scores, probs

def attention_matrix(
    queries: Tensor, 
    keys: Tensor,
    
    w_start: int,
    n_patches: int,
    mask_k: int,
    scale_up: int,
    
    BLOCK_SIZE: int = 16,
) -> Tuple[Tensor, Tensor, Tensor]:
    global DEBUG
    
    if DEBUG:
        os.makedirs('saves/models/tree_attention/', exist_ok=True)
    
    dtype = queries.dtype
    device = queries.device
    assert queries.device == keys.device
    
    N, T_DST, HID = queries.shape
    _, T_SRC, _ = keys.shape
    assert T_DST <= T_SRC
    
    # NOTE: width of last query
    w_curr = round(w_start / scale_up)
        
    # vectors
    tsrcs = torch.arange( # NOTE: store non blocked tsrc
        T_SRC-T_DST+1, T_SRC+1, 1, 
        dtype=torch.int64,
        device=device,
    )\
        .view(1, T_DST)\
        .expand(N, T_DST)\
        .contiguous()[:, ::BLOCK_SIZE]\
        .contiguous()
    ws = torch.clamp(tsrcs, 0, w_curr).to(torch.int64) # NOTE: store non blocked width
    ks = torch.ceil(ws / BLOCK_SIZE).to(torch.int64) # NOTE: store num blocks
    
    # matrices
    # NOTE: float16 -> int32 seems not possible
    mask_k_block = triton.cdiv(mask_k, BLOCK_SIZE)
    mask = (torch.arange(mask_k_block, device=device).view(1, 1, mask_k_block) / ks.unsqueeze(-1)).to(torch.float32)
    tmask = torch.zeros((mask.shape[0], mask.shape[1], mask_k_block * math.ceil(scale_up)), dtype=mask.dtype, device=device)
    
    B_SRC = triton.cdiv(T_SRC, BLOCK_SIZE)
    B_DST = triton.cdiv(T_DST, BLOCK_SIZE)
    
    def to_dense_blocked(
        indices: np.ndarray, 
        ks: np.ndarray, 
        value,
        mask, 
        N, T_DST, T_SRC, BLOCK_SIZE
    ):
        out = np.zeros((N, triton.cdiv(T_DST, BLOCK_SIZE), triton.cdiv(T_SRC, BLOCK_SIZE)))
        for idx_n in range(1):
            for idx_bdst in range(out.shape[1]):
                for k in range(indices.shape[2]):
                    if k < ks[idx_n, idx_bdst]:
                        assert out[idx_n, idx_bdst, indices[idx_n, idx_bdst, k]] == 0, f"{out[idx_n, idx_bdst, indices[idx_n, idx_bdst, k]]}, {ks[idx_n, idx_bdst]}, {indices[idx_n, idx_bdst, :]}, {mask[idx_n, idx_bdst, :]}"
                        out[idx_n, idx_bdst, indices[idx_n, idx_bdst, k]] = 1
        return out

    def to_dense(
        indices: np.ndarray, 
        ks: np.ndarray, 
        value: np.ndarray,
        N, T_DST, T_SRC, BLOCK_SIZE
    ):
        # print(indices.shape, ks.shape, value.shape, T_DST, T_SRC)
        out = np.zeros((N, T_DST, T_SRC))
        for idx_n in range(1):
            for idx_bdst in range(indices.shape[1]):
                for idx_k in range(indices.shape[2]):
                    if idx_k < ks[idx_n, idx_bdst]:
                        idx_tsrc = indices[idx_n, idx_bdst, idx_k]
                        out[
                            idx_n, 
                            idx_bdst * BLOCK_SIZE: (idx_bdst + 1) * BLOCK_SIZE, 
                            idx_tsrc: idx_tsrc + BLOCK_SIZE
                        ] = value[
                            idx_n,
                            idx_bdst * BLOCK_SIZE: (idx_bdst + 1) * BLOCK_SIZE, 
                            idx_k * BLOCK_SIZE: (idx_k + 1) * BLOCK_SIZE
                        ]
        return out
    
    def debug_print():
        indices = torch_cdiv(mask * ws.unsqueeze(-1), BLOCK_SIZE).to(torch.int32)
        indices = safe_indices(indices)
        indices = torch.clamp(indices, 0, triton.cdiv(T_SRC, BLOCK_SIZE) - 1)
        x = to_dense_blocked(
            indices.cpu().numpy(),
            ks.cpu().unsqueeze(-1).numpy(), 
            None,
            mask,
            N, T_DST, T_SRC, BLOCK_SIZE
        )[0]
        x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
        plt.imshow(x)
        path = f'saves/models/tree_attention/block_{w_curr}.png'
        # path = f'saves/models/tree_attention/block.png'
        print('saved', path)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        # input('>>>')
    
    # NOTE: Calc. Mask
    while w_curr < T_SRC:
        tmask.fill_(0)
        mask.clamp_(0, (triton.cdiv(w_curr, BLOCK_SIZE) - 1) / triton.cdiv(w_curr, BLOCK_SIZE))
        masking_iteration(
            # input matrices
            queries, keys,
            # input metrices (blocked) 
            mask, tmask, 
            # temp vectors (blocked)
            ws, ks, tsrcs, 
            # operator variables
            scale_up, triton.cdiv(n_patches, BLOCK_SIZE), triton.cdiv(mask_k, BLOCK_SIZE), 
            # input constant
            N, T_DST, T_SRC, B_DST, B_SRC, HID,
            # kernel constant
            BLOCK_SIZE
        )
        w_curr = round(w_curr * scale_up)
        
        if DEBUG:
            debug_print()
    
    # NOTE: align with blocks
    indices = torch_cdiv(mask * ws.unsqueeze(-1), BLOCK_SIZE).to(torch.int32)
    indices = safe_indices(indices)
    indices = torch.clamp(indices, 0, triton.cdiv(T_SRC, BLOCK_SIZE) - 1)
    indices = indices * BLOCK_SIZE
    
    # # NOTE: are you sure this function is the only thing can differentiate?
    scores, probs = calc_score_return_prob(
        queries=queries, keys=keys,
        indices=indices, ks=ks,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    if DEBUG:
        x = to_dense(
            indices.cpu().numpy(),
            ks.cpu().numpy(),
            probs.detach().cpu().numpy(),
            N, T_DST, T_SRC, BLOCK_SIZE,
        )[0]
        x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
        plt.imshow(x)
        path = 'saves/models/tree_attention/block_est.png'
        print('saved', path)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        
        x = np.matmul(
            queries[0].cpu().numpy(), 
            keys[0].cpu().numpy().transpose((-1, -2))
        )
        x = x + (1 - np.tri(*x.shape)) * (-32000)
        x = np.exp(x - x.max(-1, keepdims=True))
        x = x / x.sum(-1, keepdims=True)
        x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
        plt.imshow(x)
        path = 'saves/models/tree_attention/block_truth.png'
        print('saved', path)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        # print(ks)
    
    return indices, ks, probs


@triton.jit
def _sdbmm_compute(
    # inputs
    indices, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    ks, stride_ks_n, stride_ks_bdst, 
    probs, stride_probs_n, stride_probs_tdst, stride_probs_k,
    values, stride_values_n, stride_values_tsrc, stride_values_hid,
    
    # output
    context, stride_context_n, stride_context_tdst, stride_context_hid,
    
    # variables
    N, TSRC, TDST, HID, BK, BSRC, BDST,
    
    # kernel blocks
    BLOCK_SIZE,
    BLOCK_SIZE_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    idx_n = tl.program_id(0)
    
    idx_bdst = tl.program_id(1)
    idx_tdst = idx_bdst * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_PADDED)
    mask_tdst = idx_tdst < TDST
    
    pid_hid = tl.program_id(2)
    idx_hid = tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID
    mask_hid = idx_hid < HID
    
    n_bk = tl.load(
        ks +\
            idx_n * stride_ks_n+\
            idx_bdst * stride_ks_bdst,
    )
    
    scores = tl.zeros((BLOCK_SIZE_PADDED, BLOCK_HID), dtype=tl.float32)
    for idx_bk in range(n_bk):
        atten_block_indices = tl.load(
            indices +\
                idx_n * stride_indices_n +\
                idx_bdst * stride_indices_bdst +\
                idx_bk * stride_indices_bk,
        )
        # atten_indices: [BLOCK_SIZE_PADDED]
        atten_indices = atten_block_indices + tl.arange(0, BLOCK_SIZE_PADDED)
        mask_atten_indices = atten_indices < TSRC
        
        # atten_probs: [BLOCK_SIZE_PADDED, BLOCK_SIZE_PADDED]
        idx_prob_k = (idx_bk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE_PADDED))
        mask_prob_k = idx_prob_k < BK * BLOCK_SIZE
        atten_probs = tl.load(
            probs +\
                idx_n * stride_probs_n +\
                idx_tdst[:, None] * stride_probs_tdst +\
                idx_prob_k[None, :] * stride_probs_k,
            mask = mask_tdst[None, :] & mask_prob_k[:, None],
            other = 0,
        )
        
        # value: [BLOCK_SIZE_PADDED, BLOCK_HID]
        value = tl.load(
            values +\
                idx_n * stride_values_n +\
                atten_indices[:, None] * stride_values_tsrc +\
                idx_hid[None, :] * stride_values_hid,
            mask = mask_atten_indices[:, None] & mask_hid[None, :],
            other = 0,
        )
        
        prod = tl.dot(atten_probs.to(value.dtype), value)
        scores += prod
    
    tl.store(
        context +\
            idx_n * stride_context_n +\
            idx_tdst[:, None] * stride_context_tdst +\
            idx_hid[None, :] * stride_context_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        value = scores
    )


@triton.jit
def _sdbmm_compute_bwd_values(
    # input matrices
    probs, stride_probs_n, stride_probs_tdst, stride_probs_k,
    indices, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    # grad output (read)
    grad_context, stride_grad_context_n, stride_grad_context_tdst, stride_grad_context_hid,
    # grad input (write)
    grad_values, stride_grad_values_n, stride_grad_values_tsrc, stride_grad_values_hid,
    # input variables
    N, TDST, TSRC, HID, BK, K,
    # block constant
    BLOCK_SIZE,
    BLOCK_SIZE_PADDED: tl.constexpr,
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
    idx_bdst = tl.program_id(1)
    idx_bk = tl.program_id(2)
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    
    idx_block = tl.arange(0, BLOCK_SIZE_PADDED)
    mask_block = idx_block < BLOCK_SIZE
    
    idx_tdst = idx_bdst * BLOCK_SIZE + idx_block
    mask_tdst = (idx_tdst < TDST) & mask_block
    
    idx_k = idx_bk * BLOCK_SIZE + idx_block
    mask_k = (idx_k < K) & mask_block
    
    idx_tsrc = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk
    )
    idx_tsrc = idx_tsrc * BLOCK_SIZE + idx_block
    mask_tsrc = (idx_tsrc < TSRC) & mask_block
    
    # [BLOCK_SIZE_PADDED: tsrc, BLOCK_SIZE_PADDED: tdst]
    prob = tl.load(
        probs +\
            idx_n * stride_probs_n +\
            idx_tdst[None, :] * stride_probs_tdst +\
            idx_k[:, None] * stride_probs_k,
        mask = mask_tdst[None, :] & mask_k[:, None],
        other = 0
    )
    # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
    grad = tl.load(
        grad_context +\
            idx_n * stride_grad_context_n +\
            idx_tdst[:, None] * stride_grad_context_tdst +\
            idx_hid[None, :] * stride_grad_context_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        other = 0
    )
    # [BLOCK_SIZE_PADED: tsrc, BLOCK_HID: hid]
    output = tl.dot(prob, grad)
    
    tl.atomic_add(
        grad_values +\
            idx_n * stride_grad_values_n +\
            idx_tsrc[:, None] * stride_grad_values_tsrc +\
            idx_hid[None, :] * stride_grad_values_hid,
        val = output,
        mask = mask_tsrc[:, None] & mask_hid[None, :],
    )

@triton.jit
def _sdbmm_compute_bwd_probs(
    # input indices
    indices, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    values, stride_values_n, stride_values_trsc, stride_values_hid,
    # grad output (read)
    grad_context, stride_grad_context_n, stride_grad_context_tdst, stride_grad_context_hid,
    # grad input (write)
    grad_probs, stride_grad_probs_n, stride_grad_probs_tdst, stride_grad_probs_k,
    # input variables
    N, TDST, TSRC, HID, BK, K,
    # blcok constant
    BLOCK_SIZE,
    BLOCK_SIZE_PADDED: tl.constexpr,
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
    idx_bdst = tl.program_id(1)
    idx_bk = tl.program_id(2)
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    
    idx_block = tl.arange(0, BLOCK_SIZE_PADDED)
    mask_block = idx_block < BLOCK_SIZE
    
    idx_tsrc = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
    )
    idx_tsrc = idx_tsrc + idx_block
    mask_tsrc = (idx_tsrc < TSRC) & mask_block
    
    idx_tdst = idx_bdst * BLOCK_SIZE + idx_block
    mask_tdst = (idx_tdst < TDST) & mask_block
    
    # [BLOCK_HID: hid, BLOCK_SIZE_PADDED: tsrc]
    value = tl.load(
        values +\
            idx_n * stride_values_n +\
            idx_tsrc[None, :] * stride_values_trsc +\
            idx_hid[:, None] * stride_values_hid,
        mask = mask_tsrc[None, :] & mask_hid[:, None],
        other = 0,
    )
    # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
    vec_grad_context = tl.load(
        grad_context +\
            idx_n * stride_grad_context_n +\
            idx_tdst[:, None] * stride_grad_context_tdst +\
            idx_hid[None, :] * stride_grad_context_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        other = 0
    )
    # [BLOCK_SIZE_PADDED: tdst, BLOCK_SIZE_PADDED: tsrc]
    score = tl.dot(vec_grad_context, value)
    
    idx_k = idx_bk * BLOCK_SIZE + idx_block
    mask_k = (idx_k < K) & mask_block
    
    tl.store(
        grad_probs +\
            idx_n * stride_grad_probs_n +\
            idx_tdst[:, None] * stride_grad_probs_tdst +\
            idx_k[None, :] * stride_grad_probs_k,
        value = score,
        mask = mask_tdst[:, None] & mask_k[None, :]
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
        
        BLOCK_SIZE: int,
    ):
        global DEBUG
        
        ctx.save_for_backward(values, indices, ks, probs)
        ctx.BLOCK_SIZE = BLOCK_SIZE
    
        N, TSRC, HID = values.shape
        _N, BDST, BK = indices.shape
        __N, TDST, K = probs.shape
        assert N == _N
        assert N == __N
        assert ks.shape == (N, BDST)
        
        BSRC = triton.cdiv(TSRC, BLOCK_SIZE)
        
        context = torch.zeros((N, TDST, HID), dtype=values.dtype, device=values.device)
        
        BLOCK_SIZE_PADDED = next_multiple_of(BLOCK_SIZE, 16)
        BLOCK_HID = 64
        grid = (N, BDST, triton.cdiv(HID, BLOCK_HID))
        
        # NOTE: I have no idea what this sprase matrix format LOL, but for temporary
        if DEBUG:
            # print('sdbmm', grid, BLOCK_K, BLOCK_HID)
            assert indices.max() < TSRC
            assert indices.min() >= 0
            assert indices.is_contiguous()
            assert ks.is_contiguous()
            assert probs.is_contiguous()
            assert values.is_contiguous()
            assert context.is_contiguous()
            torch.cuda.synchronize()
        
        _sdbmm_compute[grid](
            # inputs
            indices, indices.stride(0), indices.stride(1), indices.stride(2),
            ks, ks.stride(0), ks.stride(1),
            probs, probs.stride(0), probs.stride(1), probs.stride(2),
            values, values.stride(0), values.stride(1), values.stride(2),
            
            # output
            context, context.stride(0), context.stride(1), context.stride(2),
            
            # input variables
            N, TSRC, TDST, HID, BK, BSRC, BDST,
            
            # blocks
            BLOCK_SIZE,
            BLOCK_SIZE_PADDED,
            BLOCK_HID,
            
            num_warps=4,
        )
        
        return context
    
    @staticmethod
    def backward(ctx, grad_context):
        values, indices, ks, probs = ctx.saved_tensors
        BLOCK_SIZE = ctx.BLOCK_SIZE
        grad_values = grad_probs = None
        
        N, T_SRC, HID = values.shape
        _, B_DST, BK = indices.shape
        _, T_DST, K = probs.shape
        assert ks.shape == (N, B_DST)
        assert probs.shape == (N, T_DST, K)
        assert indices.shape[0] == N

        # for values
        if ctx.needs_input_grad[0]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE), BK)
            BLOCK_HID = triton.next_power_of_2(HID)

            grad_values = torch.zeros(
                (N, T_SRC, HID), 
                device=values.device, 
                dtype=values.dtype,
            )
            
            _sdbmm_compute_bwd_values[grid](
                probs, probs.stride(0), probs.stride(1), probs.stride(2),
                indices, indices.stride(0), indices.stride(1), indices.stride(2),
                
                grad_context, grad_context.stride(0), grad_context.stride(1), grad_context.stride(2),
                
                grad_values, grad_values.stride(0), grad_values.stride(1), grad_values.stride(2),
                
                N, T_DST, T_SRC, HID, BK, K,
                
                BLOCK_SIZE,
                next_multiple_of(BLOCK_SIZE, 16),
                BLOCK_HID,
            )
            
            # print(grad_values.abs().sum())
        
        # for probs
        if ctx.needs_input_grad[3]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE), BK)
            BLOCK_HID = triton.next_power_of_2(HID)
            
            grad_probs = torch.zeros(
                (N, T_DST, K),
                device=probs.device,
                dtype=probs.dtype,
            )
            
            _sdbmm_compute_bwd_probs[grid](
                indices, indices.stride(0), indices.stride(1), indices.stride(2),
                values, values.stride(0), values.stride(1), values.stride(2), 
                
                grad_context, grad_context.stride(0), grad_context.stride(1), grad_context.stride(2),
                
                grad_probs, grad_probs.stride(0), grad_probs.stride(1), grad_probs.stride(2),
                
                N, T_DST, T_SRC, HID, BK, K,
                
                BLOCK_SIZE,
                next_multiple_of(BLOCK_SIZE, 16),
                BLOCK_HID,
            )

        return (
            grad_values, 
            None, 
            None, 
            grad_probs, 
            None
        )

def sparse_attention(
    # attention values
    values: Tensor,
    
    # attention matrix
    indices: Tensor,
    ks: Tensor,
    probs: Tensor,
    
    BLOCK_SIZE: int,
):
    context = SparseAttentionAutoGradFn.apply(
        values, indices, ks, probs, BLOCK_SIZE
    )
    
    return context

def tree_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor,
    
    w_start: int = None,
    n_patches: int = None,
    mask_k: int = 256,
    scale_up: float = 2,
    
    # heuristics: mask_k == n_patches * scale_up
    # heuristics: mask_k == w_start * scale_up
    
    block_size: int = 16,
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
    
    with timer('tree_attention'):
        with timer('attention_matrix'):
            indices, ks, probs = attention_matrix(
                q,
                k,
                
                w_start,
                n_patches,
                mask_k,
                scale_up,
                block_size,
            )
        
        with timer('sparse_attention'):
            context = sparse_attention(
                v,
                indices,
                ks,
                probs,
                block_size,
            )
    
    return context, (indices, ks, probs)

import torch.nn.functional as F

def torch_attention(q: Tensor, k: Tensor, v: Tensor):
    scores = torch.bmm(q, k.transpose(-1, -2))
    probs = torch.softmax(scores, dim=-1)
    context = torch.bmm(probs, v)
    return context, probs

def flash_attention(q: Tensor, k: Tensor, v: Tensor):
    context = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=1.0,
    )
    return context, None

def main_latency_benchmark():
    global DEBUG
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dups', type=int, default=2)
    parser.add_argument('--query_size', type=int, default=1)
    parser.add_argument('--method', type=str, default='tree')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--block_size', type=int, default=16)
    args = parser.parse_args()
    
    DEBUG = args.debug
    TRACE = args.trace
    BSIZE = args.batch_size
    DUPS = args.dups
    QUERY_SIZE = args.query_size
    METHOD = args.method
    n_samples = args.samples
    
    if DEBUG:
        seed()
    
    get_bench().disabled = not TRACE
    get_bench().synchronize = True
    
    q, k, v, out = load_checkouts(idx=0, window=40, seq_len=1024)
    
    q = q.repeat(BSIZE, DUPS, 1)[:, :QUERY_SIZE, :].contiguous()
    k = k.repeat(BSIZE, DUPS, 1)
    v = v.repeat(BSIZE, DUPS, 1)
    started = False
    
    samples = []
    for i in tqdm.tqdm(range(n_samples)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            if METHOD in ['torch', 'none', 'default']:
                torch_attention(q, k, v)
            elif METHOD == 'flash':
                flash_attention(q, k, v)
            elif METHOD == 'tree':
                tree_attention(
                    q,
                    k,
                    v,
                    block_size=args.block_size
                )
            else:
                raise Exception()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        
        if i > n_samples * 0.1:
            if not started:
                get_bench().reset_measures()
                get_bench().reset_trace()
                started = True
            samples.append(elapsed)
    
    if TRACE:
        print(get_bench().format_tracetree())
    
    samples = np.array(samples)
    print(f'[{METHOD}] {np.mean(samples):.4f}ms +- {np.std(samples):.4f}ms (q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)})')

def main_debug():
    global DEBUG
    DEBUG = True
    
    q, k, v, out = load_checkouts(dtype=torch.float16, seq_len=1024 * 4)
    print('q', q.shape)
    print('k', k.shape)
    print('v', v.shape)
    print('out', out.shape)
    
    context, (atten_indices, atten_ks, atten_probs) = tree_attention(
        q,
        k,
        v,
    )
    
    stderr = (out - context).abs().mean().item()
    stdcontext = torch.std_mean(context)[0].item()
    
    print(f'err = {stderr:.6f} ({stderr/stdcontext:.4f} sigma), context_std = {stdcontext:.6f}')

if __name__ == '__main__':
    # main_debug()
    main_latency_benchmark()