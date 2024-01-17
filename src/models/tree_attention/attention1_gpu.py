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
from typing import Tuple
import numba
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
import matplotlib.pyplot as plt
import tqdm
import skimage.measure
import torch.nn.functional as F

import triton
import triton.language as tl

@triton.jit
def __triton_kth_large(
    scores: tl.tensor, k: tl.tensor,
    BLOCK_SCORES: tl.constexpr,
) -> tl.tensor:
    sorted_score = tl.sort(scores)
    # tl.debug_barrier()
    sorted_score_mask = tl.arange(0, BLOCK_SCORES) < k
    return tl.max(sorted_score * sorted_score_mask + (-32000.0) * (~sorted_score_mask))

@triton.jit
def __mask_iter_compute(
    # input matrices
    queries, stride_queries_n, stride_queries_tdst, stride_queries_hid,
    keys, stride_keys_n, stride_keys_tsrc, stride_keys_hid,
    mask, stride_mask_n, stride_mask_tdst, stride_mask_k,
    tmask, stride_tmask_n, stride_tmask_tdst, stride_tmask_k,
    scores_out, stride_scores_out_n, stride_scores_out_tdst, stride_scores_out_k,
    
    # temp vectors
    ws, stride_ws_n, stride_ws_tdst,
    ks, stride_ks_n, stride_ks_tdst,
    tsrcs, stride_tsrcs_n, stride_tsrcs_tdst,
    
    # operation variables
    scale_up: float, n_patches: int, mask_k: int,
    
    # input variables
    N, T_DST, T_SRC, HID,
    
    # block constant
    BLOCK_MASK_K: tl.constexpr, 
    BLOCK_TMASK_K: tl.constexpr, 
    BLOCK_MAX_DUP: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    # TODO: we should make block across batch dim
    
    idx_n = tl.program_id(0)
    idx_tdst = tl.program_id(1)
    
    """
    # for each query
    w_old = ws[i, j, 0]
    t_src = t_srcs[i, j, 0]
    w_new = min(torch.round(w_old * scale_up), t_src)
    """
    
    w_old = tl.load(
        ws + \
            idx_n * stride_ws_n + \
            idx_tdst * stride_ws_tdst,
    )
    
    t_src = tl.load(
        tsrcs + \
            idx_n * stride_tsrcs_n + \
            idx_tdst * stride_tsrcs_tdst,
    )
    
    w_new = tl.minimum(
        tl.math.round(w_old.to(tl.float32) * scale_up.to(tl.float32)).to(tl.float32), 
        t_src
    ).to(tl.int64)
    
    """
    if w_old != w_new:
    """
    if w_old == w_new:
        return
    # return

    """
    k_old = ks[i, j, 0]
    k_new = max(n_patches, int(min(mask_k / t_src, 1.0) * w_new))
    k_new = min(t_src, max(n_patches, k_new))
    """
    
    k_old = tl.load(
        ks + \
            idx_n * stride_ks_n +\
            idx_tdst * stride_ks_tdst,
    ).to(tl.int64)
    k_new = tl.maximum(
        n_patches, 
        (
            tl.minimum(
                mask_k.to(tl.float64) / t_src.to(tl.float64), 
                1.0
            ) * w_new.to(tl.float64)
        ).to(tl.int64)
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
        mask +\
            idx_n * stride_mask_n +\
            idx_tdst * stride_mask_tdst +\
            k_old_range * stride_mask_k,
        mask = k_old_mask,
        other = 0
    )
    
    loc_idx_start_vec = (loc_vec * w_old).to(tl.int64)
    loc_idx_end_vec = loc_idx_start_vec + 1
    loc_idx_start_vec = (loc_idx_start_vec.to(tl.float64) / w_old.to(tl.float64) * w_new.to(tl.float64)).to(tl.int64)
    loc_idx_end_vec = (loc_idx_end_vec.to(tl.float64) / w_old.to(tl.float64) * w_new.to(tl.float64)).to(tl.int64)
    
    dup_pixels_vec = loc_idx_end_vec - loc_idx_start_vec
    dup_pixels_vec = dup_pixels_vec * k_old_mask
    num_pixels_vec = tl.cumsum(dup_pixels_vec)
    dup_pixels_first = tl.min(num_pixels_vec)
    num_pixels_scalar = tl.max(num_pixels_vec)
    
    dup_pixels_range = tl.arange(0, BLOCK_MAX_DUP)
    dup_pixels_mask = (dup_pixels_range[None, :] <= dup_pixels_vec[:, None]) & k_old_mask[:, None]
    # tl.debug_barrier()
    tl.store(
        tmask + \
            idx_n * stride_tmask_n +\
            idx_tdst * stride_tmask_tdst +\
            ((num_pixels_vec - dup_pixels_first)[:, None] + dup_pixels_range[None, :]) * stride_tmask_k,
        mask=dup_pixels_mask,
        value=(
            (loc_idx_start_vec[:, None] + tl.arange(0, BLOCK_MAX_DUP)[None, :]).to(tl.float32) / w_new.to(tl.float32)
        )
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
        hid_range = tl.arange(0, BLOCK_HID)
        hid_mask = hid_range < HID
        vec_q = tl.load(
            queries +\
                idx_n * stride_queries_n +\
                idx_tdst * stride_queries_tdst +\
                hid_range * stride_queries_hid,
            mask = hid_mask,
            other = 0,
        )[:, None]
        # tl.debug_barrier()
        
        num_pixels_range = tl.arange(0, BLOCK_TMASK_K)
        num_pixels_mask = num_pixels_range < num_pixels_scalar
        loc_k_vec = tl.load(
            tmask +\
                idx_n * stride_tmask_n +\
                idx_tdst * stride_tmask_tdst +\
                num_pixels_range * stride_tmask_k,
            mask = num_pixels_mask,
            other = 0,
        )
        # tl.debug_barrier()
        loc_k_vec = (loc_k_vec.to(tl.float32) * t_src.to(tl.float32)).to(tl.int64)
        vec_k_mask = num_pixels_mask[None, :] & hid_mask[:, None]
        vec_k = tl.load(
            keys +\
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
        scores = -tl.sum(
            vec_q.to(tl.float32) * vec_k.to(tl.float32), 
            axis=0,
        )
        # tl.debug_barrier()
        
        """
        _, topk_indices = torch.topk(scores[i, j, :num_pixels], k=k_new, largest=False)
        for k in range(k_new):
            mask[i, j, k] = t_mask[i, j, topk_indices[k]]
        """
        
        # select min-k from negative scores -> select top-k
        masked_scores = scores + (~num_pixels_mask) * 32000.0
        # tl.debug_barrier()
        scores_kth_large = __triton_kth_large(masked_scores, k_new, BLOCK_TMASK_K)
        # tl.debug_barrier()
        topk_mask = masked_scores <= scores_kth_large
        topk_mask_cumsum = tl.cumsum(topk_mask.to(tl.int64))
        # tl.debug_barrier()
        topk_range = tl.minimum((topk_mask_cumsum - 1) * topk_mask, k_new - 1)
        # tl.debug_barrier()
        
        temp_range = tl.arange(0, BLOCK_TMASK_K)
        temp_mask = temp_range < num_pixels_scalar
        temp = tl.load(
            tmask +\
                idx_n * stride_tmask_n +\
                idx_tdst * stride_tmask_tdst +\
                temp_range * stride_tmask_k,
            mask=temp_mask,
            other=42
        )
        # tl.debug_barrier()
        tl.store(
            mask +\
                idx_n * stride_mask_n +\
                idx_tdst * stride_mask_tdst +\
                topk_range * stride_mask_k,
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
            tmask +\
                idx_n * stride_tmask_n +\
                idx_tdst * stride_tmask_tdst +\
                temp1_range * stride_tmask_k,
            mask=temp1_mask,
        )
        
        # tl.debug_barrier()
        tl.store(
            mask +\
                idx_n * stride_mask_n +\
                idx_tdst * stride_mask_tdst +\
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
        ws +\
            idx_n * stride_ws_n +\
            idx_tdst * stride_ws_tdst,
        value = w_new
    )
    # tl.debug_barrier()
    tl.store(
        ks +\
            idx_n * stride_ks_n +\
            idx_tdst * stride_ks_tdst,
        value = tl.minimum(k_new, num_pixels_scalar)
    )
    # tl.debug_barrier()

def mask_iter(
    # input matrices
    queries: Tensor, keys: Tensor, mask: Tensor, t_mask: Tensor, scores: Tensor, 
    # temp vectors
    ws: Tensor, ks: Tensor, t_srcs: Tensor, 
    # operator variables
    scale_up: float, n_patches: int, mask_k: int, 
    # input constant
    N: int, T_DST: int, T_SRC: int, HID: int,
):
    grid = (N, T_DST)
    
    __mask_iter_compute[grid](
        # input matrices
        queries, queries.stride(0), queries.stride(1), queries.stride(2),
        keys, keys.stride(0), keys.stride(1), keys.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        t_mask, t_mask.stride(0), t_mask.stride(1), t_mask.stride(2),
        scores, scores.stride(0), scores.stride(1), scores.stride(2),
        
        # temp vectors
        ws, ws.stride(0), ws.stride(1),
        ks, ks.stride(0), ks.stride(1),
        t_srcs, t_srcs.stride(0), t_srcs.stride(1),
        
        # operation variables
        float(scale_up), int(n_patches), int(mask_k),
        
        # input variables
        N, T_DST, T_SRC, HID,
        
        # block constant
        triton.next_power_of_2(mask.shape[-1]),
        triton.next_power_of_2(t_mask.shape[-1]),
        triton.next_power_of_2(scale_up),
        triton.next_power_of_2(HID),
    )

@triton.jit
def __calc_score_compute(
    # matrices
    queries, stride_queries_n, stride_queries_tdst, stride_queries_hid,
    keys, stride_keys_n, stride_keys_tsrc, stride_keys_hid,
    indices, stride_indices_n, stride_indices_tdst, stride_indices_k,
    ks, stride_ks_n, stride_ks_tdst,
    scores_out, stride_scores_out_n, stride_scores_out_tdst, stride_scores_out_k,
    
    # input variables
    N, TDST, TSRC, HID, K,
    
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
        queries +\
            idx_n * stride_queries_n +\
            idx_tdst * stride_queries_tdst +\
            idx_hid * stride_queries_hid,
        mask = mask_hid,
        other = 0
    )
    
    # ks: [1,]
    n_k = tl.load(
        ks +\
            idx_n * stride_ks_n +\
            idx_tdst * stride_ks_tdst,
    )
    mask_n_k = idx_k < n_k
    
    # idx_keys: [BLOCK_K, ]
    idx_keys = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_tdst * stride_indices_tdst +\
            idx_k * stride_indices_k,
        mask = mask_k,
        other = 0
    )
    mask_idx_keys = mask_n_k & ((idx_keys < TSRC) & (idx_keys >= 0))
    
    # tl.debug_barrier()
    # tl.device_print("", idx_keys)
    # tl.device_print("", tl.max(idx_keys))
    
    # key: [BLOCK_HID, BLOCK_K]
    key = tl.load(
        keys +\
            idx_n * stride_keys_n +\
            idx_keys[None, :] * stride_keys_tsrc +\
            idx_hid[:, None] * stride_keys_hid,
        mask = (mask_k & mask_idx_keys)[None, :] & mask_hid[:, None],
        other = 0
    )
    
    scores = (query[:, None] * key)
    scores = tl.sum(scores, axis=0)
    
    # tl.debug_barrier()
    # tl.device_print("", tl.max(scores))
    
    tl.store(
        scores_out +\
            idx_n * stride_scores_out_n +\
            idx_tdst * stride_scores_out_tdst +\
            idx_k * stride_scores_out_k,
        mask = mask_k & mask_idx_keys,
        value = scores
    )

def calc_score_return_prob(
    # matrices
    queries: Tensor, keys: Tensor,
    # indices matrices
    indices: Tensor, ks: Tensor, 
    # output scores
    scores: Tensor,
):
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
    
    # print(
    #     queries.data_ptr(), queries.stride(0), queries.stride(1), queries.stride(2),
    #     keys.data_ptr(), keys.stride(0), keys.stride(1), keys.stride(2),
    #     indices.data_ptr(), indices.stride(0), indices.stride(1), indices.stride(2),
    #     scores.data_ptr(), scores.stride(0), scores.stride(1), scores.stride(2),
    # )
    
    assert queries.ndim == 3
    assert keys.ndim == 3
    assert indices.ndim == 3
    assert ks.ndim == 2
    assert scores.ndim == 3
    __calc_score_compute[grid](
        # matrices
        queries, queries.stride(0), queries.stride(1), queries.stride(2),
        keys, keys.stride(0), keys.stride(1), keys.stride(2),
        indices, indices.stride(0), indices.stride(1), indices.stride(2),
        ks, ks.stride(0), ks.stride(1),
        scores, scores.stride(0), scores.stride(1), scores.stride(2),
        
        # variables
        N, TDST, TSRC, HID, K,
        
        # constants
        BLOCK_K, 
        BLOCK_HID,
    )
    
    probs = scores.softmax(dim=-1)
    
    # print(scores[0,100,:], probs[0, 100, :])
    
    return probs

def to_dense(
    indices,
    ks,
    value,
    N: int,
    T_DST: int,
    T_SRC: int,
):
    print('convert to dense')
    dense = np.zeros((N, T_DST, T_SRC))
    for i in range(N):
        for j in range(T_DST):
            nonzero_k = ks[i, j].item()
            for k in range(nonzero_k):
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
    
    dtype = queries.dtype
    device = queries.device
    assert queries.device == keys.device
    
    N, T_DST, HID = queries.shape
    _, T_SRC, _ = keys.shape
    assert T_DST <= T_SRC
    
    # NOTE: width of last query
    w_curr = round(w_start / scale_up)
    
    # vectors
    tsrcs = torch.arange(
        T_SRC-T_DST+1, T_SRC+1, 1, 
        dtype=torch.int64,
        device=device,
    ).view(1, T_DST).expand(N, T_DST)
    ws = torch.clamp(tsrcs, 0, w_curr)
    ks = ws.clone()
    
    # matrices
    mask = (torch.arange(mask_k, device=device).view(1, 1, mask_k) / ks.unsqueeze(-1)).to(dtype)
    tmask = torch.zeros((N, T_DST, mask_k * math.ceil(scale_up)), dtype=dtype, device=device)
    scores = torch.ones_like(mask)
    
    # NOTE: Calc. Mask
    while w_curr < T_SRC:
        # tmask.fill_(0)
        
        mask_iter(
            # input matrices
            queries, keys, mask, tmask, scores, 
            # temp vectors
            ws, ks, tsrcs, 
            # operator variables
            scale_up, n_patches, mask_k, 
            # input constant
            N, T_DST, T_SRC, HID
        )
        w_curr = round(w_curr * scale_up)
        # print(w_curr, T_SRC)
        
        # x = to_dense(
        #     mask.cpu().numpy(), 
        #     ks.cpu().unsqueeze(-1).numpy(), 
        #     ws.cpu().unsqueeze(-1).numpy()
        # )[0]
        # x = skimage.measure.block_reduce(x, (1, 1), np.max)
        # plt.imshow(x)
        # plt.savefig('hello.png', dpi=500)
        # input('>>>')
    
    # ws = t_srcs.clone()
    # ks = (torch.logical_and(mask[:, :, 1:] > 0.0, mask[:, :, 1:] < 1.0).int().sum(dim=-1) + 1).clamp_max(mask_k)
    # print(ks[:, -16:])
    
    # NOTE: Calc. Prob.
    
    indices = torch.round(mask * ws.unsqueeze(-1)).to(torch.int32)
    indices = torch.clamp(indices, 0, T_SRC - 1)
    probs = calc_score_return_prob(
        queries=queries, keys=keys,
        indices=indices, ks=ks, 
        scores=scores,
    )
    
    if DEBUG:
        # print(scores)
        x = to_dense(
            indices.cpu().numpy(),
            ks.cpu().unsqueeze(-1).numpy(),
            probs.cpu().numpy(),
            N, T_DST, T_SRC
        )[0]
        x = skimage.measure.block_reduce(x, (4, 4), np.max) ** 0.2
        plt.imshow(x)
        plt.savefig('hello.png', dpi=200, bbox_inches='tight')
        
        x = np.matmul(
            queries[0].cpu().numpy(), 
            keys[0].cpu().numpy().transpose((-1, -2))
        )
        x = x + (1 - np.tri(*x.shape)) * (-32000)
        x = np.exp(x - x.max(-1, keepdims=True))
        x = x / x.sum(-1, keepdims=True)
        x = skimage.measure.block_reduce(x, (4, 4), np.max) ** 0.2
        plt.imshow(x)
        plt.savefig('hello_2.png', dpi=200, bbox_inches='tight')
        # print(ks)
    
    return indices, ks, probs

@triton.jit
def __sdbmm_compute(
    # inputs
    indices, stride_indices_n, stride_indices_tdst, stride_indices_k,
    ks, stride_ks_n, stride_ks_tdst, 
    probs, stride_probs_n, stride_probs_tdst, stride_probs_k,
    values, stride_values_n, stride_values_tsrc, stride_values_hid,
    
    # output
    context, stride_context_n, stride_context_tdst, stride_context_hid,
    
    # variables
    N, TSRC, TDST, HID, K,
    
    # kernel blocks
    BLOCK_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    idx_n = tl.program_id(0)
    idx_tdst = tl.program_id(1)
    
    idx_k = tl.arange(0, BLOCK_K)
    mask_k = idx_k < K
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    
    n_k = tl.load(
        ks +\
            idx_n * stride_ks_n+\
            idx_tdst * stride_ks_tdst,
    )
    mask_k = mask_k & (tl.arange(0, BLOCK_K) < n_k)
    
    # atten_indices: [BLOCK_K]
    atten_indices = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_tdst * stride_indices_tdst +\
            idx_k * stride_indices_k,
        mask = mask_k
    )
    
    # atten_probs: [BLOCK_K]
    atten_probs = tl.load(
        probs +\
            idx_n * stride_probs_n +\
            idx_tdst * stride_probs_tdst +\
            idx_k * stride_probs_k,
        mask = mask_k
    )
    
    # value: [BLOCK_K, BLOCK_HID]
    value = tl.load(
        values +\
            idx_n * stride_values_n +\
            atten_indices[:, None] * stride_values_tsrc +\
            idx_hid[None, :] * stride_values_hid,
        mask = mask_k[:, None] & mask_hid[None, :]
    )
    
    # output: [BLOCK_HID] <- atten_probs[1, BLOCK_K] @ value[BLOCK_K, BLOCK_HID]
    output = tl.sum(atten_probs[:, None] * value, axis=0)
    tl.store(
        context +\
            idx_n * stride_context_n +\
            idx_tdst * stride_context_tdst +\
            idx_hid * stride_context_hid,
        mask = mask_hid,
        value = output
    )

def sparse_attention(
    # attention values
    values: Tensor,
    
    # attention matrix
    indices: Tensor,
    ks: Tensor,
    probs: Tensor,
) -> Tensor:
    N, T_SRC, HID = values.shape
    _, T_DST, K = indices.shape
    assert ks.shape == (N, T_DST)
    assert probs.shape == indices.shape
    
    context = torch.zeros((N, T_DST, HID), dtype=values.dtype, device=values.device)
    
    grid = (N, T_DST)
    BLOCK_K = triton.next_power_of_2(K)
    BLOCK_HID = triton.next_power_of_2(HID)
    
    # NOTE: I have no idea what this sprase matrix format LOL, but for temporary
    
    __sdbmm_compute[grid](
        # inputs
        indices, indices.stride(0), indices.stride(1), indices.stride(2),
        ks, ks.stride(0), ks.stride(1),
        probs, probs.stride(0), probs.stride(1), probs.stride(2),
        values, values.stride(0), values.stride(1), values.stride(2),
        
        # output
        context, context.stride(0), context.stride(1), context.stride(2),
        
        # input variables
        N, T_SRC, T_DST, HID, K,
        
        # blocks
        BLOCK_K,
        BLOCK_HID,
    )
    
    return context

def tree_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor,
    
    w_start: int = 64,
    n_patches: int = 32,
    mask_k: int = 128,
    scale_up: int = 4,
):
    assert q.ndim == 3
    assert k.ndim == 3
    assert v.ndim == 3
    N, T_DST, HID = q.shape
    _N, T_SRC, _HID = k.shape
    assert k.shape[:-1] == v.shape[:-1]
    assert N == _N
    assert HID == _HID
    
    indices, ks, probs = attention_matrix(
        q,
        k,
        
        w_start,
        n_patches,
        mask_k,
        scale_up,
    )
    
    context = sparse_attention(
        v,
        indices,
        ks,
        probs,
    )
    
    return context, (indices, ks, probs)

def __load_checkouts():
    data_source = 'llama'
    device = 0
    if data_source == 'llama':
        state = torch.load('./cache/llama/qkvout.pth', map_location='cpu')
        q = state['q']
        k = state['k']
        v = state['v']
        out = state['out']
        N, H, T_DST, HID = q.shape
        N, H, T_SRC, HID = k.shape
        idx = 7
        q = q.view(N*H, T_DST, HID)[idx:idx+1].contiguous()
        k = k.view(N*H, T_SRC, HID)[idx:idx+1].contiguous()
        v = v.view(N*H, T_SRC, HID)[idx:idx+1].contiguous()
        out = out.view(N*H, T_DST, HID)[idx:idx+1].contiguous()
    else:
        q = torch.randn((1, 64, 4))
        k = torch.randn((1, 64, 4))
        v = k.clone()
        out = q.clone()
    
    q = q.to(device, dtype=torch.float32)
    k = k.to(device, dtype=torch.float32)
    v = v.to(device, dtype=torch.float32)
    out = out.to(device, dtype=torch.float32)
    
    return q, k, v, out

def main_debug():
    global DEBUG
    DEBUG = True
    
    q, k, v, out = __load_checkouts()
    
    context, attention_probs = tree_attention(
        q, 
        k, 
        v,
        w_start=64,
        n_patches=32,
        mask_k=128,
        scale_up=2,
    )
    
    print(
        F.mse_loss(out, context).item() ** 0.5, 
        torch.std_mean(context)
    )

def torch_attention(q: Tensor, k: Tensor, v: Tensor):
    scores = torch.bmm(q, k.transpose(-1, -2))
    probs = torch.softmax(scores, dim=-1)
    context = torch.bmm(probs, v)
    return context, probs

def flash_attention(q: Tensor, k: Tensor, v: Tensor):
    context = F.scaled_dot_product_attention(
        q, k, v, is_causal=True,
    )
    return context, None

def main_latency_benchmark():
    global DEBUG
    DEBUG = False
    
    q, k, v, out = __load_checkouts()
    
    BSIZE = 2048
    DUPS = 2
    QUERY_SIZE = 1
    q = q.repeat(BSIZE, DUPS, 1)[:, :QUERY_SIZE, :].contiguous()
    k = k.repeat(BSIZE, DUPS, 1)
    v = v.repeat(BSIZE, DUPS, 1)
    
    METHOD = 'tree'
    METHOD = 'torch'
    METHOD = 'flash'
    
    samples = []
    for i in tqdm.tqdm(range(1000)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if METHOD == 'torch':
            torch_attention(q, k, v)
        elif METHOD == 'flash':
            flash_attention(q, k, v)
        elif METHOD == 'tree':
            tree_attention(
                q,
                k,
                v,
                w_start=64,
                n_patches=32,
                mask_k=128,
                scale_up=2,
            )
        else:
            raise Exception()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        
        if i > 100:
            samples.append(elapsed)
    
    samples = np.array(samples)
    print(f'[{METHOD}] {np.mean(samples):.4f}ms +- {np.std(samples):.4f}ms (q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)})')

if __name__ == '__main__':
    main_debug()
    # main_latency_benchmark()