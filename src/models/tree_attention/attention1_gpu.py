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
import numba
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
import matplotlib.pyplot as plt
import tqdm
import skimage.measure

import triton
import triton.language as tl

@triton.jit
def __triton_kth_large(
    scores: tl.tensor, k: tl.tensor,
    BLOCK_SCORES: tl.constexpr,
) -> tl.tensor:
    sorted_score = tl.sort(scores)
    sorted_score_mask = tl.arange(0, BLOCK_SCORES) < k
    return tl.max(sorted_score * sorted_score_mask + (-32000) * (~sorted_score_mask))

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
    
    w_new = min(tl.math.round(w_old * scale_up), t_src)
    
    """
    if w_old != w_new:
    """
    if w_old == w_new:
        return
    
    """
    k_old = ks[i, j, 0]
    k_new = max(n_patches, int(min(mask_k / t_src, 1.0) * w_new))
    k_new = min(t_src, max(n_patches, k_new))
    """
    
    k_old = tl.load(
        ks + \
            idx_n * stride_ks_n +\
            idx_tdst * stride_ks_tdst,
    )
    k_new = tl.maximum(n_patches, (min(mask_k / t_src, 1.0) * w_new).to(tl.int64))
    k_new = min(t_src, tl.maximum(n_patches, k_new))
    
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
    loc_idx_start_vec = (loc_idx_start_vec / w_old * w_new).to(tl.int64)
    loc_idx_end_vec = (loc_idx_end_vec / w_old * w_new).to(tl.int64)
    
    dup_pixels_vec = loc_idx_end_vec - loc_idx_start_vec
    num_pixels_vec = tl.cumsum(dup_pixels_vec)
    dup_pixels_first = tl.min(num_pixels_vec)
    num_pixels_scalar = tl.max(num_pixels_vec)
    
    dup_pixels_range = tl.arange(0, BLOCK_MAX_DUP)[None, :]
    dup_pixels_mask = (dup_pixels_range < dup_pixels_vec[:, None]) & k_old_mask[:, None]
    tl.store(
        tmask + \
            idx_n * stride_tmask_n +\
            idx_tdst * stride_tmask_n +\
            ((num_pixels_vec - dup_pixels_first)[:, None] + dup_pixels_range) * stride_tmask_k,
        mask=dup_pixels_mask,
        value=(
            (loc_idx_start_vec[:, None] + tl.arange(0, BLOCK_MAX_DUP)[None, :]) / w_new
        )
    )
    tl.debug_barrier()
    
    """
    # t_mask -> mask (using scores)
    if k_new < num_pixels:
    """
    if k_new < num_pixels_scalar:
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
        )[None, :]
        
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
        loc_k_vec = (loc_k_vec * t_src).to(tl.int64)
        vec_k_mask = num_pixels_mask[None, :] & hid_mask[:, None]
        vec_k = tl.load(
            keys +\
                idx_n * stride_keys_n +\
                loc_k_vec[None, :] * stride_keys_tsrc + \
                hid_range[:, None] * stride_keys_hid,
            mask = vec_k_mask,
            other = 0,
        )
        
        # TODO: support tensorCore
        
        # scores = -tl.dot(vec_q, vec_k) # NOTE: negative scores
        # 1x128 @ 128x512 512x128 @ 128x1
        scores = -tl.sum(
            tl.reshape(vec_q, (BLOCK_HID, 1)) *\
            tl.reshape(vec_k, (BLOCK_HID, BLOCK_TMASK_K)), 
            axis=0
        )
        
        scores = tl.reshape(scores, (BLOCK_TMASK_K, ))
        
        """
        _, topk_indices = torch.topk(scores[i, j, :num_pixels], k=k_new, largest=False)
        for k in range(k_new):
            mask[i, j, k] = t_mask[i, j, topk_indices[k]]
        """
        
        # select min-k from negative scores -> select top-k
        masked_scores = scores + (tl.arange(0, BLOCK_TMASK_K) >= num_pixels_scalar) * 32000.0
        scores_kth_large = __triton_kth_large(masked_scores, k_new, BLOCK_TMASK_K)
        topk_mask = scores <= scores_kth_large
        topk_range = tl.cumsum(topk_mask.to(tl.int32)) - 1
        
        temp_range = tl.arange(0, BLOCK_TMASK_K)
        temp_mask = temp_range < num_pixels_scalar
        temp = tl.load(
            tmask +\
                idx_n * stride_tmask_n +\
                idx_tdst * stride_tmask_tdst +\
                temp_range * stride_tmask_k,
            mask=temp_mask,
            other=0
        )
        tl.store(
            mask +\
                idx_n * stride_mask_n +\
                idx_tdst * stride_mask_tdst +\
                topk_range * stride_mask_k,
            mask=topk_mask,
            value=temp,
        )
        # del temp, temp_range, temp_mask
    else:
        """
        else:
            mask[i, j, :num_pixels] = t_mask[i, j, :num_pixels]
        """
        temp1_range = tl.arange(0, BLOCK_MASK_K)
        temp1_mask = temp1_range < num_pixels_scalar
        temp1 = tl.load(
            tmask +\
                idx_n * stride_tmask_n +\
                idx_tdst * stride_tmask_tdst +\
                temp1_mask * stride_tmask_k,
            mask=temp1_mask,
        )
        tl.store(
            mask +\
                idx_n * stride_mask_n +\
                idx_tdst * stride_mask_tdst +\
                temp1_range * stride_mask_k,
            mask=temp1_mask,
            value=temp1
        )
        # del temp1, temp1_range, temp1_mask
    
    """
    ws[i, j, 0] = w_new
    ks[i, j, 0] = min(k_new, num_pixels)
    """
    tl.store(
        ws +\
            idx_n * stride_ws_n +\
            idx_tdst * stride_ws_tdst,
        value = w_new
    )
    tl.store(
        ks +\
            idx_n * stride_ks_n +\
            idx_tdst * stride_ks_tdst,
        value = min(k_new, num_pixels_scalar)
    )

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
    
    ws = ws.unsqueeze(-1).contiguous()
    ks = ks.unsqueeze(-1).contiguous()
    t_srcs = t_srcs.unsqueeze(-1).contiguous()
    
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

def mask(
    queries: ndarray, 
    keys: ndarray, 
    
    w_start: int = 32,
    n_patches: int = 16,
    mask_k: int = 128,
    scale_up: int = 4,
) -> ndarray:
    dtype = np.float32
    N, T_DST, HID = queries.shape
    _, T_SRC, _ = keys.shape
    assert T_DST <= T_SRC
    queries = queries.astype(dtype)
    keys = keys.astype(dtype)
    
    # NOTE: width of last query
    w_curr = round(w_start / scale_up)
    t_srcs = np.arange(T_SRC-T_DST+1, T_SRC+1, 1, dtype=np.int32).reshape((1, T_DST, 1)).repeat(N, axis=0)
    ws = t_srcs.clip(0, w_curr)
    ks = ws.copy()
    mask = (np.arange(mask_k, dtype=np.float32).reshape((1, 1, mask_k)) / ks)
    t_mask = np.zeros((N, T_DST, mask_k*math.ceil(scale_up)), dtype=np.float32)
    scores = np.zeros((N, T_DST, mask_k*math.ceil(scale_up)), dtype=dtype)
    
    def to_dense(mask, ks, ws):
        dense = np.zeros((N, T_DST, T_SRC))
        for i in range(N):
            for j in range(T_DST):
                nonzero_k = ks[i, j, 0]
                for k in range(nonzero_k):
                    dense[i, j, int(mask[i, j, k] * ws[i, j, 0])] = 1
        return dense
    
    # NOTE: to cuda
    device = 'cuda:0'
    queries = torch.tensor(queries, device=device)
    keys = torch.tensor(keys, device=device)
    mask = torch.tensor(mask, device=device)
    t_mask = torch.tensor(t_mask, device=device)
    scores = torch.tensor(scores, device=device)
    ws = torch.tensor(ws, device=device)
    ks = torch.tensor(ks, device=device)
    t_srcs = torch.tensor(t_srcs, device=device)
    
    while w_curr < T_SRC:
        mask_iter(
            # input matrices
            queries, keys, mask, t_mask, scores, 
            # temp vectors
            ws, ks, t_srcs, 
            # operator variables
            scale_up, n_patches, mask_k, 
            # input constant
            N, T_DST, T_SRC, HID
        )
        w_curr = round(w_curr * scale_up)
        print(w_curr, T_SRC)
    
    # NOTE: to numpy
    queries = queries.cpu().numpy()
    keys = keys.cpu().numpy()
    mask = mask.cpu().numpy()
    t_mask = t_mask.cpu().numpy()
    scores = scores.cpu().numpy()
    ws = ws.cpu().numpy()
    ks = ks.cpu().numpy()
    t_srcs = t_srcs.cpu().numpy()
    
    # NOTE: for debug image output
    # print mask
    mask = to_dense(mask, ks, ws)[0]
    x = skimage.measure.block_reduce(mask, (4, 4), np.max)
    plt.imshow(x)
    plt.savefig('hello.png', dpi=200)
    
    # print probabilites
    x = np.matmul(queries[0], keys[0].transpose((-1, -2)))
    x = x + (1 - np.tri(*x.shape)) * (-32000)
    x = np.exp(x - x.max(-1, keepdims=True))
    x = x / x.sum(-1, keepdims=True)
    x = skimage.measure.block_reduce(x, (8, 8), np.max) ** 0.2
    plt.imshow(x)
    plt.savefig('hello_2.png', dpi=200)
    # NOTE: end of debug output
    
    print(ks)
    
    return

def sparse_attention(q: ndarray, k: ndarray, v: ndarray, csr_mask: ndarray):
    pass

def attention(q: Tensor, k: Tensor, v: Tensor):
    assert q.ndim == 3
    assert k.ndim == 3
    assert v.ndim == 3
    N, T_DST, HID = q.shape
    _N, T_SRC, _HID = k.shape
    assert k.shape[:-1] == v.shape[:-1]
    assert N == _N
    assert HID == _HID
    
    q = q.numpy()
    k = k.numpy()
    v = v.numpy()
    csr_scores = mask(q, k)
    # out = sparse_attention(csr_scores, v)
    
    # return out

if __name__ == '__main__':
    data_source = 'llama'
    if data_source == 'llama':
        state = torch.load('./cache/llama/qkvout.pth', map_location='cpu')
        q = state['q']
        k = state['k']
        v = state['v']
        N, H, T_DST, HID = q.shape
        N, H, T_SRC, HID = k.shape
        idx = 7
        q = q.view(N*H, T_DST, HID)[idx:idx+1].contiguous()
        k = k.view(N*H, T_SRC, HID)[idx:idx+1].contiguous()
        v = v.view(N*H, T_SRC, HID)[idx:idx+1].contiguous()
    else:
        q = torch.randn((1, 64, 4))
        k = torch.randn((1, 64, 4))
        v = k.clone()
    
    print(q.shape, k.shape, v.shape)
    out = attention(q, k, v)