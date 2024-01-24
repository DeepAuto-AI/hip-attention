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
import triton
import triton.language as tl
from typing import Tuple, List
import os
import math
from src.models.tree_attention.common import load_checkouts

BLOCK_SIZE = 16

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
    
    # operation variables
    SCALE_UP: float, N_PATCHES: int, MASK_K: int,
    
    # input variables
    N: int, T_DST: int, T_SRC: int, B_DST: int, B_SRC: int, HID: int,
    
    # block constant
    GROUP_N: int,
    GROUP_BDST: int,
    BLOCK_MASK_K: tl.constexpr, 
    BLOCK_TMASK_K: tl.constexpr, 
    BLOCK_MAX_DUP: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    for _idx_n in range(GROUP_N):
        idx_n = _idx_n + GROUP_N * pid_n
        if idx_n < N:
            pid_bdst = tl.program_id(1)
            for _idx_bdst in range(GROUP_BDST):
                idx_bdst = pid_bdst * GROUP_BDST + _idx_bdst
                if idx_bdst < B_DST:
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
                    if w_old != w_new:
                        # return

                        """
                        k_old = ks[i, j, 0]
                        k_new = max(n_patches, int(min(mask_k / t_src, 1.0) * w_new))
                        k_new = min(t_src, max(n_patches, k_new))
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
                                    MASK_K.to(tl.float32) / t_src.to(tl.float32), 
                                    1.0
                                ) * w_new.to(tl.float32)
                            ).to(tl.int64)
                        )
                        k_new = tl.minimum(t_src, tl.maximum(N_PATCHES, k_new))
                        
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
                        
                        # k_old_range = 
                        # k_old_mask = tl.arange(0, BLOCK_MASK_K) < k_old
                        loc_vec = tl.load(
                            MASK +\
                                idx_n * stride_mask_n +\
                                idx_bdst * stride_mask_bdst +\
                                tl.arange(0, BLOCK_MASK_K) * stride_mask_k,
                            mask = tl.arange(0, BLOCK_MASK_K) < k_old,
                            other = 0
                        )
                        
                        # tl.device_print("a", loc_vec)
                        
                        loc_idx_start_vec = (loc_vec * w_old / BLOCK_SIZE).to(tl.int64)
                        loc_idx_end_vec = loc_idx_start_vec + 1
                        loc_idx_start_vec = (loc_idx_start_vec.to(tl.float32) / w_old.to(tl.float32) * w_new.to(tl.float32)).to(tl.int64)
                        loc_idx_end_vec = (loc_idx_end_vec.to(tl.float32) / w_old.to(tl.float32) * w_new.to(tl.float32)).to(tl.int64)
                        
                        dup_pixels_vec = loc_idx_end_vec - loc_idx_start_vec
                        dup_pixels_vec = dup_pixels_vec * (tl.arange(0, BLOCK_MASK_K) < k_old)
                        num_pixels_vec = tl.cumsum(dup_pixels_vec)
                        dup_pixels_first = tl.min(num_pixels_vec)
                        num_pixels_scalar = tl.max(num_pixels_vec)
                        
                        # tl.device_print("", num_pixels_scalar)
                        
                        dup_pixels_range = tl.arange(0, BLOCK_MAX_DUP)
                        dup_pixels_mask = (dup_pixels_range[None, :] <= dup_pixels_vec[:, None]) & (tl.arange(0, BLOCK_MASK_K) < k_old)[:, None]
                        # # tl.debug_barrier()
                        # tl.debug_barrier()
                        # tl.device_print("", k_old_mask.to(tl.int1))
                        
                        # t = (
                        #     (loc_idx_start_vec[:, None] + tl.arange(0, BLOCK_MAX_DUP)[None, :]).to(tl.float32) / w_new.to(tl.float32)
                        # )
                        # tl.device_print("ff", k_old)
                        # tl.device_print("gg", BLOCK_MASK_K)
                        tl.device_print("dd",  (tl.arange(0, BLOCK_MASK_K) + k_old) < (k_old*2))
                        
                        # tl.store(
                        #     TMASK + \
                        #         idx_n * stride_tmask_n +\
                        #         idx_bdst * stride_tmask_bdst +\
                        #         ((num_pixels_vec - dup_pixels_first)[:, None] + dup_pixels_range[None, :]) * stride_tmask_k,
                        #     mask=dup_pixels_mask,
                        #     value=(
                        #         (loc_idx_start_vec[:, None] + tl.arange(0, BLOCK_MAX_DUP)[None, :]).to(tl.float32) / w_new.to(tl.float32)
                        #     )
                        #     # value = num_pixels_scalar=
                        # )
                        
                        # # tl.debug_barrier()
                        
                        """
                        # t_mask -> mask (using scores)
                        if k_new < num_pixels:
                        """
                        # if k_new < num_pixels_scalar and False:
                        if False:
                            # """
                            # # need top_k, so compute scores
                            # vec_q = queries[i, j, :]
                            # for k in range(num_pixels):
                            #     # NOTE: nearest
                            #     loc = t_mask[i, j, k]
                            #     vec_k = keys[i, int(loc * t_src), :]
                                
                            #     score = torch.dot(vec_q, vec_k)
                            #     scores[i, j, k] = -score # NOTE: store negative store
                            # """
                            # scores = tl.zeros((BLOCK_TMASK_K,), dtype=tl.float32)
                            # for _idx_hid in range(tl.cdiv(HID, BLOCK_HID)):
                            #     hid_range = tl.arange(0, BLOCK_HID) + _idx_hid * BLOCK_HID
                            #     hid_mask = hid_range < HID
                            #     vec_q = tl.load(
                            #         QUERIES +\
                            #             idx_n * stride_queries_n +\
                            #             idx_tdst * stride_queries_tdst +\
                            #             (hid_range[None, :] + tl.arange(0, 16)[:, None]) * stride_queries_hid,
                            #         mask = (hid_mask[None, :] & (tl.arange(0, 16)[:, None] < 1)),
                            #         other = 0,
                            #     )
                            #     # tl.debug_barrier()
                                
                            #     num_pixels_range = tl.arange(0, BLOCK_TMASK_K)
                            #     num_pixels_mask = num_pixels_range < num_pixels_scalar
                            #     loc_k_vec = tl.load(
                            #         TMASK +\
                            #             idx_n * stride_tmask_n +\
                            #             idx_tdst * stride_tmask_tdst +\
                            #             num_pixels_range * stride_tmask_k,
                            #         mask = num_pixels_mask,
                            #         other = 0,
                            #     )
                            #     # tl.debug_barrier()
                            #     # NOTE: random key selection with in the block
                            #     # loc_k_vec = loc_k_vec.to(tl.float32) + tl.rand(idx_n * idx_tdst, w_old, 10) * (1.0 / w_old)
                            #     loc_k_vec = (loc_k_vec.to(tl.float32) * t_src.to(tl.float32)).to(tl.int64)
                            #     vec_k_mask = num_pixels_mask[None, :] & hid_mask[:, None]
                            #     vec_k = tl.load(
                            #         KEYS +\
                            #             idx_n * stride_keys_n +\
                            #             loc_k_vec[None, :] * stride_keys_tsrc + \
                            #             hid_range[:, None] * stride_keys_hid,
                            #         mask = vec_k_mask,
                            #         other = 0,
                            #     )
                            #     # tl.debug_barrier()
                                
                            #     # TODO: support tensorCore
                            #     # scores = -tl.dot(vec_q, vec_k) # NOTE: negative scores
                            #     # 1x128 @ 128x512 512x128 @ 128x1
                            #     # scores = -tl.sum(
                            #     #     vec_q * vec_k, 
                            #     #     axis=0,
                            #     # )
                            #     scores_partial = -tl.dot(vec_q, vec_k, allow_tf32=True)
                            #     scores_partial = tl.sum(scores_partial, axis=0)
                            #     scores_partial = scores_partial + (~num_pixels_mask) * 32000.0
                            #     scores += scores_partial.to(scores.dtype)
                            # # tl.debug_barrier()
                            # # scores = tl.zeros((BLOCK_TMASK_K,), dtype=tl.float32)
                            
                            # """
                            # _, topk_indices = torch.topk(scores[i, j, :num_pixels], k=k_new, largest=False)
                            # for k in range(k_new):
                            #     mask[i, j, k] = t_mask[i, j, topk_indices[k]]
                            # """
                            
                            # # select min-k from negative scores -> select top-k
                            # # masked_scores = scores + (~num_pixels_mask) * 32000.0
                            # masked_scores = scores
                            # # tl.debug_barrier()
                            # scores_kth_large = _triton_kth_large(masked_scores, k_new, BLOCK_TMASK_K)
                            # # tl.debug_barrier()
                            # topk_mask = masked_scores <= scores_kth_large
                            # topk_mask_cumsum = tl.cumsum(topk_mask.to(tl.int64))
                            # # tl.debug_barrier()
                            # topk_range = tl.minimum((topk_mask_cumsum - 1) * topk_mask, k_new - 1)
                            # # tl.debug_barrier()
                            
                            # temp_range = tl.arange(0, BLOCK_TMASK_K)
                            # temp_mask = temp_range < num_pixels_scalar
                            # temp = tl.load(
                            #     TMASK +\
                            #         idx_n * stride_tmask_n +\
                            #         idx_tdst * stride_tmask_tdst +\
                            #         temp_range * stride_tmask_k,
                            #     mask=temp_mask,
                            #     other=0
                            # )
                            # # tl.debug_barrier()
                            # tl.store(
                            #     MASK +\
                            #         idx_n * stride_mask_n +\
                            #         idx_tdst * stride_mask_tdst +\
                            #         topk_range * stride_mask_k,
                            #     mask=topk_mask & temp_mask,
                            #     value=temp,
                            #     # value=0.1,
                            # )
                            # # tl.debug_barrier()
                            pass
                        else:
                            """
                            else:
                                mask[i, j, :num_pixels] = t_mask[i, j, :num_pixels]
                            """
                            # temp1_range = tl.arange(0, BLOCK_MASK_K)
                            # temp1_mask = temp1_range < num_pixels_scalar
                            # # tl.debug_barrier()
                            # temp1 = tl.load(
                            #     TMASK +\
                            #         idx_n * stride_tmask_n +\
                            #         idx_bdst * stride_tmask_bdst +\
                            #         temp1_range * stride_tmask_k,
                            #     mask=temp1_mask,
                            # )
                            
                            # # tl.debug_barrier()
                            # tl.store(
                            #     MASK +\
                            #         idx_n * stride_mask_n +\
                            #         idx_bdst * stride_mask_bdst +\
                            #         temp1_range * stride_mask_k,
                            #     mask=temp1_mask,
                            #     value=temp1,
                            # )
                            # # tl.debug_barrier()
                            # # del temp1, temp1_range, temp1_mask
                        
                        """
                        ws[i, j, 0] = w_new
                        ks[i, j, 0] = min(k_new, num_pixels)
                        """
                        # # tl.debug_barrier()
                        # tl.store(
                        #     WS +\
                        #         idx_n * stride_ws_n +\
                        #         idx_bdst * stride_ws_bdst,
                        #     value = w_new
                        # )
                        # # tl.debug_barrier()
                        # tl.store(
                        #     KS +\
                        #         idx_n * stride_ks_n +\
                        #         idx_bdst * stride_ks_bdst,
                        #     value = tl.minimum(k_new, num_pixels_scalar)
                        # )
                        # # tl.debug_barrier()

DEBUG = True

def masking_iteration(
    # input matrices
    queries: Tensor, keys: Tensor,
    # input metrices (blocked) 
    mask: Tensor, t_mask: Tensor, scores: Tensor, 
    # temp vectors (blocked)
    ws: Tensor, ks: Tensor, t_srcs: Tensor, 
    # operator variables
    scale_up: float, n_patches: int, mask_k: int, 
    # input constant
    N: int, T_DST: int, T_SRC: int, B_DST: int, B_SRC: int, HID: int,
    # kernel constant
    BLOCK_SIZE: int
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
    BLOCK_HID = 32
    grid = (triton.cdiv(N, GROUP_N), triton.cdiv(B_DST, GROUP_BDST))
    
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
        GROUP_N,
        GROUP_BDST,
        triton.next_power_of_2(mask.shape[-1]),
        triton.next_power_of_2(t_mask.shape[-1]),
        triton.next_power_of_2(math.ceil(scale_up)),
        int(BLOCK_HID),
        int(BLOCK_SIZE),
        
        num_warps=8,
        num_stages=1,
        enable_warp_specialization=False,
    )

def attention_matrix(
    queries: Tensor, 
    keys: Tensor, 
    
    w_start: int,
    n_patches: int,
    mask_k: int,
    scale_up: int,
    
    BLOCK_SIZE: int = BLOCK_SIZE,
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
    tmask = torch.zeros((mask.shape[1], mask.shape[1], mask_k_block * math.ceil(scale_up)), dtype=mask.dtype, device=device)
    scores = torch.ones_like(mask, dtype=dtype)
    
    B_SRC = triton.cdiv(T_SRC, BLOCK_SIZE)
    B_DST = triton.cdiv(T_DST, BLOCK_SIZE)
    
    def to_dense(indices: np.ndarray, ks: np.ndarray, value, N, T_DST, T_SRC, BLOCK_SIZE):
        out = np.zeros((N, triton.cdiv(T_DST, BLOCK_SIZE), triton.cdiv(T_SRC, BLOCK_SIZE)))
        for idx_n in range(1):
            for idx_bdst in range(out.shape[1]):
                for k in range(indices.shape[2]):
                    out[idx_n, idx_bdst, indices[idx_n, idx_bdst, k]] = 1
        return out
    
    # NOTE: Calc. Mask
    while w_curr < T_SRC:
        tmask.fill_(0)
        mask.clamp_(0, (triton.cdiv(w_curr, BLOCK_SIZE) - 1) / triton.cdiv(w_curr, BLOCK_SIZE))
        masking_iteration(
            # input matrices
            queries, keys,
            # input metrices (blocked) 
            mask, tmask, scores, 
            # temp vectors (blocked)
            ws, ks, tsrcs, 
            # operator variables
            scale_up, n_patches, mask_k, 
            # input constant
            N, T_DST, T_SRC, B_DST, B_SRC, HID,
            # kernel constant
            BLOCK_SIZE
        )
        w_curr = round(w_curr * scale_up)
        
        if DEBUG:
            indices = torch.round(mask * ws.unsqueeze(-1) / BLOCK_SIZE).to(torch.int32)
            indices = torch.clamp(indices, 0, T_SRC - 1)
            x = to_dense(
                indices.cpu().numpy(),
                ks.cpu().unsqueeze(-1).numpy(), 
                None,
                N, T_DST, T_SRC, BLOCK_SIZE
            )[0]
            x = skimage.measure.block_reduce(x, (4, 4), np.max) ** 0.1
            plt.imshow(x)
            path = f'saves/models/tree_attention/block_{w_curr}.png'
            print('saved', path)
            plt.savefig(path, dpi=200, bbox_inches='tight')
            input('>>>')
    
    # NOTE: align with blocks
    indices = torch.round(mask * ws.unsqueeze(-1)).to(torch.int32)
    indices = indices - indices % BLOCK_SIZE
    indices.clamp_(0, (T_SRC - 1) - ((T_SRC - 1) % BLOCK_SIZE))
    
    # # NOTE: are you sure this function is the only thing can differentiate?
    # probs = calc_score_return_prob(
    #     queries=queries, keys=keys,
    #     indices=indices, ks=ks, 
    #     scores=scores,
    # )
    
    # if DEBUG:
    #     x = to_dense(
    #         indices.cpu().numpy(),
    #         ks.cpu().unsqueeze(-1).numpy(),
    #         probs.cpu().numpy(),
    #         N, T_DST, T_SRC
    #     )[0]
    #     x = skimage.measure.block_reduce(x, (4, 4), np.max) ** 0.1
    #     plt.imshow(x)
    #     path = 'saves/models/tree_attention/hello_est.png'
    #     print('saved', path)
    #     plt.savefig(path, dpi=200, bbox_inches='tight')
        
    #     x = np.matmul(
    #         queries[0].cpu().numpy(), 
    #         keys[0].cpu().numpy().transpose((-1, -2))
    #     )
    #     x = x + (1 - np.tri(*x.shape)) * (-32000)
    #     x = np.exp(x - x.max(-1, keepdims=True))
    #     x = x / x.sum(-1, keepdims=True)
    #     x = skimage.measure.block_reduce(x, (4, 4), np.max) ** 0.1
    #     plt.imshow(x)
    #     path = 'saves/models/tree_attention/hello_truth.png'
    #     print('saved', path)
    #     plt.savefig(path, dpi=200, bbox_inches='tight')
    #     print(ks)
    
    # return indices, ks, probs

def main():
    q, k, v, out = load_checkouts(dtype=torch.float32)
    attention_matrix(
        q, 
        k,
        512,
        128,
        256,
        2,
        BLOCK_SIZE
    )

if __name__ == '__main__':
    main()