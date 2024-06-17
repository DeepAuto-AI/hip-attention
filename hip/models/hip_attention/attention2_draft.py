"""
HiP v1.1
TODO:
1. Masking iteration using integer to avoid aliasing and collision
 - Convert tmask into int32
 - Reuse the computed dot products
2. Using QUEST method for b_k
2. Maximum token location predictor
 - Test oracle
 - Test estimators
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from torch import Tensor
from hip.models.hip_attention.attention1_block_gpu \
    import calc_prob_return_context \
        as block_sparse_flash_attention
from hip.models.hip_attention.attention1_block_gpu import load_checkouts
import numpy as np
from numpy import ndarray as NdArray
import numba
import math

def cdiv(a, b):
    return math.ceil(a / b)

def masking_iteration_draft_numba_kernel(
    #in
    q: NdArray, # fp32[block_size_q, HID]
    k: NdArray, # fp32[TSRC(sliced), HID]
    
    # out
    indices: NdArray, #int32[MASK_K // B_K]
    
    # param
    mask_k: int,
    mask_k_init: int,
    block_size_q: int,
    block_size_k: int,
) -> int:
    if k.shape[0] <= mask_k:
        for i in range(len(indices)):
            indices[i] = i * block_size_k
        return k.shape[0]
    else:
        pass

def masking_iteration_draft_numba(
    # in
    q: NdArray,
    k: NdArray,
    
    # out
    indices: NdArray,
    ks: NdArray,
    
    # param
    mask_k: int,
    mask_k_init: int,
    block_size_q: int,
    block_size_k: int,
):
    """
    grid = (N, TDST)
    """
    
    N, TDST, _ = q.shape
    _, TSRC, _ = k.shape
    
    for idx_n in range(N):
        for idx_tdst in range(cdiv(TDST, block_size_q)):
            final_k = masking_iteration_draft_numba_kernel(
                q[idx_n, idx_tdst * block_size_q: (idx_tdst + 1) * block_size_q, :],
                k[idx_n, :(idx_tdst + 1) * block_size_q + TSRC - TDST + 1, :],
                indices[idx_n, :],
                mask_k=mask_k,
                mask_k_init=mask_k_init,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
            )
            ks[idx_n] = final_k

def masking_iteration_draft(
    q: Tensor,
    k: Tensor,
    mask_k: int,
    block_size_q: int,
    block_size_k: int,
):
    mask_k_init = mask_k // 2
    
    device = q.device
    q = q.cpu().numpy()
    k = k.cpu().numpy()
    
    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape
    
    indices = np.zeros((
        N, 
        cdiv(TDST, block_size_q), 
        cdiv(mask_k, block_size_k)
    ), dtype=np.int32)
    ks = np.zeros((N, cdiv(TDST, block_size_q),), dtype=np.int32)
    
    masking_iteration_draft_numba(
        q, k, 
        indices, ks, 
        mask_k, mask_k_init,
        block_size_q, block_size_k,
    )
    
    return (
        torch.tensor(indices, device=device),
        torch.tensor(ks, device=device),
    )

@torch.inference_mode()
def hip_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor,
    
    mask_k: int = 512,
    
    block_size_q: int = 32,
    block_size_k: int = 2,
    
    using_sliding_window: bool = True,
    sliding_window_size: int = 128,
):
    indices, ks = masking_iteration_draft(
        q, 
        k, 
        mask_k=mask_k,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
    )
    
    context = block_sparse_flash_attention(
        q, k, v, 
        attention_mask=None,
        indices=indices,
        ks=ks,
        IS_CAUSAL=True,
        KV_REPEAT_INTERLEAVE=1,
        BLOCK_SIZE_Q=block_size_q,
        BLOCK_SIZE_K=block_size_k,
        USING_SLIDING_WINDOW=using_sliding_window,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        ROPE_METHOD='none',
        ROPE_COS=None,
        ROPE_SIN=None,
        POSITION_IDS=None,
        SELF_EXTEND_SCALE=1,
        SELF_EXTEND_WINDOW=1,
    )
    
    return context, None

if __name__ == '__main__':
    q, k, v, out = load_checkouts(idx=0, window=16)
    out_est, _ = hip_attention(q, k, v)
    print(F.mse_loss(out, out_est).item())