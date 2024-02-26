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

import json
import random
import gc
from matplotlib import pyplot as plt
import numpy as np
import skimage.measure
import skimage
import torch
from torch import Tensor
import tqdm
import triton
import triton.language as tl
from typing import Literal, Optional, Tuple, List, Union
import os
import math
from torch.autograd import Function

assert triton.__version__ in ['2.2.0']

from timber.utils import get_bench, seed
from timber.models.timber_attention.common import load_checkouts

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
    ATTEN_MASK, stride_atten_mask_n, stride_atten_mask_tsrc,
    SPARQ_INDICES, stride_sparq_indices_n, stride_sparq_indices_bdst, stride_sparq_indices_hid,
    
    # input / temp metrices (blocked)
    MASK, stride_mask_n, stride_mask_bdst, stride_mask_k,
    TMASK, stride_tmask_n, stride_tmask_bdst, stride_tmask_k,
    
    # temp vectors (blocked)
    WS, stride_ws_n, stride_ws_bdst,
    KS, stride_ks_n, stride_ks_bdst,
    TSRCS, stride_tsrcs_n, stride_tsrcs_bdst,
    
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
    HID: int, 
    SPARQ_HID: int,
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
    VLLM_BLOCK_SIZE: int,
    VLLM_X: int, 
    VLLM_HEAD_SIZE: int,
    
    BLOCK_TABLES, 
    stride_block_tables_num_seqs, 
    stride_block_tables_max_num_blocks_per_seq,
    
    CONTEXT_LENGTH,
    stride_context_length_num_seqs,
    
    # block constant
    KEY_CACHE_METHOD: tl.constexpr,
    SPARQ: tl.constexpr,
    REDUCE_METHOD: tl.constexpr,
    BLOCK_MASK_K: tl.constexpr, 
    BLOCK_TMASK_K: tl.constexpr, 
    BLOCK_MAX_DUP: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    REDUCE_STRDIE: tl.constexpr,
):
    idx_n = tl.program_id(1).to(tl.int64)
    idx_bdst = tl.program_id(0).to(tl.int64) + N_COMPLETED
    """ non blocked
    # for each query
    w_old = ws[i, j, 0]
    t_src = t_srcs[i, j, 0]
    w_new = min(torch.round(w_old * scale_up), t_src)
    """
    
    for _ in range(N_ITERATION):
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
            tl.math.round(w_old.to(tl.float32) * SCALE_UP).to(tl.float32), 
            t_src
        ).to(tl.int64)
        
        """
        if w_old != w_new:
        """
        # if w_old == w_new:
        #     return

        mask_w = w_old != w_new
        
        """
        k_old = ks[i, j, 0]
        k_new = max(n_patches, int(min(mask_k * BLOCK_SIZE / t_src, 1.0) * w_new) c/ BLOCK_SIZE)
        k_new = min(t_src c/ BLOCK_SIZE, max(n_patches, k_new))
        """
        
        k_old = tl.load(
            KS + \
                idx_n * stride_ks_n +\
                idx_bdst * stride_ks_bdst,
            mask=mask_w,
        ).to(tl.int64)
        # """
        k_new = tl.maximum(
            N_PATCHES,
            (
                tl.minimum(
                    MASK_K / tl.cdiv(t_src, BLOCK_SIZE_K).to(tl.float32),
                    1.0
                ) * tl.cdiv(w_new, BLOCK_SIZE_K)
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
        k_new = tl.minimum(tl.cdiv(t_src, BLOCK_SIZE_K), tl.maximum(N_PATCHES, k_new))
        
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
        
        k_old_range = tl.arange(0, BLOCK_MASK_K).to(tl.int64)
        k_old_mask = tl.arange(0, BLOCK_MASK_K) < k_old
        # tl.debug_barrier()
        loc_vec = tl.load(
            MASK +\
                idx_n * stride_mask_n +\
                idx_bdst * stride_mask_bdst +\
                k_old_range * stride_mask_k,
            mask = mask_w & k_old_mask,
            other = 0
        )
        k_old_mask = k_old_mask & (loc_vec < 1.0)
        
        # w_old_fp = w_old.to(tl.float32)
        # w_new_fp = w_new.to(tl.float32)
        b_old_fp = tl.cdiv(w_old, BLOCK_SIZE_K).to(tl.float32)
        b_new_fp = tl.cdiv(w_new, BLOCK_SIZE_K).to(tl.float32)
        loc_idx_start_vec = (loc_vec * b_old_fp).to(tl.int64)
        loc_idx_end_vec = loc_idx_start_vec + 1
        loc_idx_start_vec = (loc_idx_start_vec.to(tl.float32) / b_old_fp * b_new_fp).to(tl.int64)
        loc_idx_end_vec = (loc_idx_end_vec.to(tl.float32) / b_old_fp * b_new_fp).to(tl.int64)
        
        dup_pixels_vec = loc_idx_end_vec - loc_idx_start_vec
        dup_pixels_vec = dup_pixels_vec * k_old_mask
        num_pixels_vec = tl.cumsum(dup_pixels_vec)
        dup_pixels_first = tl.min(num_pixels_vec)
        num_pixels_scalar = tl.max(num_pixels_vec)
        
        num_pixels_scalar_exceed = tl.maximum(num_pixels_scalar - TMASK_K, 0)
        num_pixels_vec = tl.maximum(0, num_pixels_vec - num_pixels_scalar_exceed)
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
        
        
        for _idx in range(BLOCK_MAX_DUP):
            # _idx = BLOCK_MAX_DUP - _idx - 1
            tl.store(
                TMASK + \
                    idx_n * stride_tmask_n +\
                    idx_bdst * stride_tmask_bdst +\
                    ((num_pixels_vec - dup_pixels_first) + _idx).to(tl.int64) * stride_tmask_k,
                mask=mask_w & (_idx <= dup_pixels_vec) & k_old_mask,
                value=(
                    (loc_idx_start_vec + _idx).to(tl.float32) / tl.cdiv(w_new, BLOCK_SIZE_K).to(tl.float32)
                )
            )
        
        # idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
        # mask_block_k = idx_block_k < BLOCK_SIZE_K
        idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED).to(tl.int64)
        mask_block_q = idx_block_q < BLOCK_SIZE_Q
        
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
            scores = tl.zeros((BLOCK_TMASK_K,), dtype=tl.float32)
            
            if REDUCE_METHOD == 'first':
                assert KEY_CACHE_METHOD == 'cont'
                
                for _idx_hid in range(tl.cdiv(HID, BLOCK_HID)):
                    idx_hid = tl.arange(0, BLOCK_HID) + _idx_hid * BLOCK_HID
                    mask_hid = idx_hid < HID
                    vec_q = tl.load(
                        QUERIES +\
                            idx_n * stride_queries_n +\
                            (idx_bdst * BLOCK_SIZE_Q).to(tl.int64) * stride_queries_tdst +\
                            (idx_hid[None, :] + tl.arange(0, 16)[:, None]).to(tl.int64) * stride_queries_hid,
                        mask = mask_w & (mask_hid[None, :] & (tl.arange(0, 16)[:, None] < 1)),
                        other = 0,
                    )
                    # tl.debug_barrier()
                    
                    num_pixels_range = tl.arange(0, BLOCK_TMASK_K).to(tl.int64)
                    num_pixels_mask = num_pixels_range < num_pixels_scalar
                    idx_tsrc = tl.load(
                        TMASK +\
                            idx_n * stride_tmask_n +\
                            idx_bdst * stride_tmask_bdst +\
                            num_pixels_range * stride_tmask_k,
                        mask = mask_w & num_pixels_mask,
                        other = 0,
                    )
                    # tl.debug_barrier()
                    # NOTE: random key selection with in the block
                    # loc_k_vec = loc_k_vec.to(tl.float32) + tl.rand(idx_n * idx_tdst, w_old, 10) * (1.0 / w_old)
                    idx_tsrc = (idx_tsrc.to(tl.float32) * t_src.to(tl.float32)).to(tl.int64)
                    mask_tsrc = num_pixels_mask
                    vec_k_mask = mask_tsrc[None, :] & mask_hid[:, None]
                    
                    if ATTEN_MASK is not None:
                        key_mask = tl.load(
                            ATTEN_MASK +\
                                idx_n * stride_atten_mask_n +\
                                idx_tsrc * stride_atten_mask_tsrc,
                            mask = mask_w & num_pixels_mask,
                            other = False,
                        ).to(tl.int1)
                        vec_k_mask = vec_k_mask & key_mask[None, :]
                    
                    vec_k = tl.load(
                        KEYS +\
                            (idx_n // KV_REPEAT_INTERLEAVE) * stride_keys_n +\
                            idx_tsrc[None, :] * stride_keys_tsrc + \
                            idx_hid[:, None] * stride_keys_hid,
                        mask = mask_w & vec_k_mask,
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
                    if vec_k.dtype == tl.uint8:
                        vec_k = vec_k.to(tl.float8e5, bitcast=True).to(vec_q.dtype)
                    scores_partial = -tl.dot(vec_q, vec_k).to(scores.dtype)
                    scores_partial = tl.sum(scores_partial, axis=0)
                    scores_partial = scores_partial + (~num_pixels_mask) * 10000.0
                    scores_partial = scores_partial +\
                        ((idx_bdst * BLOCK_SIZE_Q + T_SRC - T_DST) < idx_tsrc) * 10000.0
                    
                    scores += scores_partial
            elif REDUCE_METHOD == 'max' or REDUCE_METHOD == 'sum':
                # NOTE: init scores
                if REDUCE_METHOD == 'max':
                    scores += 32000.0
                elif REDUCE_METHOD == 'sum':
                    scores *= 0.0
                
                idx_tdst = (idx_bdst * BLOCK_SIZE_Q + idx_block_q).to(tl.int64)
                mask_tdst = (idx_tdst < T_DST) & mask_block_q
                
                if ATTEN_MASK is not None:
                    query_mask = tl.load(
                        ATTEN_MASK +\
                            idx_n * stride_atten_mask_n +\
                            (idx_tdst + T_SRC - T_DST) * stride_atten_mask_tsrc,
                        mask = mask_w & mask_tdst,
                        other = False
                    ).to(tl.int1)
                # mask_tdst = mask_tdst & query_mask
                
                num_pixels_range = tl.arange(0, BLOCK_TMASK_K).to(tl.int64)
                num_pixels_mask = num_pixels_range < num_pixels_scalar
                idx_tsrc_block = tl.load(
                    TMASK +\
                        idx_n * stride_tmask_n +\
                        idx_bdst * stride_tmask_bdst +\
                        num_pixels_range * stride_tmask_k,
                    mask = mask_w & num_pixels_mask,
                    other = 0,
                )
                # tl.debug_barrier()
                # NOTE: random key selection with in the block
                # loc_k_vec = loc_k_vec.to(tl.float32) + tl.rand(idx_n * idx_tdst, w_old, 10) * (1.0 / w_old)
                idx_tsrc_block = (idx_tsrc_block.to(tl.float32) * t_src.to(tl.float32)).to(tl.int64)
                
                for _idx_block_k in range(BLOCK_SIZE_K):
                    scores_partial = tl.zeros((BLOCK_SIZE_Q_PADDED, BLOCK_TMASK_K), dtype=tl.float32)
                    
                    # [BLOCK_TMASK_K, ]
                    idx_tsrc = (idx_tsrc_block + _idx_block_k).to(tl.int64)
                    # idx_tsrc = tl.minimum(idx_tsrc + (tl.maximum(0, tl.cdiv(t_src, (k_new * BLOCK_SIZE_K)) - BLOCK_SIZE_K)).to(tl.int64), T_SRC - 1)
                    mask_tsrc = (idx_tsrc < T_SRC) & (_idx_block_k < BLOCK_SIZE_K) & ((_idx_block_k % REDUCE_STRDIE) == 0)
                    
                    # [BLOCK_TMASK_K, ]
                    if ATTEN_MASK is not None:
                        key_mask = tl.load(
                            ATTEN_MASK +\
                                idx_n * stride_atten_mask_n +\
                                idx_tsrc * stride_atten_mask_tsrc,
                            mask = mask_w & mask_tsrc,
                            other = False,
                        ).to(tl.int1)
                    # mask_tsrc = mask_tsrc & key_mask
                    
                    mask_strided_block_q = (idx_block_q % REDUCE_STRDIE) == 0
                    hidden_size = SPARQ_HID if SPARQ else HID
                    for pid_hid in range(tl.cdiv(hidden_size, BLOCK_HID)):
                        idx_hid = (tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID).to(tl.int64)
                        mask_hid = idx_hid < hidden_size
                        
                        if SPARQ:
                            idx_hid = tl.load(
                                SPARQ_INDICES +\
                                    idx_n * stride_sparq_indices_n +\
                                    idx_bdst * stride_sparq_indices_bdst +\
                                    idx_hid * stride_sparq_indices_hid,
                                mask = mask_w & mask_hid,
                                other = HID,
                            )
                        mask_hid = idx_hid < HID
                        
                        # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
                        mask_vec_q = (
                            mask_hid[None, :] &
                            mask_tdst[:, None] &
                            mask_block_q[:, None] &
                            mask_strided_block_q[:, None] &
                            True
                        )
                        if ATTEN_MASK is not None:
                            mask_vec_q = mask_vec_q & query_mask[:, None]
                        vec_q = tl.load(
                            QUERIES +\
                                idx_n * stride_queries_n +\
                                idx_tdst[:, None] * stride_queries_tdst +\
                                idx_hid[None, :] * stride_queries_hid,
                            mask = mask_w & mask_vec_q,
                            other = 0,
                        )
                        # tl.debug_barrier()
                        
                        # [BLOCK_HID: hid, BLOCK_TMASK_K: tsrc]
                        vec_k_mask = (
                            num_pixels_mask[None, :] &
                            mask_hid[:, None] &
                            mask_tsrc[None, :] &
                            # key_mask[None, :] &
                            True
                        )
                        if KEY_CACHE_METHOD == 'cont':
                            # [BLOCK_HID: hid, BLOCK_TMASK_K: tsrc]
                            vec_k = tl.load(
                                KEYS +\
                                    (idx_n // KV_REPEAT_INTERLEAVE) * stride_keys_n +\
                                    idx_tsrc[None, :] * stride_keys_tsrc + \
                                    idx_hid[:, None] * stride_keys_hid,
                                mask = mask_w & vec_k_mask,
                                other = 0,
                            )
                        elif KEY_CACHE_METHOD == 'vllm':
                            """
                            idx_block = block_tables[idx_batch, idx_tsrc // block_size]
                            offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
                            key = key_cache[idx_block, idx_head, :, offset_block, :].reshape(-1)
                            """
                            idx_batch = ((idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS).to(tl.int64)
                            idx_head = ((idx_n // KV_REPEAT_INTERLEAVE) % VLLM_NUM_KV_HEADS).to(tl.int64)
                            idx_block = tl.load(
                                BLOCK_TABLES +\
                                    idx_batch * stride_block_tables_num_seqs +\
                                    (idx_tsrc // VLLM_BLOCK_SIZE) * stride_block_tables_max_num_blocks_per_seq,
                                mask = mask_w & mask_tsrc,
                            ).to(tl.int64)
                            offset_block = (idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)).to(tl.int64)
                            
                            # [BLOCK_HID: hid, BLOCK_TMASK_K: tsrc]
                            vec_k = tl.load(
                                KEYS +\
                                    idx_block[None, :] * stride_keys_vllm_num_blcoks +\
                                    idx_head * stride_keys_vllm_num_kv_heads +\
                                    (idx_hid[:, None] // VLLM_X) * stride_keys_vllm_head_size_x +\
                                    offset_block[None, :] * stride_keys_vllm_block_size +\
                                    (idx_hid[:, None] % VLLM_X) * stride_keys_vllm_x,
                                mask = mask_w & vec_k_mask,
                                other = 0,
                            )
                        else:
                            raise Exception()
                        
                        # [BLOCK_SIZE_PADDED: tdst, BLOCK_TMASK_K: tsrc]
                        if vec_k.dtype == tl.uint8:
                            vec_k = vec_k.to(tl.float8e5, bitcast=True).to(vec_q.dtype)
                        scores_micro = -tl.dot(vec_q, vec_k)
                        scores_partial += scores_micro.to(scores_partial.dtype)
                    
                    # [BLOCK_SIZE_PADDED: tdst, BLOCK_TMASK_K: tsrc]
                    scores_partial_ignore_mask = (
                        (~num_pixels_mask[None, :]) |
                        (~mask_tdst[:, None]) |
                        (~mask_tsrc[None, :]) |
                        (~mask_block_q[:, None]) |
                        (~mask_strided_block_q[:, None]) |
                        (scores_partial == 0) |
                        False
                    )
                    
                    if IS_CAUSAL:
                        scores_partial_ignore_mask |= (
                            ((idx_tdst[:, None] + T_SRC - T_DST) < idx_tsrc[None, :]) |
                            False
                        )
                    
                    if ATTEN_MASK is not None:
                        scores_partial_ignore_mask |= (
                            (~key_mask[None, :]) |
                            (~query_mask[:, None]) |
                            False
                        )
                    
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
                    scores_partial_force_mask = False
                    
                    # tl.device_print("", idx_tdst)
                    # tl.device_print("", idx_tsrc)
                    
                    scores_partial_ignore_mask = scores_partial_ignore_mask & (~scores_partial_force_mask)
                    
                    # NOTE: reduce
                    if REDUCE_METHOD == 'max':
                        scores_partial = scores_partial + scores_partial_ignore_mask * 32000.0
                        scores_partial = scores_partial + scores_partial_force_mask * (-32000.0)
                        # scores_partial = scores_partial * (~scores_partial_force_mask)
                        scores_partial = tl.min(scores_partial, axis=0)
                        scores = tl.minimum(scores, scores_partial)
                    elif REDUCE_METHOD == 'sum':
                        scores_partial = scores_partial + scores_partial_ignore_mask * 10000.0
                        scores_partial = scores_partial + scores_partial_force_mask * (-10000.0)
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
            
            # tl.device_print("", scores)
            
            # select min-k from negative scores -> select top-k
            # masked_scores = scores + (~num_pixels_mask) * 32000.0
            masked_scores = scores
            
            scores_kth_large = _triton_kth_large(masked_scores, k_new, BLOCK_TMASK_K)
            topk_mask = masked_scores <= scores_kth_large
            # topk_mask = tl.arange(0, BLOCK_TMASK_K) <= k_new
            
            topk_mask_cumsum = tl.cumsum(topk_mask.to(tl.int64))
            # tl.debug_barrier()
            topk_range = tl.minimum((topk_mask_cumsum - 1) * topk_mask, k_new - 1).to(tl.int64)
            # tl.debug_barrier()
            
            temp_range = tl.arange(0, BLOCK_TMASK_K).to(tl.int64)
            temp_mask = temp_range < num_pixels_scalar
            temp = tl.load(
                TMASK +\
                    idx_n * stride_tmask_n +\
                    idx_bdst * stride_tmask_bdst +\
                    temp_range * stride_tmask_k,
                mask=mask_w & temp_mask,
                other=0
            )
            # tl.debug_barrier()
            tl.store(
                MASK +\
                    idx_n * stride_mask_n +\
                    idx_bdst * stride_mask_bdst +\
                    topk_range * stride_mask_k,
                mask=mask_w & topk_mask & temp_mask,
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
            temp1_range = tl.arange(0, BLOCK_MASK_K).to(tl.int64)
            temp1_mask = temp1_range < num_pixels_scalar
            # tl.debug_barrier()
            temp1 = tl.load(
                TMASK +\
                    idx_n * stride_tmask_n +\
                    idx_bdst * stride_tmask_bdst +\
                    temp1_range * stride_tmask_k,
                mask=mask_w & temp1_mask,
                other=0,
            )
            
            # tl.debug_barrier()
            tl.store(
                MASK +\
                    idx_n * stride_mask_n +\
                    idx_bdst * stride_mask_bdst +\
                    temp1_range * stride_mask_k,
                mask=mask_w & temp1_mask,
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
            mask = mask_w,
            value = w_new
        )
        # tl.debug_barrier()
        tl.store(
            KS +\
                idx_n * stride_ks_n +\
                idx_bdst * stride_ks_bdst,
            mask = mask_w,
            value = tl.minimum(k_new, num_pixels_scalar)
            # value = k_new,
            # value = num_pixels_scalar,
        )
        tl.debug_barrier()

DEBUG = os.environ.get('TIMBER_DEBUG', '0') == '1'

def next_multiple_of(x: int, multiple_by: int = 16):
    # if (x % multiple_by) == 0:
    #     return x
    # return x + multiple_by - (x % multiple_by)
    return triton.next_power_of_2(max(x, multiple_by))

def masking_iteration(
    # input matrices
    queries: Tensor, keys: Union[Tensor, "PagedKeyCacheVllmCompat"], attention_mask: Tensor,
    # input metrices (blocked) 
    mask: Tensor, t_mask: Tensor, sparq_indices, sparq_indices_strides,
    # temp vectors (blocked)
    ws: Tensor, ks: Tensor, t_srcs: Tensor, 
    # operator variables
    scale_up: float, n_patches: int, mask_k: int, is_causal: bool,
    # iteration controls
    i_iteration: int, n_iteration: int,
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
):  
    global DEBUG
    if DEBUG:
        # print(ws)
        # print(ks[0, 10])
        # print(mask[0, 10])
        # print(t_srcs)
        print(
            'masking_iteration', 
            queries.shape, queries.data_ptr(), 
            keys.shape, keys.data_ptr(), 
            mask.shape, mask.data_ptr(),
            t_mask.shape, t_mask.data_ptr(),
            ws.shape, ws.data_ptr(),
            ks.shape, ks.data_ptr(),
            t_srcs.shape, t_srcs.data_ptr(),
            N, T_DST, T_SRC, B_DST, B_SRC, HID,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
            REDUCE_METHOD,
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
    if BLOCK_TMASK_K >= 1024:
        BLOCK_HID = min(BLOCK_HID, 16)
    elif BLOCK_TMASK_K >= 512:
        BLOCK_HID = min(BLOCK_HID, 32)
    elif BLOCK_TMASK_K >= 256:
        BLOCK_HID = min(BLOCK_HID, 64)
    elif BLOCK_TMASK_K >= 128:
        BLOCK_HID = min(BLOCK_HID, 128)
    # print(BLOCK_HID, BLOCK_TMASK_K)
    
    if isinstance(keys, Tensor):
        KEY_CACHE_METHOD = 'cont'
        stride_keys_vllm = (0, 0, 0, 0, 0)
        VLLM_NUM_BLOCKS = 0
        VLLM_NUM_KV_HEADS = 0
        VLLM_HEAD_SIZE_X = 0
        VLLM_BLOCK_SIZE = 0
        VLLM_X = 0
        VLLM_HEAD_SIZE = 0
        block_tables = keys
        block_tables_stride = (0, 0)
        context_length = keys
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
        KEY_CACHE_METHOD = 'vllm'
        stride_keys_vllm = keys.key_cache.stride()
        (
            VLLM_NUM_BLOCKS, 
            VLLM_NUM_KV_HEADS, 
            VLLM_HEAD_SIZE_X, 
            VLLM_BLOCK_SIZE, 
            VLLM_X
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
    
    grid = (B_DST - N_COMPLETED, N)
    
    # HID cannot be chunked if use reduce
    # if REDUCE_METHOD in ['max', 'sum']:
    #     assert HID <= BLOCK_HID
    assert REDUCE_METHOD in ['max', 'sum', 'first']
    
    assert queries.ndim == 3
    assert keys.ndim == 3
    if attention_mask is not None:
        assert attention_mask.ndim == 2
    assert mask.ndim == 3
    assert t_mask.ndim == 3
    assert ws.ndim == 2
    assert ks.ndim == 2
    assert t_srcs.ndim == 2
    _masking_iteration_compute[grid](
        # input matrices
        queries, *queries.stride(),
        keys, *keys.stride(),
        attention_mask, *(attention_mask.stride() if attention_mask is not None else (0, 0)),
        sparq_indices, *sparq_indices_strides,
        
        # input matrices (blocked)
        mask, *mask.stride(),
        t_mask, *t_mask.stride(),
        
        # temp vectors (blocked)
        ws, *ws.stride(),
        ks, *ks.stride(),
        t_srcs, *t_srcs.stride(),
        
        # operation variables
        float(scale_up), int(n_patches), int(mask_k), int(t_mask.shape[-1]), is_causal,
        
        # input variables
        KV_REPEAT_INTERLEAVE, 
        N, 
        T_DST, 
        T_SRC, 
        int(B_DST), 
        int(B_SRC), 
        HID, 
        SPARQ_HID, 
        N_COMPLETED,
        n_iteration,
        
        # vLLM compat inputs
        *stride_keys_vllm,
        
        VLLM_NUM_BLOCKS, 
        VLLM_NUM_KV_HEADS,
        VLLM_HEAD_SIZE_X,
        VLLM_BLOCK_SIZE,
        VLLM_X, 
        VLLM_HEAD_SIZE,
        
        block_tables, *block_tables_stride,
        
        context_length, *context_length_stride,
        
        # block constant
        KEY_CACHE_METHOD,
        SPARQ,
        REDUCE_METHOD,
        BLOCK_MASK_K,
        BLOCK_TMASK_K,
        triton.next_power_of_2(math.ceil(scale_up)),
        int(BLOCK_HID),
        int(BLOCK_SIZE_Q),
        next_multiple_of(BLOCK_SIZE_Q, 16),
        int(BLOCK_SIZE_K),
        next_multiple_of(BLOCK_SIZE_K, 1),
        REDUCE_STRIDE,
        
        num_warps=min(8, max(BLOCK_TMASK_K//32, 1)) if SPARQ else 4,
        # num_warps=16,
        num_stages=2,
        enable_warp_specialization=False,
    )
    
    # if DEBUG:
    #     print('after')
    #     print(ks[0, 10])
    #     print(mask[0, 10])
    #     print('after')

def torch_cdiv(a, b):
    t1 = a.div_(b)
    t2 = torch.floor(t1)
    t1.sub_(t2)
    return t2.add_(torch.ceil_(t1))

@triton.jit
def _safe_indices_compute(
    # input tensors
    MASK, stride_mask_n, stride_mask_tdst, stride_mask_k,
    WS, stride_ws_n, stride_ws_tdst, stride_ws_k,
    
    # output tensors
    INDICES, stride_indices_n, stride_indices_tdst, stride_indices_k,
    
    N, TDST, K, BLOCK_SIZE_K,
    
    ALLOW_COLLISION: tl.constexpr,
    BLOCK_N_TDST: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    if not ALLOW_COLLISION:
        pids = tl.program_id(0) * BLOCK_N_TDST + tl.arange(0, BLOCK_N_TDST)
    
        idx_n = pids // TDST
        mask_n = idx_n < N
        
        idx_tdst = pids % TDST
        mask_tdst = idx_tdst < TDST
        
        mask = mask_n & mask_tdst
        
        last_col = tl.zeros((BLOCK_N_TDST, ), dtype=tl.int64) - 1
        for idx_k in range(K):
            mask_vec = tl.load(
                MASK +\
                    idx_n * stride_mask_n +\
                    idx_tdst * stride_mask_tdst +\
                    idx_k * stride_mask_k,
                mask = mask,
                other = 0
            ).to(tl.float32)
            ws_vec = tl.load(
                WS +\
                    idx_n * stride_ws_n +\
                    idx_tdst * stride_ws_tdst +\
                    idx_k * stride_ws_k,
                mask = mask,
                other = 0
            ).to(tl.float32)
            indices_float = mask_vec * ws_vec
            col = tl.math.ceil(indices_float / BLOCK_SIZE_K).to(tl.int32)

            # avoid collision
            col = tl.maximum(last_col + 1, col)
            last_col = col
            
            col = col * BLOCK_SIZE_K
            
            tl.store(
                INDICES +\
                    idx_n * stride_indices_n +\
                    idx_tdst * stride_indices_tdst +\
                    idx_k * stride_indices_k,
                value = col,
                mask = mask
            )
    else:
        pids_ntdst = tl.program_id(1) * BLOCK_N_TDST + tl.arange(0, BLOCK_N_TDST)
    
        idx_n = (pids_ntdst // TDST)[:, None]
        mask_n = idx_n < N
        
        idx_tdst = (pids_ntdst % TDST)[:, None]
        mask_tdst = idx_tdst < TDST
        
        idx_k = (tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K))[None, :]
        mask_k = idx_k < K
        
        mask = mask_n & mask_tdst & mask_k
        
        mask_vec = tl.load(
            MASK +\
                idx_n * stride_mask_n +\
                idx_tdst * stride_mask_tdst +\
                idx_k * stride_mask_k,
            mask = mask,
            other = 0
        ).to(tl.float32)
        ws_vec = tl.load(
            WS +\
                idx_n * stride_ws_n +\
                idx_tdst * stride_ws_tdst +\
                idx_k * stride_ws_k,
            mask = mask,
            other = 0
        ).to(tl.float32)
        
        indices_float = mask_vec * ws_vec
        col = tl.math.ceil(indices_float / BLOCK_SIZE_K).to(tl.int32)
        col = col * BLOCK_SIZE_K
        
        tl.store(
            INDICES +\
                idx_n * stride_indices_n +\
                idx_tdst * stride_indices_tdst +\
                idx_k * stride_indices_k,
            value = col,
            mask = mask
        )

def safe_indices(mask, ws, block_size_k, allow_collision=True):
    N, TDST, K = mask.shape
    ws = ws.unsqueeze(-1).expand(N, TDST, K)

    indices = torch.empty(
        (N, TDST, K), 
        dtype=torch.int32, 
        device=mask.device
    )
    
    BLOCK_N_TDST = 32
    BLOCK_K = 128
    
    if not allow_collision:
        grid = (triton.cdiv(N*TDST, BLOCK_N_TDST), )
    else:
        grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(N*TDST, BLOCK_N_TDST), )
    
    assert indices.ndim == 3
    assert mask.ndim == 3
    assert indices.ndim == 3
    _safe_indices_compute[grid](
        mask, *mask.stride(),
        ws, *ws.stride(),
        
        indices, *indices.stride(),
        
        N, TDST, K, block_size_k,
        
        allow_collision,
        BLOCK_N_TDST,
        BLOCK_K,
        
        num_warps=4 if allow_collision else 1,
    )
    
    # indices = indices.reshape(N, TDST, K)
    
    return indices

@triton.jit
def _calc_prob_return_context_compute(
    # input matrices
    Q, stride_q_n, stride_q_tdst, stride_q_hid,
    K, stride_k_n, stride_k_tsrc, stride_k_hid,
    V, stride_v_n, stride_v_tsrc, stride_v_hid,
    ATTEN_MASK, stride_atten_mask_n, stride_atten_mask_tsrc,
    
    # indices metrices
    INDICES, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    KS, stride_ks_n, stride_ks_bdst,
    
    # output matrices,
    CONTEXT, stride_context_n, stride_context_tdst, stride_context_hid,
    
    # input variables
    KV_REPEAT_INTERLEAVE, N, TDST, TSRC, HID: tl.constexpr, BDST, BSRC, BK,
    
    # vllm compat
    stride_k_vllm_num_blocks, 
    stride_k_vllm_num_kv_heads, 
    stride_k_vllm_head_size_x, 
    stride_k_vllm_block_size, 
    stride_k_vllm_x,
    
    stride_v_vllm_num_blocks,
    stride_v_vllm_num_kv_heads,
    stride_v_vllm_head_size,
    stride_v_vllm_block_size,
    
    BLOCK_TABLES,
    stride_block_tables_num_seqs,
    stride_block_tables_max_num_blocks_per_seq,
    
    VLLM_NUM_BLOCKS,
    VLLM_NUM_KV_HEADS,
    VLLM_HEAD_SIZE_X,
    VLLM_BLOCK_SIZE: tl.constexpr,
    VLLM_X: tl.constexpr,
    VLLM_HEAD_SIZE,
    
    # block constant
    CACHE_METHOD: tl.constexpr,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    pid_n = pid // BDST
    pid_bdst = pid % BDST
    
    # pid_n = tl.program_id(0).to(tl.int64)
    # pid_bdst = tl.program_id(1).to(tl.int64)
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED).to(tl.int64)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    
    idx_n = pid_n
    
    idx_bdst = pid_bdst
    idx_tdst = (idx_block_q + idx_bdst * BLOCK_SIZE_Q).to(tl.int64)
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    idx_hid = tl.arange(0, BLOCK_HID)
    if BLOCK_HID != HID:
        mask_hid = idx_hid < HID
    else:
        mask_hid = True
    
    ks = tl.load(
        KS + \
            idx_n * stride_ks_n +
            idx_bdst * stride_ks_bdst,
    ).to(tl.int64)
    
    acc = tl.zeros((BLOCK_SIZE_Q_PADDED, BLOCK_HID), dtype=tl.float32)
    # scores_rowmax_state: [BLOCK_SIZE_Q: tdst, 1: tsrc]
    m_i = tl.full((BLOCK_SIZE_Q_PADDED, 1), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_SIZE_Q_PADDED, 1), 1.0, dtype=tl.float32)
    
    for idx_bbk in range(tl.cdiv(ks, BLOCK_BK)):
        idx_bk = (tl.arange(0, BLOCK_BK) + idx_bbk * BLOCK_BK).to(tl.int64)
        mask_bk = (idx_bk < ks) & (idx_bk < BK)
        
        # [BLOCK_BK,]
        idx_tsrc_block_start = tl.load(
            INDICES +\
                idx_n * stride_indices_n +\
                idx_bdst * stride_indices_bdst +\
                idx_bk * stride_indices_bk,
            mask = mask_bk,
            other = TSRC,
        ).to(tl.int64)
        
        # [BLOCK_BK, BLOCK_SIZE_K]
        idx_tsrc = tl.arange(0, BLOCK_SIZE_K)[None, :].to(tl.int64) + idx_tsrc_block_start[:, None]
        mask_tsrc = (idx_tsrc < TSRC) & mask_bk[:, None]
        
        # [BLOCK_BK * BLOCK_SIZE_K; multiple of 16]
        idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K,))
        mask_tsrc = tl.reshape(mask_tsrc, (BLOCK_BK * BLOCK_SIZE_K,))
        
        # keys := [BLOCK_HID: hid, BLOCK_BK * BLOCK_SIZE_K: tsrc]
        # queries := [BLOCK_SIZE_Q: tdst, BLOCK_HID: hid]
        # scores := [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]
        queries = tl.load(
            Q +\
                idx_n * stride_q_n +\
                idx_tdst[:, None] * stride_q_tdst +\
                idx_hid[None, :] * stride_q_hid,
            mask = mask_tdst[:, None] & mask_hid[None, :],
            other = 0
        )

        if CACHE_METHOD == 'cont':
            keys = tl.load(
                K +\
                    (idx_n // KV_REPEAT_INTERLEAVE) * stride_k_n +\
                    idx_tsrc[None, :] * stride_k_tsrc +\
                    idx_hid[:, None] * stride_k_hid,
                mask = mask_tsrc[None, :] & mask_hid[:, None],
                other = 0,
            )
        elif CACHE_METHOD == 'vllm':
            """
            idx_block = block_tables[idx_batch, idx_tsrc // block_size]
            offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
            key = key_cache[idx_block, idx_head, :, offset_block, :].reshape(-1)
            """
            idx_batch = ((idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS).to(tl.int64)
            idx_head = ((idx_n // KV_REPEAT_INTERLEAVE) % VLLM_NUM_KV_HEADS).to(tl.int64)
            idx_block = tl.load(
                BLOCK_TABLES +\
                    idx_batch * stride_block_tables_num_seqs +\
                    (idx_tsrc // VLLM_BLOCK_SIZE) * stride_block_tables_max_num_blocks_per_seq,
                mask = mask_tsrc,
            ).to(tl.int64)
            offset_block = (idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)).to(tl.int64)
            
            # [BLOCK_HID: hid, BLOCK_BK: bk, BLOCK_SIZE_K_PADDED: tsrc]
            keys = tl.load(
                K +\
                    idx_block[None, :] * stride_k_vllm_num_blocks +\
                    idx_head * stride_k_vllm_num_kv_heads +\
                    (idx_hid[:, None] // VLLM_X) * stride_k_vllm_head_size_x +\
                    offset_block[None, :] * stride_k_vllm_block_size +\
                    (idx_hid[:, None] % VLLM_X) * stride_k_vllm_x,
                mask = mask_tsrc[None, :] & mask_hid[:, None],
                other = 0,
            )
        else:
            raise Exception()
        
        if keys.dtype == tl.uint8:
            keys = keys.to(tl.float8e5, bitcast=True).to(queries.dtype)
        
        qk = tl.dot(queries, keys).to(tl.float32) * 1.44269504
        
        if IS_CAUSAL:
            qk += (
                (idx_tdst[:, None] + TSRC - TDST) < idx_tsrc[None, :] |
                (~(mask_tdst[:, None] & mask_tsrc[None, :]))
            ) * (-1.0e-6)
        else:
            qk += (
                ~(mask_tdst[:, None] & mask_tsrc[None, :])
            ) * (-1.0e-6)
        
        # [BLOCK_SIZE_Q: tdst, 1: tsrc]
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1)[:, None])
        qk = qk - m_ij
        # [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]
        p = tl.math.exp2(qk)
        
        if IS_CAUSAL:
            p *= (
                ((idx_tdst[:, None] + TSRC - TDST) >= idx_tsrc[None, :]) &
                (mask_tdst[:, None] & mask_tsrc[None, :])
            )
        else:
            p *= (
                (mask_tdst[:, None] & mask_tsrc[None, :])
            )
        
        # [BLOCK_SIZE_Q: tdst, 1: tsrc]
        l_ij = tl.sum(p, axis=1)
        
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        # tl.device_print('ff', l_ij)
        l_i = l_i * alpha + l_ij[:, None]
        
        # -- update output accumulator --
        acc = acc * alpha
        
        if CACHE_METHOD == 'cont':
            values = tl.load(
                V +\
                    (idx_n // KV_REPEAT_INTERLEAVE) * stride_v_n +\
                    idx_tsrc[:, None] * stride_v_tsrc +\
                    idx_hid[None, :] * stride_v_hid,
                mask = mask_tsrc[:, None] & mask_hid[None, :],
                other = 0
            )
        elif CACHE_METHOD == 'vllm':
            """
            idx_block = block_tables[idx_batch, idx_tsrc // block_size]
            offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
            value = value_cache[idx_block, idx_head, :, offset_block].reshape(-1)
            """
            idx_batch = (idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS
            idx_head = (idx_n // KV_REPEAT_INTERLEAVE) % VLLM_NUM_KV_HEADS
            
            idx_block = tl.load(
                BLOCK_TABLES +\
                    idx_batch * stride_block_tables_num_seqs +\
                    (idx_tsrc // VLLM_BLOCK_SIZE) * stride_block_tables_max_num_blocks_per_seq,
                mask = mask_tsrc,
                other = 0
            ).to(tl.int64)
            mask_block = (idx_tsrc // VLLM_BLOCK_SIZE) < tl.cdiv(TSRC, VLLM_BLOCK_SIZE)
            offset_block = idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)
            
            # value: [BLOCK_SIZE_PADDED: tsrc, BLOCK_HID: hid]
            values = tl.load(
                V +\
                    idx_block[:, None] * stride_v_vllm_num_blocks+\
                    idx_head * stride_v_vllm_num_kv_heads+\
                    idx_hid[None, :].to(tl.int64) * stride_v_vllm_head_size +\
                    offset_block[:, None] * stride_v_vllm_block_size,
                mask = mask_tsrc[:, None] & mask_hid[None, :] & mask_block[:, None],
                other = 0
            )
        else:
            raise Exception()
        
        if values.dtype == tl.uint8:
            values = values.to(tl.float8e5, bitcast=True).to(tl.float16)
        
        # update acc
        acc += tl.dot(p.to(values.dtype), values).to(tl.float32)
        
        # update m_i and l_i
        m_i = m_ij
    
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i
    tl.store(
        CONTEXT +\
            idx_n * stride_context_n +\
            idx_tdst[:, None] * stride_context_tdst +\
            idx_hid[None, :] * stride_context_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        value = acc.to(CONTEXT.type.element_ty)
    )
    

def calc_prob_return_context(
    # input matrices
    queries: Tensor, 
    keys: Union[Tensor, "PagedKeyCacheVllmCompat"], 
    values: Union[Tensor, "PagedValueCacheVllmCompat"], 
    attention_mask: Optional[Tensor],
    # indices metrices
    indices: Tensor, ks: Tensor,
    # block constant
    KV_REPEAT_INTERLEAVE: int,
    BLOCK_SIZE_Q: int,
    BLOCK_SIZE_K: int,
    IS_CAUSAL: bool,
):
    """
    implement flash attention 1, not 2.
    """
    
    N, TDST, HID = queries.shape
    _N, TSRC, HID = keys.shape
    assert keys.shape == values.shape
    assert attention_mask is None or attention_mask.shape == (N, TDST)
    
    BSRC = triton.cdiv(TSRC, BLOCK_SIZE_K)
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    _, _, BK = indices.shape
    assert ks.shape == (N, BDST)
    
    # BLOCK_BK = max(1, 256 // BLOCK_SIZE_K)
    # BLOCK_BK = max(1, triton.next_power_of_2(BK) // 2)
    BLOCK_BK = triton.cdiv(64 if queries.dtype == torch.float32 else 128, BLOCK_SIZE_K)
    # print(256 // BLOCK_SIZE_K, BK)
    BLOCK_HID = triton.next_power_of_2(HID)
    BLOCK_SIZE_Q_PADDED = next_multiple_of(BLOCK_SIZE_Q, 16)
    
    # print(BK, BLOCK_BK)
    
    assert values.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.uint8]
    context = torch.zeros(
        (N, TDST, HID),
        dtype=queries.dtype,
        device=queries.device,
    )
    
    if isinstance(keys, Tensor) and isinstance(values, Tensor):
        CACHE_METHOD = 'cont'
        
        VLLM_NUM_BLOCKS =\
        VLLM_NUM_KV_HEADS =\
        VLLM_HEAD_SIZE_X =\
        VLLM_BLOCK_SIZE =\
        VLLM_X =\
        VLLM_HEAD_SIZE = 0
        
        vllm_keys_strides = (0, 0, 0, 0, 0)
        vllm_values_strides = (0, 0, 0, 0)
        
        block_tables = keys
        block_tables_strides = (0, 0)
    elif isinstance(keys, PagedKeyCacheVllmCompat) and isinstance(values, PagedValueCacheVllmCompat):
        """
        vLLM compatible paged attention
        
        q: [num_seqs, num_heads, head_size]
        k: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        v: [num_blocks, num_kv_heads, head_size, block_size]
        block_tables: [num_seqs, max_num_blocks_per_seq]
        context_lens: [num_seqs]
        """
        
        CACHE_METHOD = 'vllm'
        
        (
            VLLM_NUM_BLOCKS,
            VLLM_NUM_KV_HEADS, 
            VLLM_HEAD_SIZE_X,
            VLLM_BLOCK_SIZE,
            VLLM_X,
        ) = keys.key_cache.shape
        VLLM_HEAD_SIZE = VLLM_HEAD_SIZE_X * VLLM_X
        
        block_tables = keys.block_table
        block_tables_strides = block_tables.stride()
        assert len(block_tables_strides) == 2
        
        vllm_keys_strides = keys.key_cache.stride()
        assert len(vllm_keys_strides) == 5
        
        vllm_values_strides = values.value_cache.stride()
        assert len(vllm_values_strides) == 4
    else:
        raise Exception("not supported")
    
    # grid = (N, BDST, )
    grid = (N * BDST, )
    
    assert attention_mask is None, "attention mask is not supported yet"
    assert queries.ndim == 3
    assert keys.ndim == 3
    assert values.ndim == 3
    assert attention_mask is None or attention_mask.ndim == 3
    assert indices.ndim == 3
    assert ks.ndim == 2
    assert context.ndim == 3
    
    _calc_prob_return_context_compute[grid](
        queries, *queries.stride(),
        keys, *keys.stride(),
        values, *values.stride(),
        attention_mask, *((0, 0) if attention_mask is None else attention_mask.stride()),
        
        indices, *indices.stride(),
        ks, *ks.stride(),
        
        context, *context.stride(),
        
        KV_REPEAT_INTERLEAVE, 
        N, 
        TDST, 
        TSRC, 
        HID, 
        BDST, 
        BSRC, 
        BK,
        
        # vllm key value cache compat
        *vllm_keys_strides,
        *vllm_values_strides,
        
        block_tables,
        *block_tables_strides,
        
        VLLM_NUM_BLOCKS,
        VLLM_NUM_KV_HEADS,
        VLLM_HEAD_SIZE_X,
        VLLM_BLOCK_SIZE,
        VLLM_X,
        VLLM_HEAD_SIZE,
        
        CACHE_METHOD,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_Q_PADDED, 
        BLOCK_SIZE_K,
        BLOCK_HID,
        BLOCK_BK,
        IS_CAUSAL,
        
        num_warps=8,
        num_stages=2,
    )
    
    return context

@triton.jit
def _calc_score_compute(
    # input matrix
    QUERIES, stride_queries_n, stride_queries_tdst, stride_queries_hid,
    KEYS, stride_keys_n, stride_keys_tsrc, stride_keys_hid,
    ATTEN_MASK, stride_atten_mask_n, stride_atten_mask_tsrc,
    
    # block indices
    INDICES, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    KS, stride_ks_n, stride_ks_bdst,
    
    # out matrix
    SCORES, stride_scores_n, stride_scores_tdst, stride_scores_k,
    
    # input variables
    KV_REPEAT_INTERLEAVE, N, TDST, TSRC, HID, BK, K, BDST, BSRC, IS_CAUSAL,
    
    # vllm key cache compat
    stride_keys_vllm_num_bocks,
    stride_keys_vllm_num_kv_heads,
    stride_keys_vllm_head_size_x,
    stride_keys_vllm_block_size,
    stride_keys_vllm_x,
    
    VLLM_NUM_BLOCKS,
    VLLM_NUM_KV_HEADS,
    VLLM_HEAD_SIZE_X,
    VLLM_BLOCK_SIZE,
    VLLM_X,
    VLLM_HEAD_SIZE,
    
    BLOCK_TABLES,
    stride_block_tables_num_seqs,
    stride_block_tables_max_num_blocks_per_seq,
    
    # kernel constatnts
    KEY_CACHE_METHOD: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    idx_n = tl.program_id(0).to(tl.int64)
    idx_bdst = tl.program_id(1).to(tl.int64)
    pid_bk = tl.program_id(2).to(tl.int64)
    
    ks = tl.load(
        KS +\
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst,
    )
    
    # if (pid_bk + 1) * BLOCK_BK > ks:
    #     return
    
    idx_bk = tl.arange(0, BLOCK_BK) + pid_bk * BLOCK_BK
    mask_bk = idx_bk < ks
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_tsrc = tl.load(
        INDICES +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
        mask = mask_bk,
    )
    # [BLOCK_BK: bk, BLOCK_SIZE_K_PADDED]
    idx_tsrc = idx_tsrc[:, None] + idx_block_k[None, :]
    mask_tsrc = (idx_tsrc < TSRC) & mask_block_k[None, :] & mask_bk[:, None]
    
    # [BLOCK_BK: bk, BLOCK_SIZE_K_PADDED]
    if ATTEN_MASK is not None:
        key_mask = tl.load(
            ATTEN_MASK +\
                idx_n * stride_atten_mask_n +\
                idx_tsrc * stride_atten_mask_tsrc,
            mask = mask_tsrc,
            other = False,
        ).to(tl.int1)
        mask_tsrc = mask_tsrc & key_mask
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    if ATTEN_MASK is not None:
        query_mask = tl.load(
            ATTEN_MASK +\
                idx_n * stride_atten_mask_n +\
                (idx_tdst + TSRC - TDST) * stride_atten_mask_tsrc,
            mask = mask_tdst,
            other = False,
        ).to(tl.int1)
        mask_tdst = mask_tdst & query_mask
    
    # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_BK: bk, BLOCK_SIZE_K_PADDED: tsrc]
    scores = tl.zeros((BLOCK_SIZE_Q_PADDED, BLOCK_BK, BLOCK_SIZE_K_PADDED), dtype=tl.float32)
    for pid_hid in range(tl.cdiv(HID, BLOCK_HID)):
        idx_hid = (tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID).to(tl.int64)
        mask_hid = idx_hid < HID
        
        # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_HID: hid]
        queries = tl.load(
            QUERIES +\
                idx_n * stride_queries_n +\
                idx_tdst[:, None] * stride_queries_tdst +\
                idx_hid[None, :] * stride_queries_hid,
            mask = mask_tdst[:, None] & mask_hid[None, :],
            other = 0
        )
        
        if KEY_CACHE_METHOD == 'cont':
            # [BLOCK_HID: hid, BLOCK_BK: bk, BLOCK_SIZE_K_PADDED: tsrc]
            keys = tl.load(
                KEYS +\
                    (idx_n // KV_REPEAT_INTERLEAVE) * stride_keys_n +\
                    idx_tsrc[None, :, :] * stride_keys_tsrc +\
                    idx_hid[:, None, None] * stride_keys_hid,
                mask = mask_tsrc[None, :, :] & mask_hid[:, None, None],
                other = 0
            )
        elif KEY_CACHE_METHOD == 'vllm':
            """
            idx_block = block_tables[idx_batch, idx_tsrc // block_size]
            offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
            key = key_cache[idx_block, idx_head, :, offset_block, :].reshape(-1)
            """
            idx_batch = ((idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS).to(tl.int64)
            idx_head = ((idx_n // KV_REPEAT_INTERLEAVE) % VLLM_NUM_KV_HEADS).to(tl.int64)
            idx_block = tl.load(
                BLOCK_TABLES +\
                    idx_batch * stride_block_tables_num_seqs +\
                    (idx_tsrc // VLLM_BLOCK_SIZE) * stride_block_tables_max_num_blocks_per_seq,
                mask = mask_tsrc,
            ).to(tl.int64)
            offset_block = (idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)).to(tl.int64)
            
            # [BLOCK_HID: hid, BLOCK_BK: bk, BLOCK_SIZE_K_PADDED: tsrc]
            keys = tl.load(
                KEYS +\
                    idx_block[None, :, :] * stride_keys_vllm_num_bocks +\
                    idx_head * stride_keys_vllm_num_kv_heads +\
                    (idx_hid[:, None, None] // VLLM_X) * stride_keys_vllm_head_size_x +\
                    offset_block[None, :, :] * stride_keys_vllm_block_size +\
                    (idx_hid[:, None, None] % VLLM_X) * stride_keys_vllm_x,
                mask = mask_tsrc[None, :, :] & mask_hid[:, None, None],
                other = 0,
            )
        else:
            raise Exception()
        keys = tl.reshape(keys, (BLOCK_HID, BLOCK_BK * BLOCK_SIZE_K_PADDED))
        
        # TOOD: WIP
        
        if keys.dtype == tl.uint8:
            keys = keys.to(tl.float8e5, bitcast=True).to(queries.dtype)
        scores_mini = tl.dot(queries, keys)
        scores_mini = tl.reshape(scores_mini, (BLOCK_SIZE_Q_PADDED, BLOCK_BK, BLOCK_SIZE_K_PADDED))
        
        scores += scores_mini.to(scores.dtype)
    
    idx_scorek = (idx_bk[:, None] * BLOCK_SIZE_K + idx_block_k[None, :])
    mask_scorek = (idx_scorek < K) & mask_block_k[None, :] & mask_bk[:, None]
    
    scores_mask = (
        (mask_tdst[:, None, None] & mask_tsrc[None, :, :]) &
        mask_scorek[None, :] &
        True
    )
    
    if IS_CAUSAL:
        scores_mask = scores_mask & ((idx_tdst[:, None, None] + (TSRC - TDST)) >= idx_tsrc[None, :, :])
    
    tl.store(
        SCORES +\
            idx_n * stride_scores_n +\
            idx_tdst[:, None, None] * stride_scores_tdst +\
            idx_scorek[None, :, :] * stride_scores_k,
        mask = scores_mask,
        value = scores,
    )

@triton.jit
def _calc_score_compute_bwd_queries(
    # input matrices
    KS, stride_ks_n, stride_ks_bdst,
    INDICES, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    KEYS, stride_keys_n, stride_keys_tsrc, stride_keys_hid,
    
    # grad output (read)
    GRAD_SCORES, stride_grad_scores_n, stride_grad_scores_tdst, stride_grad_scores_k,
    
    # grad input (write)
    GRAD_QUERIES, stride_grad_queries_n, stride_grad_queries_tdst, stride_grad_queries_hid,
    
    # input variables
    N, TDST, TSRC, HID, BK, K,
    
    # block constant
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
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
        KS +\
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst
    )
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_tdst = (idx_bdst * BLOCK_SIZE_Q + idx_block_q)
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    
    accumulator = tl.zeros((BLOCK_SIZE_Q_PADDED, BLOCK_HID,), dtype=tl.float32)
    for idx_bk in range(BK):
        idx_tsrc = tl.load(
            INDICES + \
                idx_n * stride_indices_n + \
                idx_bdst * stride_indices_bdst + \
                idx_bk * stride_indices_bk,
        )
        
        idx_tsrc = idx_tsrc + idx_block_k
        mask_tsrc = (idx_tsrc < TSRC) & mask_block_k & (idx_tsrc < scalar_ks)
        
        idx_k = idx_bk * BLOCK_SIZE_K + idx_block_k
        mask_k = (idx_k < K) & mask_block_k
        
        # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_SIZE_K_PADDED: score]
        grad_score = tl.load(
            GRAD_SCORES +\
                idx_n * stride_grad_scores_n +\
                idx_tdst[:, None] * stride_grad_scores_tdst + \
                idx_k[None, :] * stride_grad_scores_k,
            mask = mask_tdst[:, None] & (mask_tsrc & mask_k)[None, :],
            other = 0,
        )
        
        # [BLOCK_SIZE_K_PADDED: score, BLOCK_HID: hid]
        key = tl.load(
            KEYS +\
                idx_n * stride_keys_n +\
                idx_tsrc[:, None] * stride_keys_tsrc +\
                idx_hid[None, :] * stride_keys_hid,
            mask = mask_hid[None, :] & (mask_tsrc & mask_k)[:, None],
            other = 0
        )
        
        # tl.device_print("", idx_tsrc)
        accumulator += tl.dot(grad_score, key).to(accumulator.dtype)
    
    tl.store(
        GRAD_QUERIES +\
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
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
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
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    idx_k = idx_bk * BLOCK_SIZE_K + idx_block_k
    mask_k = (idx_k < K) & mask_block_k
    
    # [BLOCK_SIZE_K_PADDED: tsrc, BLOCK_SIZE_Q_PADDED: tdst]
    grad_score = tl.load(
        grad_scores +\
            idx_n * stride_grad_scores_n +\
            idx_tdst[None, :] * stride_grad_scores_tdst +\
            idx_k[:, None] * stride_grad_scores_k,
        mask = mask_tdst[None, :] & mask_k[:, None],
        other = 0
    )
    # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_HID: hid]
    query = tl.load(
        queries +\
            idx_n * stride_queries_n +\
            idx_tdst[:, None] * stride_queries_tdst +\
            idx_hid[None, :] * stride_queries_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        other = 0,
    )
    # [BLOCK_SIZE_K_PADDED: tsrc, BLOCK_HID: hid]
    scores = tl.dot(grad_score, query)
    
    idx_tsrc = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
    )
    idx_tsrc = idx_tsrc + idx_block_k
    mask_tsrc = (idx_tsrc < TSRC) & mask_block_k
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
        queries: Tensor, keys: Union[Tensor, "PagedKeyCacheVllmCompat"], attention_mask: Tensor,
        # indices matrices
        indices: Tensor, ks: Tensor,
        # block constant
        KV_REPEAT_INTERLEAVE: int,
        BLOCK_SIZE_Q: int,
        BLOCK_SIZE_K: int,
        IS_CAUSAL: bool
    ):
        ctx.save_for_backward(queries, keys, indices, ks)
        ctx.BLOCK_SIZE_Q = BLOCK_SIZE_Q
        ctx.BLOCK_SIZE_K = BLOCK_SIZE_K
        
        N, TDST, HID = queries.shape
        _N, TSRC, _ = keys.shape
        _, _, BK = indices.shape
        
        BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
        BSRC = triton.cdiv(TSRC, BLOCK_SIZE_K)
        
        assert keys.shape == (_N, TSRC, HID)
        assert indices.shape == (N, BDST, BK)
        assert ks.shape == (N, BDST)
        
        K = BK * BLOCK_SIZE_K
        scores = torch.full(
            (N, TDST, K), 
            torch.finfo(queries.dtype).min,
            device=queries.device, 
            dtype=queries.dtype
        )
        
        BLOCK_SIZE_Q_PADDED = next_multiple_of(BLOCK_SIZE_Q, 16)
        BLOCK_SIZE_K_PADDED = next_multiple_of(BLOCK_SIZE_K, 1)
        BLOCK_BK = next_multiple_of(128 // BLOCK_SIZE_K_PADDED, 1)
        # BLOCK_BK = 1
        BLOCK_HID = triton.next_power_of_2(HID)
        # BLOCK_HID = max(BLOCK_SIZE_Q_PADDED, BLOCK_SIZE_K_PADDED)
        # BLOCK_HID = 32
        
        if isinstance(keys, Tensor):
            KEY_CACHE_METHOD = 'cont'
            
            VLLM_NUM_BLOCKS =\
            VLLM_NUM_KV_HEADS =\
            VLLM_HEAD_SIZE_X =\
            VLLM_BLOCK_SIZE =\
            VLLM_X =\
            VLLM_HEAD_SIZE = 0
            
            vllm_keys_strides = (0, 0, 0, 0, 0)
            
            block_tables = keys
            block_tables_strides = (0, 0)
        elif isinstance(keys, PagedKeyCacheVllmCompat):
            """
            vLLM compatible paged attention
            
            q: [num_seqs, num_heads, head_size]
            k: [num_blocks, num_kv_heads, head_size/x, block_size, x]
            v: [num_blocks, num_kv_heads, head_size, block_size]
            block_tables: [num_seqs, max_num_blocks_per_seq]
            context_lens: [num_seqs]
            """
            
            KEY_CACHE_METHOD = 'vllm'
            
            (
                VLLM_NUM_BLOCKS,
                VLLM_NUM_KV_HEADS, 
                VLLM_HEAD_SIZE_X,
                VLLM_BLOCK_SIZE,
                VLLM_X,
            ) = keys.key_cache.shape
            VLLM_HEAD_SIZE = VLLM_HEAD_SIZE_X * VLLM_X
            
            block_tables = keys.block_table
            block_tables_strides = block_tables.stride()
            assert len(block_tables_strides) == 2
            
            vllm_keys_strides = keys.key_cache.stride()
            assert len(vllm_keys_strides) == 5            
        else:
            raise Exception()
        
        grid = (N, BDST, triton.cdiv(BK, BLOCK_BK))
        
        # print(grid)
        
        assert queries.ndim == 3
        assert keys.ndim == 3
        if attention_mask is not None:
            assert attention_mask.ndim == 2
            assert attention_mask.dtype == torch.bool
        assert indices.ndim == 3
        assert ks.ndim == 2
        assert scores.ndim == 3
        with timer("_calc_score_compute"):
            _calc_score_compute[grid](
                # input matrix
                queries, *queries.stride(),
                keys, *keys.stride(),
                attention_mask, *(attention_mask.stride() if attention_mask is not None else (0, 0)),
                
                # block indices
                indices, *indices.stride(),
                ks, *ks.stride(),
                
                # out matrix
                scores, *scores.stride(),
                
                # input variables
                KV_REPEAT_INTERLEAVE, 
                N, 
                TDST, 
                TSRC, 
                HID, 
                BK, 
                K, 
                BDST, 
                BSRC, 
                IS_CAUSAL,
                
                # vllm key cache compat
                *vllm_keys_strides,
                VLLM_NUM_BLOCKS,
                VLLM_NUM_KV_HEADS,
                VLLM_HEAD_SIZE_X,
                VLLM_BLOCK_SIZE,
                VLLM_X,
                VLLM_HEAD_SIZE,
                
                block_tables, *block_tables_strides,
                
                # kernel constatnts
                KEY_CACHE_METHOD,
                BLOCK_BK,
                BLOCK_SIZE_Q,
                BLOCK_SIZE_Q_PADDED,
                BLOCK_SIZE_K,
                BLOCK_SIZE_K_PADDED,
                BLOCK_HID,
                
                num_warps=4,
                num_stages=2,
                enable_warp_specialization=False,
            )
            
        # print(scores[0, 300, :])
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        ENABLED = True
        
        queries, keys, indices, ks = ctx.saved_tensors
        BLOCK_SIZE_Q = ctx.BLOCK_SIZE_Q
        BLOCK_SIZE_K = ctx.BLOCK_SIZE_K
        grad_queries = grad_keys = None
        
        N, T_DST, HID = queries.shape
        _, T_SRC, _HID = keys.shape
        assert HID == _HID
        _, _, BK = indices.shape
        _, _, K = grad_scores.shape

        # for queries
        if ctx.needs_input_grad[0]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE_Q))
            BLOCK_HID = triton.next_power_of_2(HID)
            
            grad_queries = torch.zeros_like(queries)
            
            if ENABLED:
                assert ks.ndim == 2
                assert indices.ndim == 3
                assert keys.ndim == 3
                assert grad_scores.ndim == 3
                assert  grad_queries.ndim == 3
                
                _calc_score_compute_bwd_queries[grid](
                    ks, ks.stride(0), ks.stride(1),
                    indices, indices.stride(0), indices.stride(1), indices.stride(2), 
                    keys, keys.stride(0), keys.stride(1), keys.stride(2),
                    
                    grad_scores, grad_scores.stride(0), grad_scores.stride(1), grad_scores.stride(2),
                    
                    grad_queries, grad_queries.stride(0), grad_queries.stride(1), grad_queries.stride(2),
                    
                    N, T_DST, T_SRC, HID, BK, K,
                    
                    BLOCK_SIZE_Q,
                    next_multiple_of(BLOCK_SIZE_Q, 16),
                    BLOCK_SIZE_K,
                    next_multiple_of(BLOCK_SIZE_K, 16),
                    BLOCK_HID,
                )
        
        # for keys
        if ctx.needs_input_grad[1]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE_Q), BK)
            BLOCK_HID = triton.next_power_of_2(HID)
            
            grad_keys = torch.zeros_like(keys)
            
            if ENABLED:
                _calc_score_compute_bwd_keys[grid](
                    ks, ks.stride(0), ks.stride(1),
                    indices, indices.stride(0), indices.stride(1), indices.stride(2), 
                    queries, queries.stride(0), queries.stride(1), queries.stride(2),
                    
                    grad_scores, grad_scores.stride(0), grad_scores.stride(1), grad_scores.stride(2),
                    
                    grad_keys, grad_keys.stride(0), grad_keys.stride(1), grad_keys.stride(2),
                    
                    N, T_DST, T_SRC, HID, BK, K,
                    
                    BLOCK_SIZE_Q,
                    next_multiple_of(BLOCK_SIZE_Q, 16),
                    BLOCK_SIZE_K,
                    next_multiple_of(BLOCK_SIZE_K, 16),
                    BLOCK_HID,
                )
        
        return (
            grad_queries, 
            grad_keys, 
            None,
            None, 
            None, 
            None,
            None,
            None,
            None,
        )

def calc_score_return_prob(
    queries: Tensor, keys: Tensor, attention_mask: Tensor,
    indices: Tensor, ks: Tensor,
    
    KV_REPEAT_INTERLEAVE: int,
    BLOCK_SIZE_Q: int,
    BLOCK_SIZE_K: int,
    IS_CAUSAL: bool,
):
    scores = CalcScoreAutoGradFn.apply(
        queries, keys, attention_mask,
        indices, ks,
        
        KV_REPEAT_INTERLEAVE, BLOCK_SIZE_Q, BLOCK_SIZE_K, IS_CAUSAL
    ) # type: Tensor
    
    with timer("calc_score_return_prob.softmax"):
        probs = scores.softmax(-1).to(scores.dtype)
    
    assert probs.dtype == queries.dtype
    
    N, TDST, K = scores.shape
    if attention_mask is not None:
        _, TSRC = attention_mask.shape
        if probs.requires_grad:
            probs = probs * attention_mask[:, TSRC-TDST:, None]
        else:
            probs.masked_fill_(~attention_mask[:, TSRC-TDST:, None], 0)
    
    assert scores.dtype == queries.dtype
    assert probs.dtype == queries.dtype
    
    return scores, probs

def to_dense_blocked(
    indices: np.ndarray, 
    ks: np.ndarray, 
    value,
    mask, 
    N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K
):
    out = np.zeros((N, triton.cdiv(T_DST, BLOCK_SIZE_Q), triton.cdiv(T_SRC, BLOCK_SIZE_K)))
    for idx_n in range(1):
        for idx_bdst in range(out.shape[1]):
            for k in range(indices.shape[2]):
                if k < ks[idx_n, idx_bdst]:
                    idx_bsrc = indices[idx_n, idx_bdst, k]
                    if idx_bsrc < out.shape[2]:
                        # assert out[idx_n, idx_bdst, idx_bsrc] == 0, f"{out[idx_n, idx_bdst, idx_bsrc]}, {ks[idx_n, idx_bdst]}, {idx_bsrc}, {mask[idx_n, idx_bdst, :]}"
                        out[idx_n, idx_bdst, idx_bsrc] = 1
    return out

def to_dense(
    indices: np.ndarray, 
    ks: np.ndarray, 
    value: np.ndarray,
    N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K,
):
    # print(indices.shape, ks.shape, value.shape, T_DST, T_SRC)
    out = np.zeros((N, T_DST, T_SRC))
    for idx_n in range(1):
        for idx_bdst in range(indices.shape[1]):
            for idx_k in range(indices.shape[2]):
                if idx_k < ks[idx_n, idx_bdst]:
                    # print(idx_n, idx_bdst, idx_k)
                    idx_tsrc = indices[idx_n, idx_bdst, idx_k]
                    if idx_tsrc < T_SRC:
                        # print(
                        #     idx_n, 
                        #     idx_bdst * BLOCK_SIZE_Q, (idx_bdst + 1) * BLOCK_SIZE_Q, 
                        #     idx_tsrc, idx_tsrc + BLOCK_SIZE_K,
                        #     idx_n,
                        #     idx_bdst * BLOCK_SIZE_Q, (idx_bdst + 1) * BLOCK_SIZE_Q, 
                        #     idx_k * BLOCK_SIZE_K, idx_k * BLOCK_SIZE_K + min(BLOCK_SIZE_K, out.shape[-1] - idx_tsrc)
                        # )
                        out[
                            idx_n, 
                            idx_bdst * BLOCK_SIZE_Q: (idx_bdst + 1) * BLOCK_SIZE_Q, 
                            idx_tsrc: idx_tsrc + BLOCK_SIZE_K
                        ] = value[
                            idx_n,
                            idx_bdst * BLOCK_SIZE_Q: (idx_bdst + 1) * BLOCK_SIZE_Q, 
                            idx_k * BLOCK_SIZE_K: idx_k * BLOCK_SIZE_K + min(BLOCK_SIZE_K, out.shape[-1] - idx_tsrc)
                        ]
    return out

def debug_print(
    w_curr,
    mask, ws, ks, N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K
):
    plt.clf()
    indices = safe_indices(mask, ws, BLOCK_SIZE_K)
    # indices = torch.clamp(indices, 0, triton.cdiv(T_SRC, BLOCK_SIZE) - 1)
    x = to_dense_blocked(
        indices.cpu().numpy(),
        ks.cpu().unsqueeze(-1).numpy(), 
        None,
        mask,
        N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K,
    )[0]
    x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
    x = np.repeat(x, BLOCK_SIZE_Q, 0)
    x = np.repeat(x, BLOCK_SIZE_K, 1)
    if x.shape[0] == 1:
        x = x.repeat(32, 0)
    plt.imshow(x)
    path = f'saves/models/timber_attention/block_{w_curr}.png'
    # path = f'saves/models/timber_attention/block.png'
    print('saved', path, N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K, x.shape)
    plt.savefig(path, dpi=96, bbox_inches='tight')

def attention_matrix(
    queries: Tensor, 
    keys: Tensor,
    values: Tensor,
    attention_mask: Tensor,
    kv_repeat_interleave: int,
    
    w_start: int,
    n_patches: int,
    mask_k: int,
    scale_up: int,
    is_causal: bool,
    
    BLOCK_SIZE_Q: int = 16,
    BLOCK_SIZE_K: int = 1,
    REDUCE_METHOD: Literal['first', 'max', 'sum'] = 'max',
    REDUCE_STRIDE: int = 1,
    
    SPARQ: bool = True,
    SPARQ_START_TSRC: int = 2048,
    SPARQ_START_BK: int = 256,
    SPARQ_HID: int = 32,
    SPARQ_REDUCE_METHOD: Literal['sum', 'max'] = 'sum',
    
    IS_FLASH: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    global DEBUG
    
    if DEBUG:
        print('attention_matrix', queries.shape, keys.shape, w_start, n_patches, mask_k, scale_up, BLOCK_SIZE_Q, BLOCK_SIZE_K)
        os.makedirs('saves/models/timber_attention/', exist_ok=True)
    
    N, T_DST, HID = queries.shape
    _, T_SRC, _ = keys.shape
    assert T_DST <= T_SRC
    
    if SPARQ and ((mask_k // BLOCK_SIZE_K) < SPARQ_START_BK):
        SPARQ = False
    if SPARQ and (T_SRC < SPARQ_START_TSRC):
        SPARQ = False
    
    # SPARQ = False
    # SPARQ_HID = 16
    # SPARQ = True
    
    dtype = queries.dtype
    device = queries.device
    assert queries.device == keys.device
    
    assert isinstance(BLOCK_SIZE_Q, int)
    assert isinstance(BLOCK_SIZE_K, int)
    BLOCK_SIZE_Q = int(BLOCK_SIZE_Q)
    BLOCK_SIZE_K = int(BLOCK_SIZE_K)
    
    if attention_mask is not None:
        assert attention_mask.shape == (N, T_SRC)
        assert attention_mask.dtype == torch.bool
    
    # NOTE: width of last query
    w_curr = round(w_start / scale_up)
    assert w_curr <= mask_k
    
    with timer('matrix.setup'):
        mask_k_block = triton.cdiv(mask_k, BLOCK_SIZE_K)
        
        # vectors
        tsrcs_offset = max(BLOCK_SIZE_Q, BLOCK_SIZE_K) - 1
        tsrcs = torch.arange(
            tsrcs_offset+T_SRC-T_DST+1, tsrcs_offset+T_SRC+1, BLOCK_SIZE_Q, 
            dtype=torch.int64,
            device=device,
        )\
            .view(1, -1)\
            .expand(N, -1)\
            .contiguous()
        # tsrcs.clamp_max_(T_SRC)
        if not is_causal:
            tsrcs.fill_(T_SRC)
        # NOTE: store non blocked width
        ws = torch.clamp(tsrcs, 0, w_curr)
        # NOTE: store num blocks
        ks = torch.ceil(ws / BLOCK_SIZE_K).to(torch.int64)
        # assert tsrcs.dtype == torch.int64
        # assert ws.dtype == torch.int64
        # assert ks.dtype == torch.int64
        
        # matrices
        # NOTE: float16 -> int64 seems not possible
        mask = torch.arange(mask_k_block, device=device, dtype=torch.float32).view(1, 1, mask_k_block) / ks.unsqueeze(-1)
        tmask = torch.empty(
            (mask.shape[0], mask.shape[1], mask_k_block * math.ceil(scale_up)), 
            dtype=torch.float32, 
            device=device
        )
        
        B_SRC = triton.cdiv(T_SRC, BLOCK_SIZE_K)
        B_DST = triton.cdiv(T_DST, BLOCK_SIZE_Q)
        
        sparq_indices = None
        sparq_indices_strides = (1, 1, 1)
        if SPARQ:
            with timer('matrix.setup.sparq'):
                queries_scores = queries.abs()
                if T_DST > 1 and (B_DST * BLOCK_SIZE_Q) != T_DST:
                    queries_scores = F.pad(
                        queries_scores.unsqueeze(0), 
                        (0, 0, 0, B_DST * BLOCK_SIZE_Q - T_DST), 
                        value=0
                    ).squeeze(0)
                # print(queries_scores.shape, B_DST, BLOCK_SIZE_Q, T_DST, T_DST > 1 and (B_DST * BLOCK_SIZE_Q) != T_DST)
                # TODO: padding
                queries_scores = queries_scores.view(N, B_DST, -1, HID)
                if SPARQ_REDUCE_METHOD == 'sum':
                    queries_scores = queries_scores.sum(-2)
                elif SPARQ_REDUCE_METHOD == 'max':
                    queries_scores = queries_scores.max(-2)[0]
                else:
                    raise Exception()
                _, sparq_indices = torch.topk(
                    queries_scores, 
                    k=SPARQ_HID, 
                    dim=-1, 
                    sorted=False
                )
                sparq_indices = sparq_indices.to(torch.int16)
                # sparq_indices = torch.arange(0, SPARQ_HID, device=queries.device)[None, None, :].repeat(N, B_DST, 1)
                sparq_indices_strides = sparq_indices.stride()
    
    if DEBUG:
        debug_print(w_curr, mask, ws, ks, N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K)
        
    # NOTE: Calc. Mask
    n_iteration = 0
    _w_curr = w_curr
    while w_curr < T_SRC:
        w_curr = round(w_curr * scale_up)
        n_iteration += 1
    # w_curr = _w_curr
    
    n_completed = _w_curr
    with timer("iterations"):
        i_iteration = 0
        masking_iteration(
            # input matrices
            queries, keys, attention_mask,
            # input metrices (blocked) 
            mask, tmask, sparq_indices, sparq_indices_strides,
            # temp vectors (blocked)
            ws, ks, tsrcs, 
            # operator variables
            scale_up, triton.cdiv(n_patches, BLOCK_SIZE_K), triton.cdiv(mask_k, BLOCK_SIZE_K), is_causal,
            # iteration controls
            i_iteration, n_iteration,
            # input constant
            kv_repeat_interleave,
            N, 
            T_DST, 
            T_SRC, 
            B_DST, 
            B_SRC, 
            HID, 
            SPARQ, 
            SPARQ_HID, 
            max(0, triton.cdiv(n_completed, BLOCK_SIZE_Q) - (triton.cdiv(T_SRC, BLOCK_SIZE_Q) - triton.cdiv(T_DST, BLOCK_SIZE_Q))),
            # kernel constant
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
            REDUCE_METHOD,
            REDUCE_STRIDE,
        )
        if DEBUG:
            debug_print(w_curr, mask, ws, ks, N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K)
    
    with timer('matrix.cleanup'):
        indices = safe_indices(mask, ws, BLOCK_SIZE_K)
    
    # # NOTE: are you sure this function is the only thing can differentiate?
    with timer("score" if not IS_FLASH else "flash_atten"):
        if not IS_FLASH:
            scores, probs = calc_score_return_prob(
                queries=queries, keys=keys, attention_mask=attention_mask,
                indices=indices, ks=ks,
                KV_REPEAT_INTERLEAVE=kv_repeat_interleave,
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                IS_CAUSAL=is_causal,
            )
            assert probs.dtype == queries.dtype, f"{probs.dtype} == {queries.dtype}"
        else:
            context = calc_prob_return_context(
                queries=queries, keys=keys, values=values, attention_mask=attention_mask,
                indices=indices, ks=ks,
                KV_REPEAT_INTERLEAVE=kv_repeat_interleave,
                BLOCK_SIZE_Q=BLOCK_SIZE_Q, 
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                IS_CAUSAL=is_causal,
            )

            return indices, ks, context, None
    
    if DEBUG:
        x = to_dense(
            indices.cpu().numpy(),
            ks.cpu().numpy(),
            probs.detach().cpu().to(torch.float32).numpy(),
            N, 
            T_DST, 
            T_SRC, 
            BLOCK_SIZE_Q, 
            BLOCK_SIZE_K,
        )[0]
        x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
        if x.shape[0] == 1:
            x = x.repeat(32, 0)
        plt.imshow(x)
        path = 'saves/models/timber_attention/block_est.png'
        print('saved', path)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        
        # x = np.matmul(
        #     queries[0].cpu().numpy(), 
        #     keys[0].cpu().numpy().transpose((-1, -2))
        # )
        if isinstance(keys, Tensor):
            x = (queries[0] @ keys[0].transpose(-1, -2)).detach().to(torch.float32).cpu().numpy()
            if is_causal:
                x = x + (1 - np.tri(*x.shape, T_SRC-T_DST)) * (-10000)
            x = np.exp(x - x.max(-1, keepdims=True))
            x = x / x.sum(-1, keepdims=True)
            x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
            plt.imshow(x)
            path = 'saves/models/timber_attention/block_truth.png'
            print('saved', path)
            plt.savefig(path, dpi=200, bbox_inches='tight')
            # print(ks)
            # input('>>>')
    
    return indices, ks, probs, scores


@triton.jit
def _sdbmm_compute(
    # inputs
    INDICES, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    KS, stride_ks_n, stride_ks_bdst, 
    PROBS, stride_probs_n, stride_probs_tdst, stride_probs_k,
    VALUES, stride_values_n, stride_values_tsrc, stride_values_hid,
    
    # output
    CONTEXT, stride_context_n, stride_context_tdst, stride_context_hid,
    
    # variables
    KV_REPEAT_INTERLEAVE, N, TSRC, TDST, HID, K, BK, BSRC, BDST,
    
    # vllm value cache compat,
    stride_values_vllm_num_blocks,
    stride_values_vllm_num_kv_heads,
    stride_values_vllm_head_size,
    stride_values_vllm_block_size,
    
    VLLM_NUM_BLOCKS,
    VLLM_NUM_KV_HEADS,
    VLLM_HEAD_SIZE,
    VLLM_BLOCK_SIZE,
    
    BLOCK_TABLES,
    stride_block_tables_num_seqs,
    stride_block_tables_max_num_blocks_per_seq,
    
    # kernel blocks
    VALUE_CACHE_METHOD: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    idx_n = tl.program_id(0)
    # if idx_n >= N: return
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_bdst = tl.program_id(1)
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    pid_hid = tl.program_id(2)
    idx_hid = tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID
    mask_hid = idx_hid < HID
    
    n_bk = tl.load(
        KS +\
            idx_n * stride_ks_n+\
            idx_bdst * stride_ks_bdst,
    )
    
    scores = tl.zeros((BLOCK_SIZE_Q_PADDED, BLOCK_HID), dtype=tl.float32)
    for idx_bk in range(BK):
        mask_bk = idx_bk < n_bk
        _idx_tsrc = tl.load(
            INDICES +\
                idx_n * stride_indices_n +\
                idx_bdst * stride_indices_bdst +\
                idx_bk * stride_indices_bk,
            mask = mask_bk,
            # other = TSRC,
        ).to(tl.int64)
        # atten_indices: [BLOCK_SIZE_PADDED]
        idx_tsrc = _idx_tsrc + idx_block_k
        mask_tsrc = (idx_tsrc < TSRC) & mask_block_k & mask_bk
        
        # atten_probs: [BLOCK_SIZE_PADDED: tdst, BLOCK_SIZE_PADDED: tsrc]
        idx_prob_k = (idx_bk * BLOCK_SIZE_K + idx_block_k)
        mask_prob_k = (idx_prob_k < K) & mask_block_k & mask_bk
        atten_probs = tl.load(
            PROBS +\
                idx_n * stride_probs_n +\
                idx_tdst[:, None] * stride_probs_tdst +\
                idx_prob_k[None, :] * stride_probs_k,
            mask = \
                mask_tdst[:, None] &\
                mask_prob_k[None, :] &\
                ((idx_tdst[:, None] + TSRC - TDST) >= idx_tsrc[None, :]) & \
                mask_bk,
            other = 0,
        )
        # DEBUG: tl.device_assert(tl.max(idx_tsrc * mask_tsrc) < TSRC, "TSRC")
        
        if VALUE_CACHE_METHOD == 'cont':
            # value: [BLOCK_SIZE_PADDED: tsrc, BLOCK_HID: hid]
            value = tl.load(
                VALUES +\
                    (idx_n // KV_REPEAT_INTERLEAVE).to(tl.int64) * stride_values_n +\
                    idx_tsrc[:, None].to(tl.int64) * stride_values_tsrc +\
                    idx_hid[None, :].to(tl.int64) * stride_values_hid,
                mask = mask_tsrc[:, None] & mask_hid[None, :] & mask_bk,
                other = 0,
            )
        elif VALUE_CACHE_METHOD == 'vllm':
            """
            idx_block = block_tables[idx_batch, idx_tsrc // block_size]
            offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
            value = value_cache[idx_block, idx_head, :, offset_block].reshape(-1)
            """
            idx_batch = (idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS
            idx_head = (idx_n // KV_REPEAT_INTERLEAVE) % VLLM_NUM_KV_HEADS
            
            idx_block = tl.load(
                BLOCK_TABLES +\
                    idx_batch * stride_block_tables_num_seqs +\
                    (idx_tsrc // VLLM_BLOCK_SIZE) * stride_block_tables_max_num_blocks_per_seq,
                mask = mask_tsrc & mask_bk,
                other = 0
            ).to(tl.int64)
            mask_block = (idx_tsrc // VLLM_BLOCK_SIZE) < tl.cdiv(TSRC, VLLM_BLOCK_SIZE)
            offset_block = idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)
            
            # value: [BLOCK_SIZE_PADDED: tsrc, BLOCK_HID: hid]
            value = tl.load(
                VALUES +\
                    idx_block[:, None] * stride_values_vllm_num_blocks+\
                    idx_head * stride_values_vllm_num_kv_heads+\
                    idx_hid[None, :].to(tl.int64) * stride_values_vllm_head_size +\
                    offset_block[:, None] * stride_values_vllm_block_size,
                mask = mask_tsrc[:, None] & mask_hid[None, :] & mask_bk & mask_block[:, None],
                other = 0
            )
        else:
            raise Exception()
        
        # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
        if value.dtype == tl.uint8:
            value = value.to(tl.float8e5, bitcast=True).to(atten_probs.dtype)
        scores_mini = tl.dot(atten_probs, value)
        scores += scores_mini.to(scores.dtype)
        
        # scores += tl.sum(value)
        
    tl.store(
        CONTEXT +\
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
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
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
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    idx_k = idx_bk * BLOCK_SIZE_K + idx_block_k
    mask_k = (idx_k < K) & mask_block_k
    
    idx_tsrc = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk
    )
    idx_tsrc = idx_tsrc + idx_block_k
    mask_tsrc = (idx_tsrc < TSRC) & mask_block_k
    
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
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
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
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_tsrc = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
    )
    idx_tsrc = idx_tsrc + idx_block_k
    mask_tsrc = (idx_tsrc < TSRC) & mask_block_k
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
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
    
    idx_k = idx_bk * BLOCK_SIZE_K + idx_block_k
    mask_k = (idx_k < K) & mask_block_k
    
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
        values: Union[Tensor, "PagedValueCacheVllmCompat"],
        
        # attention matrix
        indices: Tensor,
        ks: Tensor,
        probs: Tensor,
        
        KV_REPEAT_INTERLEAVE: int,
        BLOCK_SIZE_Q: int,
        BLOCK_SIZE_K: int,
    ):
        global DEBUG
        
        ctx.save_for_backward(values, indices, ks, probs)
        ctx.BLOCK_SIZE_Q = BLOCK_SIZE_Q
        ctx.BLOCK_SIZE_K = BLOCK_SIZE_K
    
        N, BDST, BK = indices.shape
        _N, TDST, K = probs.shape
        __N, TSRC, HID = values.shape
        assert N == _N
        assert N == (__N * KV_REPEAT_INTERLEAVE)
        # assert N == __N
        assert ks.shape == (N, BDST)
        
        BSRC = triton.cdiv(TSRC, BLOCK_SIZE_K)
        
        context_dtype = values.dtype
        if context_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            context_dtype = probs.dtype
        assert context_dtype in [torch.float16, torch.bfloat16, torch.float32]
        context = torch.zeros((N, TDST, HID), dtype=context_dtype, device=values.device)
        
        BLOCK_SIZE_Q_PADDED = next_multiple_of(BLOCK_SIZE_Q, 16)
        BLOCK_SIZE_K_PADDED = next_multiple_of(BLOCK_SIZE_K, 16)
        BLOCK_HID = triton.next_power_of_2(HID)
        
        if isinstance(values, Tensor):
            VALUE_CACHE_METHOD = 'cont'
            
            block_tables = values
            block_tables_strides = (0, 0)
            
            VLLM_NUM_BLOCKS =\
            VLLM_NUM_KV_HEADS =\
            VLLM_HEAD_SIZE =\
            VLLM_BLOCK_SIZE = 0
            
            vllm_values_strides = (0, 0, 0, 0)
        elif isinstance(values, PagedValueCacheVllmCompat):
            """
            vLLM compatible paged attention
            
            q: [num_seqs, num_heads, head_size]
            k: [num_blocks, num_kv_heads, head_size/x, block_size, x]
            v: [num_blocks, num_kv_heads, head_size, block_size]
            block_tables: [num_seqs, max_num_blocks_per_seq]
            context_lens: [num_seqs]
            """
            
            VALUE_CACHE_METHOD = 'vllm'
            
            block_tables = values.block_table
            block_tables_strides = block_tables.stride()
            assert len(block_tables_strides) == 2
            
            (
                VLLM_NUM_BLOCKS,
                VLLM_NUM_KV_HEADS,
                VLLM_HEAD_SIZE,
                VLLM_BLOCK_SIZE
            ) = values.value_cache.shape
            vllm_values_strides = values.value_cache.stride()
            assert len(vllm_values_strides) == 4 
        else:
            raise Exception()
        
        grid = (N, BDST, triton.cdiv(HID, BLOCK_HID))
        # grid = (1, 1, 1)
        
        # NOTE: I have no idea what this sprase matrix format LOL, but for temporary
        if DEBUG:
            # print('sdbmm', grid, BLOCK_K, BLOCK_HID)
            # assert indices.max() < TSRC
            assert indices.min() >= 0
            assert indices.is_contiguous()
            assert ks.is_contiguous()
            assert probs.is_contiguous()
            # assert values.is_contiguous()
            assert context.is_contiguous()
            torch.cuda.synchronize()
        
        # print(values.shape[0] * values.stride(0))
        
        assert indices.shape[0] == N
        assert ks.shape[0] == N
        assert probs.shape[0] == N, f'{probs.shape} == {N}'
        # assert values.shape[0] == N
        assert context.shape[0] == N
        assert ks.ndim == 2
        assert probs.ndim == 3
        assert values.ndim == 3
        assert context.ndim == 3
        # assert values.dtype == probs.dtype, f"{values.dtype} == {probs.dtype}"
        # assert values.dtype == context.dtype
        _sdbmm_compute[grid](
            # inputs
            indices, *indices.stride(),
            ks, *ks.stride(),
            probs, *probs.stride(),
            values, *values.stride(),
            
            # output
            context, *context.stride(),
            
            # input variables
            KV_REPEAT_INTERLEAVE, N, TSRC, TDST, HID, K, BK, BSRC, BDST,
            
            # vllm value cache compat
            *vllm_values_strides,
            VLLM_NUM_BLOCKS,
            VLLM_NUM_KV_HEADS,
            VLLM_HEAD_SIZE,
            VLLM_BLOCK_SIZE,
            
            block_tables,
            *block_tables_strides,
            
            # blocks
            VALUE_CACHE_METHOD,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_Q_PADDED,
            BLOCK_SIZE_K,
            BLOCK_SIZE_K_PADDED,
            BLOCK_HID,
            
            num_warps=BLOCK_HID//32,
        )
        
        return context
    
    @staticmethod
    def backward(ctx, grad_context):
        ENABLED_VALUES = True
        ENABLED_PROBS = True
        
        values, indices, ks, probs = ctx.saved_tensors
        BLOCK_SIZE_Q = ctx.BLOCK_SIZE_Q
        BLOCK_SIZE_K = ctx.BLOCK_SIZE_K
        grad_values = grad_probs = None
        
        N, T_SRC, HID = values.shape
        _, B_DST, BK = indices.shape
        _, T_DST, K = probs.shape
        assert ks.shape == (N, B_DST)
        assert probs.shape == (N, T_DST, K)
        assert indices.shape[0] == N

        # for values
        if ctx.needs_input_grad[0]:
            grid = (N, B_DST, BK)
            BLOCK_HID = triton.next_power_of_2(HID)

            grad_values = torch.zeros(
                (N, T_SRC, HID), 
                device=values.device, 
                dtype=values.dtype,
            )
            
            if ENABLED_VALUES:
                _sdbmm_compute_bwd_values[grid](
                    probs, probs.stride(0), probs.stride(1), probs.stride(2),
                    indices, indices.stride(0), indices.stride(1), indices.stride(2),
                    
                    grad_context, grad_context.stride(0), grad_context.stride(1), grad_context.stride(2),
                    
                    grad_values, grad_values.stride(0), grad_values.stride(1), grad_values.stride(2),
                    
                    N, T_DST, T_SRC, HID, BK, K,
                    
                    BLOCK_SIZE_Q,
                    next_multiple_of(BLOCK_SIZE_Q, 16),
                    BLOCK_SIZE_K,
                    next_multiple_of(BLOCK_SIZE_K, 16),
                    BLOCK_HID,
                )
            
            # print(grad_values.abs().sum())
        
        # for probs
        if ctx.needs_input_grad[3]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE_Q), BK)
            BLOCK_HID = triton.next_power_of_2(HID)
            
            grad_probs = torch.zeros(
                (N, T_DST, K),
                device=probs.device,
                dtype=probs.dtype,
            )
            
            if ENABLED_PROBS:
                _sdbmm_compute_bwd_probs[grid](
                    indices, indices.stride(0), indices.stride(1), indices.stride(2),
                    values, values.stride(0), values.stride(1), values.stride(2), 
                    
                    grad_context, grad_context.stride(0), grad_context.stride(1), grad_context.stride(2),
                    
                    grad_probs, grad_probs.stride(0), grad_probs.stride(1), grad_probs.stride(2),
                    
                    N, T_DST, T_SRC, HID, BK, K,
                    
                    BLOCK_SIZE_Q,
                    next_multiple_of(BLOCK_SIZE_Q, 16),
                    BLOCK_SIZE_K,
                    next_multiple_of(BLOCK_SIZE_K, 16),
                    BLOCK_HID,
                )

        return (
            grad_values, 
            None, 
            None, 
            grad_probs, 
            None,
            None,
            None,
        )

def sparse_attention(
    # attention values
    values: Tensor,
    
    # attention matrix
    indices: Tensor,
    ks: Tensor,
    probs: Tensor,
    
    KV_REPEAT_INTERLEAVE: int,
    BLOCK_SIZE_Q: int,
    BLOCK_SIZE_K: int,
):
    context = SparseAttentionAutoGradFn.apply(
        values, indices, ks, probs, 
        KV_REPEAT_INTERLEAVE, BLOCK_SIZE_Q, BLOCK_SIZE_K,
    )
    
    return context

import numba
@numba.njit
def to_dense(
    indices: np.ndarray, 
    ks: np.ndarray, 
    value: np.ndarray,
    N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K
):
    # print(indices.shape, ks.shape, value.shape, T_DST, T_SRC)
    out = np.zeros((N, T_DST, T_SRC), dtype=value.dtype)
    for idx_n in numba.prange(N):
        for idx_bdst in range(indices.shape[1]):
            for idx_k in range(indices.shape[2]):
                if idx_k < ks[idx_n, idx_bdst]:
                    idx_tsrc = indices[idx_n, idx_bdst, idx_k]
                    out[
                        idx_n, 
                        idx_bdst * BLOCK_SIZE_Q: (idx_bdst + 1) * BLOCK_SIZE_Q, 
                        idx_tsrc: idx_tsrc + BLOCK_SIZE_K
                    ] = value[
                        idx_n,
                        idx_bdst * BLOCK_SIZE_Q: (idx_bdst + 1) * BLOCK_SIZE_Q, 
                        idx_k * BLOCK_SIZE_K: (idx_k + 1) * BLOCK_SIZE_K
                    ]
    return out

class VllmCompat:
    pass

class PagedKeyCacheVllmCompat(VllmCompat):
    # interface
    dtype: torch.dtype
    device: torch.device
    shape: Tuple[int, int, int]
    ndim: int
    
    # vllm compat
    key_cache: Tensor
    block_table: Tensor
    context_length: Tensor
    max_context_length: int
    block_size: int
    
    def __init__(
        self,
        key_cache: Tensor,
        block_table: Tensor,
        context_length: Tensor,
        max_context_length: int,
    ):
        self.key_cache = key_cache
        self.block_table = block_table
        self.context_length = context_length
        self.max_context_length = max_context_length
        
        self.dtype = key_cache.dtype
        self.device = key_cache.device
        
        BATCH_SIZE, MAX_NUM_BLOCKS = block_table.shape
        assert context_length.shape == (BATCH_SIZE,)
        assert isinstance(max_context_length, int)
        
        NUM_BLOCKS, NUM_HEADS, HEAD_SIZE_DIV_X, BLOCK_SIZE, X = key_cache.shape
        HEAD_SIZE = HEAD_SIZE_DIV_X * X
        
        assert NUM_BLOCKS >= MAX_NUM_BLOCKS
        assert (BLOCK_SIZE * NUM_BLOCKS) >= max_context_length
        
        self.shape = (BATCH_SIZE * NUM_HEADS, max_context_length, HEAD_SIZE)
        self.block_size = BLOCK_SIZE
        self.ndim = 3
    
    def stride(self):
        return tuple([1, ] * len(self.shape))

    def data_ptr(self):
        return self.key_cache.data_ptr()

class PagedValueCacheVllmCompat(VllmCompat):
    # interface
    dtype: torch.dtype
    device: torch.device
    shape: Tuple[int, int, int]
    ndim: int
    
    # vllm compat
    value_cache: Tensor
    block_table: Tensor
    context_length: Tensor
    max_context_length: int
    block_size: int
    
    def __init__(
        self,
        key_cache: "PagedKeyCacheVllmCompat",
        value_cache: Tensor,
    ):
        self.block_size = key_cache.block_size
        block_table = key_cache.block_table
        context_length = key_cache.context_length
        max_context_length = key_cache.max_context_length
        
        self.value_cache = value_cache
        self.block_table = block_table
        self.context_length = context_length
        self.max_context_length = max_context_length
        
        self.dtype = value_cache.dtype
        self.device = value_cache.device
        
        BATCH_SIZE, MAX_NUM_BLOCKS = block_table.shape
        assert context_length.shape == (BATCH_SIZE,)
        assert isinstance(max_context_length, int)
        
        NUM_BLOCKS, NUM_HEADS, HEAD_SIZE, BLOCK_SIZE = value_cache.shape
        
        assert NUM_BLOCKS >= MAX_NUM_BLOCKS
        assert BLOCK_SIZE == self.block_size
        
        self.shape = (BATCH_SIZE * NUM_HEADS, max_context_length, HEAD_SIZE)
        self.ndim = 3
    
    def stride(self):
        return tuple([1, ] * len(self.shape))

    def data_ptr(self):
        return self.value_cache.data_ptr()

def paged_timber_attention(
    q: Tensor, 
    q_scale: float,
    k: Tensor, 
    v: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    max_context_len: int,
    # optional mask
    attention_mask: Tensor = None,
    
    # heuristics: w_start == mask_k * scale_up
    w_start: int = None,
    # heuristics: n_patches == mask_k // scale_up
    n_patches: int = None,
    mask_k: int = 512,
    scale_up: float = 2,
    
    block_size_q: int = 8,
    block_size_k: int = 1,
    reduce_method: str = 'max',
    reduce_stride: int = 1,
):
    """
    vLLM compatible paged attention
    
    q: [num_seqs, num_heads, head_size]
    k: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    v: [num_blocks, num_kv_heads, head_size, block_size]
    block_tables: [num_seqs, max_num_blocks_per_seq]
    context_lens: [num_seqs]
    """
    
    with timer('scaling'):
        q = q * q_scale
        q = q.view(q.shape[0] * q.shape[1], 1, q.shape[2])
    
    with timer('compat'):
        paged_k = PagedKeyCacheVllmCompat(
            key_cache=k,
            block_table=block_tables,
            context_length=context_lens,
            max_context_length=max_context_len,
        )
        
        paged_v = PagedValueCacheVllmCompat(
            key_cache=paged_k,
            value_cache=v,
        )
    
    # print('paged qkv cache shape', q.shape, paged_k.shape, paged_v.shape)
    
    return timber_attention(
        q=q,
        k=paged_k,
        v=paged_v,
        attention_mask=attention_mask,
        w_start=w_start,
        n_patches=n_patches,
        mask_k=mask_k,
        scale_up=scale_up,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        reduce_method=reduce_method,
        reduce_stride=reduce_stride,
    )

def timber_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor,
    # optional mask
    attention_mask: Tensor = None,
    
    # heuristics: w_start == mask_k * scale_up
    w_start: int = None,
    # heuristics: n_patches == mask_k // scale_up
    n_patches: int = None,
    mask_k: int = 512,
    scale_up: float = 2,
    is_causal: bool = True,
    
    block_size_q: int = 8,
    block_size_k: int = 1,
    reduce_method: str = 'max',
    reduce_stride: int = 1,
    
    chunking: bool = False,
    chunk_size: int = 2048,
    
    is_flash: bool = True,
    enable_sparq: bool = True,
):
    CHUNKING = chunking
    CHUNK_SIZE = chunk_size
    if q.shape[1] > CHUNK_SIZE and CHUNKING:
        N, T_DST, HID = q.shape
        N, T_SRC, HID = k.shape
        
        contexts = []
        
        for ichunk in range(triton.cdiv(T_DST, CHUNK_SIZE)):
            q_chunk = q[:, ichunk*CHUNK_SIZE:(ichunk+1)*CHUNK_SIZE, :]
            cache_chunk_end = T_SRC-T_DST+(ichunk+1)*CHUNK_SIZE
            k_chunk = k[:, :cache_chunk_end, :]
            v_chunk = v[:, :cache_chunk_end, :]
            if attention_mask is not None:
                attention_mask_chunk = attention_mask[:, :cache_chunk_end]
            else:
                attention_mask_chunk = None
            
            context, _ = timber_attention(
                q_chunk, k_chunk, v_chunk, 
                attention_mask=attention_mask_chunk,
                w_start=w_start,
                n_patches=n_patches,
                mask_k=mask_k,
                scale_up=scale_up,
                is_causal=is_causal,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                reduce_method=reduce_method,
                reduce_stride=reduce_stride,
                is_flash=is_flash,
                enable_sparq=enable_sparq,
            )
            contexts.append(context)
            
        contexts = torch.cat(contexts, dim=1)    
        
        return contexts, None    
    
    global DEBUG
    DENSE_SPARSE_ATTENTION = False
    
    if w_start is None:
        w_start = math.ceil(mask_k * scale_up)
        # w_start = math.ceil(mask_k * scale_up * scale_up)
        # w_start = math.ceil(mask_k / scale_up)
        # w_start = mask_k
    if n_patches is None:
        n_patches = math.ceil(mask_k / scale_up)
        # n_patches = mask_k / scale_up
    
    assert q.ndim == 3
    assert k.ndim == 3
    assert v.ndim == 3
    N, T_DST, HID = q.shape
    _N, T_SRC, _HID = k.shape
    assert k.shape[:-1] == v.shape[:-1]
    assert (N % _N) == 0
    assert HID == _HID
    KV_REPEAT_INTERLEAVE = N // _N
    
    # assert q.dtype == k.dtype, f'{q.dtype} == {k.dtype}'
    # assert q.dtype == v.dtype
    
    # if attention_mask is None:
    #     attention_mask = torch.full((N, T_SRC), True, dtype=torch.bool, device=q.device)
    # if attention_mask.dtype != torch.bool:
    #     # mask should mark alive token as True
    #     attention_mask = attention_mask > 0.5
    # assert attention_mask.dtype == torch.bool
    
    assert isinstance(block_size_q, int)
    assert isinstance(block_size_k, int)
    
    block_size_q = min(block_size_q, triton.next_power_of_2(T_DST))
    block_size_k = min(block_size_k, triton.next_power_of_2(T_SRC))
    
    if DEBUG:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    with timer('timber_attention'):
        with timer('attention_matrix'):
            indices, ks, probs_or_context, scores = attention_matrix(
                queries=q,
                keys=k,
                values=v,
                attention_mask=attention_mask,
                kv_repeat_interleave=KV_REPEAT_INTERLEAVE,
                
                w_start=w_start,
                n_patches=n_patches,
                mask_k=mask_k,
                scale_up=scale_up,
                is_causal=is_causal,
                
                BLOCK_SIZE_Q=block_size_q,
                BLOCK_SIZE_K=block_size_k,
                REDUCE_METHOD=reduce_method,
                REDUCE_STRIDE=reduce_stride,
                
                IS_FLASH=is_flash,
                SPARQ=enable_sparq,
            )
            
            if is_flash:
                return probs_or_context, (indices, ks, None)
            else:
                probs = probs_or_context
            
            # assert probs.dtype == v.dtype, f"{probs.dtype} == {v.dtype}"
        
        with timer('sparse_attention'):
            if DENSE_SPARSE_ATTENTION:
                probs_dense = torch.tensor(to_dense(
                    indices.to(torch.float32).cpu().numpy(), 
                    ks.to(torch.float32).cpu().numpy(), 
                    probs.to(torch.float32).cpu().numpy(), 
                    N, T_DST, T_SRC, block_size_q, block_size_k,
                )).to(v.dtype).to(indices.device)
                
                # scores_dense = to_dense(
                #     indices.cpu(), ks.cpu(), scores.cpu(),
                #     N, T_DST, T_SRC, block_size_q, block_size_k,
                # ).to(indices.device)
                
                mask_dense = probs_dense <= 1e-7
                mask_dense = mask_dense.to(probs.dtype) * torch.finfo(probs.dtype).min
                
                scores_truth = torch.bmm(q, k.transpose(-1, -2))
                probs_truth = (scores_truth + mask_dense).softmax(dim=-1)
                
                context = torch.bmm(probs_truth, v)
            else:
                context = sparse_attention(
                    v,
                    indices,
                    ks,
                    probs,
                    KV_REPEAT_INTERLEAVE=KV_REPEAT_INTERLEAVE,
                    BLOCK_SIZE_Q=block_size_q,
                    BLOCK_SIZE_K=block_size_k,
                )
    
    return context, (indices, ks, probs)

import torch.nn.functional as F

def torch_attention(q: Tensor, k: Tensor, v: Tensor):
    scores = torch.bmm(q, k.transpose(-1, -2))
    probs = torch.softmax(scores, dim=-1)
    context = torch.bmm(probs, v)
    return context, probs

def flash_attention(q: Tensor, k: Tensor, v: Tensor, is_causal=True):
    # context = F.scaled_dot_product_attention(
    #     q, k, v, is_causal=False, scale=None,
    # )
    # return context, None
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_with_kvcache

    return flash_attn_with_kvcache(q, k, v, causal=is_causal), None

def landmark_attention(q: Tensor, k: Tensor, v: Tensor):
    """
    https://arxiv.org/pdf/2305.16300.pdf
    this paper claimed, they are faster than original attetnion... but seems not?
    """
    from timber.models.landmark_attention import fused_landmark_attention
    
    seqlen_k = k.shape[1]
    block_size = 64
    is_mem = torch.arange(0, seqlen_k, device=q.device) % block_size == (block_size - 1)
    return fused_landmark_attention(q, k, v, is_mem, block_size=block_size)

def main_latency_benchmark():
    global DEBUG
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dups', type=int, default=2)
    parser.add_argument('--query_size', type=int, default=1)
    parser.add_argument('--method', type=str, default='timber')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--block_size_q', type=int, default=16)
    parser.add_argument('--block_size_k', type=int, default=1)
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--scale_up', type=int, default=2)
    parser.add_argument('--not_causal', action='store_true')
    args = parser.parse_args()
    
    DEBUG = args.debug
    TRACE = args.trace
    BSIZE = args.batch_size
    DUPS = args.dups
    QUERY_SIZE = args.query_size
    METHOD = args.method
    n_samples = args.samples
    is_causal = not args.not_causal
    
    if DEBUG:
        seed()
    
    get_bench().disabled = not TRACE
    get_bench().synchronize = True

    CHUNK_LEN = 1024
    q, k, v, out = load_checkouts(idx=0, window=40, seq_len=CHUNK_LEN)
    HID = q.shape[-1]
    
    q = q.cpu()
    k = k.cpu()
    v = v.cpu()
    
    q = q.repeat(BSIZE, max(1, triton.cdiv(QUERY_SIZE, 1024)), 1)[:, :QUERY_SIZE, :].contiguous()
    k = k.repeat(BSIZE, DUPS, 1)
    v = v.repeat(BSIZE, DUPS, 1)
    started = False
    
    if METHOD in 'flash':
        q = q.view(BSIZE, -1, QUERY_SIZE, HID).permute(0, 2, 1, 3).contiguous()
        k = k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).permute(0, 2, 1, 3).contiguous()
        v = v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).permute(0, 2, 1, 3).contiguous()
    elif METHOD in 'landmark':
        q = q.view(BSIZE, -1, QUERY_SIZE, HID).contiguous()
        k = k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).contiguous()
        v = v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).contiguous()
    
    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    
    timber_attention_mask = torch.full((q.shape[0], k.shape[1]), True, dtype=torch.bool, device=q.device)
    
    def sample():
        with torch.no_grad():
            if METHOD in ['torch', 'none', 'default']:
                torch_attention(q, k, v)
            elif METHOD == 'flash':
                flash_attention(q, k, v, is_causal=is_causal)
            elif METHOD == 'landmark':
                landmark_attention(q, k, v)
            elif METHOD == 'timber':
                timber_attention(
                    q,
                    k,
                    v,
                    # attention_mask=timber_attention_mask,
                    mask_k=args.k,
                    block_size_q=args.block_size_q,
                    block_size_k=args.block_size_k,
                    scale_up=args.scale_up,
                    is_causal=is_causal,
                )
            else:
                raise Exception()
    
    s = torch.cuda.Stream()
    graph = None
    samples = []
    for i in tqdm.tqdm(range(n_samples)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        if i < 3:
            s.wait_stream(torch.cuda.current_stream())
            sample()
            torch.cuda.current_stream().wait_stream(s)
        elif graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                sample()
        else:
            graph.replay()
        
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
    
    os.makedirs('./cache/attention1_block_gpu/', exist_ok=True)
    with open('./cache/attention1_block_gpu/result.json', 'w') as f:
        json.dump({
            'method': METHOD,
            'mean': np.mean(samples),
            'std': np.std(samples),
            'query_length': q.shape[-2],
            'keyvalue_length': k.shape[-2],
        }, f, indent=2)

def main_debug():
    global DEBUG
    DEBUG = True
    
    block = 1024
    # block = 8
    q, k, v, out = load_checkouts(
        dtype=torch.float32, 
        seq_len=block * 4, 
        idx=26, 
        window=1
    )
    
    q = q[:, block * 2:, :]
    out = out[:, block * 2:, :]
    
    print('q', q.shape)
    print('k', k.shape)
    print('v', v.shape)
    print('out', out.shape)
    
    context, (
        atten_indices, 
        atten_ks, 
        atten_probs
    ) = timber_attention(
        q,
        k,
        v,
        mask_k=512,
        block_size_q=16,
        block_size_k=2,
    )
    
    stderr = (out - context).abs().mean().item()
    stdcontext = torch.std_mean(out)[0].item()
    
    print(f'err = {stderr:.6f} ({stderr/stdcontext:.4f} sigma), out_std = {stdcontext:.6f}')

def main_debug_mask():
    global DEBUG
    DEBUG = True
    
    seed()
    q, k, v, out = load_checkouts(dtype=torch.float16, seq_len=1024 * 2, idx=24, window=1)
    
    q = q[:, 512:, :]
    out = out[:, 512:, :]
    
    N, TSRC, HID = k.shape
    mask = torch.full((N, TSRC), 1, dtype=torch.float32, device=k.device)
    for i in range(N):
        mask[i, :1024] = 0
    
    context, (atten_indices, atten_ks, atten_probs) = timber_attention(
        q,
        k,
        v,
        attention_mask=mask,
    )
    
    stderr = (out - context).abs().mean().item()
    stdcontext = torch.std_mean(out)[0].item()
    
    print(f'err = {stderr:.6f} ({stderr/stdcontext:.4f} sigma), out_std = {stdcontext:.6f}')

if __name__ == '__main__':
    import sys
    if sys.argv[-1] == 'debug':
        main_debug()
    elif sys.argv[-1] == 'debug_mask':
        main_debug_mask()
    else:
        main_latency_benchmark()