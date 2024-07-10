"""
HiP v1.1
TODO:
1. Masking iteration using integer to avoid aliasing and collision
 - Convert tmask into int32 (good!)
 - Reuse the computed dot products
2. Using QUEST method for b_k (not very good)
3. Maximum token location predictor
 - Test oracle (not very good)
 - Test estimators
4. sifters? (not very good)
5. masking -> allocate cells (not very good)
6. StreamLLM based traverse (use Self-Extend instead of SLLM)
7. causal-batch
8. 2d support
9. support backward across tree
10. chunk-wise BPTT
"""

# normal                    PPL: 9.7576
# bk 1 bkg 4                PPL: 9.3042
# bk 4 bkg 1                PPL: 9.1336
# bk 1 bkg 4 oracle_rep 
# bk 4 bkg 1 oracle_rep     PPL: 9.1336
# bk 4 bkg 1 2 sifters      PPL: 9.2236
# bk 4 bkg 1 recurse 2 lv   PPL: 9.1364
# bk 4 bkg 1 recurse 3 lv   PPL: 9.1930

# topk_head_group
# 

import random, os
from numba.cuda.cudadrv.devicearray import DeviceNDArray as NdArrayCuda
import warnings
import numba.cuda as cuda
import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
from torch import Tensor
from hip.models.hip_attention.attention1_block_gpu import load_checkouts, to_dense
import numpy as np
from numpy import ndarray as NdArray
import numba
import math

def cdiv_python(a, b):
    return math.ceil(a / b)

@numba.njit
def cdiv_numba(a, b):
    return math.ceil(a / b)

@cuda.jit(device=True, inline=True)
def cdiv_cuda(a, b):
    return math.ceil(a / b)

@numba.njit
def de_rope(vec_rope, cos, sin):
    assert len(vec_rope.shape) == 1
    assert vec_rope.shape == cos.shape
    assert cos.shape == sin.shape
    out = np.zeros_like(vec_rope)
    half = len(vec_rope) // 2
    c0 = cos[:half]
    ch = cos[half:]
    s0 = sin[:half]
    sh = sin[half:]
    vr0 = vec_rope[:half]
    vrh = vec_rope[half:]
    out[:half] = (vrh * s0 + vr0 * ch) / (c0 * ch + sh * s0 + 1e-20)
    out[half:] = (out[:half] * c0 - vr0) / (s0 + 1e-20)
    return out

@numba.njit
def rotate_half(vec):
    assert len(vec.shape) == 1
    out = np.zeros_like(vec)
    x1 = vec[:len(vec) // 2]
    x2 = vec[len(vec) // 2:]
    out[:len(vec) // 2] = -x2
    out[len(vec) // 2:] = x1
    return out

@numba.njit
def apply_rope(vec, cos, sin):
    assert vec.shape == cos.shape
    assert cos.shape == sin.shape
    vec_rope = (vec * cos) + (rotate_half(vec) * sin)
    return vec_rope

@numba.njit
def masking_iteration_draft_numba_kernel(
    #in
    q: NdArray, # fp32[block_size_q, HID]
    k: NdArray, # fp32[TSRC(sliced), HID]
    
    # out
    indices: NdArray, #int32[MASK_K // B_K]
    
    # param
    mask_k: int,
    block_size_q: int,
    block_size_k: int,
    block_size_k_group: int, 
    sliding_window_size: int, 
    using_extend: bool,
    rope_cos: Optional[Tensor],
    rope_sin: Optional[Tensor],
    idx_bdst: int,
    TDST: int,
    MAX_TSRC: int,
    
    self_extend_neighboor_window: int,
    self_extend_group_size: int,
    
    topk_head_group_size: int,
) -> int:
    mask_block_k = cdiv_numba(mask_k, block_size_k)
    TSRC = max(0, k.shape[1] - sliding_window_size)
    BSRC = cdiv_numba(TSRC, block_size_k)
    MAX_BSRC = cdiv_numba(MAX_TSRC, block_size_k)
    
    if TSRC <= mask_k:
        k_out = 0
        for i in range(topk_head_group_size):
            for j in range(BSRC):
                indices[k_out] = i * MAX_TSRC + j * block_size_k
                k_out += 1
        return k_out
    else:
        # initialize
        group_sizes = np.zeros_like(indices)
        for i in range(topk_head_group_size):
            for j in range(mask_block_k):
                indices[i * mask_block_k + j] = MAX_BSRC * i + int(BSRC / mask_block_k * j)
                group_sizes[i * mask_block_k + j] = min(BSRC, int(BSRC / mask_block_k * (i + 1))) - int(BSRC / mask_block_k * i)
        
        group_size = BSRC / mask_block_k

        # until converge
        while group_size > 1:
            # ----------------- (dup and score)
            # divide
            dupped_indices = indices.repeat(2).copy()
            dupped_indices[1::2] = (dupped_indices[1::2] + group_sizes * 0.5).astype(np.int32)
            dupped_group_sizes = group_sizes.repeat(2).copy()
            dupped_group_sizes[0::2] = dupped_indices[1::2] - dupped_indices[0::2]
            dupped_group_sizes[1::2] = dupped_indices[0::2] + group_sizes - dupped_indices[1::2]
            dupped_mask = dupped_group_sizes >= 1
            
            scores = np.zeros_like(dupped_indices, dtype=np.float32)
            for i in range(len(scores)):
                if not dupped_mask[i]:
                    continue
                
                idx_tgsrc = dupped_indices[i] * block_size_k
                idx_group = idx_tgsrc // MAX_TSRC
                idx_tsrc = idx_tgsrc % MAX_TSRC
                
                if block_size_k_group > 1:
                    assert not using_extend
                    queries = q[idx_group, 1::2, :].copy()
                    keys_min = k[idx_group, idx_tsrc:idx_tsrc+block_size_k, :q.shape[-1]]
                    keys_max = k[idx_group, idx_tsrc:idx_tsrc+block_size_k, q.shape[-1]:]
                    t_1 = np.ascontiguousarray(queries) @ np.ascontiguousarray(keys_min.T)
                    t_2 = np.ascontiguousarray(queries) @ np.ascontiguousarray(keys_max.T)
                    scores[i] = max(t_1.max(), t_2.max())
                else:
                    if not oracle_rep:
                        queries = q[idx_group, 1::2, :].copy()
                        keys = k[idx_group, idx_tsrc:idx_tsrc+block_size_k, :].copy()
                        
                        if using_extend:
                            assert rope_cos is not None
                            assert rope_sin is not None
                            for j in range(len(keys)):
                                old_idx = idx_tsrc + j
                                
                                # StreamingLLM (not working well)
                                # new_idx = i * block_size_k + j
                                
                                # Self Extend (working great)
                                if idx_tsrc >= (idx_bdst - self_extend_neighboor_window):
                                    new_idx = old_idx
                                else:
                                    new_idx = old_idx // self_extend_group_size
                                
                                keys[j] = de_rope(
                                    keys[j], 
                                    rope_cos[old_idx], 
                                    rope_sin[old_idx]
                                )
                                keys[j] = apply_rope(
                                    keys[j], 
                                    rope_cos[new_idx], 
                                    rope_sin[new_idx]
                                )
                            
                            for j in range(len(queries)):
                                old_idx = idx_bdst + j + TSRC - TDST
                                
                                # new_idx = len(scores) * block_size_k - block_size_q + j
                                
                                if idx_tsrc >= (idx_bdst - self_extend_neighboor_window):
                                    new_idx = old_idx
                                else:
                                    new_idx = old_idx // self_extend_group_size
                                
                                queries[j] = de_rope(
                                    queries[j],
                                    rope_cos[old_idx], 
                                    rope_sin[old_idx]
                                )
                                queries[j] = apply_rope(
                                    queries[j],
                                    rope_cos[new_idx], 
                                    rope_sin[new_idx]
                                )
                        
                        t = np.ascontiguousarray(queries) @ np.ascontiguousarray(keys.T)
                        scores[i] = t.max()
                    else:
                        assert not using_extend
                        queries = q[idx_group, 1::2, :].copy()
                        for shift in range(dupped_group_sizes[i]):
                            keys = k[idx_group, idx_tsrc+shift*block_size_k:idx_tsrc+shift*block_size_k+block_size_k, :]
                            t = np.ascontiguousarray(queries) @ np.ascontiguousarray(keys.T)
                            scores[i] = max(scores[i], t.max())
            scores[:] += -32000.0 * ~dupped_mask
            # -----------------
            
            # ----------------- (torch.argsort)
            # select
            topk_indices = np.argsort(-scores)[:mask_block_k * topk_head_group_size]
            # -----------------
            
            # ----------------- (gather)
            indices[:] = dupped_indices[topk_indices]
            group_sizes[:] = dupped_group_sizes[topk_indices]
            # print(group_size, indices, topk_indices, group_sizes)
            # -----------------
            
            # ----------------- torch.mul_
            group_size = group_size / 2
            # -----------------
        
        indices[:] = np.sort(indices) * block_size_k
        return mask_block_k * topk_head_group_size


@numba.njit(parallel=True)
def masking_iteration_draft_numba(
    # in
    q: NdArray,
    k: NdArray,
    
    # out
    indices: NdArray,
    ks: NdArray,
    
    # param
    mask_k: int,
    block_size_q: int,
    block_size_k: int,
    block_size_k_group: int,
    oracle_rep: bool,
    sliding_window_size: int,
    using_extend: bool,
    rope_cos: Optional[Tensor],
    rope_sin: Optional[Tensor],
    self_extend_neighboor_window: int,
    self_extend_group_size: int,
    
    topk_head_group_size: int, 
):
    """
    grid = (N, TDST)
    """
    
    N, G, TDST, HID = q.shape
    _, _, TSRC, _ = k.shape
    
    for idx_n in numba.prange(N):
        for idx_bdst in numba.prange(cdiv_numba(TDST, block_size_q)):
            q_chunk = q[
                idx_n,
                :, 
                idx_bdst * block_size_q: (idx_bdst + 1) * block_size_q, 
                :
            ]
            k_chunk = k[
                idx_n, 
                :, 
                :((idx_bdst + 1) * block_size_q + TSRC * block_size_k_group - TDST) // block_size_k_group, 
                :
            ]
            final_k = masking_iteration_draft_numba_kernel(
                q_chunk,
                k_chunk,
                indices[idx_n, idx_bdst, :],
                mask_k=mask_k,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                block_size_k_group=block_size_k_group,
                sliding_window_size=sliding_window_size,
                using_extend=using_extend,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                idx_bdst=idx_bdst,
                TDST=TDST,
                MAX_TSRC=k.shape[-2],
                self_extend_neighboor_window=self_extend_neighboor_window,
                self_extend_group_size=self_extend_group_size,
                topk_head_group_size=topk_head_group_size, 
            )
            ks[idx_n, idx_bdst] = final_k

@triton.jit
def masking_iteration_draft_cuda_initialize(
    # in
    INDICES_SEED, 
    stride_indices_seed_b, 
    stride_indices_seed_bdst, 
    stride_indices_seed_bk,
    KS_SEED,
    stride_ks_seed_b,
    stride_ks_seed_bdst,
    POS, stride_pos_tdst,
    
    # out
    INDICES, stride_indices_b, stride_indices_bdst, stride_indices_bk,
    KS, stride_ks_b, stride_ks_bdst,
    GROUP_SIZE, stride_group_size_b, stride_group_size_bdst, stride_group_size_bk,
    
    # temp
    T_GROUP_SIZE, stride_t_group_size_b, stride_t_group_size_bdst,
    
    # param
    mask_k: int,
    block_size_q: tl.constexpr,
    block_size_k: tl.constexpr,
    
    sliding_window_size: int,
    
    G, MAX_TDST, MAX_TSRC, 
    
    BLOCK_MASK_BLOCK_K: tl.constexpr,
):
    idx_b = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_group = tl.program_id(2)
    idx_tdst = tl.arange(0, block_size_q) + idx_bdst * block_size_q
    mask_tdst = idx_tdst < MAX_TDST
    
    mask_block_k = tl.cdiv(mask_k, block_size_k)
    pos_tdst = tl.load(
        POS +\
            idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
    )
    TSRC = tl.max(pos_tdst)
    TSRC = tl.maximum(0, TSRC - sliding_window_size)
    BSRC = tl.cdiv(TSRC, block_size_k)
    MAX_BSRC = tl.cdiv(MAX_TSRC, block_size_k)
    
    if TSRC <= mask_k:
        idx_bk = tl.arange(0, BLOCK_MASK_BLOCK_K)
        mask_bk = idx_bk < BSRC
        tl.store(
            INDICES +\
                idx_b * stride_indices_b +\
                idx_bdst * stride_indices_bdst +\
                (idx_group * BSRC + idx_bk) * stride_indices_bk,
            value = idx_group * MAX_BSRC + idx_bk,
            mask = mask_bk,
        )
        
        if idx_group == 0:
            tl.store(
                KS +\
                    idx_b * stride_ks_b +\
                    idx_bdst * stride_ks_bdst,
                value = BSRC * G
            )
    else:
        idx_bk = tl.arange(0, BLOCK_MASK_BLOCK_K)
        mask_bk = idx_bk < mask_block_k
        
        ks = 0
        if KS_SEED is not None:
            ks = tl.load(
                KS_SEED +\
                    idx_b * stride_ks_seed_b +\
                    idx_bdst * stride_ks_seed_bdst,
            )
        
        indices = (MAX_BSRC * idx_group + (BSRC / mask_block_k * idx_bk)).to(tl.int32)
        group_sizes = tl.minimum(
            BSRC, 
            (
                BSRC / mask_block_k * (idx_bk + 1).to(tl.int32) -\
                (BSRC / mask_block_k * idx_bk).to(tl.int32)
            )
        ).to(tl.int32)
        if INDICES_SEED is not None:
            if ks == (mask_block_k * G):
                indices = tl.load(
                    INDICES_SEED +\
                        idx_b * stride_indices_seed_b +\
                        idx_bdst * stride_indices_seed_bdst +\
                        (idx_group * mask_block_k + idx_bk) * stride_indices_seed_bk,
                    mask=(
                        mask_bk &
                        (idx_bk > 0)
                    ),
                    other=idx_group * MAX_BSRC,
                )
                indicse_next = tl.load(
                    INDICES_SEED +\
                        idx_b * stride_indices_seed_b +\
                        idx_bdst * stride_indices_seed_bdst +\
                        (idx_group * mask_block_k + idx_bk + 1) * stride_indices_seed_bk,
                    mask=(
                        mask_bk &
                        (idx_bk < (BLOCK_MASK_BLOCK_K - 1))
                    ),
                    other=idx_group * MAX_BSRC + BSRC,
                )
                group_sizes = indicse_next - indices
        
        tl.store(
            INDICES +\
                idx_b * stride_indices_b +\
                idx_bdst * stride_indices_bdst +\
                (idx_group * mask_block_k + idx_bk) * stride_indices_bk,
            value = indices,
            mask=mask_bk,
        )
        tl.store(
            GROUP_SIZE +\
                idx_b * stride_group_size_b +\
                idx_bdst * stride_group_size_bdst +\
                (idx_group * mask_block_k + idx_bk) * stride_group_size_bk,
            value = group_sizes,
            mask=mask_bk,
        )
        
        tl.atomic_max(
            T_GROUP_SIZE +\
                idx_b * stride_t_group_size_b +\
                idx_bdst * stride_t_group_size_bdst,
            val = tl.max(group_sizes)
        )
        tl.atomic_add(
            KS +\
                idx_b * stride_ks_b +\
                idx_bdst * stride_ks_bdst,
            val = mask_block_k
        )

@triton.jit
def split_half(x: tl.tensor, T: tl.constexpr, HID: tl.constexpr):
    x = x.reshape(T, 2, HID // 2)
    x = x.trans(0, 2, 1)
    return x.split()

@triton.jit
def merge_half(left: tl.tensor, right: tl.tensor, T: tl.constexpr, HID: tl.constexpr):
    assert left.shape == right.shape
    x = tl.join(left, right)
    x = x.trans(0, 2, 1)
    x = x.reshape(T, HID)
    return x

@triton.jit
def de_rope(vec: tl.tensor, cos: tl.tensor, sin: tl.tensor, T: tl.constexpr, HID: tl.constexpr):
    c0, ch = split_half(cos, T, HID)
    s0, sh = split_half(sin, T, HID)
    vr0, vrh = split_half(vec, T, HID)
    
    out0 = (vrh * s0 + vr0 * ch) / (c0 * ch + sh * s0 + 1e-20)
    outh = (out0 * c0 - vr0) / (s0 + 1e-20)
    out = merge_half(out0, outh, T, HID)
    return out

@triton.jit
def rotate_half(vec: tl.tensor, T: tl.constexpr, HID: tl.constexpr):
    left, right = split_half(vec, T, HID)
    out0 = -right
    outh = left
    return merge_half(out0, outh, T, HID)

@triton.jit
def apply_rope(vec: tl.tensor, cos: tl.tensor, sin: tl.tensor, T: tl.constexpr, HID: tl.constexpr):
    vec = vec * cos + rotate_half(vec, T, HID) * sin
    return vec

@triton.jit
def adjust_rope(
    tokens: tl.tensor,
    old_t: tl.tensor,
    new_t: tl.tensor,
    idx_hid: tl.tensor,
    
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    
    T: tl.constexpr, HID: tl.constexpr,
):
    cos_old = tl.load(
        COS +\
            old_t[:, None] * stride_cos_t +\
            idx_hid[None, :] * stride_cos_hid
    )
    sin_old = tl.load(
        SIN +\
            old_t[:, None] * stride_sin_t +\
            idx_hid[None, :] * stride_sin_hid
    )
    
    cos_new = tl.load(
        COS +\
            new_t[:, None] * stride_cos_t +\
            idx_hid[None, :] * stride_cos_hid
    )
    sin_new = tl.load(
        SIN +\
            new_t[:, None] * stride_sin_t +\
            idx_hid[None, :] * stride_sin_hid
    )
    
    tokens = de_rope(tokens, cos_old, sin_old, T, HID)
    tokens = apply_rope(tokens, cos_new, sin_new, T, HID)
    
    return tokens

@triton.jit
def masking_iteration_draft_cuda_dup_and_score_calc_score(
    dupped_indices_for_keys,
    
    Q, stride_q_b, stride_q_g, stride_q_tdst, stride_q_hid,
    K, stride_k_b, stride_k_g, stride_k_tsrc, stride_k_hid,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    
    idx_b, 
    idx_tdst, mask_tdst, pos_tdst,
    dupped_mask,
    
    G, MAX_TSRC, HID: tl.constexpr,
    
    USING_EXTEND: tl.constexpr,
    extend_window_size,
    extend_group_size,
    
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    REDUCE_METHOD: tl.constexpr,
):
    idx_tsrc = (
        (dupped_indices_for_keys * BLOCK_SIZE_K)[:, None]\
        + tl.arange(0, BLOCK_SIZE_K)[None, :]
    ).reshape(BLOCK_BK * 2 * BLOCK_SIZE_K)
    idx_group = idx_tsrc // MAX_TSRC
    idx_tsrc = idx_tsrc % MAX_TSRC
    
    acc = tl.zeros((BLOCK_SIZE_Q, BLOCK_BK * 2 * BLOCK_SIZE_K), dtype=tl.float32)
    idx_hid = tl.arange(0, HID)
    for i_group in range(G):
        mask_keys = (dupped_mask[:, None] & (idx_group == i_group).reshape(BLOCK_BK * 2, BLOCK_SIZE_K))\
            .reshape(1, BLOCK_BK * 2 * BLOCK_SIZE_K)
        queries = tl.load(
            Q +\
                idx_b * stride_q_b +\
                i_group * stride_q_g +\
                idx_tdst[:, None] * stride_q_tdst +\
                idx_hid[None, :] * stride_q_hid,
            mask = mask_tdst[:, None],
            other = 0
        )
        keys = tl.load(
            K +\
                idx_b * stride_k_b +\
                idx_group[None, :] * stride_k_g +\
                idx_tsrc[None, :] * stride_k_tsrc +\
                idx_hid[:, None] * stride_k_hid,
            mask = mask_keys,
            other = 0,
        )
        
        if USING_EXTEND:
            if tl.min(pos_tdst) > extend_window_size:
                assert COS is not None
                assert SIN is not None
                
                idx_tsrc_calib = tl.maximum(0, tl.min(pos_tdst) - extend_window_size - BLOCK_SIZE_K)
                idx_tsrc_calib = idx_tsrc_calib + tl.arange(0, 16)
                mask_tsrc_calib = idx_tsrc_calib < MAX_TSRC
                keys_calib_old = tl.load(
                    K +\
                        idx_b * stride_k_b +\
                        i_group * stride_k_g +\
                        idx_tsrc_calib[None, :] * stride_k_tsrc +\
                        idx_hid[:, None] * stride_k_hid,
                    mask=mask_tsrc_calib[None, :],
                    other=0
                )
                
                keys_calib_new = adjust_rope(
                    keys_calib_old.trans(1, 0), idx_tsrc_calib, idx_tsrc_calib // extend_group_size, idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    16, HID,
                ).trans(1, 0)
                
                old_tsrc = idx_tsrc
                mask_tsrc_window = idx_tsrc >= (tl.min(tl.where(mask_tdst, pos_tdst, 9999999)) - extend_window_size)
                new_tsrc = tl.where(
                    mask_tsrc_window,
                    old_tsrc,
                    old_tsrc // extend_group_size
                )
                
                keys = keys.trans(1, 0)
                keys = adjust_rope(
                    keys, old_tsrc, new_tsrc, idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    BLOCK_BK * 2 * BLOCK_SIZE_K, HID,
                ).to(keys.dtype)
                keys = tl.trans(keys, 1, 0)
                keys = (keys * mask_keys).to(keys.dtype)
                
                old_tdst = pos_tdst
                new_tdst = old_tdst // extend_group_size
                
                queries_grouped = adjust_rope(
                    queries, old_tdst, new_tdst, idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    BLOCK_SIZE_Q, HID,
                ).to(queries.dtype)
                
                t_calib_old = tl.dot(
                    queries, keys_calib_old.to(queries.dtype),
                )
                t_calib_new = tl.dot(
                    queries_grouped, keys_calib_new.to(queries.dtype),
                )
                calibration = tl.sum(t_calib_new - t_calib_old, axis=-1) / 16
                
                t_window = tl.dot(
                    queries, keys.to(queries.dtype),
                )
                t_grouped = tl.dot(
                    queries_grouped, keys.to(queries.dtype),
                ) - calibration[:, None]
                # NOTE: this calibration trick is very important.
                t = tl.where(
                    mask_tsrc_window[None, :],
                    t_window,
                    t_grouped,
                ).to(tl.float32)
            else:
                t = tl.dot(
                    queries, keys,
                ).to(tl.float32)
        else:
            t = tl.dot(
                queries, keys,
            ).to(tl.float32)
        acc += t
    acc = tl.where(
        (
            (acc == 0.0) |
            (idx_tsrc[None, :] > pos_tdst[:, None]) |
            False
        ), 
        -32000.0 if REDUCE_METHOD == 'max' else 32000.0, 
        acc
    )
    scores = tl.reshape(acc, (BLOCK_SIZE_Q, BLOCK_BK * 2, BLOCK_SIZE_K))
    if REDUCE_METHOD == 'max':
        scores = tl.max(
            scores,
            axis=0
        )
        scores = tl.max(
            scores,
            axis=-1,
        )
    elif REDUCE_METHOD == 'min':
        scores = tl.min(
            scores,
            axis=0
        )
        scores = tl.min(
            scores,
            axis=-1,
        )
    else:
        raise Exception()
    scores = tl.where(dupped_mask, scores, float('-inf'))
    
    return scores

@triton.jit
def masking_iteration_draft_cuda_dup_and_score(
    Q, stride_q_b, stride_q_g, stride_q_tdst, stride_q_hid,
    K, stride_k_b, stride_k_g, stride_k_tsrc, stride_k_hid,
    POS, stride_pos_tdst,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    
    INDICES, stride_indices_b, stride_indices_bdst, stride_indices_bk,
    KS, stride_ks_b, stride_ks_bdst,
    GROUP_SIZE, stride_group_size_b, stride_group_size_bdst, stride_group_size_bk,
    
    DUPPED_INDICES, 
    stride_dupped_indices_b, 
    stride_dupped_indices_bdst, 
    stride_dupped_indices_bk,
    DUPPED_GROUP_SIZE, 
    stride_dupped_group_size_b, 
    stride_dupped_group_size_bdst, 
    stride_dupped_group_size_bk,
    SCORES,
    stride_scores_b,
    stride_scores_bdst,
    stride_scores_bk,
    
    T_GROUP_SIZE, stride_t_group_size_b, stride_t_group_size_bdst,
    
    mask_k,
    
    sliding_window_size,
    
    G: tl.constexpr, MAX_TDST, MAX_TSRC, BK, HID: tl.constexpr,
    RAND_SEED,
    SAMPLE_METHOD: tl.constexpr,
    BRANCH_METHOD: tl.constexpr,
    
    USING_EXTEND: tl.constexpr,
    extend_window_size,
    extend_group_size,
    
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
):
    idx_b = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    mask_tdst = idx_tdst < MAX_TDST
    idx_bk = tl.program_id(2) * BLOCK_BK + tl.arange(0, BLOCK_BK)
    mask_bk = idx_bk < (BK * G)
    idx_bk_dup = tl.program_id(2) * BLOCK_BK * 2 + tl.arange(0, BLOCK_BK * 2)
    mask_bk_dup = idx_bk_dup < (BK * 2 * G)
    idx_n = idx_b * G + tl.arange(0, G)
    
    mask_block_k = tl.cdiv(mask_k, BLOCK_SIZE_K)
    pos_tdst = tl.load(
        POS +\
            idx_tdst * stride_pos_tdst,
        mask=mask_tdst
    )
    TSRC = tl.max(pos_tdst)
    TSRC = tl.maximum(0, TSRC - sliding_window_size)
    BSRC = tl.cdiv(TSRC, BLOCK_SIZE_K)
    MAX_BSRC = tl.cdiv(MAX_TSRC, BLOCK_SIZE_K)
    
    if TSRC <= mask_k:
        return
    
    t_group_size = tl.load(
        T_GROUP_SIZE +\
            idx_b * stride_t_group_size_b +\
            idx_bdst * stride_t_group_size_bdst
    )
    if t_group_size <= 1.0:
        return

    # int[BLOCK_BK]
    indices = tl.load(
        INDICES +\
            idx_b * stride_indices_b +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
        mask=mask_bk,
        other=0,
    )
    
    # int[BLOCK_BK]
    group_sizes = tl.load(
        GROUP_SIZE +\
            idx_b * stride_group_size_b +\
            idx_bdst * stride_group_size_bdst +\
            idx_bk * stride_group_size_bk,
        mask=mask_bk,
        other=0,
    )
    
    # int[BLOCK_BK * 2]
    dupped_indices = tl.reshape(
        tl.join(indices, indices),
        (BLOCK_BK * 2,),
    )
    dupped_group_sizes = tl.reshape(
        tl.join(group_sizes, group_sizes),
        (BLOCK_BK * 2,)
    )
    if BRANCH_METHOD == 'half':
        dupped_indices = tl.where(
            (tl.arange(0, BLOCK_BK * 2) % 2) == 0,
            dupped_indices,
            (dupped_indices + dupped_group_sizes * 0.5).to(tl.int32)
        )
    elif BRANCH_METHOD == 'random':
        dupped_indices = tl.where(
            (tl.arange(0, BLOCK_BK * 2) % 2) == 0,
            dupped_indices,
            tl.where(
                dupped_group_sizes == 0,
                dupped_indices,
                tl.maximum(
                    dupped_indices + 1,
                    dupped_indices +\
                        dupped_group_sizes * 0.5 +\
                        dupped_group_sizes * (0.2 * tl.random.rand(
                            RAND_SEED, 
                            tl.arange(0, BLOCK_BK * 2) +\
                                tl.program_id(0) * 7 +\
                                tl.program_id(1) * 53 +\
                                tl.program_id(2) * 157
                            ) * 0.99 - 0.1
                        )
                ).to(tl.int32)
            )
        )
    else:
        raise Exception(BRANCH_METHOD)
    flipped_dupped_indices = tl.reshape(
        tl.flip(
            tl.reshape(
                dupped_indices, 
                (BLOCK_BK, 2)
            ), 
        ),
        (BLOCK_BK * 2),
    )
    dupped_group_sizes = tl.where(
        (tl.arange(0, BLOCK_BK * 2) % 2) == 0,
        flipped_dupped_indices - dupped_indices,
        flipped_dupped_indices + dupped_group_sizes - dupped_indices,
    )
    dupped_mask = (dupped_group_sizes >= 1) & mask_bk_dup
    
    dupped_indices_for_keys = dupped_indices
    if SAMPLE_METHOD == 'random':
        offsets = tl.where(
            dupped_group_sizes > 4,
            0, 
            (
                tl.randint(
                    RAND_SEED, 
                    dupped_indices + \
                        tl.program_id(0) * 3 + \
                        tl.program_id(1) * 71 + \
                        tl.program_id(2) * 13
                    ) % dupped_group_sizes.to(tl.uint32)
            ).to(tl.int32)
        )
        dupped_indices_for_keys += offsets
    elif SAMPLE_METHOD == 'last':
        dupped_indices_for_keys = dupped_indices + tl.where(
            dupped_group_sizes == 0,
            0,
            dupped_group_sizes - 1,
        )
    elif SAMPLE_METHOD == 'center':
        dupped_indices_for_keys = dupped_indices + tl.maximum(
            0, dupped_group_sizes // 2
        )
    elif SAMPLE_METHOD == 'oracle':
        # NOTE: perform linear scan inside of the chunk, this will cost O(T^2)
        dupped_indices_for_keys_start = dupped_indices_for_keys
        dupped_indices_for_keys_end = dupped_indices_for_keys + tl.maximum(dupped_group_sizes - 1, 0)
        max_scores = tl.zeros((BLOCK_BK * 2, ), dtype=tl.float32) - 32000.0
        for i_shift in range(0, tl.cdiv(BSRC, mask_block_k)):
            t_dupped_indices_for_keys = tl.where(
                i_shift < dupped_group_sizes,
                dupped_indices_for_keys_start + i_shift,
                dupped_indices_for_keys_end
            ).to(tl.int32)
            t_scores = masking_iteration_draft_cuda_dup_and_score_calc_score(
                t_dupped_indices_for_keys,
                
                Q, stride_q_b, stride_q_g, stride_q_tdst, stride_q_hid,
                K, stride_k_b, stride_k_g, stride_k_tsrc, stride_k_hid,
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                
                idx_b, 
                idx_tdst, mask_tdst, pos_tdst,
                dupped_mask,
                
                G, MAX_TSRC, HID,
                
                USING_EXTEND,
                extend_window_size,
                extend_group_size,
                
                BLOCK_SIZE_Q,
                BLOCK_SIZE_K,
                BLOCK_BK,
                'max',
            )
            dupped_indices_for_keys = tl.where(
                t_scores > max_scores,
                t_dupped_indices_for_keys,
                dupped_indices_for_keys,
            )
            max_scores = tl.minimum(max_scores, t_scores)
    else:
        # this should be first
        assert SAMPLE_METHOD == 'first'
    
    scores = masking_iteration_draft_cuda_dup_and_score_calc_score(
        dupped_indices_for_keys,
        
        Q, stride_q_b, stride_q_g, stride_q_tdst, stride_q_hid,
        K, stride_k_b, stride_k_g, stride_k_tsrc, stride_k_hid,
        COS, stride_cos_t, stride_cos_hid,
        SIN, stride_sin_t, stride_sin_hid,
        
        idx_b, 
        idx_tdst, mask_tdst, pos_tdst,
        dupped_mask,
        
        G, MAX_TSRC, HID,
        
        USING_EXTEND,
        extend_window_size,
        extend_group_size,
        
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
        BLOCK_BK,
        'max',
    )
    
    tl.store(
        SCORES +\
            idx_b * stride_scores_b +\
            idx_bdst * stride_scores_bdst +\
            idx_bk_dup * stride_scores_bk,
        value=scores,
        mask=mask_bk_dup,
    )
    tl.store(
        DUPPED_INDICES +\
            idx_b * stride_dupped_indices_b +\
            idx_bdst * stride_dupped_indices_bdst +\
            idx_bk_dup * stride_dupped_indices_bk,
        value=dupped_indices,
        mask=mask_bk_dup,
    )
    tl.store(
        DUPPED_GROUP_SIZE +\
            idx_b * stride_dupped_group_size_b +\
            idx_bdst * stride_dupped_group_size_bdst +\
            idx_bk_dup * stride_dupped_group_size_bk,
        value=dupped_group_sizes,
        mask=mask_bk_dup,
    )

@triton.jit
def masking_iteration_draft_cuda_gather(
    INDICES, 
    stride_indices_b, 
    stride_indices_bdst, 
    stride_indices_bk,
    GROUP_SIZES, 
    stride_group_sizes_b, 
    stride_group_sizes_bdst, 
    stride_group_sizes_bk,
    
    DUPPED_INDICES, 
    stride_dupped_indices_b, 
    stride_dupped_indices_bdst, 
    stride_dupped_indices_bk,
    DUPPED_GROUP_SIZE, 
    stride_dupped_group_size_b, 
    stride_dupped_group_size_bdst, 
    stride_dupped_group_size_bk,
    
    TOPK_INDICES,
    stride_topk_indices_b,
    stride_topk_indices_bdst,
    stride_topk_indices_bk,
    
    T_GROUP_SIZE,
    stride_t_group_size_b, 
    stride_t_group_size_bdst,
    
    G, BK,
    
    BLOCK_BK: tl.constexpr,
):
    idx_b = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_bk = tl.program_id(2) * BLOCK_BK + tl.arange(0, BLOCK_BK)
    mask_bk = idx_bk < (BK * G)
    
    t_group_size = tl.load(
        T_GROUP_SIZE +\
            idx_b * stride_t_group_size_b +\
            idx_bdst * stride_t_group_size_bdst,
    )
    if t_group_size <= 1.0:
        return
    
    topk_indices = tl.load(
        TOPK_INDICES +\
            idx_b * stride_topk_indices_b +\
            idx_bdst * stride_topk_indices_bdst +\
            idx_bk * stride_topk_indices_bk,
        mask=mask_bk,
    )
    
    dupped_indices = tl.load(
        DUPPED_INDICES +\
            idx_b * stride_dupped_indices_b +\
            idx_bdst * stride_dupped_indices_bdst +\
            topk_indices * stride_dupped_indices_bk,
        mask=mask_bk
    )
    dupped_group_size = tl.load(
        DUPPED_GROUP_SIZE +\
            idx_b * stride_dupped_group_size_b +\
            idx_bdst * stride_dupped_group_size_bdst +\
            topk_indices * stride_dupped_group_size_bk,
        mask=mask_bk
    )
    
    tl.store(
        INDICES +\
            idx_b * stride_indices_b +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
        value=dupped_indices,
        mask=mask_bk
    )
    tl.store(
        GROUP_SIZES +\
            idx_b * stride_group_sizes_b +\
            idx_bdst * stride_group_sizes_bdst +\
            idx_bk * stride_group_sizes_bk,
        value=dupped_group_size,
        mask=mask_bk
    )

@triton.jit
def masking_iteration_draft_cuda_epiloge(
    INDICES, 
    stride_indices_b, 
    stride_indices_bdst, 
    stride_indices_bk,
    KS,
    stride_ks_b,
    stride_ks_bdst,
    
    KS_COUNT, 
    stride_ks_count_b, 
    stride_ks_count_bdst, 
    stride_ks_count_g,
    KS_START_END, 
    stride_ks_start_end_b,
    stride_ks_start_end_bdst,
    stride_ks_start_end_g,
    
    BK, MAX_TSRC, 
    
    G: tl.constexpr,
    BLOCK_BK: tl.constexpr,
):
    idx_b = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_bk = tl.program_id(2) * BLOCK_BK + tl.arange(0, BLOCK_BK)
    
    ks = tl.load(
        KS + \
            idx_b * stride_ks_b +\
            idx_bdst * stride_ks_bdst,
    )
    mask_bk = idx_bk < ks
    
    indices = tl.load(
        INDICES +\
            idx_b * stride_indices_b +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
        mask=mask_bk,
        other=0
    )
    
    hist = tl.histogram(indices // MAX_TSRC, G)
    hist -= (tl.arange(0, G) == 0).to(tl.int32) * (tl.sum((~mask_bk).to(tl.int32)))
    
    hist_cumsum = tl.cumsum(hist)
    
    idx_g = tl.arange(0, G)
    
    tl.atomic_add(
        KS_COUNT +\
            idx_b * stride_ks_count_b +\
            idx_bdst * stride_ks_count_bdst +\
            idx_g * stride_ks_count_g,
        val=hist
    )
    tl.atomic_add(
        KS_START_END +\
            idx_b * stride_ks_start_end_b +\
            idx_bdst * stride_ks_start_end_bdst +\
            (idx_g + 1) * stride_ks_start_end_g,
        val=hist_cumsum
    )

@triton.jit
def masking_iteration_draft_cuda_partial_softmax(
    SCORES, stride_scores_b, stride_scores_bdst, stride_scores_bk,
    DUPPED_INDICES, 
    stride_dupped_indices_b, 
    stride_dupped_indices_bdst, 
    stride_dupped_indices_bk,
    DUPPED_GROUP_SIZES,
    stride_dupped_group_sizes_b,
    stride_dupped_group_sizes_bdst,
    stride_dupped_group_sizes_bk,
    
    SINK_TOKEN_SIZE,
    MASK_BLOCK_K,
    G, BK, MAX_BSRC,
    BLOCK_SIZE_K,
    
    BLOCK_SCORE: tl.constexpr,
):
    idx_b = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    grid_bk = tl.num_programs(2)
    idx_bk = tl.arange(0, BLOCK_SCORE) * grid_bk + tl.program_id(2)
    mask_bk = idx_bk < BK
    
    indices = tl.load(
        DUPPED_INDICES +\
            idx_b * stride_dupped_indices_b +\
            idx_bdst * stride_dupped_indices_bdst +\
            idx_bk * stride_dupped_indices_bk,
        mask=mask_bk,
        other=MAX_BSRC * G
    )
    group_sizes = tl.load(
        DUPPED_GROUP_SIZES +\
            idx_b * stride_dupped_group_sizes_b +\
            idx_bdst * stride_dupped_group_sizes_bdst +\
            idx_bk * stride_dupped_group_sizes_bk,
        mask=mask_bk,
        other=MAX_BSRC * G
    )
    groups = indices // MAX_BSRC
    scores = tl.load(
        SCORES +\
            idx_b * stride_scores_b +\
            idx_bdst * stride_scores_bdst +\
            idx_bk * stride_scores_bk,
        mask=mask_bk,
        other=float('-inf')
    ).to(tl.float32)
    
    for i_group in range(G):
        mask_softmax = groups == i_group
        count = tl.max(mask_softmax.to(tl.int32))
        t = count / BK
        scores_masked = tl.where(mask_softmax, scores, float('-inf'))
        scores_softmax = tl.softmax(scores_masked * t)
        neg_scores_softmax_sorted = tl.sort(-scores_softmax)
        scores_promote_thresh = -tl.min(neg_scores_softmax_sorted * (tl.arange(0, BLOCK_SCORE) == (MASK_BLOCK_K * 0.5).to(tl.int32)))
        scores_softmax = tl.where(scores_softmax >= scores_promote_thresh, scores_softmax + 1, scores_softmax)
        scores = tl.where(mask_softmax, scores_softmax, scores)
    
    scores = tl.where((indices % MAX_BSRC) < tl.cdiv(SINK_TOKEN_SIZE, BLOCK_SIZE_K), 2, scores)
    scores = tl.where(group_sizes == 0, -1, scores)
    
    tl.store(
        SCORES +\
            idx_b * stride_scores_b +\
            idx_bdst * stride_scores_bdst +\
            idx_bk * stride_scores_bk,
        value=scores,
        mask=mask_bk,
    )

def masking_iteration_draft(
    q: Tensor,
    k: Tensor,
    position_ids: Tensor,
    mask_k: int,
    block_size_q: int,
    block_size_k: int,
    block_size_k_group: int,
    sliding_window_size: int,
    sink_token_size: int,
    using_extend: bool,
    rope_cos: Optional[Tensor],
    rope_sin: Optional[Tensor],
    self_extend_neighboor_window: int,
    self_extend_group_size: int,
    topk_head_group_size: int,
    sample_method: str,
    branch_method: str,
    score_head_group_size: int,
    
    indices_seed: Optional[Tensor] = None,
    ks_seed: Optional[Tensor] = None,
):
    assert q.device == k.device
    assert isinstance(q, Tensor)
    assert isinstance(k, Tensor)
    
    if rope_cos is not None:
        assert rope_cos.ndim == 2
        assert rope_cos.shape[-1] == q.shape[-1]
        assert isinstance(rope_cos, Tensor)
    
    if rope_sin is not None:
        assert rope_sin.ndim == 2
        assert rope_sin.shape[-1] == q.shape[-1]
        assert isinstance(rope_sin, Tensor)
        assert isinstance(rope_sin, Tensor)
    
    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape
    BDST = cdiv_python(TDST, block_size_q)
    BSRC = cdiv_python(TSRC, block_size_k)
    
    assert (N % topk_head_group_size) == 0, 'batch * n_head should divisible by head group size'
    
    # split batch-head dim into head groups
    q = q.view(N // topk_head_group_size, topk_head_group_size, TDST, HID)
    k = k.view(N // topk_head_group_size, topk_head_group_size, TSRC, HID)
    
    B, G, TDST, HID = q.shape
    _, _, TSRC, _ = k.shape
    mask_block_k = cdiv_python(mask_k, block_size_k)
    
    assert block_size_k_group == 1
    if block_size_k_group > 1:
        warnings.warn('K grouping is inefficient right now.')
        k_group = k.view(B, G, TSRC // block_size_k_group, block_size_k_group, HID)
        k_group_min = torch.min(k_group, dim=-2)
        k_group_max = torch.max(k_group, dim=-2)
        k = torch.concat([k_group_min, k_group_max], dim=-1)
    del block_size_k_group
    
    indices = torch.full((
        B,
        cdiv_python(TDST, block_size_q), 
        # head group is merged as single sequence
        G * mask_block_k,
    ), fill_value=(BSRC + block_size_k + block_size_q) * G, dtype=torch.int32, device=q.device)
    
    ks = torch.zeros((
        B, 
        cdiv_python(TDST, block_size_q),
    ), dtype=torch.int32, device=q.device)
    
    group_sizes = torch.zeros_like(indices)
    t_group_sizes = torch.zeros((B, BDST), dtype=torch.float32, device=q.device)
    
    assert len(q.stride()) == 4
    assert len(k.stride()) == 4
    assert len(indices.stride()) == 3
    assert len(ks.stride()) == 2
    assert len(group_sizes.stride()) == 3
    assert len(t_group_sizes.stride()) == 2
    if indices_seed is not None:
        assert len(indices_seed.stride()) == 3
        assert len(ks_seed.stride()) == 2
        assert indices_seed.shape == indices.shape
        assert ks_seed.shape == ks.shape
        indices_seed = indices_seed // block_size_k
    if rope_cos is not None:
        assert len(rope_cos.stride()) == 2
        assert len(rope_sin.stride()) == 2
    
    assert sample_method in ['first', 'last', 'random', 'oracle', 'center']
    
    # launch kernels
    BLOCK_MASK_BLOCK_K = triton.next_power_of_2(mask_block_k)
    grid = (B, BDST, G)
    masking_iteration_draft_cuda_initialize[grid](
        indices_seed, *(indices_seed.stride() if indices_seed is not None else (0, 0, 0)),
        ks_seed, *(ks_seed.stride() if ks_seed is not None else (0, 0)),
        position_ids, *position_ids.stride(),
        
        indices, *indices.stride(),
        ks, *ks.stride(),
        group_sizes, *group_sizes.stride(),
        
        t_group_sizes, *t_group_sizes.stride(),
        
        mask_k,
        block_size_q, 
        block_size_k, 
        
        sliding_window_size,
        
        G, TDST, TSRC, 
        
        BLOCK_MASK_BLOCK_K,
        
        num_warps=min(max(cdiv_python(BLOCK_MASK_BLOCK_K, 32), 1), 32),
        num_stages=1,
    )
    
    dupped_indices = torch.empty(
        (B, BDST, indices.shape[-1] * 2),
        dtype=torch.int32, device=q.device,
    )
    dupped_group_sizes = torch.empty(
        (B, BDST, indices.shape[-1] * 2),
        dtype=torch.int32, device=q.device,
    )
    scores = torch.zeros_like(dupped_indices, dtype=q.dtype)
    
    BLOCK_BK = 16 // 2 // block_size_k
    assert BLOCK_BK > 0
    BLOCK_HID = 64
    assert (HID % BLOCK_HID) == 0
    
    # print(indices[0, -10])
    # print(ks[0, -10])
    # assert indices[0, -10].shape == torch.unique(indices[0, -10]).shape, f'{indices[0, -10].shape} == {torch.unique(indices[0, -10]).shape}'
    
    # NOTE: this should be removed
    max_group_size = torch.max(t_group_sizes).item()
    while max_group_size >= 1:
        # print('ind', indices[0, 32, :10])
        # print('gs', group_sizes[0, 32, :10])
        
        grid = (B, BDST, triton.cdiv(indices.shape[-1], BLOCK_BK))
        masking_iteration_draft_cuda_dup_and_score[grid](
            q, *q.stride(),
            # NOTE: need experiment, sink token based normalization is working well for head grouping
            k if G < 2 else (k - k[:, :, :2, :].mean(-2, keepdim=True)), 
            # k,
            *k.stride(),
            # (k - k[:, :, :2, :].mean(-2, keepdim=True)), *k.stride(),
            position_ids, *position_ids.stride(),
            rope_cos, *(rope_cos.stride() if rope_cos is not None else (0, 0)),
            rope_sin, *(rope_sin.stride() if rope_sin is not None else (0, 0)),
            
            indices, *indices.stride(),
            ks, *ks.stride(),
            group_sizes, *group_sizes.stride(),
            
            dupped_indices, *dupped_indices.stride(),
            dupped_group_sizes, *dupped_group_sizes.stride(),
            scores, *scores.stride(),
            
            t_group_sizes, *t_group_sizes.stride(),
            
            mask_k,
            
            sliding_window_size,
            
            G, TDST, TSRC, mask_block_k, HID,
            random.randint(0, 1024*1024),
            sample_method,
            branch_method,
            
            using_extend,
            self_extend_neighboor_window,
            self_extend_group_size,
            
            block_size_q,
            block_size_k,
            BLOCK_BK,
            
            num_stages=1,
        )
        
        # print(indices[0, -10])
        # print(ks[0, -10])
        
        # assert indices[0, -10].shape == torch.unique(indices[0, -10]).shape, f'{indices[0, -10].shape} == {torch.unique(indices[0, -10]).shape}'
        
        # scores_max = scores.max()
        # scores = torch.where((dupped_indices % BSRC) == 0, scores_max, scores)
        # scores = torch.where((dupped_indices % BSRC) == 1, scores_max, scores)
        # scores = torch.where((dupped_indices % BSRC) == 2, scores_max, scores)
        # scores = torch.where((dupped_indices % BSRC) == 3, scores_max, scores)
        
        # NOTE: because of softmax, we cannot fuse everything...
        # BLOCK_SCORE = min(1024, mask_block_k * G)
        BLOCK_SCORE = triton.next_power_of_2(scores.shape[-1])
        grid = (B, BDST, triton.cdiv(scores.shape[-1], BLOCK_SCORE))
        masking_iteration_draft_cuda_partial_softmax[grid](
            scores, *scores.stride(),
            dupped_indices, *dupped_indices.stride(),
            dupped_group_sizes, *dupped_group_sizes.stride(),
            
            sink_token_size,
            mask_block_k,
            G, scores.shape[-1], BSRC, block_size_k,
            
            BLOCK_SCORE,
            
            num_warps=min(32, BLOCK_SCORE//32),
        )
        
        if score_head_group_size > 1:
            assert score_head_group_size <= B
            assert (B  % score_head_group_size) == 0
            scores_max = scores\
                .view(B // score_head_group_size, score_head_group_size, BDST, scores.shape[-1])\
                .min(1, keepdim=True)[0]
            # print(scores_max[0, :, 0])
            # input()
            scores = scores_max\
                .repeat(1, score_head_group_size, 1, 1)\
                .view(-1, scores_max.shape[-2], scores_max.shape[-1])
        
        # print(dupped_indices[0, -10])
        # print(scores[0, -10])
        # print(scores.sum(-1)[0, -1])
        
        # NOTE: because of sort, we cannot fuse everything...
        topk_indices = torch.argsort(
            scores, dim=-1, descending=True, stable=False
        )[:, :, :mask_block_k * G]
        
        grid = (B, BDST, triton.cdiv(indices.shape[-1], BLOCK_BK))
        masking_iteration_draft_cuda_gather[grid](
            indices, *indices.stride(),
            group_sizes, *group_sizes.stride(),
            
            dupped_indices, *dupped_indices.stride(),
            dupped_group_sizes, *dupped_group_sizes.stride(),
            
            topk_indices, *topk_indices.stride(),
            
            t_group_sizes, *t_group_sizes.stride(),
            
            G, mask_block_k, 
            
            BLOCK_BK,
        )
        
        if branch_method == 'random':
            t_group_sizes.mul_(0.7)
            max_group_size = max_group_size * 0.7
        else:
            t_group_sizes.mul_(0.5)
            max_group_size = max_group_size * 0.5
        # print('a', max_group_size, t_group_sizes.max())
    
    # torch.cuda.synchronize()
    # print('done', flush=True)
    
    # print(indices[0, :5])
    # print(ks[0, :5])
    
    indices.mul_(block_size_k)
    
    # NOTE: before this sort, indices are sorted by imporatnce of each block
    indices, _ = torch.sort(indices, dim=-1, stable=False)
    
    if G > 1:
        ks_count = torch.zeros((B, BDST, G), dtype=torch.int32, device=q.device)
        ks_start_end = torch.zeros((B, BDST, G + 1), dtype=torch.int32, device=q.device)
        
        BLOCK_BK = 128
        grid = (B, BDST, triton.cdiv(indices.shape[-1], BLOCK_BK))
        masking_iteration_draft_cuda_epiloge[grid](
            indices, *indices.stride(),
            ks, *ks.stride(),
            
            ks_count, *ks_count.stride(),
            ks_start_end, *ks_start_end.stride(),
            
            mask_block_k, TSRC, 
            
            G,
            BLOCK_BK,
        )
        # print(indices[0, -1] // TSRC)
        # print(ks_count[0, -1], ks_start_end[0, -1])
        # print(ks_count.float().mean(1).int()[0])
    else:
        ks_count = ks[:, :, None]
        ks_start_end = torch.zeros((B, BDST, G + 1), dtype=torch.int32, device=q.device)
        ks_start_end[:, :, -1] = ks
        
    # assert indices[0, -10].shape == torch.unique(indices[0, -10]).shape, f'{indices[0, -10].shape} == {torch.unique(indices[0, -10]).shape}'
    # t = indices[0, 16]
    # c = ks[0, 16]
    # tu = torch.unique(t)
    # print(t)
    # print(tu)
    # print(t.shape, tu.shape, c)
    
    return indices, ks, ks_count, ks_start_end

@numba.njit(parallel=True)
def block_sparse_attention_numba(
    q: NdArray,
    k: NdArray,
    v: NdArray,
    
    indices: NdArray,
    ks: NdArray,
    
    block_size_q: int,
    block_size_k: int,
    mask_k: int,
    
    topk_head_group_size: int,
) -> NdArray:
    output = np.zeros_like(q)
    
    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape
    
    G = topk_head_group_size
    B = N // G
    assert N == (B * G)
    
    _, BDST = ks.shape
    _, _, GKS = indices.shape
    
    for idx_n in numba.prange(B):
        for idx_bdst in numba.prange(BDST):
            # compute scores for each index
            idx_tdst = idx_bdst * block_size_q
            scores = np.zeros((block_size_q, GKS * block_size_k), dtype=q.dtype) - 32000.0
            for i in range(min(ks[idx_n, idx_bdst], GKS)):
                idx_index = indices[idx_n, idx_bdst, i]
                idx_group = idx_index // TSRC
                idx_tsrc = idx_index % TSRC
                queries = q[idx_n * G + idx_group, idx_tdst: idx_tdst + block_size_q, :]
                keys = k[idx_n * G + idx_group, idx_tsrc: idx_tsrc + block_size_k, :]
                t = np.ascontiguousarray(queries) @ np.ascontiguousarray(keys.T)
                for m in range(t.shape[0]):
                    for n in range(t.shape[1]):
                        if (idx_tsrc + n) > (idx_tdst + TSRC - TDST + m):
                            t[m, n] = -32000.0
                scores[:, i * block_size_k: (i + 1) * block_size_k] = t
            # compute exp
            scores_max = np.max(scores)
            scores = np.exp(scores - scores_max).astype(np.float32)
            # sum of each head
            scores_sum = np.zeros((topk_head_group_size, block_size_q), dtype=q.dtype)
            for i in range(min(ks[idx_n, idx_bdst], GKS)):
                idx_index = indices[idx_n, idx_bdst, i]
                idx_group = idx_index // TSRC
                scores_sum[idx_group, :] += np.sum(
                    scores[:, i * block_size_k: (i + 1) * block_size_k],
                    axis=-1
                )
            # divide by sum of each
            for i in range(min(ks[idx_n, idx_bdst], GKS)):
                idx_index = indices[idx_n, idx_bdst, i]
                idx_group = idx_index // TSRC
                for j in range(block_size_k):
                    scores[:, i * block_size_k + j] /= \
                        (scores_sum[idx_group, :] + 1e-12)
            # multiply and acc for each head
            for i in range(min(ks[idx_n, idx_bdst], GKS)):
                idx_index = indices[idx_n, idx_bdst, i]
                idx_group = idx_index // TSRC
                idx_tsrc = idx_index % TSRC
                values = np.ascontiguousarray(v[idx_n * G + idx_group, idx_tsrc: idx_tsrc + block_size_k, :])
                t = np.ascontiguousarray(scores[:, i * block_size_k: (i + 1) * block_size_k]) @ values
                output[
                    idx_n * G + idx_group, 
                    idx_bdst * block_size_q: (idx_bdst + 1) * block_size_q, 
                    :
                ] += t
    
    return output

@triton.jit
def block_sparse_attention_cuda_step(
    # QKV
    queries,
    keys,
    values,
    
    #indices
    idx_tsrc, mask_tsrc,
    idx_tdst, mask_tdst,
    
    # rolling value
    acc, l_i, m_i,
    
    TDST,
    TSRC,
    
    sliding_window_size,
    EXCLUDE_SLIDING_WINDOW: tl.constexpr,
    
    USING_EXTEND: tl.constexpr,
    extend_window_size,
    extend_group_size,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    pos_tdst,
    idx_hid, HID: tl.constexpr, BLOCK_TQ, BLOCK_TK,
):
    # keys := [BLOCK_HID: hid, BLOCK_BK * BLOCK_SIZE_K: tsrc]
    # queries := [BLOCK_SIZE_Q: tdst, BLOCK_HID: hid]
    # scores := [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]

    # keys = tl.load(
    #     K +\
    #         (idx_n // KV_REPEAT_INTERLEAVE) * stride_k_n +\
    #         idx_tsrc[None, :] * stride_k_tsrc +\
    #         idx_hid[:, None] * stride_k_hid,
    #     mask = mask_tsrc[None, :] & mask_hid[:, None],
    #     other = 0,
    # )
    
    # queries_max = tl.maximum(1.0, tl.max(tl.abs(queries)).to(tl.float32))
    # keys_max = tl.maximum(1.0, tl.max(tl.abs(keys)).to(tl.float32))
    # queries_scale = (1.0 / queries_max)
    # keys_scale = (1.0 / keys_max)
    # qk = tl.dot(
    #     # (queries * queries_scale).to(queries.dtype),
    #     # (keys * keys_scale).to(keys.dtype),
    #     queries, keys,
    #     allow_tf32=True,
    # ).to(tl.float32) * 1.44269504 # * queries_max * keys_max)
    
    if USING_EXTEND:
        assert COS is not None
        assert SIN is not None
        
        old_tsrc = idx_tsrc
        mask_tsrc_window = idx_tsrc >= (tl.min(tl.where(mask_tdst, pos_tdst, 9999999)) - extend_window_size)
        new_tsrc = tl.where(
            mask_tsrc_window,
            old_tsrc,
            old_tsrc // extend_group_size
        )
        
        keys = keys.trans(1, 0)
        keys = adjust_rope(
            keys, old_tsrc, new_tsrc, idx_hid,
            COS, stride_cos_t, stride_cos_hid,
            SIN, stride_sin_t, stride_sin_hid,
            BLOCK_TK, HID,
        )
        keys = tl.trans(keys, 1, 0)
        keys = keys * mask_tsrc[None, :]
        
        old_tdst = pos_tdst
        new_tdst = old_tdst // extend_group_size
        
        queries_grouped = adjust_rope(
            queries, old_tdst, new_tdst, idx_hid,
            COS, stride_cos_t, stride_cos_hid,
            SIN, stride_sin_t, stride_sin_hid,
            BLOCK_TQ, HID,
        )
        queries_grouped = queries_grouped * mask_tdst[:, None]
        
        t_window = tl.dot(
            queries, keys.to(queries.dtype),
            allow_tf32=True,
        )
        t_grouped = tl.dot(
            queries_grouped.to(queries.dtype), keys.to(queries.dtype),
            allow_tf32=True,
        )
        qk = tl.where(
            mask_tsrc_window[None, :],
            t_window,
            t_grouped,
        ).to(tl.float32) * 1.44269504
    else:
        qk = tl.dot(
            queries, keys,
            allow_tf32=True,
        ).to(tl.float32) * 1.44269504
    
    # qk_mask = (
    #     ((idx_tdst[:, None] + TSRC - TDST) < (idx_tsrc)[None, :]) |
    #     (~(mask_tdst[:, None] & mask_tsrc[None, :]))
    # )
    
    if EXCLUDE_SLIDING_WINDOW:
        qk_mask = (
            ((idx_tdst[:, None] + TSRC - TDST) < (idx_tsrc + sliding_window_size)[None, :]) |
            (~(mask_tdst[:, None] & mask_tsrc[None, :]))
        )
    else:
        qk_mask = (
            ((idx_tdst[:, None] + TSRC - TDST) < idx_tsrc[None, :]) |
            ((idx_tdst[:, None] + TSRC - TDST) >= (idx_tsrc + sliding_window_size)[None, :]) |
            (~(mask_tdst[:, None] & mask_tsrc[None, :]))
        )
    
    qk += qk_mask * (-1.0e+6)
    
    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1)[:, None])
    qk = qk - m_ij
    # [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]
    p = tl.math.exp2(qk)
    
    p *= ~qk_mask
    
    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    l_ij = tl.sum(p, axis=1)
    
    # -- update m_i and l_i
    alpha = tl.math.exp2(m_i - m_ij)
    # tl.device_print('ff', l_ij)
    l_i = l_i * alpha + l_ij[:, None]
    
    # -- update output accumulator --
    acc = acc * alpha
    
    # values = tl.load(
    #     V +\
    #         (idx_n // KV_REPEAT_INTERLEAVE) * stride_v_n +\
    #         idx_tsrc[:, None] * stride_v_tsrc +\
    #         idx_hid[None, :] * stride_v_hid,
    #     mask = mask_tsrc[:, None] & mask_hid[None, :],
    #     other = 0
    # )
    
    # update acc
    acc += tl.dot(p.to(values.dtype), values).to(tl.float32)
    
    # update m_i and l_i
    m_i = m_ij
    
    return acc, l_i, m_i

@triton.jit
def block_sparse_attention_cuda(
    Q, stride_q_n, stride_q_tdst, stride_q_hid,
    K, stride_k_n, stride_k_tsrc, stride_k_hid,
    V, stride_v_n, stride_v_tsrc, stride_v_hid,
    
    INDICES, 
    stride_indices_b, stride_indices_bdst, stride_indices_bk,
    
    KS_START_END,
    stride_ks_start_end_b, stride_ks_start_end_bdst, stride_ks_start_end_g,
    
    CONTEXT,
    stride_context_n, stride_context_tdst, stride_context_hid,
    
    G, BK, MAX_TDST, MAX_TSRC,
    
    sliding_window_size: int,
    
    USING_EXTEND: tl.constexpr,
    extend_window_size,
    extend_group_size,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    
    HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
):
    idx_b = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_tdst = BLOCK_SIZE_Q * idx_bdst + tl.arange(0, BLOCK_SIZE_Q)
    mask_tdst = idx_tdst < MAX_TDST
    idx_g = tl.program_id(2)
    idx_n = idx_b * G + idx_g
    
    acc = tl.zeros((BLOCK_SIZE_Q, HID), dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_Q, 1), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_SIZE_Q, 1), 1.0, dtype=tl.float32)
    
    range_start = tl.load(
        KS_START_END + \
            idx_b * stride_ks_start_end_b +\
            idx_bdst * stride_ks_start_end_bdst +\
            idx_g * stride_ks_start_end_g
    )
    range_end = tl.load(
        KS_START_END + \
            idx_b * stride_ks_start_end_b +\
            idx_bdst * stride_ks_start_end_bdst +\
            (idx_g + 1) * stride_ks_start_end_g
    )
    
    idx_hid = tl.arange(0, HID)
    
    queries = tl.load(
        Q +\
            idx_n * stride_q_n +\
            idx_tdst[:, None] * stride_q_tdst +\
            idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0,
    )
    
    for i_bk in range(range_start, range_end, BLOCK_BK):
        idx_bk = i_bk + tl.arange(0, BLOCK_BK)
        mask_bk = idx_bk < (BK * G)
        
        idx_tsrc_start = tl.load(
            INDICES +\
                idx_b * stride_indices_b +\
                idx_bdst * stride_indices_bdst +\
                idx_bk * stride_indices_bk,
            mask=mask_bk,
            # other=(MAX_TSRC + 1) * G,
        )
        idx_tsrc_start = tl.where(mask_bk, idx_tsrc_start, MAX_TSRC * G + 1)
        idx_tsrc = idx_tsrc_start[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :]
        idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K))
        mask_tsrc = (idx_tsrc < (MAX_TSRC * (idx_g + 1))) & (idx_tsrc >= (MAX_TSRC * idx_g))
        # mask_tsrc = True
        # mask_tsrc = idx_tsrc > 0
        # idx_group = idx_tsrc // MAX_TSRC
        idx_tsrc = idx_tsrc % MAX_TSRC
        
        # idx_n = idx_b * G + idx_group
        keys = tl.load(
            K +\
                idx_n * stride_k_n +\
                idx_tsrc[None, :] * stride_k_tsrc +\
                idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc[None, :],
            other=0
        )
        values = tl.load(
            V +\
                idx_n * stride_v_n +\
                idx_tsrc[:, None] * stride_v_tsrc +\
                idx_hid[None, :] * stride_v_hid,
            mask=mask_tsrc[:, None],
            other=0
        )
        
        acc, l_i, m_i = block_sparse_attention_cuda_step(
            queries,
            keys,
            values,
            
            idx_tsrc, mask_tsrc,
            idx_tdst, mask_tdst,
            
            acc, l_i, m_i,
            
            MAX_TDST,
            MAX_TSRC,
            
            sliding_window_size,
            True,
            
            USING_EXTEND,
            extend_window_size,
            extend_group_size,
            COS, stride_cos_t, stride_cos_hid,
            SIN, stride_sin_t, stride_sin_hid,
            idx_tdst + MAX_TSRC - MAX_TDST,
            idx_hid, HID, 
            BLOCK_SIZE_Q, 
            BLOCK_BK * BLOCK_SIZE_K,
        )
    
    CURR_TSRC = (idx_bdst + 1) * BLOCK_SIZE_Q + MAX_TSRC - MAX_TDST
    for i_tsrc in range(tl.maximum(0, CURR_TSRC - sliding_window_size - BLOCK_SIZE_Q), CURR_TSRC, BLOCK_BK):
        idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK)
        mask_tsrc = idx_tsrc < MAX_TSRC
        
        # idx_n = idx_b * G + idx_group
        keys = tl.load(
            K +\
                idx_n * stride_k_n +\
                idx_tsrc[None, :] * stride_k_tsrc +\
                idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc[None, :],
            other=0
        )
        values = tl.load(
            V +\
                idx_n * stride_v_n +\
                idx_tsrc[:, None] * stride_v_tsrc +\
                idx_hid[None, :] * stride_v_hid,
            mask=mask_tsrc[:, None],
            other=0
        )
        
        acc, l_i, m_i = block_sparse_attention_cuda_step(
            queries,
            keys,
            values,
            
            idx_tsrc, mask_tsrc,
            idx_tdst, mask_tdst,
            
            acc, l_i, m_i,
            
            MAX_TDST,
            MAX_TSRC,
            
            sliding_window_size,
            False,
            
            USING_EXTEND,
            extend_window_size,
            extend_group_size,
            COS, stride_cos_t, stride_cos_hid,
            SIN, stride_sin_t, stride_sin_hid,
            idx_tdst + MAX_TSRC - MAX_TDST,
            idx_hid, HID, 
            BLOCK_SIZE_Q, 
            BLOCK_BK,
        )
    
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = (acc / (tl.where(l_i == 0.0, 1e-20, l_i)))
    
    tl.store(
        CONTEXT +\
            idx_n * stride_context_n +\
            idx_tdst[:, None] * stride_context_tdst +\
            idx_hid[None, :] * stride_context_hid,
        mask = mask_tdst[:, None],
        value = acc.to(CONTEXT.type.element_ty),
        # value = l_i
    )

def block_sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    
    indices: Tensor,
    ks: Tensor,
    ks_count: Tensor,
    ks_start_end: Tensor,
    
    block_size_q: int,
    block_size_k: int,
    mask_k: int,
    sliding_window_size: int,
    
    topk_head_group_size: int,
    
    using_extend: bool,
    extend_window_size: int,
    extend_group_size: int,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
):
    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape
    assert q.shape == k.shape
    BDST = cdiv_python(TDST, block_size_q)
    BSRC = cdiv_python(TSRC, block_size_k)
    
    G = topk_head_group_size
    B = N // G
    assert (B * G) == N
    BK = cdiv_python(mask_k, block_size_k)
    
    context = torch.zeros_like(q)
    
    BLOCK_BK = 64 // block_size_k
    assert BLOCK_BK > 0
    
    if rope_cos is not None:
        assert len(rope_cos.stride()) == 2
        assert len(rope_sin.stride()) == 2
    
    grid = (B, BDST, G)
    block_sparse_attention_cuda[grid](
        q, *q.stride(),
        k, *k.stride(),
        v, *v.stride(),
        
        indices, *indices.stride(),
        
        ks_start_end, *ks_start_end.stride(),
        
        context, *context.stride(),
        
        G, BK, TDST, TSRC,
        
        sliding_window_size,
        
        using_extend,
        extend_window_size,
        extend_group_size,
        rope_cos, *(rope_cos.stride() if rope_cos is not None else (0, 0)),
        rope_sin, *(rope_sin.stride() if rope_sin is not None else (0, 0)),
        
        HID,
        block_size_q,
        block_size_k,
        BLOCK_BK,
        
        num_stages=1,
    )
    
    return context

@torch.inference_mode()
def hip_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor,
    
    mask_k: int = 512,
    
    block_size_q: int = 32,
    block_size_k: int = 1,
    block_size_k_group: int = 8,
    
    sliding_window_size: int = 128,
    sink_token_size: int = 4,
    
    using_extend: bool = False,
    rope_cos: Optional[Tensor] = None,
    rope_sin: Optional[Tensor] = None,
    self_extend_neighboor_window: int = 1024,
    self_extend_group_size: int = 8,
    topk_head_group_size: int = 8,
    sample_method: str = 'first',
    branch_method: str = 'half',
    
    traverse_from_last_step: bool = True,
    step_size: int = 1,
    chunk_size: Optional[int] = None,
    num_samples: int = 1,
    
    score_head_group_size: int = 1,
):
    indices_blocks = []
    ks_blocks = []
    ks_count_blocks = []
    ks_start_end_blocks = []
    indices_seed = ks_seed = None
    for idx_tdst in range(0, q.shape[1], block_size_q * step_size):
        for idx_sample in range(num_samples):
            indices, ks, ks_count, ks_start_end = masking_iteration_draft(
                q[:, idx_tdst: idx_tdst+block_size_q * step_size, :], 
                k[:, :, :], 
                position_ids=torch.arange(
                    idx_tdst, 
                    idx_tdst+block_size_q * step_size, 
                    device=q.device
                ),
                mask_k=mask_k,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                block_size_k_group=block_size_k_group,
                sliding_window_size=sliding_window_size,
                sink_token_size=sink_token_size,
                using_extend=using_extend,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                self_extend_neighboor_window=self_extend_neighboor_window,
                self_extend_group_size=self_extend_group_size,
                topk_head_group_size=topk_head_group_size,
                sample_method=sample_method,
                branch_method=branch_method,
                score_head_group_size=score_head_group_size,
                indices_seed=indices_seed,
                ks_seed=ks_seed,
            )
            indices_seed = indices
            ks_seed = ks
        
        if not traverse_from_last_step:
            indices_seed = ks_seed = None
        if (chunk_size is not None) and (((idx_tdst // block_size_q + 1) % (chunk_size // block_size_q)) == 0):
            indices_seed = ks_seed = None
        
        indices_blocks.append(indices)
        ks_blocks.append(ks)
        ks_count_blocks.append(ks_count)
        ks_start_end_blocks.append(ks_start_end)
    indices = torch.cat(indices_blocks, dim=1)
    ks = torch.cat(ks_blocks, dim=1)
    ks_count = torch.cat(ks_count_blocks, dim=1)
    ks_start_end = torch.cat(ks_start_end_blocks, dim=1)
    
    # print(indices[0, 4+16-1])
    # print(indices[0, 4+16])
    # print(indices[0, 4+16+1])
    
    # print(ks[0, 4+16-1])
    # print(ks[0, 4+16])
    # print(ks[0, 4+16+1])
    
    if os.getenv('HIP_DEBUG', '0') == '1':
        N, TDST, HID = q.shape
        _, TSRC, _ = k.shape
        debug_mask = to_dense(
            indices.cpu().numpy(),
            ks.cpu().numpy(),
            None,
            cdiv_python(N, topk_head_group_size),
            TDST, 
            TSRC * topk_head_group_size, 
            block_size_q, 
            block_size_k * block_size_k_group,
        )
        # print(debug_mask)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4*topk_head_group_size, 4))
        plt.imshow(debug_mask[0])
        plt.savefig('dummy.png', dpi=96)
        print('saved dummy.png')
        # input()
    
    context = block_sparse_attention(
        q, k, v, 
        
        indices, ks, ks_count, ks_start_end,
        
        block_size_q, 
        block_size_k, 
        mask_k, 
        sliding_window_size,
        
        topk_head_group_size,
        
        using_extend,
        # False,
        self_extend_neighboor_window,
        self_extend_group_size,
        rope_cos,
        rope_sin,
    )
    
    # print(context[0, :320, 0])
    
    return context, None
    
    # context = block_sparse_attention_numba(
    #     q.cpu().float().numpy(), 
    #     k.cpu().float().numpy(), 
    #     v.cpu().float().numpy(),
        
    #     indices.cpu().long().numpy(), 
    #     ks.cpu().long().numpy(),
        
    #     block_size_q=block_size_q,
    #     block_size_k=block_size_k,
    #     mask_k=mask_k,
        
    #     topk_head_group_size=topk_head_group_size,
    # )
    # context = torch.tensor(context, device=q.device)
    
    # return context, None

if __name__ == '__main__':
    q, k, v, out, cos, sin = load_checkouts(idx=0, window=40, seq_len=4096, return_cos_sin=True, dtype=torch.float32)
    
    # q = q[:, -32:, :]
    # out = out[:, -32:, :]
    
    os.environ['HIP_DEBUG'] = '1'
    
    context, _ = hip_attention(
        q, k, v, 
        
        mask_k=512, 
        
        block_size_k=4,
        block_size_k_group=1,
        block_size_q=32,
        
        sliding_window_size=128,
        sink_token_size=16,
        
        using_extend=True,
        rope_cos=cos,
        rope_sin=sin,
        self_extend_neighboor_window=1024,
        self_extend_group_size=4,
        
        topk_head_group_size=1,
        sample_method='first',
        branch_method='half',
        
        traverse_from_last_step=True,
        num_samples=2,
        
        score_head_group_size=1,
    )
    
    if context is not None:
        stderr = (out - context).abs().mean().item()
        stdcontext = torch.std_mean(out)[0].item()
        
        print(f'err = {stderr:.8f} ({stderr/stdcontext:.6f} sigma), out_std = {stdcontext:.8f}')