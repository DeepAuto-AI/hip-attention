"""
HiP v1.1
TODO:
1. Masking iteration using integer to avoid aliasing and collision
 - Convert tmask into int32 (good!)
 - Reuse the computed dot products (good!)
2. Using QUEST method for b_k (not very good)
3. Maximum token location predictor
 - Test oracle (not very good, sometimes worse)
 - Test estimators
4. sifters? (not very good) (num_unions, num_samples handle this)
5. masking -> allocate cells (num_samples, traverse_from_last_step)
6. StreamLLM based traverse (use Self-Extend instead of SLLM)
7. causal-batch (fine, topk_head_group_size)
8. 2d support
9. support backward across tree
10. chunk-wise BPTT
"""

import copy
import cv2
import matplotlib.pyplot as plt
import numba
from dataclasses import dataclass
from importlib import metadata
import nvtx
import cupy as cp
import random, os
import warnings
import tqdm
import triton
import triton.language as tl
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union
from torch import Tensor
from hip.models.hip_attention.attention1_block_gpu import load_checkouts, to_dense
import numpy as np
from numpy import ndarray as NdArray
import math
from hip.utils.triton_argsort import argsort as tl_argsort

def cdiv_python(a, b):
    return math.ceil(float(a) / float(b))

DEFAULT_CACHE_MODIFIER = tl.constexpr('.cg')

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
    POS,
    stride_pos_n,
    stride_pos_tdst,
    
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
    
    G, MAX_TDST, MAX_TSRC, HEAD,
    
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
            (idx_b * G // HEAD) * stride_pos_n +\
            idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    TSRC = tl.max(pos_tdst)
    tl.debug_barrier()
    TSRC = tl.maximum(0, TSRC - sliding_window_size)
    BSRC = tl.cdiv(TSRC, block_size_k)
    MAX_BSRC = tl.cdiv(MAX_TSRC, block_size_k)
    
    if TSRC <= mask_k:
        idx_bk = tl.arange(0, BLOCK_MASK_BLOCK_K)
        mask_bk = idx_bk < BSRC
        if INDICES is not None:
            tl.store(
                INDICES +\
                    idx_b * stride_indices_b +\
                    idx_bdst * stride_indices_bdst +\
                    (idx_group * BSRC + idx_bk) * stride_indices_bk,
                value = idx_group * MAX_BSRC + idx_bk,
                mask = mask_bk,
            )
        
        if idx_group == 0:
            if KS is not None:
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
            ).to(tl.int32)
        
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
                    mask=mask_bk,
                    other=idx_group * MAX_BSRC,
                ).to(tl.int32)
                indices_next = tl.load(
                    INDICES_SEED +\
                        idx_b * stride_indices_seed_b +\
                        idx_bdst * stride_indices_seed_bdst +\
                        (idx_group * mask_block_k + idx_bk + 1) * stride_indices_seed_bk,
                    mask=(
                        mask_bk &
                        ((idx_group * mask_block_k + idx_bk + 1) < (BLOCK_MASK_BLOCK_K * G))
                    ),
                    other=G * MAX_BSRC,
                ).to(tl.int32)
                indices_group_id = indices // MAX_BSRC
                indices_next_group_id = indices_next // MAX_BSRC
                group_sizes = tl.where(
                    indices_group_id == indices_next_group_id,
                    indices_next - indices,
                    indices_group_id * MAX_BSRC + BSRC - indices,
                ).to(tl.int32)
        
        if INDICES is not None:
            tl.store(
                INDICES +\
                    idx_b * stride_indices_b +\
                    idx_bdst * stride_indices_bdst +\
                    (idx_group * mask_block_k + idx_bk) * stride_indices_bk,
                value=indices,
                mask=mask_bk,
            )
        if GROUP_SIZE is not None:
            tl.store(
                GROUP_SIZE +\
                    idx_b * stride_group_size_b +\
                    idx_bdst * stride_group_size_bdst +\
                    (idx_group * mask_block_k + idx_bk) * stride_group_size_bk,
                value=group_sizes,
                mask=mask_bk,
            )
        
        if T_GROUP_SIZE is not None:
            tl.atomic_max(
                T_GROUP_SIZE +\
                    idx_b * stride_t_group_size_b +\
                    idx_bdst * stride_t_group_size_bdst,
                val = tl.max(group_sizes)
                # value = tl.minimum(
                #     tl.max(group_sizes), 
                #     tl.maximum(tl.cdiv(BSRC, mask_block_k), 8)
                # )
            )
        if KS is not None:
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
def load_tokens(
    K, 
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE, 
    stride_k_cache_page, 
    stride_k_cache_offset, 
    stride_k_cache_kv_head, 
    stride_k_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    
    idx_bsz,
    idx_tsrc,
    idx_kv_head,
    idx_hid,
    
    mask_keys,
):
    if not USING_PAGES:
        keys = tl.load(
            K +\
                idx_bsz.to(tl.int64) * stride_k_bsz +\
                idx_tsrc.to(tl.int64) * stride_k_tsrc +\
                idx_kv_head.to(tl.int64) * stride_k_head +\
                idx_hid.to(tl.int64) * stride_k_hid,
            mask = mask_keys,
            other = 0,
            # cache_modifier='.cs', # TODO: uncomment this
        )
    else:
        seq_len = tl.load(
            CACHE_SEQ_LENS +\
                idx_bsz.to(tl.int64) * stride_cache_seq_lens_b,
        )
        tl.debug_barrier()
        mask_tsrc = idx_tsrc < seq_len
        tl.debug_barrier()
        ptrs = BLOCK_TABLE +\
            idx_bsz.to(tl.int64) * stride_block_table_bsz + \
            (idx_tsrc // PAGE_SIZE).to(tl.int64) * stride_block_table_page
        tl.debug_barrier()
        idx_page = tl.load(
            ptrs,
            mask=mask_tsrc,
            other=0,
        )
        offset_page = idx_tsrc % PAGE_SIZE
        
        keys = tl.load(
            K_CACHE +\
                idx_page.to(tl.int64) * stride_k_cache_page +\
                offset_page.to(tl.int64) * stride_k_cache_offset +\
                idx_kv_head.to(tl.int64) * stride_k_cache_kv_head +\
                idx_hid.to(tl.int64) * stride_k_cache_hid,
            mask=mask_keys,
            other=0,
        )
    
    if keys.dtype == tl.uint8:
        keys = keys.to(tl.float8e5, bitcast=True).to(tl.float16)
    
    return keys

@triton.jit
def masking_iteration_draft_cuda_dup_and_score_calc_score(
    dupped_indices_for_keys,
    KEY_DUP: tl.constexpr,
    
    Q, stride_q_bsz, stride_q_tdst, stride_q_bh, stride_q_g, stride_q_hid,
    K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    KEY_ACCESS_LOG, 
    stride_key_access_log_b, 
    stride_key_access_log_bdst, 
    stride_key_access_log_t,
    KEY_ACCESS_COUNT, 
    stride_key_access_count_b,
    stride_key_access_count_bdst, 
    MAX_ACCESS_COUNT,
    
    idx_b, 
    idx_bdst,
    idx_tdst, mask_tdst, pos_tdst,
    dupped_mask,
    
    BH: tl.constexpr,
    G: tl.constexpr, 
    MAX_TSRC, 
    HID: tl.constexpr,
    KV_HEAD_REPEAT: tl.constexpr,
    
    USING_EXTEND: tl.constexpr,
    extend_window_size,
    extend_group_size,
    
    USING_SPARQ: tl.constexpr,
    SPARQ_HID: tl.constexpr,
    Q_IND, 
    stride_q_ind_b, 
    stride_q_ind_g, 
    stride_q_ind_bdst, 
    stride_q_ind_k,
    
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE, 
    stride_k_cache_page, 
    stride_k_cache_offset, 
    stride_k_cache_kv_head, 
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    REDUCE_METHOD: tl.constexpr,
    
    NUM_CALIB: tl.constexpr = 32
):
    idx_tsrc = (
        (dupped_indices_for_keys * BLOCK_SIZE_K)[:, None]\
        + tl.arange(0, BLOCK_SIZE_K // BLOCK_STRIDE_K)[None, :] * BLOCK_STRIDE_K + BLOCK_STRIDE_K - 1
    )
    idx_tsrc = tl.ravel(idx_tsrc)
    idx_tsrc_grouped = idx_tsrc
    idx_group = idx_tsrc // MAX_TSRC
    idx_tsrc = idx_tsrc % MAX_TSRC
    idx_bsz = idx_b // BH
    idx_bh = idx_b % BH
    
    if KEY_ACCESS_LOG is not None:
        mask_access = tl.ravel(tl.broadcast_to(
            dupped_mask[:, None], 
            BLOCK_BK * KEY_DUP, BLOCK_SIZE_K // BLOCK_STRIDE_K
        ))
        len_access = tl.sum(mask_access.to(tl.int32))
        key_access_location = tl.atomic_add(
            KEY_ACCESS_COUNT +\
                idx_b * stride_key_access_count_b +\
                idx_bdst * stride_key_access_count_bdst,
            val=len_access,
        )
        idx_access = (key_access_location + tl.cumsum(mask_access.to(tl.int32)) - 1) % MAX_ACCESS_COUNT
        # idx_access = tl.arange(0, BLOCK_BK * KEY_DUP * BLOCK_SIZE_K // BLOCK_STRIDE_K)
        tl.store(
            KEY_ACCESS_LOG +\
                idx_b * stride_key_access_log_b +\
                idx_bdst * stride_key_access_log_bdst +\
                idx_access * stride_key_access_log_t,
            value=idx_tsrc_grouped,
            mask=mask_access,
            # eviction_policy='evict_first'
        )
    
    acc = tl.zeros((
        BLOCK_SIZE_Q // BLOCK_STRIDE_Q, 
        BLOCK_BK * KEY_DUP * BLOCK_SIZE_K // BLOCK_STRIDE_K
    ), dtype=tl.float32)
    idx_hid = tl.arange(0, HID)
    for i_group in tl.range(0, G):
        queries = tl.load(
            Q +\
                idx_bsz.to(tl.int64) * stride_q_bsz +\
                idx_tdst[:, None].to(tl.int64) * stride_q_tdst +\
                idx_bh.to(tl.int64) * stride_q_bh +\
                i_group.to(tl.int64) * stride_q_g +\
                idx_hid[None, :].to(tl.int64) * stride_q_hid,
            mask = mask_tdst[:, None],
            other = 0,
            # cache_modifier='.cs', # TODO: uncomment this (do not uncomment others)
            # eviction_policy='evict_last'
        )
        # queries = (idx_tdst[:, None] + idx_hid[None, :]).to(tl.float16)
        
        if queries.dtype == tl.uint8:
            queries = queries.to(tl.float8e5, bitcast=True).to(tl.float16)
        if G == 1:
            mask_keys = tl.broadcast_to(
                dupped_mask[:, None],
                BLOCK_BK * KEY_DUP, 
                BLOCK_SIZE_K // BLOCK_STRIDE_K
            )
            mask_keys = tl.ravel(mask_keys)[None, :]
            mask_keys = mask_keys & (idx_tsrc_grouped < MAX_TSRC)
        else:
            mask_keys = (
                dupped_mask[:, None] &\
                (idx_group == i_group).reshape(
                    BLOCK_BK * KEY_DUP, 
                    BLOCK_SIZE_K // BLOCK_STRIDE_K
                )
            )
            mask_keys = tl.ravel(mask_keys)[None, :]
        idx_head = idx_bh.to(tl.int64) * G + idx_group[None, :].to(tl.int64)
        idx_kv_head = idx_head // KV_HEAD_REPEAT
        keys = load_tokens(
            K, 
            stride_k_bsz, 
            stride_k_tsrc, 
            stride_k_head, 
            stride_k_hid,
            
            USING_PAGES, 
            PAGE_SIZE,
            K_CACHE,
            stride_k_cache_page,
            stride_k_cache_offset,
            stride_k_cache_kv_head,
            stride_k_cache_hid,
            BLOCK_TABLE,
            stride_block_table_bsz,
            stride_block_table_page,
            CACHE_SEQ_LENS,
            stride_cache_seq_lens_b,
            
            idx_bsz,
            idx_tsrc[None, :],
            idx_kv_head,
            idx_hid[:, None],
            mask_keys,
        )
        # keys = (idx_tsrc[None, :] + idx_hid[:, None]).to(tl.float16)
        if keys.dtype == tl.uint8:
            keys = keys.to(tl.float8e5, bitcast=True).to(tl.float16)
        
        if USING_EXTEND:
            if tl.min(pos_tdst) > (extend_window_size + NUM_CALIB // 2):
                assert COS is not None
                assert SIN is not None
                
                # dynamic_group_size = tl.maximum(1.0, tl.math.floor(tl.max(pos_tdst / 3072)))
                dynamic_group_size = extend_group_size
                
                idx_tsrc_calib = tl.maximum(0, tl.min(pos_tdst) - (extend_window_size + NUM_CALIB // 2))
                idx_tsrc_calib = idx_tsrc_calib + tl.arange(0, NUM_CALIB)
                mask_tsrc_calib = idx_tsrc_calib < MAX_TSRC
                keys_calib_old = tl.load(
                    K +\
                        idx_bsz.to(tl.int64) * stride_k_bsz +\
                        idx_tsrc_calib[None, :] * stride_k_tsrc +\
                        (idx_bh * BH + i_group) * stride_k_head +\
                        idx_hid[:, None] * stride_k_hid,
                    mask=mask_tsrc_calib[None, :],
                    other=0
                )
                
                keys_calib_new = adjust_rope(
                    keys_calib_old.trans(1, 0), 
                    idx_tsrc_calib, 
                    # idx_tsrc_calib // extend_group_size,
                    (idx_tsrc_calib / dynamic_group_size).to(tl.int32),
                    idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    NUM_CALIB, HID,
                ).trans(1, 0)
                
                old_tsrc = idx_tsrc
                mask_tsrc_window = idx_tsrc >= (tl.min(tl.where(mask_tdst, (pos_tdst - 1), 9999999)) - extend_window_size)
                new_tsrc = tl.where(
                    mask_tsrc_window,
                    old_tsrc,
                    # old_tsrc // extend_group_size
                    (old_tsrc / dynamic_group_size).to(tl.int32)
                )
                
                keys = keys.trans(1, 0)
                keys = adjust_rope(
                    keys, old_tsrc, new_tsrc, idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    BLOCK_BK * KEY_DUP * BLOCK_SIZE_K // BLOCK_STRIDE_K, HID,
                ).to(keys.dtype)
                keys = tl.trans(keys, 1, 0)
                keys = (keys * mask_keys).to(keys.dtype)
                
                old_tdst = (pos_tdst - 1)
                # new_tdst = old_tdst // extend_group_size
                new_tdst = (old_tdst / dynamic_group_size).to(tl.int32)
                
                queries_grouped = adjust_rope(
                    queries, old_tdst, new_tdst, idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    BLOCK_SIZE_Q // BLOCK_STRIDE_Q, HID,
                ).to(queries.dtype)
                
                t_calib_old = tl.dot(
                    queries, keys_calib_old.to(queries.dtype),
                )
                t_calib_new = tl.dot(
                    queries_grouped, keys_calib_new.to(queries.dtype),
                )
                
                calibration = tl.sum(t_calib_new - t_calib_old, axis=-1) / NUM_CALIB
                
                # calib_old_mean = tl.sum(t_calib_old, axis=-1) / NUM_CALIB
                # calib_old_std = tl.sqrt(tl.sum(tl.extra.cuda.libdevice.pow(t_calib_old - calib_old_mean[:, None], 2), axis=-1) / NUM_CALIB)
                # calib_new_mean = tl.sum(t_calib_new, axis=-1) / NUM_CALIB
                # calib_new_std = tl.sqrt(tl.sum(tl.extra.cuda.libdevice.pow(t_calib_new - calib_new_mean[:, None], 2), axis=-1) / NUM_CALIB)
                
                t_window = tl.dot(
                    queries, keys.to(queries.dtype),
                )
                
                t_grouped = tl.dot(
                    queries_grouped, keys.to(queries.dtype),
                )
                
                # NOTE: this calibration trick is very important.
                # > w/o std
                t_grouped = t_grouped - calibration[:, None]
                # > with std
                # t_grouped = ((t_grouped - calib_new_mean[:, None]) / calib_new_std[:, None]) * calib_old_std[:, None] + calib_old_mean[:, None]
                
                t = tl.where(
                    mask_tsrc_window[None, :],
                    t_window,
                    t_grouped,
                ).to(tl.float32)
            else:
                t = tl.dot(
                    queries.to(tl.float16),
                    keys.to(tl.float16),
                    out_dtype=tl.float16,
                ).to(tl.float32)
        else:
            if not USING_SPARQ:
                t = tl.dot(
                    queries.to(tl.float16), 
                    keys.to(tl.float16),
                    out_dtype=tl.float16,
                )
            else:
                idx_sparq_hid = tl.arange(0, SPARQ_HID)
                
                idx_sparq_hid = tl.load(
                    Q_IND +\
                        idx_b * stride_q_ind_b +\
                        i_group * stride_q_ind_g +\
                        idx_bdst * stride_q_ind_bdst +\
                        idx_sparq_hid * stride_q_ind_k
                )
                
                q_sparq = tl.load(
                    Q +\
                        idx_bsz * stride_q_bsz +\
                        idx_tdst[:, None] * stride_q_tdst +\
                        idx_bh * stride_q_bh +\
                        i_group * stride_q_g +\
                        idx_sparq_hid[None, :] * stride_q_hid,
                    mask = mask_tdst[:, None],
                    other = 0
                )
                k_sparq = tl.load(
                    K +\
                        idx_b * stride_k_bsz +\
                        idx_tsrc[None, :] * stride_k_tsrc +\
                        (idx_bh * BH + idx_group[None, :]) * stride_k_head +\
                        idx_sparq_hid[:, None] * stride_k_hid,
                    mask = mask_keys,
                    other = 0,
                )
                
                t = tl.dot(
                    q_sparq, 
                    k_sparq,
                ).to(tl.float32)
        acc += t.to(acc.dtype)
        # acc += tl.sum(queries)
        # acc += tl.sum(keys)
    acc = tl.where(
        (
            (acc == 0.0) |
            (idx_tsrc[None, :] > (pos_tdst - 1)[:, None]) |
            False
        ), 
        -32000.0 if REDUCE_METHOD == 'max' else 32000.0, 
        acc
    )
    scores = tl.reshape(
        acc, (
            BLOCK_SIZE_Q // BLOCK_STRIDE_Q, 
            BLOCK_BK * KEY_DUP, 
            BLOCK_SIZE_K // BLOCK_STRIDE_K
        )
    )
    if REDUCE_METHOD == 'max':
        scores = tl.max(
            scores,
            axis=0,
        )
        scores = tl.max(
            scores,
            axis=-1,
        )
    elif REDUCE_METHOD == 'min':
        scores = tl.min(
            scores,
            axis=0,
        )
        scores = tl.min(
            scores,
            axis=-1,
        )
    else:
        raise Exception()
    scores = tl.where(dupped_mask, scores, float('-inf'))
    
    return scores

# @triton.autotune(
#     configs=[
#         triton.Config({}, num_warps=1),
#         triton.Config({}, num_warps=2),
#         triton.Config({}, num_warps=4),
#         triton.Config({}, num_warps=8),
#         triton.Config({}, num_warps=16),
#     ],
#     key=[
#         'max_group_size', 
#         'i_iteration',
#         'BLOCK_BK'
#     ],
#     restore_value=[
#         'DUPPED_INDICES', 
#         'DUPPED_GROUP_SIZE', 
#         'SCORES',
#         'T_GROUP_SIZE',
#     ]
# )
@triton.jit
def masking_iteration_draft_cuda_dup_and_score(
    Q, stride_q_bsz, stride_q_tdst, stride_q_bh, stride_q_g, stride_q_hid,
    K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
    POS, stride_pos_bsz, stride_pos_tdst,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    KEY_ACCESS_LOG, 
    stride_key_access_log_b, 
    stride_key_access_log_bdst, 
    stride_key_access_log_t,
    KEY_ACCESS_COUNT,
    stride_key_access_count_b,
    stride_key_access_count_bdst,
    MAX_ACCESS_COUNT,
    
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
    SCORES_FINAL,
    stride_scores_final_b,
    stride_scores_final_bdst,
    stride_scores_final_bk,
    SCORES_CACHED: tl.constexpr,
    
    T_GROUP_SIZE, 
    stride_t_group_size_b, 
    stride_t_group_size_bdst,
    INDICES_TDST,
    stride_indices_tdst_t,
    
    mask_k,
    
    sliding_window_size,
    
    BH: tl.constexpr,
    G: tl.constexpr, 
    MAX_TDST, 
    MAX_TSRC, 
    BK, 
    HID: tl.constexpr,
    RAND_SEED,
    SAMPLE_METHOD: tl.constexpr,
    BRANCH_METHOD: tl.constexpr,
    KV_HEAD_REPEAT: tl.constexpr,
    
    USING_EXTEND: tl.constexpr,
    extend_window_size,
    extend_group_size,
    
    USING_SPARQ: tl.constexpr,
    SPARQ_HID: tl.constexpr,
    Q_IND, 
    stride_q_ind_b, 
    stride_q_ind_g, 
    stride_q_ind_bdst, 
    stride_q_ind_k,
    
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE, 
    stride_k_cache_page, 
    stride_k_cache_offset, 
    stride_k_cache_kv_head, 
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_bsz,
    
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    
    max_group_size, # just for autotune
    i_iteration, # just for autotune
    
    pid_0=None,
    pid_1=None,
    pid_2=None,
):
    if pid_2 is None:
        pid_b = tl.program_id(2)
    else:
        pid_b = pid_2
    
    if pid_1 is None:
        pid_bdst = tl.program_id(1)
    else:
        pid_bdst = pid_1
    
    if pid_0 is None:
        pid_bbk = tl.program_id(0)
    else:
        pid_bbk = pid_0
    
    idx_b = pid_b
    idx_bdst = pid_bdst
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
    idx_tdst_no_proj = idx_tdst
    mask_tdst = idx_tdst < MAX_TDST
    if INDICES_TDST is not None:
        idx_tdst = tl.load(
            INDICES_TDST +\
                idx_tdst.to(tl.int64) * stride_indices_tdst_t,
            mask=mask_tdst,
            other=MAX_TDST,
            cache_modifier=DEFAULT_CACHE_MODIFIER,
        ).to(tl.int64)
    
    idx_bk = pid_bbk * BLOCK_BK + tl.arange(0, BLOCK_BK)
    mask_bk = idx_bk < (BK * G)
    idx_bk_dup = pid_bbk * BLOCK_BK * 2 + tl.arange(0, BLOCK_BK * 2)
    mask_bk_dup = idx_bk_dup < (BK * 2 * G)
    idx_n = idx_b * G + tl.arange(0, G)
    
    mask_block_k = tl.cdiv(mask_k, BLOCK_SIZE_K)
    pos_tdst = tl.load(
        POS +\
            (idx_b // BH) * stride_pos_bsz +\
            idx_tdst_no_proj * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    TSRC = tl.max(pos_tdst)
    TSRC = tl.maximum(0, TSRC - sliding_window_size)
    BSRC = tl.cdiv(TSRC, BLOCK_SIZE_K)
    # MAX_BSRC = tl.cdiv(MAX_TSRC, BLOCK_SIZE_K)
    
    if TSRC <= mask_k:
        return
    
    t_group_size = tl.load(
        T_GROUP_SIZE +\
            idx_b * stride_t_group_size_b +\
            idx_bdst * stride_t_group_size_bdst,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
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
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    
    # int[BLOCK_BK]
    group_sizes = tl.load(
        GROUP_SIZE +\
            idx_b * stride_group_size_b +\
            idx_bdst * stride_group_size_bdst +\
            idx_bk * stride_group_size_bk,
        mask=mask_bk,
        other=0,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
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
    dupped_mask = (dupped_group_sizes > 0) & mask_bk_dup
    
    dupped_indices_for_keys = dupped_indices
    if SAMPLE_METHOD == 'random':
        offsets = tl.where(
            dupped_group_sizes > 4,
            0,
            (
                tl.randint(
                    RAND_SEED, 
                    dupped_indices + \
                        tl.program_id(0) * 31 + \
                        tl.program_id(1) * 7 + \
                        tl.program_id(2) * 1371
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
                
                Q, stride_q_bsz, stride_q_tdst, stride_q_bh, stride_q_g, stride_q_hid,
                K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                KEY_ACCESS_LOG, 
                stride_key_access_log_b, 
                stride_key_access_log_bdst, 
                stride_key_access_log_t,
                KEY_ACCESS_COUNT,
                stride_key_access_count_b,
                stride_key_access_count_bdst,
                MAX_ACCESS_COUNT,
                
                idx_b, 
                idx_bdst,
                idx_tdst, mask_tdst, pos_tdst,
                dupped_mask,
                
                BH, G, MAX_TSRC, HID, KV_HEAD_REPEAT,
                
                USING_EXTEND,
                extend_window_size,
                extend_group_size,
                
                USING_SPARQ,
                SPARQ_HID,
                Q_IND, 
                stride_q_ind_b, 
                stride_q_ind_g, 
                stride_q_ind_bdst, 
                stride_q_ind_k,
                
                # paged attention args template
                USING_PAGES,
                PAGE_SIZE,
                K_CACHE, 
                stride_k_cache_page, 
                stride_k_cache_offset, 
                stride_k_cache_kv_head, 
                stride_k_cache_hid,
                V_CACHE,
                stride_v_cache_page,
                stride_v_cache_offset,
                stride_v_cache_kv_head,
                stride_v_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_bsz,
                
                BLOCK_SIZE_Q,
                BLOCK_STRIDE_Q,
                BLOCK_SIZE_K,
                BLOCK_STRIDE_K,
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
    
    if SCORES_CACHED:
        cached_scores = tl.load(
            SCORES_FINAL +\
                idx_b * stride_scores_final_b+\
                idx_bdst * stride_scores_final_bdst+\
                idx_bk * stride_scores_final_bk,
            mask = mask_bk,
            cache_modifier=DEFAULT_CACHE_MODIFIER,
        )
        _, indices_to_sample = dupped_indices_for_keys\
            .reshape(BLOCK_BK, 2)\
            .split()
        _, mask_to_sample = dupped_mask\
            .reshape(BLOCK_BK, 2)\
            .split()
        
        # t1 = indices_to_sample.to(tl.uint16).to(tl.uint32)
        # t2 = mask_to_sample.to(tl.int1)
        # t3 = tl.arange(0, BLOCK_BK).to(tl.uint16).to(tl.uint32)
        # # t2 (1bit) | -- t3 (15bit) -- | -- t1 (16bit) --
        # t = (t2 << 31) | ((t3 << 17) >> 1) | t1
        
        # # _, indices_to_sample_sorted = tl_argsort(cached_scores, indices_to_sample, 0, False)
        # # _, mask_to_sample_sorted = tl_argsort(cached_scores, mask_to_sample.to(tl.int32), 0, False)
        # # _, mapping = tl_argsort(cached_scores, tl.arange(0, BLOCK_BK), 0, False)
        
        # _, t_sorted = tl_argsort(cached_scores, t, 0, False)
        # mask_to_sample_sorted = (t_sorted >> 31)
        # mapping = ((t_sorted << 1) >> 17).to(tl.int32)
        # indices_to_sample_sorted = ((t_sorted << 16) >> 16).to(tl.int32)
        
        # indices_to_sample_sorted, indices_to_not_sample_sorted = \
        #     indices_to_sample_sorted\
        #         .reshape(2, BLOCK_BK // 2)\
        #         .trans(1, 0)\
        #         .split()
        
        # mask_to_sample_sorted, mask_to_not_sample = \
        #     mask_to_sample_sorted\
        #         .reshape(2, BLOCK_BK // 2)\
        #         .trans(1, 0)\
        #         .split()
        # mask_to_sample_sorted = mask_to_sample_sorted.to(tl.int1)
        
        # indices_to_sample = indices_to_sample_sorted
        # mask_to_sample = mask_to_sample_sorted
        
        scores_sampled = masking_iteration_draft_cuda_dup_and_score_calc_score(
            indices_to_sample,
            1,
            
            Q, stride_q_bsz, stride_q_tdst, stride_q_bh, stride_q_g, stride_q_hid,
            K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
            COS, stride_cos_t, stride_cos_hid,
            SIN, stride_sin_t, stride_sin_hid,
            KEY_ACCESS_LOG, 
            stride_key_access_log_b, 
            stride_key_access_log_bdst, 
            stride_key_access_log_t,
            KEY_ACCESS_COUNT,
            stride_key_access_count_b,
            stride_key_access_count_bdst,
            MAX_ACCESS_COUNT,
            
            idx_b, 
            idx_bdst,
            idx_tdst, mask_tdst, pos_tdst,
            mask_to_sample,
            
            BH, G, MAX_TSRC, HID, KV_HEAD_REPEAT,
            
            USING_EXTEND,
            extend_window_size,
            extend_group_size,
            
            USING_SPARQ,
            SPARQ_HID,
            Q_IND, 
            stride_q_ind_b, 
            stride_q_ind_g, 
            stride_q_ind_bdst, 
            stride_q_ind_k,
            
            # paged attention args template
            USING_PAGES,
            PAGE_SIZE,
            K_CACHE, 
            stride_k_cache_page, 
            stride_k_cache_offset, 
            stride_k_cache_kv_head, 
            stride_k_cache_hid,
            V_CACHE,
            stride_v_cache_page,
            stride_v_cache_offset,
            stride_v_cache_kv_head,
            stride_v_cache_hid,
            BLOCK_TABLE,
            stride_block_table_bsz,
            stride_block_table_page,
            CACHE_SEQ_LENS,
            stride_cache_seq_lens_bsz,
            
            BLOCK_SIZE_Q,
            BLOCK_STRIDE_Q,
            BLOCK_SIZE_K,
            BLOCK_STRIDE_K,
            BLOCK_BK,
            # BLOCK_BK // 2,
            'max',
        )
        
        # scores_not_sampled = tl.full((BLOCK_BK // 2,), float('-inf'), dtype=scores_sampled.dtype)
        
        # scores_sorted = tl.join(scores_sampled, scores_not_sampled)\
        #     .trans(1, 0)\
        #     .reshape(BLOCK_BK)
        
        # _, scores_sampled = tl_argsort(mapping, scores_sorted.to(tl.float32).to(tl.int32, bitcast=True), 0, False)
        # scores_sampled = scores_sampled.to(tl.float32, bitcast=True)
        
        scores = tl.join(
            cached_scores.to(SCORES.dtype.element_ty), 
            scores_sampled.to(SCORES.dtype.element_ty)
        ).reshape(BLOCK_BK * 2)
    else:
        indices_to_sample = dupped_indices_for_keys
        mask_to_sample = dupped_mask

        scores_sampled = masking_iteration_draft_cuda_dup_and_score_calc_score(
            indices_to_sample,
            2,
            
            Q, stride_q_bsz, stride_q_tdst, stride_q_bh, stride_q_g, stride_q_hid,
            K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
            COS, stride_cos_t, stride_cos_hid,
            SIN, stride_sin_t, stride_sin_hid,
            KEY_ACCESS_LOG, 
            stride_key_access_log_b, 
            stride_key_access_log_bdst, 
            stride_key_access_log_t,
            KEY_ACCESS_COUNT,
            stride_key_access_count_b,
            stride_key_access_count_bdst,
            MAX_ACCESS_COUNT,
            
            idx_b, 
            idx_bdst,
            idx_tdst, mask_tdst, pos_tdst,
            mask_to_sample,
            
            BH, G, MAX_TSRC, HID, KV_HEAD_REPEAT,
            
            USING_EXTEND,
            extend_window_size,
            extend_group_size,
            
            USING_SPARQ,
            SPARQ_HID,
            Q_IND, 
            stride_q_ind_b, 
            stride_q_ind_g, 
            stride_q_ind_bdst, 
            stride_q_ind_k,
            
            # paged attention args template
            USING_PAGES,
            PAGE_SIZE,
            K_CACHE, 
            stride_k_cache_page, 
            stride_k_cache_offset, 
            stride_k_cache_kv_head, 
            stride_k_cache_hid,
            V_CACHE,
            stride_v_cache_page,
            stride_v_cache_offset,
            stride_v_cache_kv_head,
            stride_v_cache_hid,
            BLOCK_TABLE,
            stride_block_table_bsz,
            stride_block_table_page,
            CACHE_SEQ_LENS,
            stride_cache_seq_lens_bsz,
            
            BLOCK_SIZE_Q,
            BLOCK_STRIDE_Q,
            BLOCK_SIZE_K,
            BLOCK_STRIDE_K,
            BLOCK_BK,
            'max',
        )
        scores = scores_sampled.to(SCORES.dtype.element_ty)
    
    tl.store(
        SCORES +\
            idx_b * stride_scores_b +\
            idx_bdst * stride_scores_bdst +\
            idx_bk_dup * stride_scores_bk,
        value=scores,
        mask=mask_bk_dup,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    tl.store(
        DUPPED_INDICES +\
            idx_b * stride_dupped_indices_b +\
            idx_bdst * stride_dupped_indices_bdst +\
            idx_bk_dup * stride_dupped_indices_bk,
        value=dupped_indices,
        mask=mask_bk_dup,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    tl.store(
        DUPPED_GROUP_SIZE +\
            idx_b * stride_dupped_group_size_b +\
            idx_bdst * stride_dupped_group_size_bdst +\
            idx_bk_dup * stride_dupped_group_size_bk,
        value=dupped_group_sizes,
        mask=mask_bk_dup,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
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
    SCORES_FINAL,
    stride_scores_final_b,
    stride_scores_final_bdst,
    stride_scores_final_bk,
    
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
    
    TOPK_INDICES,
    stride_topk_indices_b,
    stride_topk_indices_bdst,
    stride_topk_indices_bk,
    
    T_GROUP_SIZE,
    stride_t_group_size_b, 
    stride_t_group_size_bdst,
    
    G: tl.constexpr, BK,
    
    BLOCK_BK: tl.constexpr,
    
    pid_0=None,
    pid_1=None,
    pid_2=None,
):
    if pid_0 is not None:
        pid_b = pid_2
        pid_bdst = pid_1
        pid_bk = pid_0
    else:
        pid_b = tl.program_id(2)
        pid_bdst = tl.program_id(1)
        pid_bk = tl.program_id(0)
    
    idx_b = pid_b
    idx_bdst = pid_bdst
    idx_bk = pid_bk * BLOCK_BK + tl.arange(0, BLOCK_BK)
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
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    
    dupped_indices = tl.load(
        DUPPED_INDICES +\
            idx_b * stride_dupped_indices_b +\
            idx_bdst * stride_dupped_indices_bdst +\
            topk_indices * stride_dupped_indices_bk,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    dupped_group_size = tl.load(
        DUPPED_GROUP_SIZE +\
            idx_b * stride_dupped_group_size_b +\
            idx_bdst * stride_dupped_group_size_bdst +\
            topk_indices * stride_dupped_group_size_bk,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    scores = tl.load(
        SCORES +\
            idx_b * stride_scores_b +\
            idx_bdst * stride_scores_bdst +\
            topk_indices * stride_scores_bk,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    
    tl.store(
        INDICES +\
            idx_b * stride_indices_b +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
        value=dupped_indices,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    tl.store(
        GROUP_SIZES +\
            idx_b * stride_group_sizes_b +\
            idx_bdst * stride_group_sizes_bdst +\
            idx_bk * stride_group_sizes_bk,
        value=dupped_group_size,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    tl.store(
        SCORES_FINAL +\
            idx_b * stride_scores_final_b +\
            idx_bdst * stride_scores_final_bdst +\
            idx_bk * stride_scores_final_bk,
        value=scores,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
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
    ).to(tl.int32)
    
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
    SCORES, 
    stride_scores_b, 
    stride_scores_bdst, 
    stride_scores_bk,
    DUPPED_INDICES, 
    stride_dupped_indices_b, 
    stride_dupped_indices_bdst, 
    stride_dupped_indices_bk,
    DUPPED_GROUP_SIZES,
    stride_dupped_group_sizes_b,
    stride_dupped_group_sizes_bdst,
    stride_dupped_group_sizes_bk,
    
    PROBS,
    stride_probs_b,
    stride_probs_bdst,
    stride_probs_bk,
    
    SINK_TOKEN_SIZE,
    MASK_BLOCK_K,
    G: tl.constexpr, BK, MAX_BSRC,
    BLOCK_SIZE_K,
    
    BLOCK_SCORE: tl.constexpr,
    
    pid_0 = None,
    pid_1 = None,
    CARRYING: tl.constexpr = False,
):
    if pid_0 is None:
        pid_0 = tl.program_id(0)
    if pid_1 is None:
        pid_1 = tl.program_id(1)
    
    idx_b = pid_1
    idx_bdst = pid_0
    idx_bk = tl.arange(0, BLOCK_SCORE)
    mask_bk = idx_bk < BK
    
    indices = tl.load(
        DUPPED_INDICES +\
            idx_b * stride_dupped_indices_b +\
            idx_bdst * stride_dupped_indices_bdst +\
            idx_bk * stride_dupped_indices_bk,
        mask=mask_bk,
        other=MAX_BSRC * G,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    group_sizes = tl.load(
        DUPPED_GROUP_SIZES +\
            idx_b * stride_dupped_group_sizes_b +\
            idx_bdst * stride_dupped_group_sizes_bdst +\
            idx_bk * stride_dupped_group_sizes_bk,
        mask=mask_bk,
        other=MAX_BSRC * G,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    groups = indices // MAX_BSRC
    scores = tl.load(
        SCORES +\
            idx_b * stride_scores_b +\
            idx_bdst * stride_scores_bdst +\
            idx_bk * stride_scores_bk,
        mask=mask_bk,
        other=float('-inf'),
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    ).to(tl.float32)
    
    one = tl.zeros((1, ), dtype=tl.float32) + 1
    for i_group in range(G):
        mask_softmax = groups == i_group
        scores_masked = tl.where(mask_softmax, scores, float('-inf'))
        if G == 1:
            scores_softmax = tl.sigmoid(scores_masked)
        else:
            count = tl.max(mask_softmax.to(tl.int32)).to(tl.float32)
            t = count / (BK * G)
            scores_softmax = tl.softmax(scores_masked * t)
            neg_scores_softmax_sorted = tl.sort(-scores_softmax)
            scores_promote_thresh = -tl.min(neg_scores_softmax_sorted * (tl.arange(0, BLOCK_SCORE) == (MASK_BLOCK_K * 0.5 * one).to(tl.int32)))
            scores_softmax = tl.where(scores_softmax >= scores_promote_thresh, scores_softmax + 1, scores_softmax)
        scores = tl.where(mask_softmax, scores_softmax, scores)
    
    scores = tl.where((indices % MAX_BSRC) < tl.cdiv(SINK_TOKEN_SIZE, BLOCK_SIZE_K), 2, scores)
    scores = tl.where(group_sizes == 0, -1, scores)
    
    tl.store(
        PROBS +\
            idx_b * stride_scores_b +\
            idx_bdst * stride_scores_bdst +\
            idx_bk * stride_scores_bk,
        value=scores,
        mask=mask_bk,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )

@triton.jit
def masking_iteration_draft_cuda_argsort(
    PROBS, stride_probs_b, stride_probs_bdst, stride_probs_bk,
    IDS, stride_ids_b, stride_ids_bdst, stride_ids_bk,
    
    T_GROUP_SIZES, stride_t_group_size_b, stride_t_group_size_bdst,
    
    BDST,
    
    BK: tl.constexpr,
    TOP_BK: tl.constexpr,
    BLOCK_BDST: tl.constexpr,
    
    pid_0=None,
    pid_1=None,
    CARRYING: tl.constexpr = False,
    carried_probs = None,
):
    if pid_0 is None:
        pid_0 = tl.program_id(0)
    if pid_1 is None:
        pid_1 = tl.program_id(1)
    
    idx_b = pid_1
    idx_bdst = pid_0 * BLOCK_BDST + tl.arange(0, BLOCK_BDST)
    mask_bdst = idx_bdst < BDST
    idx_bk = tl.arange(0, BK)
    
    t_group_size = tl.load(
        T_GROUP_SIZES +\
            idx_b * stride_t_group_size_b +\
            idx_bdst * stride_t_group_size_bdst,
        mask=mask_bdst,
        other=1.0,
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    if tl.max(t_group_size) < 1.0:
        return

    probs = tl.load(
        PROBS +\
            idx_b * stride_probs_b +\
            idx_bdst[:, None] * stride_probs_bdst +\
            idx_bk[None, :] * stride_probs_bk,
        mask=mask_bdst[:, None],
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )
    ids = tl.broadcast_to(tl.arange(0, BK)[None, :], (BLOCK_BDST, BK)).to(tl.int32)
    
    # ids_low, ids_high = tl.split(tl.reshape(ids, TOP_BK, 2))
    # probs_low, probs_high = tl.split(tl.reshape(probs.to(tl.float32), TOP_BK, 2))
    # probs_low, ids_low = tl_argsort(probs_low, ids_low, 0, True)
    # probs_high, ids_high = tl_argsort(probs_high, ids_high, 0, True)
    # tl.store(
    #     IDS +\
    #         idx_b * stride_ids_b +\
    #         idx_bdst[:, None] * stride_ids_bdst +\
    #         tl.arange(0, TOP_BK)[None, :] * stride_ids_bk,
    #     value=tl.where(
    #         probs_low > probs_high,
    #         ids_low,
    #         ids_high,
    #     )[None, :],
    #     mask=mask_bdst[:, None],
    #     cache_modifier=DEFAULT_CACHE_MODIFIER,
    # )
    
    _, ids = tl_argsort(probs.to(tl.float32), ids, 1, True)
    # ids, _ = tl.split(tl.trans(tl.reshape(ids, 2, TOP_BK), 1, 0))
    
    tl.store(
        IDS +\
            idx_b * stride_ids_b +\
            idx_bdst[:, None] * stride_ids_bdst +\
            idx_bk[None, :] * stride_ids_bk,
        value=ids,
        mask=(idx_bk < TOP_BK)[None, :] & mask_bdst[:, None],
        cache_modifier=DEFAULT_CACHE_MODIFIER,
    )

def masking_iteration_draft_python_epilog(
    indices: Tensor, ks: Tensor, 
    
    mask_block_k, MAX_TSRC,
    B, BDST, G
):
    if G > 1:
        ks_count = torch.zeros((B, BDST, G), dtype=torch.int32, device=indices.device)
        ks_start_end = torch.zeros((B, BDST, G + 1), dtype=torch.int32, device=indices.device)
        
        BLOCK_BK = 128
        grid = (B, BDST, triton.cdiv(indices.shape[-1], BLOCK_BK))
        pre_device = torch.get_default_device()
        torch.set_default_device(indices.device)
        masking_iteration_draft_cuda_epiloge[grid](
            indices, *indices.stride(),
            ks, *ks.stride(),
            
            ks_count, *ks_count.stride(),
            ks_start_end, *ks_start_end.stride(),
            
            mask_block_k, MAX_TSRC, 
            
            G,
            BLOCK_BK,
        )
        torch.set_default_device(pre_device)
        # print(indices[0, -1] // TSRC)
        # print(ks_count[0, -1], ks_start_end[0, -1])
        # print(ks_count.float().mean(1).int()[0])
        # if topk_indices is not None:
        #     scores_final = scores\
        #         .gather(index=topk_indices, dim=-1)\
        #         .gather(index=indices_sort_mapping, dim=-1)
        # else:
        #     scores_final = scores[:, :, :indices_sort_mapping.shape[-1]]\
        #         .gather(index=indices_sort_mapping, dim=-1)
    else:
        ks_count = ks[:, :, None]
        ks_start_end = torch.zeros((B, BDST, G + 1), dtype=torch.int32, device=indices.device)
        ks_start_end[:, :, -1] = ks
        # if topk_indices is not None:
        #     scores_final = scores\
        #         .gather(index=topk_indices, dim=-1)\
        #         .gather(index=indices_sort_mapping, dim=-1)
        # else:
        #     scores_final = scores[:, :, :indices_sort_mapping.shape[-1]]\
        #         .gather(index=indices_sort_mapping, dim=-1)
    
    return ks_count, ks_start_end

def get_masking_iteration_draft_cuda_fused_configs():
    warnings.warn('triton autotune will slow down startup!')
    configs = []
    for num_warps in [2, 4, 8]:
    # for num_warps in [4,]:
        for num_stages in [2,]:
        # for num_stages in [2]:
            for num_regs in [64, 128, 256]:
            # for num_regs in [256]:
                configs.append(triton.Config(
                    {}, 
                    num_warps=num_warps, 
                    num_stages=num_stages,
                    maxnreg=num_regs,
                ))
    return configs

@triton.autotune(
    configs=get_masking_iteration_draft_cuda_fused_configs(),
    key=[
        'BLOCK_BK',
        'BLOCK_SIZE_K', 
        'BLOCK_SIZE_Q', 
        'HID'
    ],
    restore_value=[
        'KEY_ACCESS_LOG',
        'KEY_ACCESS_COUNT',
        'INDICES',
        'KS',
        'GROUP_SIZE',
        'DUPPED_INDICES',
        'DUPPED_GROUP_SIZE',
        'SCORES', 
        'SCORES_FINAL',
        'PROBS',
        'TOPK_IDS',
        'T_GROUP_SIZE',
    ]
)
@triton.jit
def masking_iteration_draft_cuda_fused(
    Q, 
    stride_q_bsz, 
    stride_q_tdst,
    stride_q_bh, 
    stride_q_g, 
    stride_q_hid,
    K, 
    stride_k_bsz, 
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    POS, 
    stride_pos_bsz,
    stride_pos_tdst,
    KEY_ACCESS_LOG, 
    stride_key_access_log_b, 
    stride_key_access_log_bdst, 
    stride_key_access_log_t,
    KEY_ACCESS_COUNT, 
    stride_key_access_count_b,
    stride_key_access_count_bdst, 
    MAX_ACCESS_COUNT,
    
    INDICES, 
    stride_indices_b, 
    stride_indices_bdst, 
    stride_indices_bk,
    KS, 
    stride_ks_b, 
    stride_ks_bdst,
    GROUP_SIZE, 
    stride_group_size_b, 
    stride_group_size_bdst, 
    stride_group_size_bk,
    
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
    SCORES_FINAL,
    stride_scores_final_b,
    stride_scores_final_bdst,
    stride_scores_final_bk,
    SCORES_CACHED: tl.constexpr,
    PROBS,
    stride_probs_b,
    stride_probs_bdst,
    stride_probs_bk,
    TOPK_IDS, 
    stride_topk_ids_b, 
    stride_topk_ids_bdst, 
    stride_topk_ids_bk,
    
    T_GROUP_SIZE, 
    stride_t_group_size_b, 
    stride_t_group_size_bdst,
    INDICES_TDST,
    stride_indices_tdst_t,
    
    mask_k,
    
    sink_token_size,
    sliding_window_size,
    
    BH: tl.constexpr,
    G: tl.constexpr, 
    MAX_TDST, 
    MAX_TSRC,
    MAX_BDST,
    MAX_BSRC,
    BK: tl.constexpr,
    HID: tl.constexpr,
    RAND_SEED,
    SAMPLE_METHOD: tl.constexpr,
    BRANCH_METHOD: tl.constexpr,
    KV_HEAD_REPEAT: tl.constexpr,
    
    USING_EXTEND: tl.constexpr,
    extend_window_size,
    extend_group_size,
    COS, 
    stride_cos_t, 
    stride_cos_hid,
    SIN, 
    stride_sin_t, 
    stride_sin_hid,
    
    USING_SPARQ: tl.constexpr,
    SPARQ_HID: tl.constexpr,
    Q_IND, 
    stride_q_ind_b, 
    stride_q_ind_g, 
    stride_q_ind_bdst, 
    stride_q_ind_k,
    
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE, 
    stride_k_cache_page, 
    stride_k_cache_offset, 
    stride_k_cache_kv_head, 
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_bsz,
    
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    BLOCK_SCORE: tl.constexpr,
    GROUP_BDST,
    GROUP_BH,
    
    indices_bk_len: tl.constexpr,
    probs_bk_len: tl.constexpr,
):
    # _pid = tl.program_id(0)
    # #(BBH, GDST, GBH, BSZ)
    # _grid_bbh = GROUP_BH
    _grid_gdst = tl.cdiv(MAX_BDST, GROUP_BDST)
    # _grid_gbh = BH // GROUP_BH
    
    # _pid_bbh = _pid % _grid_bbh
    # _pid_gdst = (_pid // _grid_bbh) % _grid_gdst
    # _pid_gbh = (_pid // (_grid_bbh * _grid_gdst)) % _grid_gbh
    # _pid_bsz = _pid // (_grid_bbh * _grid_gdst * _grid_gbh)
    
    # # BH
    # _pid_0 = (_pid_bbh + _pid_gbh * GROUP_BH)
    # # BDST / GROUP BDST
    # _pid_1 = _pid_gdst
    # # BSZ
    # _pid_2 = _pid_bsz
    
    _pid_0 = tl.program_id(0) % GROUP_BH + tl.program_id(1) * GROUP_BH
    _pid_1 = (tl.program_id(0) // GROUP_BH) % _grid_gdst
    _pid_2 = tl.program_id(2)
    
    # _pid_0 = _pid % BH
    # _pid_1 = (_pid // BH) % _grid_gdst
    # _pid_2 = _pid // (BH * _grid_gdst)
    
    # _pid_0 = tl.program_id(0)
    # _pid_1 = tl.program_id(1)
    # _pid_2 = tl.program_id(2)
    
    pid_1 = _pid_2 * BH + _pid_0
    
    num_groups = tl.minimum(GROUP_BDST, (MAX_BDST - _pid_1 * GROUP_BDST))
    for i_group in range(num_groups):
        # originally bdst dim, before vectorize head
        pid_0 = _pid_1 * GROUP_BDST + i_group
        idx_b = pid_1
        idx_bdst = pid_0
        
        max_group_size = tl.load(
            T_GROUP_SIZE +\
                idx_b * stride_t_group_size_b +\
                idx_bdst * stride_t_group_size_bdst,
        ).to(tl.float32)
        
        while max_group_size > 1:
            n_program = tl.cdiv(indices_bk_len, BLOCK_BK)
            for i_program in range(n_program):
                masking_iteration_draft_cuda_dup_and_score(
                    Q, stride_q_bsz, stride_q_tdst, stride_q_bh, stride_q_g, stride_q_hid,
                    K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
                    POS, stride_pos_bsz, stride_pos_tdst,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    KEY_ACCESS_LOG, 
                    stride_key_access_log_b, 
                    stride_key_access_log_bdst, 
                    stride_key_access_log_t,
                    KEY_ACCESS_COUNT, 
                    stride_key_access_count_b,
                    stride_key_access_count_bdst, 
                    MAX_ACCESS_COUNT,
                    
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
                    SCORES_FINAL,
                    stride_scores_final_b,
                    stride_scores_final_bdst,
                    stride_scores_final_bk,
                    SCORES_CACHED,
                    
                    T_GROUP_SIZE, 
                    stride_t_group_size_b, 
                    stride_t_group_size_bdst,
                    INDICES_TDST,
                    stride_indices_tdst_t,
                    
                    mask_k,
                    
                    sliding_window_size,
                    
                    BH,
                    G, 
                    MAX_TDST, 
                    MAX_TSRC, 
                    BK, 
                    HID,
                    RAND_SEED,
                    SAMPLE_METHOD,
                    BRANCH_METHOD,
                    KV_HEAD_REPEAT,
                    
                    USING_EXTEND,
                    extend_window_size,
                    extend_group_size,
                    
                    USING_SPARQ,
                    SPARQ_HID,
                    Q_IND, 
                    stride_q_ind_b, 
                    stride_q_ind_g, 
                    stride_q_ind_bdst, 
                    stride_q_ind_k,
                    
                    # paged attention args template
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE, 
                    stride_k_cache_page, 
                    stride_k_cache_offset, 
                    stride_k_cache_kv_head, 
                    stride_k_cache_hid,
                    V_CACHE,
                    stride_v_cache_page,
                    stride_v_cache_offset,
                    stride_v_cache_kv_head,
                    stride_v_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_bsz,
                    
                    BLOCK_SIZE_Q,
                    BLOCK_STRIDE_Q,
                    BLOCK_SIZE_K,
                    BLOCK_STRIDE_K,
                    BLOCK_BK,
                    
                    0,
                    0,
                    
                    pid_0=i_program,
                    pid_1=pid_0,
                    pid_2=pid_1,
                )
            # end for
            tl.debug_barrier()
            
            # same grid with master (BDST, B)
            masking_iteration_draft_cuda_partial_softmax(
                SCORES, 
                stride_scores_b, 
                stride_scores_bdst, 
                stride_scores_bk,
                DUPPED_INDICES, 
                stride_dupped_indices_b, 
                stride_dupped_indices_bdst, 
                stride_dupped_indices_bk,
                DUPPED_GROUP_SIZE,
                stride_dupped_group_size_b,
                stride_dupped_group_size_bdst,
                stride_dupped_group_size_bk,
                
                PROBS,
                stride_probs_b,
                stride_probs_bdst,
                stride_probs_bk,
                
                sink_token_size,
                BK,
                G, 
                probs_bk_len, 
                MAX_BSRC,
                BLOCK_SIZE_K,
                
                BLOCK_SCORE,
                
                pid_0=pid_0,
                pid_1=pid_1,
            )
            tl.debug_barrier()
            
            # TODO: support score_head_group_size
            
            # same grid with master (BDST, B)
            masking_iteration_draft_cuda_argsort(
                PROBS,
                stride_probs_b, 
                stride_probs_bdst, 
                stride_probs_bk,
                TOPK_IDS, 
                stride_topk_ids_b, 
                stride_topk_ids_bdst, 
                stride_topk_ids_bk,
                
                T_GROUP_SIZE, 
                stride_t_group_size_b, 
                stride_t_group_size_bdst,
                
                MAX_BDST,
                
                probs_bk_len,
                BK * G,
                1,
                
                pid_0=pid_0,
                pid_1=pid_1,
            )
            tl.debug_barrier()
            
            # num_program = tl.cdiv(indices_bk_len, BLOCK_BK)
            # for i_program in range(num_program):
            masking_iteration_draft_cuda_gather(
                INDICES, 
                stride_indices_b, 
                stride_indices_bdst, 
                stride_indices_bk,
                GROUP_SIZE, 
                stride_group_size_b, 
                stride_group_size_bdst, 
                stride_group_size_bk,
                SCORES_FINAL,
                stride_scores_final_b,
                stride_scores_final_bdst,
                stride_scores_final_bk,
                
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
                
                TOPK_IDS,
                stride_topk_ids_b,
                stride_topk_ids_bdst,
                stride_topk_ids_bk,
                
                T_GROUP_SIZE,
                stride_t_group_size_b, 
                stride_t_group_size_bdst,
                
                G, BK, 
                
                indices_bk_len,
                
                pid_0=0,
                pid_1=pid_0,
                pid_2=pid_1,
            )
            
            tl.debug_barrier()
            
            # SCORES_CACHED = True
            
            if BRANCH_METHOD == 'random':
                max_group_size *= 0.7
            else:
                max_group_size *= 0.5
        tl.store(
            T_GROUP_SIZE +\
                idx_b * stride_t_group_size_b +\
                idx_bdst * stride_t_group_size_bdst,
            value=max_group_size
        )
        tl.debug_barrier()

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_BK': 16}, num_warps=1),
#         triton.Config({'BLOCK_BK': 32}, num_warps=1),
#         # triton.Config({'BLOCK_BK': 64}, num_warps=1),
#         # triton.Config({'BLOCK_BK': 128}, num_warps=1),
        
#         # triton.Config({'BLOCK_BK': 16}, num_warps=2),
#         triton.Config({'BLOCK_BK': 32}, num_warps=2),
#         triton.Config({'BLOCK_BK': 64}, num_warps=2),
#         # triton.Config({'BLOCK_BK': 128}, num_warps=2),
        
#         # triton.Config({'BLOCK_BK': 16}, num_warps=4),
#         # triton.Config({'BLOCK_BK': 32}, num_warps=4),
#         triton.Config({'BLOCK_BK': 64}, num_warps=4),
#         triton.Config({'BLOCK_BK': 128}, num_warps=4),
        
#         # triton.Config({'BLOCK_BK': 16}, num_warps=8),
#         # triton.Config({'BLOCK_BK': 32}, num_warps=8),
#         triton.Config({'BLOCK_BK': 64}, num_warps=8),
#         triton.Config({'BLOCK_BK': 128}, num_warps=8),
        
#         # triton.Config({'BLOCK_BK': 16}, num_warps=16),
#         # triton.Config({'BLOCK_BK': 32}, num_warps=16),
#         triton.Config({'BLOCK_BK': 64}, num_warps=16),
#         triton.Config({'BLOCK_BK': 128}, num_warps=16),
#     ],
#     key=['BLOCK_SIZE_K', 'BLOCK_SIZE_Q'],
#     rep=200,
#     use_cuda_graph=True,
# )
@triton.jit
def masking_iteration_draft_cuda_initialize_score(
    Q, stride_q_bsz, stride_q_tdst, stride_q_bh, stride_q_g, stride_q_hid,
    K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
    POS, stride_pos_bsz, stride_pos_tdst,
    KEY_ACCESS_LOG, 
    stride_key_access_log_b, 
    stride_key_access_log_bdst, 
    stride_key_access_log_t,
    KEY_ACCESS_COUNT, 
    stride_key_access_count_b,
    stride_key_access_count_bdst,
    MAX_ACCESS_COUNT,
    
    INDICES, 
    stride_indices_b, 
    stride_indices_bdst, 
    stride_indices_bk,
    
    SCORES,
    stride_scores_b,
    stride_scores_bdst,
    stride_scores_bk,
    
    T_GROUP_SIZE, 
    stride_t_group_size_b, 
    stride_t_group_size_bdst,
    INDICES_TDST,
    stride_indices_tdst_t,
    
    sliding_window_size,
    indices_bk_len,
    BH: tl.constexpr, 
    G: tl.constexpr, 
    MAX_TDST, 
    MAX_TSRC, 
    HID: tl.constexpr,
    KV_HEAD_REPEAT: tl.constexpr,
                
    USING_EXTEND: tl.constexpr,
    extend_window_size,
    extend_group_size,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    
    USING_SPARQ: tl.constexpr,
    SPARQ_HID: tl.constexpr,
    Q_IND, 
    stride_q_ind_b, 
    stride_q_ind_g, 
    stride_q_ind_bdst, 
    stride_q_ind_k,
    
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE, 
    stride_k_cache_page, 
    stride_k_cache_offset, 
    stride_k_cache_kv_head, 
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_bsz,
    
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    
    KEY_DUP: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    
    grid_bh = BH
    grid_bk = tl.cdiv(indices_bk_len, BLOCK_BK)
    grid_bdst = tl.cdiv(MAX_TDST, BLOCK_SIZE_Q)
    
    pid_bh = tl.program_id(0) % BH
    pid_bk = tl.program_id(0) // BH
    pid_bdst = tl.program_id(1)
    pid_bsz = tl.program_id(2)
    
    idx_bk = pid_bk * BLOCK_BK + tl.arange(0, BLOCK_BK)
    mask_bk = idx_bk < indices_bk_len
    idx_bdst = pid_bdst
    idx_b = pid_bsz * BH + pid_bh
    
    t_group_size = tl.load(
        T_GROUP_SIZE +\
            idx_b * stride_t_group_size_b +\
            idx_bdst * stride_t_group_size_bdst,
    )
    if t_group_size <= 1.0:
        return
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
    idx_tdst_no_proj = idx_tdst
    mask_tdst = idx_tdst < MAX_TDST
    if INDICES_TDST is not None:
        idx_tdst = tl.load(
            INDICES_TDST +\
                idx_tdst.to(tl.int64) * stride_indices_tdst_t,
            mask=mask_tdst,
            other=MAX_TDST,
        ).to(tl.int64)
    
    pos_tdst = tl.load(
        POS +\
            (idx_b // BH) * stride_pos_bsz +\
            idx_tdst_no_proj * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    TSRC = tl.max(pos_tdst)
    TSRC = tl.maximum(0, TSRC - sliding_window_size)
    BSRC = tl.cdiv(TSRC, BLOCK_SIZE_K)
    
    indices = tl.load(
        INDICES +\
            idx_b * stride_indices_b +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
        mask=mask_bk,
        other=0
    )
    
    scores = masking_iteration_draft_cuda_dup_and_score_calc_score(
        indices,
        KEY_DUP,
        
        Q, stride_q_bsz, stride_q_tdst, stride_q_bh, stride_q_g, stride_q_hid,
        K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
        COS, stride_cos_t, stride_cos_hid,
        SIN, stride_sin_t, stride_sin_hid,
        KEY_ACCESS_LOG, 
        stride_key_access_log_b, 
        stride_key_access_log_bdst, 
        stride_key_access_log_t,
        KEY_ACCESS_COUNT,
        stride_key_access_count_b,
        stride_key_access_count_bdst,
        MAX_ACCESS_COUNT,
        
        idx_b,
        idx_bdst,
        idx_tdst, mask_tdst, pos_tdst,
        mask_bk,
        
        BH, G, MAX_TSRC, HID, KV_HEAD_REPEAT,
                
        USING_EXTEND,
        extend_window_size,
        extend_group_size,
        
        USING_SPARQ,
        SPARQ_HID,
        Q_IND, 
        stride_q_ind_b, 
        stride_q_ind_g, 
        stride_q_ind_bdst, 
        stride_q_ind_k,
        
        # paged attention args template
        USING_PAGES,
        PAGE_SIZE,
        K_CACHE, 
        stride_k_cache_page, 
        stride_k_cache_offset, 
        stride_k_cache_kv_head, 
        stride_k_cache_hid,
        V_CACHE,
        stride_v_cache_page,
        stride_v_cache_offset,
        stride_v_cache_kv_head,
        stride_v_cache_hid,
        BLOCK_TABLE,
        stride_block_table_bsz,
        stride_block_table_page,
        CACHE_SEQ_LENS,
        stride_cache_seq_lens_bsz,
        
        BLOCK_SIZE_Q,
        BLOCK_STRIDE_Q,
        BLOCK_SIZE_K,
        BLOCK_STRIDE_K,
        BLOCK_BK,
        'max',
    )
    
    tl.store(
        SCORES +\
            idx_b * stride_scores_b +\
            idx_bdst * stride_scores_bdst +\
            idx_bk * stride_scores_bk,
        mask=mask_bk,
        value=scores,
    )

@nvtx.annotate('masking_iteration_draft')
def masking_iteration_draft( 
    q: Tensor,
    k: Optional[Tensor],
    position_ids: Tensor,
    args: "HiPAttentionArgs",
    
    # seeds
    indices_seed: Optional[Tensor] = None,
    ks_seed: Optional[Tensor] = None,
    scores_seed: Optional[Tensor] = None,
    group_size_seed: Optional[Tensor] = None,
    max_group_size_seed: Optional[float] = None,
    indices_tdst: Optional[Tensor] = None,
):
    assert isinstance(q, Tensor)
    if k is not None:
        assert q.device == k.device
        assert isinstance(k, Tensor)
    
    if args.rope_cos is not None and args.using_extend:
        assert args.rope_cos.ndim == 2
        assert args.rope_cos.shape[-1] == q.shape[-1]
        assert isinstance(args.rope_cos, Tensor)
    
    if args.rope_sin is not None and args.using_extend:
        assert args.rope_sin.ndim == 2
        assert args.rope_sin.shape[-1] == q.shape[-1]
        assert isinstance(args.rope_sin, Tensor)
        assert isinstance(args.rope_sin, Tensor)
    
    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        _, TSRC, KV_HEAD, HID = k.shape
    else:
        assert args.k_cache is not None
        N_PAGES, PAGE_SIZE, KV_HEAD, HID = args.k_cache.shape
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD
    N = BSZ * HEAD
    if indices_tdst is not None:
        TDST = len(indices_tdst)
        assert indices_tdst.ndim == 1
        indices_tdst_stride = indices_tdst.stride()
    else:
        indices_tdst_stride = (0,)
    BDST = cdiv_python(TDST, args.block_size_q)
    if k is not None:
        _, TSRC, _, _ = k.shape
        BSRC = cdiv_python(TSRC, args.block_size_k)
        MAX_TSRC = TSRC
        MAX_BSRC = BSRC
    else:
        TSRC = BSRC = None
        MAX_TSRC = N_PAGES * PAGE_SIZE
        MAX_BSRC = cdiv_python(MAX_TSRC, args.block_size_k)
    
    assert (N % args.topk_head_group_size) == 0, 'batch * n_head should divisible by head group size'
    
    # split batch-head dim into head groups
    q = q.view(BSZ, -1, HEAD // args.topk_head_group_size, args.topk_head_group_size, HID)
    if k is not None:
        k = k.view(BSZ, TSRC, KV_HEAD, HID)
    
    BSZ, _, BH, G, HID = q.shape
    B = BSZ * BH
    mask_block_k = cdiv_python(args.mask_k, args.block_size_k)
    
    assert args.block_size_k_group == 1
    if args.block_size_k_group > 1:
        warnings.warn('K grouping is inefficient right now.')
        k_group = k.view(BSZ, triton.cdiv(TSRC, args.block_size_k_group), args.block_size_k_group, BH, G, HID)
        k_group_min = torch.min(k_group, dim=2)
        k_group_max = torch.max(k_group, dim=2)
        k = torch.concat([k_group_min, k_group_max], dim=-1)
    
    indices = torch.full(
        (
            B,
            cdiv_python(TDST, args.block_size_q), 
            # head group is merged as single sequence
            G * mask_block_k,
        ), 
        fill_value=(MAX_BSRC + args.block_size_k + args.block_size_q) * G, 
        dtype=torch.int32, 
        device=q.device
    )
    
    ks = torch.zeros((
        B, 
        cdiv_python(TDST, args.block_size_q),
    ), dtype=torch.int32, device=q.device)
    
    group_sizes = torch.zeros_like(indices)
    t_group_sizes = torch.zeros((B, BDST), dtype=torch.float32, device=q.device)
    
    if max_group_size_seed is None:
        max_group_strategy = 'worst'
        
        if indices_seed is None:
            # always chunks are evenly distributed. fastest.
            max_group_strategy = 'best'
        
        if k is not None:
            if max_group_strategy == 'oracle':
                # > oracle      5.1117  18.4503 sec
                # This is impossible at this point, because t_group_size is initilized by following kernel
                raise NotImplementedError() 
                max_group_size = torch.max(t_group_sizes).item()
            elif max_group_strategy == 'best':
                # > best case   5.1218  10.3745 sec
                #   (not complete search if you gave seed)
                max_group_size = triton.cdiv(BSRC, mask_block_k)
            elif max_group_strategy == 'worst':
                # > worst case  5.1097  17.6545 sec
                #   (always complete search)
                max_group_size = triton.cdiv(BSRC, args.block_size_k)
            elif max_group_strategy == 'greedy':
                # > greedy      5.1202  11.4861 sec
                #   (slightly generous then best stratgy)
                max_group_size = triton.cdiv(BSRC, mask_block_k) * 2
            elif max_group_strategy == 'constant':
                # TODO: test this
                max_group_size = min(triton.cdiv(BSRC, args.block_size_k), 8)
            else:
                raise Exception()
        else:
            assert args.k_cache is not None
            max_group_size = None
    else:
        max_group_size = max_group_size_seed
    
    if max_group_size is not None:
        KEY_ACCESS_LEN = args.mask_k * math.ceil(math.log2(max_group_size))
    else:
        KEY_ACCESS_LEN = args.mask_k * math.ceil(math.log2(triton.cdiv(MAX_BSRC, args.block_size_k)))
    if args.output_key_access_log:
        key_access_log = torch.empty(
            (B, BDST, KEY_ACCESS_LEN,), dtype=torch.int32, 
            # fill_value=torch.iinfo(torch.int32).max,
            device=q.device,
        )
        key_access_count = torch.zeros(
            (B, BDST, ), 
            dtype=torch.long,
            device=q.device,
        )
    else:
        key_access_log = None
        key_access_count = None
    
    assert len(q.stride()) == 5 # BSZ, MAX_TDST, BH, G, HID
    if k is not None:
        assert len(k.stride()) == 4 # BSZ, MAX_TSRC, KV_HEAD, HID
    if args.k_cache is not None:
        assert args.k_cache.ndim == 4 # N_PAGES, PAGE_SIZE, KV_HEAD, HID
        assert args.block_table.ndim == 2 # BSZ, N_PAGES
        assert args.cache_seq_lens.ndim == 1 # BSZ
    assert len(indices.stride()) == 3
    assert len(ks.stride()) == 2
    assert len(group_sizes.stride()) == 3
    assert len(t_group_sizes.stride()) == 2
    if indices_seed is not None:
        assert len(indices_seed.stride()) == 3
        assert len(ks_seed.stride()) == 2
        assert indices_seed.shape == indices.shape
        assert ks_seed.shape == ks.shape
        indices_seed = indices_seed // args.block_size_k
    if args.rope_cos is not None:
        assert len(args.rope_cos.stride()) == 2, args.rope_cos.shape
        assert len(args.rope_sin.stride()) == 2, args.rope_cos.shape
    
    assert args.sample_method in ['first', 'last', 'random', 'oracle', 'center']
    assert position_ids.ndim == 2, position_ids.shape
    
    # launch kernels
    # print('init in', indices[0, -1, :10])
    # if indices_seed is not None:
    #     print('init ins', indices_seed[0, -1, :10])
    BLOCK_MASK_BLOCK_K = triton.next_power_of_2(mask_block_k)
    
    if group_size_seed is None:
        grid = (B, BDST, G)
        # print('init grid', grid)
        pre_device = torch.get_default_device()
        torch.set_default_device(indices.device)
        masking_iteration_draft_cuda_initialize[grid](
            indices_seed, *(indices_seed.stride() if indices_seed is not None else (0, 0, 0)),
            ks_seed, *(ks_seed.stride() if ks_seed is not None else (0, 0)),
            position_ids, *position_ids.stride(),
            
            indices, *indices.stride(),
            ks, *ks.stride(),
            group_sizes, *group_sizes.stride(),
            
            t_group_sizes, *t_group_sizes.stride(),
            
            args.mask_k,
            args.block_size_q, 
            args.block_size_k, 
            
            args.sliding_window_size,
            
            G, TDST, MAX_TSRC, HEAD,
            
            BLOCK_MASK_BLOCK_K,
            
            # num_warps=min(max(cdiv_python(BLOCK_MASK_BLOCK_K, 32), 1), 32),
            num_warps=1,
            num_stages=1,
        )
        torch.set_default_device(pre_device)
    else:
        indices.copy_(indices_seed)
        ks.copy_(ks_seed)
        group_sizes.copy_(group_size_seed)
        t_group_sizes = group_sizes.max(dim=-1)[0].float()
    # print('init in after', indices[0, 0, :10])
    # print('init in after', indices[0, -1, :10])
    # print('init gs after', group_sizes[0, 0, :10])
    # print('init gs after', group_sizes[0, :, 0])
    # print('init ks after', ks[0, :])
    # print('init pos', position_ids[:])
    
    dupped_indices = torch.empty(
        (B, BDST, indices.shape[-1] * 2),
        dtype=torch.int32, device=q.device,
    )
    dupped_group_sizes = torch.empty(
        (B, BDST, indices.shape[-1] * 2),
        dtype=torch.int32, device=q.device,
    )
    scores = torch.empty_like(dupped_indices, dtype=torch.bfloat16)
    probs = torch.empty_like(scores)
    if (scores_seed is not None) and args.sample_method == 'first':
        scores_final = scores_seed.clone()
    else:
        scores_final = torch.zeros_like(indices, dtype=torch.bfloat16)
        
        # BLOCK_BK = 128 // block_size_k
        # grid = (triton.cdiv(indices.shape[-1], BLOCK_BK), BDST, B)
        
        BLOCK_BK = 256 // (args.block_size_k // args.block_stride_k) * G
        
        assert B == BSZ * BH
        grid = (
            BH * triton.cdiv(indices.shape[-1], BLOCK_BK),
            BDST, 
            BSZ,
        )
        
        # BUG: autotune ruin the access log
        # grid = lambda META: (triton.cdiv(indices.shape[-1], META['BLOCK_BK']), BDST, B)
        pre_device = torch.get_default_device()
        torch.set_default_device(q.device)
        masking_iteration_draft_cuda_initialize_score[grid](
            q, *q.stride(),
            k, *args.safe_stride(k, 4),
            position_ids, *position_ids.stride(),
            key_access_log, *(key_access_log.stride() if key_access_log is not None else (0, 0, 0)),
            key_access_count, *(key_access_count.stride() if key_access_count is not None else (0, 0)),
            KEY_ACCESS_LEN,
            
            indices, *indices.stride(),
            
            scores_final, *scores_final.stride(),
            
            t_group_sizes, *t_group_sizes.stride(),
            indices_tdst, *indices_tdst_stride,
            
            args.sliding_window_size,
            indices.shape[-1],
            BH, G, TDST, MAX_TSRC, HID, KV_HEAD_REPEAT,
            
            *args.args_extend(),
            *args.args_sparq(),
            *args.args_paged_kv_cache(),
            *args.args_bq_bsq_bk_bsk(),
            
            BLOCK_BK,
            
            num_warps=2,
            num_stages=1,
        )
        torch.set_default_device(pre_device)
        
        # print('-- after initialize')
        # print(scores.shape, key_access_log.shape, key_access_count.shape)
        # print('access count', key_access_count[0])
        # print('access log', key_access_log[0, -1, :key_access_count[0, -1].item()].tolist())
    scores_cached = args.sample_method == 'first'
    # scores_cached = False
    
    BLOCK_BK = 256 // 2 // args.block_size_k
    assert BLOCK_BK > 0
    BLOCK_HID = HID
    assert (HID % BLOCK_HID) == 0
    
    # print(indices[0, -10])
    # print(ks[0, -10])
    # assert indices[0, -10].shape == torch.unique(indices[0, -10]).shape, f'{indices[0, -10].shape} == {torch.unique(indices[0, -10]).shape}'
    
    topk_indices = None
    
    # max_group_size = max_group_size
    
    topk_indices = torch.empty(
        (probs.shape[0], probs.shape[1], mask_block_k * G),
        device=probs.device,
        dtype=torch.int32,
    )
    BLOCK_SCORE = triton.next_power_of_2(scores.shape[-1])
    
    using_fused_iteration = True
    if using_fused_iteration:
        assert args.score_head_group_size == 1
        
        if not scores_cached:
            BLOCK_BK = 512 // (args.block_size_k // args.block_stride_k) * G // 4 * G
        else:
            BLOCK_BK = 512 // (args.block_size_k // args.block_stride_k) * G // 8 * G
        # BLOCK_BK = indices.shape[-1]
        # BLOCK_BK = indices.shape[-1] // 4
        
        # BLOCK_BK = indices.shape[-1] // 4
        
        GROUP_BDST = 1
        GROUP_BH = 1
        
        assert (BH % GROUP_BH) == 0
        assert B == BSZ * BH
        
        # grid = (BH, triton.cdiv(BDST, GROUP_BDST), BSZ,)
        # grid = (triton.cdiv(BDST, GROUP_BDST), BSZ, BH,)
        # grid = (B, triton.cdiv(BDST, GROUP_BDST),)
        
        # grid = (
        #     triton.cdiv(BDST, GROUP_BDST) * BH * BSZ,
        # )
        
        grid = (
            GROUP_BH * triton.cdiv(BDST, GROUP_BDST),
            BH // GROUP_BH,
            BSZ
        )
        
        pre_device = torch.get_default_device()
        torch.set_default_device(q.device)
        masking_iteration_draft_cuda_fused[grid](
            q, *q.stride(),
            k, *args.safe_stride(k, 4),
            position_ids, *position_ids.stride(),
            key_access_log, *args.safe_stride(key_access_log, 3),
            key_access_count, *args.safe_stride(key_access_count, 2),
            KEY_ACCESS_LEN,
            
            indices, *indices.stride(),
            ks, *ks.stride(),
            group_sizes, *group_sizes.stride(),
            
            dupped_indices, *dupped_indices.stride(),
            dupped_group_sizes, *dupped_group_sizes.stride(),
            scores, *scores.stride(),
            scores_final, *scores_final.stride(),
            scores_cached,
            probs, *probs.stride(),
            topk_indices, *topk_indices.stride(),
            
            t_group_sizes, *t_group_sizes.stride(),
            indices_tdst, *indices_tdst_stride,
            
            args.mask_k,
            
            args.sink_token_size,
            args.sliding_window_size,
            
            BH,
            G, 
            # TDST, 
            # TSRC,
            # cdiv_python(TDST, args.block_size_q),
            # cdiv_python(TSRC, args.block_size_k),
            TDST,
            MAX_TSRC,
            cdiv_python(TDST, args.block_size_q),
            MAX_BSRC,
            mask_block_k, 
            HID,
            random.randint(0, 1024*1024),
            args.sample_method,
            args.branch_method,
            KV_HEAD_REPEAT,
            
            *args.args_extend(),
            *args.args_sparq(),
            *args.args_paged_kv_cache(),
            *args.args_bq_bsq_bk_bsk(),
            
            BLOCK_BK,
            BLOCK_SCORE,
            GROUP_BDST,
            GROUP_BH,
            
            indices_bk_len=indices.shape[-1],
            probs_bk_len=probs.shape[-1],
            
            # num_warps=4,
            # num_stages=2,
        )
        torch.set_default_device(pre_device)
    else:
        raise NotImplementedError()
        i_iteration = 0
        while max_group_size > 1:
            BLOCK_BK = 128 // block_size_k
            grid = (triton.cdiv(indices.shape[-1], BLOCK_BK), BDST, B,)
            masking_iteration_draft_cuda_dup_and_score[grid](
                q, *q.stride(),
                k, *k.stride(),
                position_ids, *position_ids.stride(),
                rope_cos, *(rope_cos.stride() if rope_cos is not None else (0, 0)),
                rope_sin, *(rope_sin.stride() if rope_sin is not None else (0, 0)),
                key_access_log, *(key_access_log.stride() if key_access_log is not None else (0, 0, 0)),
                key_access_count, *(key_access_count.stride() if key_access_count is not None else (0, 0)),
                KEY_ACCESS_LEN,
                
                indices, *indices.stride(),
                ks, *ks.stride(),
                group_sizes, *group_sizes.stride(),
                
                dupped_indices, *dupped_indices.stride(),
                dupped_group_sizes, *dupped_group_sizes.stride(),
                scores, *scores.stride(),
                scores_final, *scores_final.stride(),
                scores_cached,
                
                t_group_sizes, *t_group_sizes.stride(),
                indices_tdst, *indices_tdst_stride,
                
                mask_k,
                
                sliding_window_size,
                
                G, TDST, TSRC, mask_block_k, HID,
                random.randint(0, 1024*1024),
                sample_method,
                branch_method,
                
                using_extend,
                self_extend_neighboor_window,
                self_extend_group_size,
                
                using_sparq,
                sparq_hid,
                sparq_ind, *(sparq_ind.stride() if sparq_ind is not None else (0, 0, 0, 0)),
                
                block_size_q,
                block_stride_q,
                block_size_k,
                block_stride_k,
                BLOCK_BK,
                
                max_group_size,
                i_iteration,
                
                num_warps=(2 if scores_cached else 4) * G,
                num_stages=max(1, 4 // G),
            )
            
            # NOTE: because of softmax, we cannot fuse everything...
            # BLOCK_SCORE = min(1024, mask_block_k * G)
            grid = (BDST, B)
            masking_iteration_draft_cuda_partial_softmax[grid](
                scores, *scores.stride(),
                dupped_indices, *dupped_indices.stride(),
                dupped_group_sizes, *dupped_group_sizes.stride(),
                
                probs, *probs.stride(),
                
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
                scores = scores_max\
                    .repeat(1, score_head_group_size, 1, 1)\
                    .view(-1, scores_max.shape[-2], scores_max.shape[-1])
            
            # also villan
            BLOCK_BDST = 1
            grid = (triton.cdiv(BDST, BLOCK_BDST), B,)
            masking_iteration_draft_cuda_argsort[grid](
                probs, *probs.stride(),
                topk_indices, *topk_indices.stride(),
                
                t_group_sizes, *t_group_sizes.stride(),
                
                BDST,
                
                probs.shape[-1],
                mask_block_k * G,
                BLOCK_BDST,
                
                num_warps=min(32, max(1, (probs.shape[-1] * BLOCK_BDST) // 256)),
                num_stages=8,
            )
            
            BLOCK_BK = indices.shape[-1]
            grid = (triton.cdiv(indices.shape[-1], BLOCK_BK), BDST, B,)
            masking_iteration_draft_cuda_gather[grid](
                indices, *indices.stride(),
                group_sizes, *group_sizes.stride(),
                scores_final, *scores_final.stride(),
                
                dupped_indices, *dupped_indices.stride(),
                dupped_group_sizes, *dupped_group_sizes.stride(),
                scores, *scores.stride(),
                
                topk_indices, *topk_indices.stride(),
                
                t_group_sizes, *t_group_sizes.stride(),
                
                G, mask_block_k, 
                
                BLOCK_BK,
            )
            
            # indices, indices_sort_mapping = torch.sort(indices, dim=-1, stable=False)
            # scores_final = scores_final\
            #     .gather(index=indices_sort_mapping, dim=-1)
            # group_sizes = group_sizes\
            #     .gather(index=indices_sort_mapping, dim=-1)
            
            if sample_method in ['first', 'last', 'center', 'half']:
                scores_cached = True
            
            if branch_method == 'random':
                max_group_size = max_group_size * 0.7
                if max_group_size > 1.0:
                    t_group_sizes.mul_(0.7)
            else:
                max_group_size = max_group_size * 0.5
                if max_group_size > 1.0:
                    t_group_sizes.mul_(0.5)
            i_iteration += 1
    
    indices.mul_(args.block_size_k)
    
    # NOTE: before this sort, indices are sorted by imporatnce of each block
    indices, indices_sort_mapping = torch.sort(indices, dim=-1, stable=False)
    
    scores_final = scores_final\
        .gather(index=indices_sort_mapping, dim=-1)
    
    # scores_final = None
    
    ks_count, ks_start_end = masking_iteration_draft_python_epilog(
        indices, ks, 
        mask_block_k, MAX_TSRC,
        B, BDST, G
    )
    
    # assert indices[0, -10].shape == torch.unique(indices[0, -10]).shape, f'{indices[0, -10].shape} == {torch.unique(indices[0, -10]).shape}'
    # t = indices[0, 16]
    # c = ks[0, 16]
    # tu = torch.unique(t)
    # print(t)
    # print(tu)
    # print(t.shape, tu.shape, c)
    
    return indices, ks, ks_count, ks_start_end, scores_final, group_sizes, key_access_log, key_access_count

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
    
    # TDST,
    # TSRC,
    
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
        mask_tsrc_window = idx_tsrc >= (tl.min(tl.where(mask_tdst, (pos_tdst - 1), 987654321)) - extend_window_size)
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
        
        old_tdst = (pos_tdst - 1)
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
            queries, 
            keys,
            allow_tf32=True,
            # out_dtype=tl.float16,
        ).to(tl.float32) * 1.44269504
    
    # qk_mask = (
    #     ((idx_tdst[:, None] + TSRC - TDST) < (idx_tsrc)[None, :]) |
    #     (~(mask_tdst[:, None] & mask_tsrc[None, :]))
    # )
    
    if EXCLUDE_SLIDING_WINDOW:
        qk_mask = (
            ((pos_tdst - 1)[:, None] < idx_tsrc[None, :]) |
            ((pos_tdst - 1)[:, None] < (idx_tsrc + sliding_window_size)[None, :]) |
            (~(mask_tdst[:, None] & mask_tsrc[None, :]))
        )
    else:
        qk_mask = (
            ((pos_tdst - 1)[:, None] < idx_tsrc[None, :]) |
            ((pos_tdst - 1)[:, None] >= (idx_tsrc + sliding_window_size)[None, :]) |
            (~(mask_tdst[:, None] & mask_tsrc[None, :]))
        )
    
    # qk = tl.where(
    #     qk_mask,
    #     float('-inf'),
    #     qk
    # )
    
    # qk += qk_mask * (-1.0e+6)
    
    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1)[:, None])
    qk = qk - m_ij
    # [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]
    p = tl.math.exp2(qk)
    
    p = tl.where(qk_mask, 0, p)
    # p *= ~qk_mask
    
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

def get_block_sparse_attention_configs():
    warnings.warn('triton autotuning is activated. this should be disabled for faster startup.')
    configs = []
    # for block_bk in [4, 8, 16, 32]:
    for block_bk in [8, 16, 32]:
        for max_nreg in [128, 256, 512]:
            for num_warps in [4]:
                for num_stages in [2]:
                    configs.append(triton.Config(
                        {
                            'BLOCK_BK': block_bk
                        }, 
                        num_warps=num_warps, 
                        num_stages=num_stages, 
                        maxnreg=max_nreg
                    ))
    return configs

@triton.autotune(
    configs=get_block_sparse_attention_configs(),
    key=[
        'BLOCK_SIZE_K',
        'BLOCK_SIZE_Q',
        'HID',
    ],
)
@triton.jit
def block_sparse_attention_cuda(
    Q, stride_q_bsz, stride_q_tdst, stride_q_head, stride_q_hid,
    K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
    V, stride_v_bsz, stride_v_tsrc, stride_v_head, stride_v_hid,
    POS, stride_pos_bsz, stride_pos_tdst,
    
    INDICES, 
    stride_indices_b, stride_indices_bdst, stride_indices_bk,
    
    KS_START_END,
    stride_ks_start_end_b, stride_ks_start_end_bdst, stride_ks_start_end_g,
    
    CONTEXT,
    stride_context_bsz, 
    stride_context_tdst,
    stride_context_head, 
    stride_context_hid,
    
    HEAD: tl.constexpr, 
    G: tl.constexpr, 
    BK: tl.constexpr, 
    MAX_TDST, 
    MAX_TSRC,
    KV_HEAD_REPEAT: tl.constexpr,
    
    sliding_window_size: tl.constexpr,
    
    USING_EXTEND: tl.constexpr,
    extend_window_size,
    extend_group_size,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE, 
    stride_k_cache_page, 
    stride_k_cache_offset, 
    stride_k_cache_kv_head, 
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    
    HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_BK: tl.constexpr,
):
    pid_bsz = tl.program_id(2)
    pid_bdst = tl.program_id(1)
    pid_head = tl.program_id(0)
    
    idx_bsz = pid_bsz.to(tl.int64)
    idx_head = pid_head
    idx_n = idx_bsz * HEAD + idx_head
    idx_b = idx_n // G
    idx_g = idx_n % G
    
    idx_bdst = pid_bdst
    idx_tdst = BLOCK_SIZE_Q * idx_bdst + tl.arange(0, BLOCK_SIZE_Q)
    mask_tdst = idx_tdst < MAX_TDST
    pos_tdst = tl.load(
        POS +\
            idx_bsz * stride_pos_bsz +\
            idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    
    idx_hid = tl.arange(0, HID)
    
    acc = tl.zeros((BLOCK_SIZE_Q, HID), dtype=tl.float32)
    m_i = tl.full((BLOCK_SIZE_Q, 1), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_SIZE_Q, 1), 1.0, dtype=tl.float32)
    
    queries = tl.load(
        Q +\
            idx_bsz * stride_q_bsz +\
            idx_tdst[:, None] * stride_q_tdst +\
            idx_head * stride_q_head +\
            idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0,
        cache_modifier='.cg',
        # eviction_policy='evict_last',
        # volatile=True,
    )
    
    if BK > 0:
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
            keys = load_tokens(
                K, 
                stride_k_bsz, 
                stride_k_tsrc, 
                stride_k_head, 
                stride_k_hid,
                
                USING_PAGES, 
                PAGE_SIZE,
                K_CACHE,
                stride_k_cache_page,
                stride_k_cache_offset,
                stride_k_cache_kv_head,
                stride_k_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                
                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[:, None],
                mask_tsrc[None, :],
            )
            
            values = load_tokens(
                V, 
                stride_v_bsz, 
                stride_v_tsrc, 
                stride_v_head, 
                stride_v_hid,
                
                USING_PAGES, 
                PAGE_SIZE,
                V_CACHE,
                stride_v_cache_page,
                stride_v_cache_offset,
                stride_v_cache_kv_head,
                stride_v_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                
                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[None, :],
                mask_tsrc[:, None],
            )
            
            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries,
                keys,
                values,
                
                idx_tsrc, mask_tsrc,
                idx_tdst, mask_tdst,
                
                acc, l_i, m_i,
                
                sliding_window_size,
                True,
                
                USING_EXTEND,
                extend_window_size,
                extend_group_size,
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                
                pos_tdst,
                idx_hid, HID, 
                BLOCK_SIZE_Q, 
                BLOCK_BK * BLOCK_SIZE_K,
            )
    
    if sliding_window_size > 0:
        CURR_TSRC = tl.max(pos_tdst)
        # CURR_TSRC = (idx_bdst + 1) * BLOCK_SIZE_Q + MAX_TSRC - MAX_TDST
        for i_tsrc in range(tl.maximum(0, CURR_TSRC - sliding_window_size - BLOCK_SIZE_Q), CURR_TSRC, BLOCK_BK * BLOCK_SIZE_K):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < MAX_TSRC
            
            # idx_n = idx_b * G + idx_group
            keys = load_tokens(
                K, 
                stride_k_bsz, 
                stride_k_tsrc, 
                stride_k_head, 
                stride_k_hid,
                
                USING_PAGES, 
                PAGE_SIZE,
                K_CACHE,
                stride_k_cache_page,
                stride_k_cache_offset,
                stride_k_cache_kv_head,
                stride_k_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                
                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[:, None],
                mask_tsrc[None, :],
            )
            
            values = load_tokens(
                V, 
                stride_v_bsz, 
                stride_v_tsrc, 
                stride_v_head, 
                stride_v_hid,
                
                USING_PAGES, 
                PAGE_SIZE,
                V_CACHE,
                stride_v_cache_page,
                stride_v_cache_offset,
                stride_v_cache_kv_head,
                stride_v_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                
                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[None, :],
                mask_tsrc[:, None],
            )
            
            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries,
                keys,
                values,
                
                idx_tsrc, mask_tsrc,
                idx_tdst, mask_tdst,
                
                acc, l_i, m_i,
                
                sliding_window_size,
                False,
                
                USING_EXTEND,
                extend_window_size,
                extend_group_size,
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                
                pos_tdst,
                idx_hid, HID, 
                BLOCK_SIZE_Q, 
                BLOCK_BK * BLOCK_SIZE_K,
            )
    
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = (acc / (tl.where(l_i == 0.0, 1e-20, l_i)))
    
    tl.store(
        CONTEXT +\
            idx_bsz * stride_context_bsz +\
            idx_tdst[:, None] * stride_context_tdst +\
            idx_head * stride_context_head +\
            idx_hid[None, :] * stride_context_hid,
        mask = mask_tdst[:, None],
        value = acc.to(CONTEXT.type.element_ty),
        # eviction_policy='evict_first',
        # cache_modifier='.cs', # TODO: uncomment this
        # value = l_i
    )

def block_sparse_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    position_ids: Tensor,
    
    indices: Tensor,
    ks: Tensor,
    ks_count: Tensor,
    ks_start_end: Tensor,
    
    args: "HiPAttentionArgs",
):
    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        _, TSRC, KV_HEAD, _ = k.shape
        BSRC = cdiv_python(TSRC, args.block_size_k)
        MAX_TSRC = TSRC
        MAX_BSRC = BSRC
    else:
        NUM_PAGE, PAGE_SIZE, KV_HEAD, _ = args.k_cache.shape
        TSRC = None
        BSRC = None
        MAX_TSRC = NUM_PAGE * PAGE_SIZE
        MAX_BSRC = cdiv_python(MAX_TSRC, args.block_size_k)
    N = BSZ * HEAD
    # assert q.shape == k.shape
    BDST = cdiv_python(TDST, args.block_size_q)
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD == HEAD
    
    G = args.topk_head_group_size
    B = N // G
    assert (B * G) == N
    BK = cdiv_python(args.mask_k, args.block_size_k)
    
    context = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    
    # BLOCK_BK = 64 // block_size_k
    # if block_size_k > 4:
    #     BLOCK_BK = 128 // block_size_k
    # elif block_size_k > 8:
    #     BLOCK_BK = 256 // block_size_k
    BLOCK_BK = 64 // args.block_size_k
    assert BLOCK_BK > 0
    
    # sliding_window_size = min(sliding_window_size, block_size_k * 16)
    
    if args.rope_cos is not None:
        assert len(args.rope_cos.stride()) == 2
        assert len(args.rope_sin.stride()) == 2
    
    assert context.ndim == 4
    if ks_start_end is not None:
        assert ks_start_end.ndim == 3
    if indices is not None:
        assert indices.ndim == 3
    assert q.ndim == 4
    if k is not None:
        assert k.ndim == 4
        assert v.ndim == 4
    elif args.using_paged_cache:
        assert args.k_cache.ndim == 4
        assert args.v_cache.ndim == 4
    else:
        raise Exception()
    assert position_ids.ndim == 2
    
    grid = (HEAD, BDST, BSZ)
    pre_device = torch.get_default_device()
    torch.set_default_device(q.device)
    block_sparse_attention_cuda[grid](
        q, *q.stride(),
        k, *args.safe_stride(k, 4),
        v, *args.safe_stride(v, 4),
        position_ids, *position_ids.stride(),
        
        indices, *args.safe_stride(indices, 3),
        
        ks_start_end, *args.safe_stride(ks_start_end, 3),
        
        context, *context.stride(),
        
        HEAD, G, BK, TDST, MAX_TSRC, KV_HEAD_REPEAT,
        
        args.sliding_window_size,
        
        *args.args_extend(),
        *args.args_paged_kv_cache(),
        
        HID,
        args.block_size_q,
        args.block_size_k,
        # BLOCK_BK,
        
        # num_warps=4,
        # num_stages=2 if not using_extend else 1,
    )
    torch.set_default_device(pre_device)
    
    return context

@nvtx.annotate("masking_step_loop")
def masking_step_loop(
    q: Tensor,
    k: Tensor,
    chunk_offset: int,
    args: "HiPAttentionArgs"
):
    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        _, TSRC, _, _ = k.shape
    else:
        TSRC = None
    N = BSZ * HEAD
    
    # NOTE: this make ppl worse
    # with nvtx.annotate('k_adjust'):
    #     if topk_head_group_size > 1:
    #         k = k - k[:, :2, :].mean(-2, keepdim=True)
    
    indices_blocks = []
    ks_blocks = []
    ks_count_blocks = []
    ks_start_end_blocks = []
    scores_blocks = []
    key_access_log_blocks = []
    key_access_count_blocks = []
    indices_seed = ks_seed = None
    for i_chunk_tdst in range(0, args.chunk_size, args.block_size_q * args.step_size):
        idx_tdst = torch.arange(
            i_chunk_tdst, 
            i_chunk_tdst + args.block_size_q * args.step_size, 
            device=q.device
        )[None, :] + torch.arange(
            0,
            TDST,
            args.chunk_size,
            device=q.device,
        )[:, None] + chunk_offset
        idx_tdst = idx_tdst % TDST
        idx_tdst = idx_tdst.reshape(-1)
        if TSRC is not None:
            pos_tdst = (idx_tdst[None, :] + TSRC - TDST).expand(BSZ, -1) + 1
        else:
            pos_tdst = idx_tdst[None, :] + args.cache_seq_lens[:, None] - TDST + 1
        scores_seed = None
        with nvtx.annotate(f'masking_samples(seed={tuple(indices_seed.shape) if indices_seed is not None else None})'):
            for idx_sample in range(args.num_samples):
                with nvtx.annotate(f'masking_iteration_draft(idx_sample={idx_sample})'):
                    if args.low_res_sample_scale <= 1 and args.low_res_oversample_rate <= 1:
                        (
                            indices, 
                            ks, 
                            ks_count, 
                            ks_start_end, 
                            scores, 
                            group_sizes, 
                            key_access_log, 
                            key_access_count
                        ) = masking_iteration_draft(
                            q, 
                            k, 
                            position_ids=pos_tdst,
                            indices_seed=indices_seed,
                            ks_seed=ks_seed,
                            scores_seed=scores_seed,
                            indices_tdst=idx_tdst,
                            
                            args=args,
                        )
                        
                        indices_seed = indices
                        ks_seed = ks
                        scores_seed = scores
                        if key_access_log is not None:
                            key_access_log_blocks.append(key_access_log)
                        if key_access_count is not None:
                            key_access_count_blocks.append(key_access_count)
                    else:
                        assert isinstance(args.low_res_sample_scale, int)
                        low_mask_k = args.mask_k * args.low_res_oversample_rate
                        low_block_size_k = args.block_size_k * args.low_res_oversample_rate * args.low_res_sample_scale
                        
                        assert args.low_res_sample_scale >= 1
                        assert args.low_res_oversample_rate >= 1
                        assert isinstance(args.low_res_sample_scale, int)
                        assert isinstance(args.low_res_oversample_rate, int)
                        
                        # low_res_oversample_rate == group_size
                        # low_res_sample_scale == num block split
                        
                        # NOTE: following code is for downsample the seed from last step
                        """
                        # need to be num element low_mask_k // low_block_size_k
                        stride = low_res_oversample_rate * low_res_sample_scale
                        assert stride > 1
                        if indices_seed is not None:
                            indices_seed = indices_seed[:, :, ::stride]
                        if scores_seed is not None:
                            scores_seed = scores_seed[:, :, ::stride]
                        
                        if low_res_sample_scale > 1:
                            if ks_seed is not None:
                                ks_seed = torch.ceil(ks_seed / low_res_sample_scale).to(torch.int32)
                        
                        if low_res_oversample_rate > 1:
                            if indices_seed is not None:
                                scores_seed = None
                                indices_seed = indices_seed\
                                    .repeat_interleave(low_res_oversample_rate, dim=-1)\
                                    .view(*indices_seed.shape, 2)
                                indices_seed = indices_seed +\
                                    torch.arange(
                                        0, 
                                        low_res_oversample_rate * low_block_size_k, 
                                        low_block_size_k, 
                                        device=indices_seed.device
                                    )[None, None, None, :]
                                indices_seed = indices_seed.view(
                                    indices_seed.shape[0],
                                    indices_seed.shape[1],
                                    indices_seed.shape[2] * low_res_oversample_rate
                                )
                        """
                        
                        low_res_sample_config = args.clone()
                        low_res_sample_config.mask_k = low_mask_k
                        low_res_sample_config.block_size_k = low_block_size_k
                        low_res_sample_config.block_stride_k = args.low_res_oversample_block_stride_k
                        
                        with nvtx.annotate('low_res_sample'):
                            # TODO: reduce initial seeds
                            indices, ks, ks_count, ks_start_end, scores, group_sizes, key_access_log, key_access_count = masking_iteration_draft(
                                q[:, :, :], 
                                k[:, :, :], 
                                position_ids=pos_tdst,
                                indices_seed=indices_seed,
                                ks_seed=ks_seed,
                                scores_seed=scores_seed,
                                indices_tdst=idx_tdst,
                                
                                args=low_res_sample_config,
                            )
                            
                            indices_seed = indices
                            ks_seed = ks
                            scores_seed = scores
                            
                            # indices_for_seed = indices
                            # scores_for_seed = scores
                            # ks_for_seed = ks
                            
                            # NOTE: if we recurrent on low res, then upsampling is ignored for few steps
                            if args.num_samples > 1 and idx_sample < (args.num_samples - 1):
                                continue
                        
                        with nvtx.annotate('sample_division'):
                            if args.low_res_sample_scale > 1:
                                indices = indices[:, :, :, None] +\
                                    torch.arange(
                                        0, low_block_size_k, args.block_size_k * args.low_res_oversample_rate, 
                                        device=indices.device
                                    )[None, None, None, :]
                                indices = indices.view(indices.shape[0], indices.shape[1], -1)
                                ks = ks.mul(args.low_res_sample_scale)
                                group_sizes = torch.repeat_interleave(
                                    group_sizes, args.low_res_sample_scale, dim=-1
                                )
                                
                                # NOTE: block is break down, this is not accurate
                                scores = scores[:, :, :, None]\
                                    .expand(-1, -1, -1, 2)\
                                    .contiguous()\
                                    .view(scores.shape[0], scores.shape[1], -1)
                                
                                ks_count, ks_start_end = masking_iteration_draft_python_epilog(
                                    indices, ks, 
                                    cdiv_python(args.mask_k, args.block_size_k), 
                                    TSRC,
                                    ks.shape[0], 
                                    ks.shape[1], 
                                    args.topk_head_group_size
                                )
                        
                        with nvtx.annotate('downsample'):
                            if args.low_res_oversample_rate > 1:
                                init_indices = torch.full_like(
                                    indices, 
                                    fill_value=(cdiv_python(TSRC, args.block_size_k) + args.block_size_k + args.block_size_q) * args.topk_head_group_size
                                )
                                init_ks = torch.zeros_like(ks)
                                init_group_sizes = torch.zeros_like(group_sizes)
                                grid = (N // args.topk_head_group_size, init_group_sizes.shape[1], args.topk_head_group_size)
                                pre_device = torch.get_default_device()
                                torch.set_default_device(pos_tdst.device)
                                masking_iteration_draft_cuda_initialize[grid](
                                    None, *(0, 0, 0),
                                    None, *(0, 0),
                                    pos_tdst, *pos_tdst.stride(),
                                    
                                    init_indices, *init_indices.stride(),
                                    init_ks, *init_ks.stride(),
                                    init_group_sizes, *init_group_sizes.stride(),
                                    
                                    None, *(0, 0,),
                                    
                                    args.mask_k,
                                    args.block_size_q, 
                                    args.block_size_k, 
                                    
                                    args.sliding_window_size,
                                    
                                    args.topk_head_group_size, len(idx_tdst), TSRC, HEAD,
                                    
                                    cdiv_python(args.mask_k, args.block_size_k),
                                    
                                    # num_warps=min(max(cdiv_python(BLOCK_MASK_BLOCK_K, 32), 1), 32),
                                    num_warps=1,
                                    num_stages=1,
                                )
                                torch.set_default_device(pre_device)
                                
                                # init_indices.mul_(block_size_k)
                                
                                group_sizes_scaled = torch.maximum(group_sizes.float(), torch.ones_like(group_sizes)) * args.low_res_oversample_rate
                                
                                # print(init_group_sizes[0, idx_tdst[::32] < 1024, :10])
                                # print(group_sizes_scaled[0, idx_tdst[::32] < 1024, :10])
                                
                                mask_tdst = pos_tdst[::args.block_size_q] < args.mask_k * 2
                                group_sizes = torch.where(
                                    mask_tdst[None, :, None],
                                    init_group_sizes,
                                    group_sizes_scaled,
                                )
                                indices = torch.where(
                                    mask_tdst[None, :, None],
                                    init_indices * args.block_size_k,
                                    indices,
                                )
                                ks = torch.where(
                                    mask_tdst[None, :],
                                    init_ks,
                                    ks,
                                )
                                
                                indices, ks, ks_count, ks_start_end, scores, group_sizes, key_access_log, key_access_count = masking_iteration_draft(
                                    q[:, :, :], 
                                    k[:, :, :], 
                                    position_ids=pos_tdst,
                                    indices_seed=indices_seed,
                                    ks_seed=ks_seed,
                                    scores_seed=None,
                                    indices_tdst=idx_tdst,
                                    
                                    args=args,
                                )
                        
                        # use this indices for cache, if you want to downsample
                        """
                        indices_seed = indices
                        ks_seed = ks
                        scores_seed = scores
                        """
        
        if not args.traverse_from_last_step:
            indices_seed = ks_seed = None
        # if (chunk_size is not None) and ((((i_chunk_tdst + chunk_offset) // block_size_q + 1) % (chunk_size // block_size_q)) == 0):
        # if ((i_chunk_tdst + 1) % (chunk_size - chunk_offset)) == 0:
            # indices_seed = ks_seed = None
        
        indices_blocks.append(indices)
        ks_blocks.append(ks)
        ks_count_blocks.append(ks_count)
        ks_start_end_blocks.append(ks_start_end)
        scores_blocks.append(scores)
    
    if len(indices_blocks) == 1:
        indices = indices_blocks[0]
        ks = ks_blocks[0]
        ks_count = ks_count_blocks[0]
        ks_start_end = ks_start_end_blocks[0]
        scores = scores_blocks[0]
    else:
        indices = torch.cat(indices_blocks, dim=1)
        ks = torch.cat(ks_blocks, dim=1)
        ks_count = torch.cat(ks_count_blocks, dim=1)
        ks_start_end = torch.cat(ks_start_end_blocks, dim=1)
        scores = torch.cat(scores_blocks, dim=1)
        
    if len(key_access_log_blocks) == 0:
        key_access_log = None
        key_access_count = None
    elif len(key_access_log_blocks) == 1:
        key_access_log = key_access_log_blocks[0]
        key_access_count = key_access_count_blocks[0]
    else:
        key_access_log = torch.cat(key_access_log_blocks, dim=1)
        key_access_count = torch.cat(key_access_count_blocks, dim=1)
    
    # print(indices.shape)
    # print(ks.shape)
    # print(ks_count.shape)
    # print(ks_start_end.shape)
    # print(scores.shape)
    # torch.Size([32, 256, 256])
    # torch.Size([32, 256])
    # torch.Size([32, 256, 1])
    # torch.Size([32, 256, 2])
    # torch.Size([32, 256, 256])
    
    num_chunks = triton.cdiv(TDST, args.chunk_size)
    
    if num_chunks > 1:
        def permute_3d(x: Tensor):
            N, BDST, K = x.shape
            return x.view(N, triton.cdiv(BDST, num_chunks), num_chunks, K)\
                .permute(0, 2, 1, 3)\
                .reshape(N, BDST, K)
        
        indices = permute_3d(indices)
        ks = permute_3d(ks.unsqueeze(-1)).squeeze(-1)
        ks_count = permute_3d(ks_count)
        ks_start_end = permute_3d(ks_start_end)
        scores = permute_3d(scores)
    
    return indices, ks, ks_count, ks_start_end, scores, key_access_log, key_access_count

@numba.njit(parallel=True)
def access_log_to_dense(
    key_access_log: NdArray,
    key_access_count: NdArray,
    TSRC,
):
    B, BDST, K = key_access_log.shape
    out = np.zeros((B, BDST, TSRC), dtype=np.int32)
    for ib in numba.prange(B):
        for ibdst in numba.prange(BDST):
            nk = key_access_count[ib, ibdst]
            for ik in range(nk):
                out[ib, ibdst, key_access_log[ib, ibdst, ik]] += 1
    return out

@numba.njit
def img_reduce(
    img: NdArray,
    rh: int, rw: int
):
    H, W = img.shape
    RH = H // rh
    RW = W // rw
    out = np.zeros((RH, RW))
    for ih in range(RH):
        for iw in range(RW):
            chunk = img[ih * rh: ih * rh + rh, iw * rw: iw * rw + rw]
            scaler = np.mean(chunk)
            out[ih, iw] = scaler
    return out

@numba.njit
def incr_first_iteration(
    mask: NdArray, 
    block_size_q: int,
    mask_k: int,
    block_size_k: int,
    block_stride_k: int,
    sliding_window_size: int,
):
    B, BDST, TSRC = mask.shape
    for ib in range(B):
        for ibdst in range(BDST):
            _mask_k = mask_k * max(1, int(math.log2(ibdst * block_size_q / 8192)))
            for ibk in range(_mask_k // block_size_k):
                tsrc = ((ibdst + 1) * block_size_q - sliding_window_size) * (ibk / (_mask_k // block_size_k))
                tsrc = round(tsrc)
                if tsrc >= 0:
                    tsrc = tsrc - (tsrc % block_size_k)
                    for ioffset in range(block_stride_k - 1, block_size_k, block_stride_k):
                        mask[ib, ibdst, tsrc + ioffset] += 1

@numba.njit
def perform_lru(
    key_access_map,
    key_access_log,
    key_access_count,
    lru_budget,
):
    B, BDST, K = key_access_log.shape
    
    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = np.zeros((B, lru_budget,), dtype=np.int32) - 1
    loaded_key_timestamp = np.zeros((B, lru_budget,), dtype=np.int32)
    
    for ib in numba.prange(B): #prange
        for ibdst in range(1, BDST):
            last_accessed = key_access_log[:, ibdst-1, :]
            last_accessed_count = key_access_count[:, ibdst-1]
            # try to add last accessed to LRU cache
            for ik in range(last_accessed_count[ib]):
                current_pointer = last_accessed[ib, ik]
                in_cache = False
                least_timestamp_val = 999999999
                least_timestamp_idx = -1
                for icache in range(lru_budget):
                    if loaded_key_list[ib, icache] == current_pointer:
                        loaded_key_timestamp[ib, icache] = ibdst
                        # if in LRU cache, update life
                        in_cache = True
                    else:
                        if loaded_key_timestamp[ib, icache] < least_timestamp_val:
                            least_timestamp_val = loaded_key_timestamp[ib, icache]
                            least_timestamp_idx = icache
                # else, evict victim
                if not in_cache:
                    loaded_key_list[ib, least_timestamp_idx] = current_pointer
                    loaded_key_timestamp[ib, least_timestamp_idx] = ibdst
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ib, icache]
                if idx > 0:
                    loaded_key_mask[ib, ibdst, idx] = 1
    
    return loaded_key_mask

@numba.njit
def perform_lfu(
    key_access_map,
    key_access_log,
    key_access_count,
    lru_budget,
):
    B, BDST, K = key_access_log.shape
    
    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_list = np.zeros((B, lru_budget,), dtype=np.int32) - 1
    loaded_key_freq = np.zeros((B, lru_budget,), dtype=np.int32)
    
    for ib in numba.prange(B): #prange
        for ibdst in range(1, BDST):
            last_accessed = key_access_log[:, ibdst-1, :]
            last_accessed_count = key_access_count[:, ibdst-1]
            for icache in range(lru_budget):
                loaded_key_freq[ib, icache] -= 1
            # try to add last accessed to LRU cache
            for ik in range(last_accessed_count[ib]):
                current_pointer = last_accessed[ib, ik]
                in_cache = False
                least_freq_val = 999999999
                least_freq_idx = -1
                for icache in range(lru_budget):
                    if loaded_key_list[ib, icache] == current_pointer:
                        loaded_key_freq[ib, icache] += 1
                        # if in cache, update life
                        in_cache = True
                    else:
                        if loaded_key_freq[ib, icache] < least_freq_val:
                            least_freq_val = loaded_key_freq[ib, icache]
                            least_freq_idx = icache
                # else, evict victim
                if not in_cache:
                    loaded_key_list[ib, least_freq_idx] = current_pointer
                    loaded_key_freq[ib, least_freq_idx] = 1
            # submit to mask for debug
            for icache in range(lru_budget):
                idx = loaded_key_list[ib, icache]
                if idx >= 0:
                    loaded_key_mask[ib, ibdst, idx] = 1
    
    return loaded_key_mask

@numba.njit(parallel=True)
def perform_lru_heuristic(
    key_access_map,
    key_access_log,
    key_access_count,
    lru_budget,
    block_size_q = 32,
    block_size_k = 8,
    sliding_window_size = 512,
):
    B, BDST, K = key_access_log.shape
    
    loaded_key_mask = np.zeros_like(key_access_map)
    loaded_key_value = np.zeros((B, lru_budget,), dtype=np.int32) - 1
    loaded_key_first_value = np.zeros((B, lru_budget,), dtype=np.int32) - 1
    loaded_key_first_stamp = np.zeros((B, lru_budget,), dtype=np.int32)
    loaded_key_importance = np.zeros((B, lru_budget,), dtype=np.int32)
    
    for ib in numba.prange(B): #prange
        for ibdst in range(1, BDST):
            last_accessed = key_access_log[:, ibdst-1, :]
            last_accessed_count = key_access_count[:, ibdst-1]
            
            # prefetch keys using scaling
            if ibdst > (sliding_window_size // block_size_q):
                for _icache in range(lru_budget):
                    icache = lru_budget - _icache - 1
                    current_pointer = loaded_key_value[ib, icache]
                    if current_pointer >= 0:
                        first_ibdst = loaded_key_first_stamp[ib, icache]
                        first_value = loaded_key_first_value[ib, icache]
                        first_offset = first_value % block_size_k
                        new_position = (first_value // block_size_k) / first_ibdst * ibdst
                        new_position = math.ceil(new_position) * block_size_k + first_offset
                        
                        if new_position not in loaded_key_value[ib]:
                            loaded_key_value[ib, icache] = new_position
                        else:
                            loaded_key_value[ib, icache] = current_pointer
                            if new_position == current_pointer:
                                # when keep position
                                loaded_key_importance[ib, icache] -= 0
                            else:
                                # when collide
                                loaded_key_importance[ib, icache] -= 1
            # try to add last accessed to LRU cache
            # loaded_key_importance[ib] -= 1 # decay freq if LFU
            for ik in range(last_accessed_count[ib]):
                current_pointer = last_accessed[ib, ik]
                in_cache = False
                least_timestamp_val = 999999999
                least_timestamp_idx = -1
                for icache in range(lru_budget):
                    if loaded_key_value[ib, icache] == current_pointer:
                        loaded_key_importance[ib, icache] = ibdst
                        # loaded_key_importance[ib, icache] += 3
                        # if in LRU cache, update life
                        in_cache = True
                    else:
                        if loaded_key_importance[ib, icache] < least_timestamp_val:
                            least_timestamp_val = loaded_key_importance[ib, icache]
                            least_timestamp_idx = icache
                # else, evict victim
                if not in_cache:
                    new_position = (current_pointer // block_size_k) / (ibdst - 1) * ibdst
                    new_position = math.ceil(new_position) * block_size_k + (current_pointer % block_size_k)
                    if new_position not in loaded_key_value[ib, :]:
                        loaded_key_value[ib, least_timestamp_idx] = new_position
                        loaded_key_first_value[ib, least_timestamp_idx] = current_pointer
                        loaded_key_first_stamp[ib, least_timestamp_idx] = ibdst - 1
                        loaded_key_importance[ib, least_timestamp_idx] = ibdst
                    else:
                        for i in range(len(loaded_key_value[ib, :])):
                            if loaded_key_value[ib, i] == new_position:
                                loaded_key_value[ib, i] = new_position
                                loaded_key_first_value[ib, i] = current_pointer
                                loaded_key_first_stamp[ib, i] = ibdst - 1
                                loaded_key_importance[ib, i] = ibdst
                        loaded_key_value[ib, least_timestamp_idx] = current_pointer
                        loaded_key_first_value[ib, least_timestamp_idx] = current_pointer
                        loaded_key_first_stamp[ib, least_timestamp_idx] = ibdst - 1
                        loaded_key_importance[ib, least_timestamp_idx] = ibdst
            # submit to mask for debug, in realworld, time to fetch
            for icache in range(lru_budget):
                idx = loaded_key_value[ib, icache]
                if idx > 0:
                    loaded_key_mask[ib, ibdst, idx] = 1
    
    return loaded_key_mask

@nvtx.annotate('hip_masking')
def hip_masking(
    q: Tensor, 
    k: Optional[Tensor],
    args: "HiPAttentionArgs",
):
    assert (k is None and args.k_cache is not None) or (k is not None and args.k_cache is None)
    assert q.ndim == 4
    if k is not None:
        assert k.ndim == 4
    BSZ, TDST, HEAD, HID = q.shape
    G = args.topk_head_group_size
    B = BSZ * HEAD // G
    N = BSZ * HEAD
    
    args = args.clone()
    
    assert args.num_unions > 0
    if args.chunk_size is None:
        args.chunk_size = q.shape[1]
    assert args.chunk_size > 0
    assert args.chunk_size >= args.num_unions
    
    if args.step_size is None:
        args.step_size = cdiv_python(q.shape[1], args.block_size_q)
    assert args.step_size > 0
    assert args.step_size <= cdiv_python(q.shape[1], args.block_size_q), f'{args.step_size} <= {cdiv_python(q.shape[1], args.block_size_q)}'
    
    if args.using_sparq:
        raise Exception('vectorized head not support SparQ')
        BSZ, T, HEAD, D = q.shape
        q_score = q.view(
            BSZ, 
            triton.cdiv(T, block_size_q),
            block_size_k, 
            HEAD // topk_head_group_size, 
            topk_head_group_size, 
            D
        )
        _, sparq_ind = q_score\
            .abs()\
            .sum(dim=2)\
            .topk(k=sparq_hid, dim=-1, largest=True, sorted=False)
        sparq_ind, _ = torch.sort(sparq_ind, dim=-1)
    else:
        sparq_ind = None
    
    indices_sampled = []
    ks_sampled = []
    ks_count_sampled = []
    ks_start_end_sampled = []
    scores_sampled = []
    key_access_log_sampled = []
    key_access_count_sampled = []
    for i_chunk_offset in range(0, args.chunk_size, args.chunk_size // args.num_unions):
        (
            indices, 
            ks, 
            ks_count, 
            ks_start_end, 
            scores, 
            key_access_log, 
            key_access_count
        ) = masking_step_loop(
            q=q,
            k=k,
            chunk_offset=i_chunk_offset,
            args=args,
        )
        
        indices_sampled.append(indices)
        ks_sampled.append(ks)
        ks_count_sampled.append(ks_count)
        ks_start_end_sampled.append(ks_start_end)
        scores_sampled.append(scores)
        if key_access_log is not None:
            key_access_log_sampled.append(key_access_log)
        if key_access_count is not None:
            key_access_count_sampled.append(key_access_count)
    
    if len(indices_sampled) > 1:
        ignore_ranage = max(cdiv_python(args.mask_k, args.block_size_q), cdiv_python(args.chunk_size, args.block_size_q * args.num_unions)) * 2
        compute_range = cdiv_python(q.shape[1], args.block_size_q) - ignore_ranage
        
        bcs = args.chunk_size // args.block_size_q
        bcs_step = bcs // args.num_unions
        indices = torch.cat([
            x[:, bcs - bcs_step * ix: x.shape[1] - bcs_step * ix] 
            for ix, x in enumerate(indices_sampled)
        ], dim=-1)[:, -compute_range:]
        scores = torch.cat([
            x[:, bcs - bcs_step * ix: x.shape[1] - bcs_step * ix] 
            for ix, x in enumerate(scores_sampled)
        ], dim=-1)[:, -compute_range:]
        
        indices_to_sorted = torch.argsort(indices, dim=-1)
        
        indices = indices.gather(dim=-1, index=indices_to_sorted)
        scores = scores.gather(dim=-1, index=indices_to_sorted)
        
        unique_indices_mask = indices != torch.roll(indices, shifts=(1,), dims=(2,))
        scores.masked_fill_(~unique_indices_mask, float('-inf'))
        
        scores_to_highest = torch.argsort(
            scores, dim=-1, descending=True
        )[:, :, :triton.cdiv((args.mask_k * args.topk_head_group_size), args.block_size_k)]
        
        indices = indices.gather(dim=-1, index=scores_to_highest)
        scores = scores.gather(dim=-1, index=scores_to_highest)
        
        top_indices_to_sorted = torch.argsort(indices, dim=-1)
        
        indices = indices.gather(dim=-1, index=top_indices_to_sorted)
        scores = scores.gather(dim=-1, index=top_indices_to_sorted)
        
        indices_sampled[0][:, ignore_ranage:, :] = indices
        
        indices = indices_sampled[0]
        ks = ks_sampled[0]
        # ks_count = ks_count_sampled[0]
        # ks_start_end = ks_start_end_sampled[0]
        
        BSZ, TDST, H, _ = q.shape
        _, TSRC, _, _ = k.shape
        BDST = triton.cdiv(TDST, args.block_size_q)
        mask_block_k = triton.cdiv(args.mask_k, args.block_size_k)
        
        ks_count = torch.zeros((B, BDST, G), dtype=torch.int32, device=q.device)
        ks_start_end = torch.zeros((B, BDST, G + 1), dtype=torch.int32, device=q.device)
        
        BLOCK_BK = 128
        grid = (B, BDST, triton.cdiv(indices.shape[-1], BLOCK_BK))
        pre_device = torch.get_default_device()
        torch.set_default_device(indices.device)
        masking_iteration_draft_cuda_epiloge[grid](
            indices, *indices.stride(),
            ks, *ks.stride(),
            
            ks_count, *ks_count.stride(),
            ks_start_end, *ks_start_end.stride(),
            
            mask_block_k, TSRC, 
            
            G,
            BLOCK_BK,
        )
        torch.set_default_device(pre_device)
        
        ks = ks_count.sum(-1)
        if len(key_access_log_sampled) > 0:
            key_access_log = torch.cat(key_access_log_sampled, dim=1)
            key_access_count = torch.cat(key_access_count_sampled, dim=1)
        else:
            key_access_log = None
            key_access_count = None
    else:
        indices = indices_sampled[0]
        ks = ks_sampled[0]
        ks_count = ks_count_sampled[0]
        ks_start_end = ks_start_end_sampled[0]
        if len(key_access_log_sampled) > 0:
            key_access_log = key_access_log_sampled[0]
            key_access_count = key_access_count_sampled[0]
        else:
            key_access_log = None
            key_access_count = None
    
    if os.getenv('HIP_DEBUG', '0') == '1':
        B, TDST, H, HID = q.shape
        if k is not None:
            _, TSRC, _, _ = k.shape
        else:
            TSRC = torch.max(args.cache_seq_lens).item()
        N = B * H
        def render_mask():
            debug_mask = to_dense(
                indices.cpu().numpy(),
                ks.cpu().numpy(),
                None,
                cdiv_python(N, args.topk_head_group_size),
                TDST, 
                TSRC * args.topk_head_group_size, 
                args.block_size_q, 
                args.block_size_k * args.block_size_k_group,
            )
            plt.figure(figsize=(4*args.topk_head_group_size, 4))
            plt.imshow(debug_mask[0])
            plt.tight_layout()
            plt.savefig('dummy.png', dpi=96, bbox_inches='tight')
            print('saved dummy.png')
        render_mask()
        
        if key_access_log is not None:
            key_access_map = access_log_to_dense(
                key_access_log.cpu().numpy(),
                key_access_count.cpu().numpy(),
                TSRC,
            )
            key_access_mask = np.clip(key_access_map, 0, 1)
            
            def render_access_map_fullres():
                # mat = cv2.applyColorMap(key_access_map[0], cv2.COLORMAP_JET)
                for i in range(key_access_map.shape[0]):
                    path = f'dummy_access_map_fullres_{i}.png'
                    cv2.imwrite(path, (key_access_map[i] * 255).astype(np.uint8))
                    print(f'saved {path}')
            # render_access_map_fullres()
            
            # plot key access map
            def render_access_map():
                img = key_access_map[0]
                img = img_reduce(img, 1, args.block_size_q)
                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.colorbar()
                plt.title(f'avg access count (T={TSRC}, bq={args.block_size_q}, bk={args.block_size_k})')
                plt.tight_layout()
                plt.savefig('dummy_access.png', dpi=96, bbox_inches='tight')
                print('saved dummy_access.png')
            # render_access_map()
            
            def render_remain(window_size, prefetch_next_tokens, prefetch_middle_tokens):
                plt.figure(figsize=(6, 4))
                key_access_mask = (key_access_map > 0).astype(np.int32)
                key_remain_mask = key_access_mask[:, window_size-1:, :].copy()
                
                # rule 1: keep past N steps
                for i in range(window_size - 1):
                    key_remain_mask += key_access_mask[:, i:key_access_mask.shape[1] - window_size + 1 + i, :]
                key_remain_mask = np.clip(key_remain_mask, 0, 1)
                
                # rule 2: prefetch next tokens
                if prefetch_next_tokens:
                    shift = args.block_size_k
                    key_remain_mask[:, 1:, shift:] = np.clip(key_remain_mask[:, 1:, shift:] + key_access_mask[:, window_size-1:-1, :-shift], 0, 1)
                
                # rule 3: prefetch middle tokens
                if prefetch_middle_tokens:
                    pass
                
                unique_remain_issue = np.clip(key_remain_mask[:, 1:, :] - key_remain_mask[:, :-1, :], 0, 1)
                unique_remain_issue = np.sum(unique_remain_issue, axis=-1)
                xs = np.arange(args.block_size_q * window_size, TDST, args.block_size_q)
                for i in range(unique_remain_issue.shape[0]):
                    plt.plot(xs, unique_remain_issue[i])
                
                cache_keys = np.sum(key_remain_mask, axis=-1)[:, 1:]
                for i in range(cache_keys.shape[0]):
                    plt.plot(xs, cache_keys[i])
                
                last_unique_remain_issue = unique_remain_issue[:, -1].mean()
                last_cache_keys = cache_keys[:, -1].mean()
                
                plt.title(f'newly requested key\n(T={TSRC}, wnd={window_size * args.block_size_q} steps, last cache size={int(last_cache_keys)}, last request size={int(last_unique_remain_issue)})')
                plt.grid()
                plt.tight_layout()
                path = f'dummy_unique_remain_wnd{window_size}_pnext{prefetch_next_tokens}.png'
                plt.savefig(path, dpi=96, bbox_inches='tight')
                print(f'saved {path}')
            # render_remain(1, False, False)
            # render_remain(2, False, False)
            # render_remain(4, False, False)
            # render_remain(8, False, False)
            # render_remain(16, False, False)
            
            # render_remain(1, True, False)
            # render_remain(2, True, False)
            
            def plot_stats(
                name,
                loaded_key_mask,
            ):
                # calc fetchs
                fetched_key_mask = loaded_key_mask[:, 1:, :] - loaded_key_mask[:, :-1, :]
                fetched_key_mask = np.clip(fetched_key_mask, 0, 1)
                
                # calc misses
                missed_key_mask = np.clip(key_access_mask[:, 1:, :] - loaded_key_mask[:, 1:, :], 0, 1)
                
                cv2.imwrite(f'dummy_{name}_loaded.png', loaded_key_mask[0, 1:, :] * 255)
                cv2.imwrite(f'dummy_{name}_fetched.png', fetched_key_mask[0] * 255)
                cv2.imwrite(f'dummy_{name}_missed.png', missed_key_mask[0] * 255)
                cv2.imwrite(f'dummy_{name}_accessed.png', key_access_mask[0, 1:, :] * 255)
                
                # 0 (black): not loaded
                # 1 (white): loaded but not used
                # 2 (green): cache hit
                # 3 (red): missed
                load_map = cache_map = loaded_key_mask[0, 1:, :]
                access_map = key_access_map[0, 1:, :]
                cache_map = np.where(load_map * access_map, 2, cache_map)
                cache_map = np.where((1 - load_map) * access_map, 3, cache_map)
                colormap = np.array([
                    [0, 0, 0],
                    [255,255,255],
                    [0,255,0],
                    [0,0,255],
                ], dtype=np.int64)
                H, W = cache_map.shape
                cache_image = np.take(colormap.reshape(1, 1, 4, 3), cache_map.reshape(H, W, 1, 1), axis=-2)
                cache_image = np.reshape(cache_image, (H, W, 3))
                cv2.imwrite(f'dummy_{name}_combined.png', cache_image)
                
                accessed_key_counts = key_access_mask[:, 1:, :].sum(axis=-1)
                loaded_key_counts = loaded_key_mask[:, 1:, :].sum(axis=-1)
                fetched_key_counts = fetched_key_mask.sum(axis=-1)
                missed_key_counts = missed_key_mask.sum(axis=-1)
                xs = np.arange(args.block_size_q, TDST, args.block_size_q)
                
                plt.figure(figsize=(8, 12))
                plt.plot(xs, loaded_key_counts.T, color='gray')
                plt.plot(xs, fetched_key_counts.T, color='green')
                plt.plot(xs, missed_key_counts.T, color='red')
                plt.plot(xs, fetched_key_counts.T + missed_key_counts.T, color='orange')
                plt.plot(xs, accessed_key_counts.T, color='blue')
                plt.axhline(TSRC / args.block_stride_k, color='darkgray')
                plt.grid()
                filename = f'dummy_{name}_stats'
                path = f'{filename}.png'
                plt.savefig(path, dpi=96)
                print(f'saved {path}')
                
                accessed_count = accessed_key_counts.T[-1].mean()
                missed_count = missed_key_counts.T[-1].mean()
                print(f'cache hit ratio: {(1 - missed_count / accessed_count) * 100:.4f}')
                
                fetched_count = fetched_key_counts.T[-1].mean()
                fetched_mb = fetched_count * 32 * 32 * 128 / 1024 / 1024
                print(f'fetched tokens: {fetched_count:.1f}, {fetched_mb:.4f} MB, took {fetched_mb / 32:.2f} ms (bsz=1) / {fetched_mb / 32 * 16:.2f} ms (bsz=16) in PCIe 4.0')
            
            def render_heuristics():
                loaded_key_mask = np.zeros_like(key_access_mask)

                # load pre N steps
                keep_past_n_steps = 1
                for i in range(1, keep_past_n_steps+1):
                    loaded_key_mask[:, i:, :] += key_access_mask[:, :-i, :]
                loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
                
                # prefetch few iteration representatives and shift others
                first_iteration_mask = np.zeros_like(loaded_key_mask)
                incr_first_iteration(
                    first_iteration_mask,
                    args.block_size_q,
                    args.mask_k,
                    args.block_size_k,
                    args.block_stride_k,
                    args.sliding_window_size,
                )
                
                # uncomment this, for adding shift
                loaded_key_mask = np.clip(loaded_key_mask - first_iteration_mask, 0, 1)
                # loaded_key_mask[:, 1:, block_size_k:] += loaded_key_mask[:, :-1, :-block_size_k].copy()
                
                # union pre N steps
                union_past_n_steps = 1
                loaded_key_mask_no_union = loaded_key_mask.copy()
                for i in range(1, union_past_n_steps+1):
                    loaded_key_mask[:, i:, :] += loaded_key_mask_no_union[:, :-i, :]
                loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
                
                loaded_key_mask += first_iteration_mask
                loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
                
                # plot stats
                plot_stats('heuristic', loaded_key_mask)
            # render_heuristics()
            
            def render_lru(lru_budget=1024):
                loaded_key_mask = perform_lru(
                    key_access_map, 
                    key_access_log.cpu().numpy(), 
                    key_access_count.cpu().numpy(), 
                    lru_budget
                )
                # incr_first_iteration(
                #     loaded_key_mask,
                #     block_size_q,
                #     mask_k,
                #     block_size_k,
                #     block_stride_k,
                #     sliding_window_size,
                # )
                loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
                plot_stats(f'lru_{lru_budget}', loaded_key_mask)
            # render_lru(1024) # 3.12%
            # render_lru(2048) # 6.25%
            
            def render_lfu(lfu_budget=1024):
                loaded_key_map = perform_lfu(
                    key_access_map, 
                    key_access_log.cpu().numpy(), 
                    key_access_count.cpu().numpy(), 
                    lfu_budget
                )
                # incr_first_iteration(
                #     loaded_key_mask,
                #     block_size_q,
                #     mask_k,
                #     block_size_k,
                #     block_stride_k,
                #     sliding_window_size,
                # )
                loaded_key_mask = np.clip(loaded_key_map, 0, 1)
                plot_stats(f'lfu_{lfu_budget}', loaded_key_mask)
            # render_lfu(1024) # 3.12%
            # render_lfu(2048) # 6.25%
            
            def render_lru_heuristic(lru_budget=1024):
                loaded_key_mask = perform_lru_heuristic(
                    key_access_map, 
                    key_access_log.cpu().numpy(), 
                    key_access_count.cpu().numpy(), 
                    lru_budget, 
                    args.block_size_q,
                    args.block_size_k,
                    args.sliding_window_size,
                )
                # incr_first_iteration(
                #     loaded_key_mask,
                #     block_size_q,
                #     mask_k,
                #     block_size_k,
                #     block_stride_k,
                #     sliding_window_size,
                # )
                loaded_key_mask = np.clip(loaded_key_mask, 0, 1)
                plot_stats(f'lru_heuristic_{lru_budget}', loaded_key_mask)
            render_lru_heuristic(1024) # 12.5%
            render_lru_heuristic(2048) # 25.0%
            render_lru_heuristic(4096) # 50.0%
        # input('>>>')
    
    return indices, ks, ks_count, ks_start_end, key_access_log, key_access_count

@dataclass
class HiPAttentionOutputMetadata:
    indices: Tensor
    ks: Tensor
    ks_count: Tensor
    ks_start_end: Tensor
    key_access_log: Optional[Tensor]
    key_access_count: Optional[Tensor]

@dataclass
class HiPAttentionArgs:
    mask_k: int = 512
    
    block_size_q: int = 32
    block_stride_q: int = 2
    block_size_k: int = 8
    block_stride_k: int = 4
    block_size_k_group: int = 1
    
    sliding_window_size: int = 512
    sink_token_size: int = 32
    
    using_extend: bool = False
    rope_cos: Optional[Tensor] = None
    rope_sin: Optional[Tensor] = None
    self_extend_neighboor_window: int = 1024
    self_extend_group_size: int = 8
    
    topk_head_group_size: int = 1
    sample_method: str = 'first'
    branch_method: str = 'half'
    
    traverse_from_last_step: bool = False
    step_size: Optional[int] = None
    num_samples: int = 1
    chunk_size: Optional[int] = None
    num_unions: int = 1
    
    score_head_group_size: int = 1
    
    using_sparq: bool = False
    sparq_hid: int = 32
    
    low_res_sample_scale: int = 1
    low_res_oversample_rate: int = 1
    low_res_oversample_block_stride_k: int = 1
    
    output_key_access_log: bool = False
    
    q_quant: Optional[Tensor] = None
    k_quant: Optional[Tensor] = None
    
    sparq_ind: Optional[Tensor] = None
    
    k_cache: Optional[Tensor] = None
    v_cache: Optional[Tensor] = None
    cache_seq_lens: Optional[Tensor] = None
    block_table: Optional[Tensor] = None
    
    def __post_init__(self):
        if self.rope_cos is not None and self.rope_cos.ndim == 3:
            self.rope_cos = self.rope_cos.view(-1, self.rope_cos.shape[-1])
            self.rope_sin = self.rope_sin.view(-1, self.rope_sin.shape[-1])
        if self.q_quant is not None:
            assert self.q_quant.ndim == 4
            assert self.k_quant.ndim == 4
        self.using_paged_cache = self.k_cache is not None

    def clone(self):
        return copy.copy(self)

    def safe_stride(self, x: Optional[Tensor], ndim: int):
        if x is None:
            return tuple([0,] * ndim)
        else:
            return x.stride()

    def get_q_quant(self, q: Tensor):
        return self.q_quant if self.q_quant is not None else q

    def get_k_quant(self, k: Tensor):
        return self.k_quant if self.k_quant is not None else k

    def args_extend(self):
        return (
            self.using_extend,
            self.self_extend_neighboor_window,
            self.self_extend_group_size,
            *self.args_rope_cos(),
            *self.args_rope_sin(),
        )
    
    def args_rope_cos(self):
        return self.rope_cos, *self.safe_stride(self.rope_cos, 2),
    
    def args_rope_sin(self):
        return self.rope_sin, *self.safe_stride(self.rope_sin, 2),

    def args_sparq(self):
        if self.sparq_ind is None:
            using_sparq = False
            sparq_hid = 0
        else:
            using_sparq = True
            sparq_hid = self.sparq_ind.shape[-1]
            assert self.sparq_ind.ndim == 4
        
        return (
            using_sparq,
            sparq_hid,
            self.sparq_ind, *self.safe_stride(self.sparq_ind, 4),
        )
    
    def args_bq_bsq_bk_bsk(self):
        return (
            self.block_size_q,
            self.block_stride_q,
            self.block_size_k,
            self.block_stride_k,
        )
    
    def args_paged_kv_cache(self):
        using_page = self.using_paged_cache
        
        if using_page:
            assert self.v_cache is not None
            assert self.k_cache.ndim == self.v_cache.ndim
            assert self.k_cache.ndim == 4
            assert self.block_table is not None
            assert self.block_table.ndim == 2
            assert self.cache_seq_lens is not None
            assert self.cache_seq_lens.ndim == 1
            page_size = self.k_cache.shape[1]
        else:
            page_size = 0
        
        return (
            using_page,
            page_size,
            self.k_cache, *self.safe_stride(self.k_cache, 4),
            self.v_cache, *self.safe_stride(self.v_cache, 4),
            self.block_table, *self.safe_stride(self.block_table, 2),
            self.cache_seq_lens, *self.safe_stride(self.cache_seq_lens, 1),
        )

@nvtx.annotate('hip_attention')
@torch.inference_mode()
def hip_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor,
    
    args: Optional[HiPAttentionArgs] = None,  
    **kwargs,
):
    if args is None:
        args = HiPAttentionArgs(**kwargs)
    
    assert q.ndim == 4
    assert k.ndim == 4
    
    indices, ks, ks_count, ks_start_end, key_access_log, key_access_count = hip_masking(
        # TODO(heejun): apply PCA topk
        q=args.get_q_quant(q),
        k=args.get_k_quant(k),
        args=args,
    )
    
    HIP_RANDOM_MASK = os.getenv('HIP_RANDOM_MASK', '0') == '1'
    if HIP_RANDOM_MASK:
        for ib in tqdm.tqdm(range(indices.shape[0])):
            for ibdst in range(indices.shape[1]):
                assert args.topk_head_group_size == 1
                K = indices.shape[-1]
                tsrc = (ibdst + 1) * args.block_size_q - args.sliding_window_size + k.shape[-2] - q.shape[-2]
                tsrc = tsrc - (tsrc % args.block_size_q)
                if tsrc > args.mask_k:
                    rand_ids = torch.arange(args.block_size_k, tsrc, args.block_size_k, device=indices.device)
                    rp = torch.randperm(len(rand_ids), device=indices.device)
                    rand_ids = rand_ids[rp][:K-1] * args.block_size_k
                    indices[ib, ibdst, 1:len(rand_ids)+1] = rand_ids
                    indices[ib, ibdst, 0] = 0
    
    # return None, None
    
    position_ids = (
        torch.arange(0, q.shape[1], device=q.device) + k.shape[1] - q.shape[1] + 1
    )[None, :].expand(q.shape[0], -1)
    context = block_sparse_attention(
        q=q, 
        k=k, 
        v=v,
        position_ids=position_ids,
        
        indices=indices, 
        ks=ks, 
        ks_count=ks_count, 
        ks_start_end=ks_start_end,
        
        args=args
    )
    
    return context, HiPAttentionOutputMetadata(
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        key_access_log=key_access_log,
        key_access_count=key_access_count,
    )

@nvtx.annotate('paged_hip_attention')
def paged_hip_attention(
    q: Tensor,
    softmax_scale: float,
    args: HiPAttentionArgs
):
    B, TDST, HEAD, HID = q.shape
    assert args.k_cache.shape[-1] == HID
    N_PAGES, PAGE_SIZE, HEAD_KV, HID = args.k_cache.shape
    assert args.v_cache.shape == args.k_cache.shape
    
    assert args.block_table.shape[0] == B
    assert args.cache_seq_lens.shape[0] == B
    
    q = q * softmax_scale
    
    (
        indices, 
        ks, 
        ks_count, 
        ks_start_end, 
        key_access_log, 
        key_access_count
    ) = hip_masking(
        q=q,
        k=None,
        args=args
    )
    
    position_ids = torch.arange(0, TDST, device=q.device)[None, :] +\
        args.cache_seq_lens[:, None] - TDST + 1
    context = block_sparse_attention(
        q=q,
        k=None,
        v=None,
        position_ids=position_ids,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args,
    )
    
    return context, HiPAttentionOutputMetadata(
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        key_access_log=key_access_log,
        key_access_count=key_access_count,
    )

@nvtx.annotate('varlen_hip_attention')
def varlen_hip_attention(
    q: Tensor,
    softmax_scale: float,
    k: Tensor,
    v: Tensor,
    seq_lens: List[int],
    args: HiPAttentionArgs,
):
    q = q * softmax_scale
    
    outs = []
    total_length = 0
    for seq_len in seq_lens:
        seq_start = total_length
        seq_end = seq_start + seq_len
        total_length = seq_end
        
        # torch.cuda.synchronize()
        
        out, _ = hip_attention(
            q=q[seq_start:seq_end].unsqueeze(0), 
            k=k[seq_start:seq_end].unsqueeze(0), 
            v=v[seq_start:seq_end].unsqueeze(0),
            args=args.clone(),
        )
        
        # torch.cuda.synchronize()
        # from flash_attn import flash_attn_func
        # out_flash = flash_attn_func(
        #     q=q[seq_start:seq_end].unsqueeze(0), 
        #     k=k[seq_start:seq_end].unsqueeze(0), 
        #     v=v[seq_start:seq_end].unsqueeze(0),
        #     softmax_scale=1,
        #     causal=True,
        # )
        # print('varlen', seq_start, seq_end, out.shape, F.mse_loss(out, out_flash))

        outs.append(out.squeeze(0))
    
    return torch.cat(outs, dim=0)

@nvtx.annotate('paged_varlen_hiop_attention')
def paged_varlen_hip_attention(
    q: Tensor,
    softmax_scale: float,
    seq_lens: List[int],
    args: HiPAttentionArgs,
):
    # q = q * softmax_scale
    
    outs = []
    total_length = 0
    total_page_length = 0
    for idx_batch, seq_len in enumerate(seq_lens):
        seq_start = total_length
        seq_end = seq_start + seq_len
        
        page_start = total_page_length
        page_end = page_start + cdiv_python(seq_len, args.k_cache.shape[1])
        
        total_length = seq_len
        total_page_length = page_end
        
        curr_args = args.clone()
        curr_args.block_table = args.block_table[idx_batch:idx_batch+1]
        curr_args.cache_seq_lens = args.cache_seq_lens[idx_batch:idx_batch+1]
        
        out, _ = paged_hip_attention(
            q=q[seq_start:seq_end].unsqueeze(0),
            softmax_scale=softmax_scale,
            args=curr_args,
        )
        # print('varlen', seq_start, seq_end, out.shape)
        outs.append(out.squeeze(0))
    
    return torch.cat(outs, dim=0)

def main():
    debug_only = True
    seq_len = 32768
    seq_repeat = 1
    batch_repeat = 1
    if os.getenv('HIP_DEBUG', '1') == '0':
        seq_len = 32768
        # seq_len = 16384
        # seq_len = 8192
        seq_repeat = 1
        batch_repeat = 1
        debug_only = False
    
    q, k, v, out, cos, sin = load_checkouts(
        idx=0, 
        window=40, 
        seq_len=seq_len, 
        return_cos_sin=True, 
        dtype=torch.bfloat16
    )
    HEAD = q.shape[0]
    
    if seq_repeat > 1 or batch_repeat > 1:
        q = q.repeat(batch_repeat, seq_repeat, 1)
        k = k.repeat(batch_repeat, seq_repeat, 1)
        v = v.repeat(batch_repeat, seq_repeat, 1)
        out = out.repeat(batch_repeat, seq_repeat, 1)
        cos = cos.repeat(seq_repeat, 1)
        sin = sin.repeat(seq_repeat, 1)
    
    def reshape(x):
        N, T, H = x.shape
        x = x.contiguous()\
            .view(N // HEAD, HEAD, T, H)\
            .permute(0, 2, 1, 3)\
            .contiguous()
        assert x.shape == (N // HEAD, T, HEAD, H)
        assert x.is_contiguous()
        return x

    q = reshape(q)
    k = reshape(k)
    v = reshape(v)
    out = reshape(out)
    q_quant = q.to(torch.float8_e5m2).view(torch.uint8)#[...,::2]
    k_quant = k.to(torch.float8_e5m2).view(torch.uint8)#[...,::2]
    # q_quant = q
    # k_quant = k
    
    # num_queries = 1
    # q = q[:, -num_queries:]
    # q_quant = q_quant[:, -num_queries:]
    # out = out[:, -num_queries:,]
    
    print(q.shape, k.shape, v.shape)
    
    def fn():
        return hip_attention(
            q, k, v, 
            
            mask_k=512,
            
            block_size_q=32,
            block_stride_q=2,
            block_size_k=8,
            block_stride_k=4,
            block_size_k_group=1,
            
            sliding_window_size=512,
            sink_token_size=32,
            
            using_extend=False,
            rope_cos=cos,
            rope_sin=sin,
            self_extend_neighboor_window=1024,
            self_extend_group_size=4,
            
            topk_head_group_size=1,
            sample_method='first',
            branch_method='half',
            
            traverse_from_last_step=False,
            step_size=None,
            num_samples=1,
            chunk_size=None,
            num_unions=1,
            
            score_head_group_size=1,
            
            using_sparq=False,
            sparq_hid=64,
            
            low_res_sample_scale=1,
            low_res_oversample_rate=1,
            low_res_oversample_block_stride_k=1,
            
            q_quant=q_quant,
            k_quant=k_quant,
            
            # NOTE: change this to True to simulate key cache algorithms
            output_key_access_log=False,
        )
    
    if 'HIP_DEBUG' not in os.environ:
        os.environ['HIP_DEBUG'] = '1'
    
    context, metadata = fn()
    
    if context is not None:
        stderr = (out - context).abs().mean().item()
        stdcontext = torch.std_mean(out)[0].item()
        
        print(f'err = {stderr:.8f} ({stderr/stdcontext:.6f} sigma), out_std = {stdcontext:.8f}')
    
    if debug_only:
        return
    
    os.environ['HIP_DEBUG'] = '0'
    
    torch.cuda.synchronize()
    
    graph = None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    sample = 0
    elapsed = 0
    for i in range(50):
        if graph is None:
            for _ in range(3):
                fn()
            
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            
            print('graph compiled')
        
        if i > 3:
            start.record()
        graph.replay()
        if i > 3:
            end.record()
        
        if i > 3:
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(end)
            sample += 1
    
    if sample > 0:
        print(f'latency: {elapsed/sample:.6f} ms')

if __name__ == '__main__':
    main()