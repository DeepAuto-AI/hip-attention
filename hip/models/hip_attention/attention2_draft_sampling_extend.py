from dataclasses import dataclass
import math
import os
import time

from matplotlib import pyplot as plt
from hip.models.hip_attention.attention2_draft_prefetch import (
    hip_attention, 
    masking_iteration_draft,
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
    load_checkouts,
    block_sparse_attention,
    to_dense,
    adjust_rope,
    load_tokens,
)
# from hip.models.hip_attention.new_block_sparse import (
#     block_sparse_attention
# )
import torch
from torch import Tensor
from typing import List, Dict, Literal, Optional, Tuple
import triton
import triton.language as tl
import numpy as np
import cv2
import numba

@dataclass
class Stage:
    stage_block_size_q: int
    stage_block_stride_q: int
    stage_chunk_size: int
    stage_k: Optional[int]
    stage_stride: int
    
    require_realign_index: bool = False
    require_reset_score: bool = False
    require_post_sort: bool = False

@dataclass
class NopStage(Stage):
    require_realign_index: bool = True
    require_reset_score: bool = False
    require_post_sort: bool = True

@dataclass
class EvalScoreStage(Stage):
    block_chunk: int = 64
    stage_extend_backend: Optional[str] = None
    require_reset_score: bool = True
    require_post_sort: bool = True

@dataclass
class ScanStage(Stage):
    stage_extend_backend: Optional[str] = None
    require_realign_index: bool = True
    require_reset_score: bool = True
    require_post_sort: bool = True

@dataclass
class EnsembleScoreStage(Stage):
    reduce_method: str = 'sum'
    require_reset_score: bool = True
    require_post_sort: bool = True

@numba.njit(parallel=True)
def render_plot(
    out_indices_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q
):
    for i in numba.prange(out_indices_cpu.shape[1]):
        for j in range(out_indices_cpu.shape[-1]):
            # if j >= out_indices_cpu.shape[-1]: continue
            t = out_indices_cpu[0, i, DEBUG_HEAD, j] // BLOCK_SIZE_Q
            debug[i, t:t+1] = 1

@numba.njit(parallel=True)
def render_plot_dynamic(
    out_indices_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q, stage_k, chunk_size,
):
    for i in numba.prange(out_indices_cpu.shape[1]):
        for j in range(math.ceil(stage_k / chunk_size)):
            if j >= out_indices_cpu.shape[-1]: continue
            t = out_indices_cpu[0, i, DEBUG_HEAD, j] // BLOCK_SIZE_Q
            debug[i, t:t+math.ceil(chunk_size / BLOCK_SIZE_Q)] = 1

@numba.njit(parallel=True)
def render_plot_sampled(
    out_indices_cpu, debug, DEBUG_HEAD, BLOCK_CHUNK, chunk_count, TDST, sink_token_size, 
):
    for i in numba.prange(out_indices_cpu.shape[1]):
        t_chunk_size = math.ceil(TDST / chunk_count * BLOCK_CHUNK)
        # print(i, t_chunk_size)
        for j in range(max(
            0,
            out_indices_cpu.shape[-1]
        )):
            if j >= out_indices_cpu.shape[-1]: continue
            t = (out_indices_cpu[0, i, DEBUG_HEAD, j] - sink_token_size) // BLOCK_CHUNK + sink_token_size // BLOCK_CHUNK
            t = t // t_chunk_size * t_chunk_size
            debug[i, t:t+t_chunk_size] = 1

@numba.njit(parallel=True)
def render_plot_ks(
    indices, ks, debug, DEBUG_HEAD, BLOCK_SIZE_Q
):
    for i in numba.prange(indices.shape[1]):
        k = ks[DEBUG_HEAD, i]
        for j in range(indices.shape[-1]):
            if j >= k: continue
            t = indices[DEBUG_HEAD, i, j] // BLOCK_SIZE_Q
            debug[i, t:t+1] = 1

DEBUG = (os.getenv('HIP_DEBUG', '0') == '1')
DEBUG_RENDER = (os.getenv('HIP_DEBUG_RENDER', '1') == '1')

@triton.jit
def load_keys_with_rope(
    K, 
    stride_k_bsz, 
    stride_k_tsrc, 
    stride_k_head_kv, 
    stride_k_hid,
    
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    
    # paged attention args template
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
    
    # offload cache args template
    USING_OFFLOAD_CACHE,
    OFFLOAD_CACHE_METHOD,
    OFFLOAD_CACHE_BUDGET,
    OFFLOAD_CACHE_KV_HEAD,
    OFFLOAD_CACHE_K_TABLES,
    stride_offload_cache_k_tables_n,
    stride_offload_cache_k_tables_t,
    OFFLOAD_CACHE_K_BANKS,
    stride_offload_cache_k_banks_n,
    stride_offload_cache_k_banks_page,
    stride_offload_cache_k_banks_offset,
    stride_offload_cache_k_banks_hid,
    OFFLOAD_CACHE_K_BANK_STATS,
    stride_offload_cache_k_bank_stats_n,
    stride_offload_cache_k_bank_stats_page,
    stride_offload_cache_k_bank_stats_k,
    OFFLOAD_CACHE_COUNTERS,
    stride_offload_cache_counters_n,
    stride_offload_cache_counters_k,
    
    queries,
    
    idx_bsz,
    idx_tsrc,
    idx_head_kv,
    idx_hid,
    idx_chunk,
    mask_tsrc_active,
    mask_tdst,
    mask_hid,
    
    real_pos_tdst_min,
    model_context_length,
    num_sinks,
    
    USING_EXTEND,
    EXTEND_BACKEND,
    NEED_APPLY_ROPE,
    BLOCK_CHUNK,
    BLOCK_HID,
    IS_RIGHT,
):
    keys_left = load_tokens(
        K, 
        stride_k_bsz,
        stride_k_tsrc,
        stride_k_head_kv,
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
        
        USING_OFFLOAD_CACHE,
        OFFLOAD_CACHE_METHOD,
        OFFLOAD_CACHE_BUDGET,
        OFFLOAD_CACHE_KV_HEAD,
        True,
        OFFLOAD_CACHE_K_TABLES,
        stride_offload_cache_k_tables_n,
        stride_offload_cache_k_tables_t,
        OFFLOAD_CACHE_K_BANKS,
        stride_offload_cache_k_banks_n,
        stride_offload_cache_k_banks_page,
        stride_offload_cache_k_banks_offset,
        stride_offload_cache_k_banks_hid,
        OFFLOAD_CACHE_K_BANK_STATS,
        stride_offload_cache_k_bank_stats_n,
        stride_offload_cache_k_bank_stats_page,
        stride_offload_cache_k_bank_stats_k,
        OFFLOAD_CACHE_COUNTERS,
        stride_offload_cache_counters_n,
        stride_offload_cache_counters_k,
        
        idx_bsz,
        idx_tsrc[None, :],
        idx_head_kv,
        idx_hid[:, None],
        
        mask_tsrc_active[None, :] & mask_hid[:, None],
        
        BLOCK_CHUNK,
    ).to(queries.dtype)
    
    if USING_EXTEND:
        real_pos_tdst_max = tl.sum(mask_tdst.to(tl.int32)) + real_pos_tdst_min
        tsrc_extend = tl.maximum(0, real_pos_tdst_max - model_context_length)
        if NEED_APPLY_ROPE or (tsrc_extend >= 0):
            old_tsrc = idx_tsrc
            
            if EXTEND_BACKEND == 'dynamic_extend':
                window = model_context_length // 4
                
                new_tsrc = tl.where(
                    (idx_tsrc >= (real_pos_tdst_max - window)) | (real_pos_tdst_max <= model_context_length),
                    idx_tsrc,
                    # idx_tsrc * 0 + real_pos_tdst_max,
                    ((idx_tsrc.to(tl.float32) - (real_pos_tdst_min - window)) * ((model_context_length - window) / (real_pos_tdst_min - window)).to(tl.float32)).to(tl.int32) + (real_pos_tdst_min - window)
                )
                # new_tsrc = idx_tsrc * 0 + real_pos_tdst_max
                new_tsrc = tl.maximum(real_pos_tdst_max - model_context_length, new_tsrc)
            elif EXTEND_BACKEND == 'self_extend':
                window = 8192
                group_size = 16
                
                new_tsrc = tl.where(
                    idx_tsrc >= (real_pos_tdst_max - window),
                    idx_tsrc,
                    tl.where(
                        real_pos_tdst_max <= model_context_length,
                        idx_tsrc,
                        (idx_tsrc - real_pos_tdst_min) // group_size + real_pos_tdst_min
                    )
                )
                new_tsrc = tl.maximum(0, new_tsrc)
            elif EXTEND_BACKEND == 'relative':
                new_tsrc = idx_chunk * 0
                if IS_RIGHT:
                    new_tsrc += 1
            elif EXTEND_BACKEND == 'streaming':
                # streaming
                new_tsrc = idx_chunk
            else:
                raise Exception()
            
            if not NEED_APPLY_ROPE:
                tl.static_assert(False)
                keys_left = keys_left.trans(1, 0)
                keys_left = adjust_rope(
                    keys_left,
                    old_tsrc,
                    new_tsrc,
                    mask_tsrc_active,
                    idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    BLOCK_CHUNK,
                    BLOCK_HID,
                    NEED_APPLY_ROPE,
                ).to(keys_left.dtype)
                keys_left = tl.trans(keys_left, 1, 0)
                keys_left = (keys_left * mask_tsrc_active[None, :]).to(keys_left.dtype)
            else:
                keys_left_rot = load_tokens(
                    K, 
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head_kv,
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
                    
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_METHOD,
                    OFFLOAD_CACHE_BUDGET,
                    OFFLOAD_CACHE_KV_HEAD,
                    True,
                    OFFLOAD_CACHE_K_TABLES,
                    stride_offload_cache_k_tables_n,
                    stride_offload_cache_k_tables_t,
                    OFFLOAD_CACHE_K_BANKS,
                    stride_offload_cache_k_banks_n,
                    stride_offload_cache_k_banks_page,
                    stride_offload_cache_k_banks_offset,
                    stride_offload_cache_k_banks_hid,
                    OFFLOAD_CACHE_K_BANK_STATS,
                    stride_offload_cache_k_bank_stats_n,
                    stride_offload_cache_k_bank_stats_page,
                    stride_offload_cache_k_bank_stats_k,
                    OFFLOAD_CACHE_COUNTERS,
                    stride_offload_cache_counters_n,
                    stride_offload_cache_counters_k,
                    
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head_kv,
                    ((idx_hid + BLOCK_HID // 2) % BLOCK_HID)[:, None],
                    
                    mask_tsrc_active[None, :],
                    
                    BLOCK_CHUNK,
                ).to(queries.dtype)
                
                # TODO: multiply -right
                # keys_left_rot = tl.where(
                #     (idx_hid + BLOCK_HID // 2)[:, None] < BLOCK_HID,
                #     -keys_left_rot,
                #     keys_left_rot
                # )
                keys_left_rot = keys_left_rot * (((idx_hid + BLOCK_HID // 2)[:, None] < BLOCK_HID) * (-2) + 1).to(keys_left_rot.dtype)
                
                cos_new = tl.load(
                    COS +\
                        new_tsrc[None, :].to(tl.int64) * stride_cos_t +\
                        (idx_hid % (BLOCK_HID // 2))[:, None] * stride_cos_hid,
                    mask=mask_tsrc_active[None, :],
                    other=0.0,
                ).to(keys_left.dtype)
                sin_new = tl.load(
                    SIN +\
                        new_tsrc[None, :].to(tl.int64) * stride_sin_t +\
                        (idx_hid % (BLOCK_HID // 2))[:, None] * stride_sin_hid,
                    mask=mask_tsrc_active[None, :],
                    other=0.0,
                ).to(keys_left.dtype)
                
                keys_left = keys_left * cos_new + keys_left_rot * sin_new
                # keys_left = keys_left * cos_new + keys_left_rot * sin_new
                # keys_left = keys_left * keys_left + keys_left * keys_left
        # else:
        #     if NEED_APPLY_ROPE:
        #         keys_left = keys_left.trans(1, 0)
        #         keys_left = adjust_rope(
        #             keys_left,
        #             idx_tsrc,
        #             idx_tsrc,
        #             mask_tsrc_active,
        #             idx_hid,
        #             COS, stride_cos_t, stride_cos_hid,
        #             SIN, stride_sin_t, stride_sin_hid,
        #             BLOCK_CHUNK,
        #             BLOCK_HID,
        #             NEED_APPLY_ROPE,
        #         ).to(keys_left.dtype)
        #         keys_left = tl.trans(keys_left, 1, 0)
    
    return keys_left

@triton.jit
def chunk_controllable_sampling_mask_cuda(
    Q, 
    stride_q_bsz, 
    stride_q_tdst, 
    stride_q_head, 
    stride_q_hid,
    K, 
    stride_k_bsz, 
    stride_k_tsrc, 
    stride_k_head_kv, 
    stride_k_hid,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    
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
    
    # offload cache args template
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_METHOD: tl.constexpr,
    OFFLOAD_CACHE_BUDGET: tl.constexpr,
    OFFLOAD_CACHE_KV_HEAD: tl.constexpr,
    OFFLOAD_CACHE_K_TABLES,
    stride_offload_cache_k_tables_n,
    stride_offload_cache_k_tables_t,
    OFFLOAD_CACHE_K_BANKS,
    stride_offload_cache_k_banks_n,
    stride_offload_cache_k_banks_page,
    stride_offload_cache_k_banks_offset,
    stride_offload_cache_k_banks_hid,
    OFFLOAD_CACHE_K_BANK_STATS,
    stride_offload_cache_k_bank_stats_n,
    stride_offload_cache_k_bank_stats_page,
    stride_offload_cache_k_bank_stats_k,
    OFFLOAD_CACHE_COUNTERS,
    stride_offload_cache_counters_n,
    stride_offload_cache_counters_k,
    
    INDICES_LEFT, 
    stride_indices_left_bsz, 
    stride_indices_left_bdst, 
    stride_indices_left_head,
    stride_indices_left_chunk,
    
    INDICES_RIGHT, 
    stride_indices_right_bsz, 
    stride_indices_right_bdst, 
    stride_indices_right_head,
    stride_indices_right_chunk,
    
    OUT_SCORES, 
    stride_out_scores_bsz, 
    stride_out_scores_bdst, 
    stride_out_scores_head,
    stride_out_scores_chunk,
    
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    
    CHUNK_COUNT: int,
    MAX_TSRC: int,
    TDST: int,
    HEAD: int,
    sliding_window_size: int,
    num_sinks: int,
    model_context_length: int,
    
    BLOCK_HID: tl.constexpr = 128,
    BLOCK_SIZE_Q: tl.constexpr = 32,
    STRIDE_Q: tl.constexpr = 1,
    BLOCK_CHUNK: tl.constexpr = 32,
    HEAD_GROUP: tl.constexpr = 4,
    REDUCE: tl.constexpr = 'max',
    USING_EXTEND: tl.constexpr = False,
    EXTEND_BACKEND: tl.constexpr = 'dynamic_extend',
    NEED_APPLY_ROPE: tl.constexpr = False,
    TERMINATE_SIZE: tl.constexpr = 1,
    SCAN_STRIDE: tl.constexpr = 1,
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)
    BDST_SCAN = tl.cdiv(BDST, SCAN_STRIDE)
    BCHUNK = tl.cdiv(CHUNK_COUNT, BLOCK_CHUNK)
    
    pid = tl.program_id(0).to(tl.int64)
    
    idx_bdst_scan = pid % BDST_SCAN
    pid = pid // BDST_SCAN
    idx_bchunk = pid % BCHUNK
    pid = pid // BCHUNK
    idx_head = pid % HEAD
    pid = pid // HEAD
    idx_bsz = pid
    
    # idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // STRIDE_Q) * STRIDE_Q
    # mask_tdst = idx_tdst < TDST
    idx_tdst = ((BDST - 1) - (BDST_SCAN - 1) * SCAN_STRIDE + idx_bdst_scan * SCAN_STRIDE) * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // STRIDE_Q) * STRIDE_Q
    mask_tdst = (idx_tdst < TDST) & (idx_tdst >= 0)
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < BLOCK_HID #(tl.arange(0, BLOCK_HID) % 4) == 0
    
    pos_tdst = tl.load(
        POS +\
            idx_bsz * stride_pos_bsz +\
            idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    
    # real_pos_tdst_min = idx_bdst * BLOCK_SIZE_Q + TSRC - TDST
    real_pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst, 99999999999))
    real_pos_tdst_min = tl.where(tl.sum(mask_tdst.to(tl.int32)) > 0, real_pos_tdst_min, -1)
    
    if real_pos_tdst_min < 0:
        return
    
    pos_tdst_min = (real_pos_tdst_min - sliding_window_size).to(tl.int32)
    pos_tdst_min = tl.maximum(pos_tdst_min, 0)
    
    idx_chunk = idx_bchunk * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
    mask_chunk = idx_chunk < CHUNK_COUNT
    
    idx_tsrc_left = tl.load(
        INDICES_LEFT +\
            idx_bsz * stride_indices_left_bsz +\
            idx_bdst_scan * stride_indices_left_bdst +\
            idx_head * stride_indices_left_head +\
            idx_chunk * stride_indices_left_chunk,
        mask=mask_chunk,
        other=MAX_TSRC,
    ).to(tl.int32)
    
    idx_tsrc_right = tl.load(
        INDICES_RIGHT +\
            idx_bsz * stride_indices_right_bsz +\
            idx_bdst_scan * stride_indices_right_bdst +\
            idx_head * stride_indices_right_head +\
            idx_chunk * stride_indices_right_chunk,
        mask=mask_chunk,
        other=MAX_TSRC,
    ).to(tl.int32)
    
    if (real_pos_tdst_min + BLOCK_SIZE_Q * SCAN_STRIDE) < tl.min(idx_tsrc_left):
        return
    
    # mask_chunk = mask_chunk & (idx_tsrc_left < (real_pos_tdst_min - sliding_window_size + BLOCK_SIZE_Q))

    # max_chunk_size = tl.ceil(MAX_TSRC / CHUNK_COUNT).to(tl.float32)
    max_chunk_size = tl.max(idx_tsrc_right - idx_tsrc_left).to(tl.float32)
    
    scores = tl.zeros((BLOCK_CHUNK,), dtype=tl.float32) - 32000.0
    
    queries_sum = tl.zeros((BLOCK_SIZE_Q // STRIDE_Q, BLOCK_HID), dtype=tl.float32)
    queries_counter = tl.zeros((BLOCK_SIZE_Q // STRIDE_Q,), dtype=tl.int32)
    tl.static_assert(BLOCK_SIZE_Q // STRIDE_Q > 0)
    
    for i_offset in tl.static_range(STRIDE_Q):
        idx_tdst_iter = idx_tdst + i_offset
        mask_tdst_iter = mask_tdst & (idx_tdst_iter < TDST)
        queries_iter = tl.load(
            Q + \
                idx_bsz * stride_q_bsz +\
                idx_tdst_iter[:, None] * stride_q_tdst +\
                idx_head * stride_q_head +\
                idx_hid[None, :] * stride_q_hid,
            mask=mask_tdst_iter[:, None] & mask_hid[None, :],
            other=0
        )
        
        if USING_EXTEND:
            if NEED_APPLY_ROPE or (real_pos_tdst_min >= model_context_length):
                old_tdst = pos_tdst
                if EXTEND_BACKEND == 'dynamic_extend':
                    new_tdst = pos_tdst
                elif EXTEND_BACKEND == 'self_extend':
                    new_tdst = pos_tdst
                elif EXTEND_BACKEND == 'relative':
                    new_tdst = pos_tdst * 0 + 1 + sliding_window_size
                elif EXTEND_BACKEND == 'streaming':
                    # streaming
                    new_tdst = tl.minimum(pos_tdst, CHUNK_COUNT + sliding_window_size)
                else:
                    raise Exception()
                
                if NEED_APPLY_ROPE:
                    queries_rot = tl.load(
                        Q +\
                            idx_bsz * stride_q_bsz +\
                            idx_tdst_iter[:, None] * stride_q_tdst +\
                            idx_head * stride_q_head +\
                            ((idx_hid + BLOCK_HID // 2) % BLOCK_HID)[None, :] * stride_q_hid,
                        mask=mask_tdst_iter[:, None],
                        other=0,
                        # cache_modifier='.cg',
                        # eviction_policy='evict_last',
                        # volatile=True,
                    )
                    
                    cos_new = tl.load(
                        COS +\
                            new_tdst[:, None].to(tl.int64) * stride_cos_t +\
                            (idx_hid % (BLOCK_HID // 2))[None, :] * stride_cos_hid,
                        mask=mask_tdst_iter[:, None],
                        other=0.0,
                    ).to(queries_iter.dtype)
                    sin_new = tl.load(
                        SIN +\
                            new_tdst[:, None].to(tl.int64) * stride_sin_t +\
                            (idx_hid % (BLOCK_HID // 2))[None, :] * stride_sin_hid,
                        mask=mask_tdst_iter[:, None],
                        other=0.0,
                    ).to(queries_iter.dtype)
                    
                    # queries_rot = tl.where(
                    #     (idx_hid + BLOCK_HID // 2)[None, :] < BLOCK_HID,
                    #     -queries_rot,
                    #     queries_rot
                    # )
                    queries_rot = queries_rot * (((idx_hid + BLOCK_HID // 2)[None, :] < BLOCK_HID) * (-2) + 1).to(queries_rot.dtype)
                    
                    queries_iter = (queries_iter * cos_new + queries_rot * sin_new).to(queries_iter.dtype)
                else:
                    queries_iter = adjust_rope(
                        queries,
                        old_tdst,
                        new_tdst,
                        mask_tdst_iter,
                        idx_hid,
                        COS, stride_cos_t, stride_cos_hid,
                        SIN, stride_sin_t, stride_sin_hid,
                        BLOCK_SIZE_Q // STRIDE_Q, 
                        BLOCK_HID,
                        NEED_APPLY_ROPE,
                    ).to(queries_iter.dtype)
                    queries_iter = (queries_iter * mask_tdst_iter[:, None]).to(queries_iter.dtype)
        
        queries_sum += queries_iter
        queries_counter += mask_tdst_iter.to(tl.int32)
    
    queries = ((queries_sum / (queries_counter[:, None] + 1e-12)) * mask_tdst[:, None]).to(Q.dtype.element_ty)
    
    while (max_chunk_size >= TERMINATE_SIZE):
        max_chunk_size /= 2.0
        mask_tsrc_active = mask_chunk & (idx_tsrc_left < idx_tsrc_right) & (idx_tsrc_left <= pos_tdst_min)
        # mask_tsrc_active = mask_tsrc_active & (idx_tsrc_left < (real_pos_tdst_min - sliding_window_size + BLOCK_SIZE_Q))
        idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2
        
        idx_tsrc = (idx_tsrc_left + idx_tsrc_center) // 2
        keys_left = load_keys_with_rope(
            K, 
            stride_k_bsz, 
            stride_k_tsrc, 
            stride_k_head_kv, 
            stride_k_hid,
            
            COS, stride_cos_t, stride_cos_hid,
            SIN, stride_sin_t, stride_sin_hid,
            
            # paged attention args template
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
            
            # offload cache args template
            USING_OFFLOAD_CACHE,
            OFFLOAD_CACHE_METHOD,
            OFFLOAD_CACHE_BUDGET,
            OFFLOAD_CACHE_KV_HEAD,
            OFFLOAD_CACHE_K_TABLES,
            stride_offload_cache_k_tables_n,
            stride_offload_cache_k_tables_t,
            OFFLOAD_CACHE_K_BANKS,
            stride_offload_cache_k_banks_n,
            stride_offload_cache_k_banks_page,
            stride_offload_cache_k_banks_offset,
            stride_offload_cache_k_banks_hid,
            OFFLOAD_CACHE_K_BANK_STATS,
            stride_offload_cache_k_bank_stats_n,
            stride_offload_cache_k_bank_stats_page,
            stride_offload_cache_k_bank_stats_k,
            OFFLOAD_CACHE_COUNTERS,
            stride_offload_cache_counters_n,
            stride_offload_cache_counters_k,
            
            queries,
            
            idx_bsz,
            idx_tsrc,
            idx_head // HEAD_GROUP,
            idx_hid,
            idx_chunk,
            mask_tsrc_active,
            mask_tdst,
            mask_hid,
            
            real_pos_tdst_min,
            model_context_length,
            num_sinks,
            
            USING_EXTEND,
            EXTEND_BACKEND,
            NEED_APPLY_ROPE,
            BLOCK_CHUNK,
            BLOCK_HID,
            False,
        )
            
        scores_left = tl.dot(
            (queries * (tl.sqrt(BLOCK_HID * 1.0) / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(queries.dtype)).to(queries.dtype),
            (keys_left.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(queries.dtype)).to(queries.dtype),
            allow_tf32=True,
            out_dtype=tl.float32,
        ).to(queries.dtype)
        
        if REDUCE == 'max':
            scores_left = tl.where(mask_tdst[:, None], scores_left, float('-inf'))
            scores_left = tl.max(scores_left, axis=0).to(scores_left.dtype)
        elif REDUCE == 'mean':
            scores_left = tl.where(mask_tdst[:, None], scores_left, float('0'))
            scores_left = tl.sum(scores_left, axis=0).to(scores_left.dtype)
            scores_left = (scores_left / tl.sum(mask_tdst.to(tl.float32))).to(scores_left.dtype)
        else:
            raise Exception()
        scores_left = tl.where(mask_tsrc_active, scores_left, float('-inf')).to(scores_left.dtype)
        
        idx_tsrc = (idx_tsrc_center + idx_tsrc_right) // 2
        keys_right = load_keys_with_rope(
            K, 
            stride_k_bsz, 
            stride_k_tsrc, 
            stride_k_head_kv, 
            stride_k_hid,
            
            COS, stride_cos_t, stride_cos_hid,
            SIN, stride_sin_t, stride_sin_hid,
            
            # paged attention args template
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
            
            # offload cache args template
            USING_OFFLOAD_CACHE,
            OFFLOAD_CACHE_METHOD,
            OFFLOAD_CACHE_BUDGET,
            OFFLOAD_CACHE_KV_HEAD,
            OFFLOAD_CACHE_K_TABLES,
            stride_offload_cache_k_tables_n,
            stride_offload_cache_k_tables_t,
            OFFLOAD_CACHE_K_BANKS,
            stride_offload_cache_k_banks_n,
            stride_offload_cache_k_banks_page,
            stride_offload_cache_k_banks_offset,
            stride_offload_cache_k_banks_hid,
            OFFLOAD_CACHE_K_BANK_STATS,
            stride_offload_cache_k_bank_stats_n,
            stride_offload_cache_k_bank_stats_page,
            stride_offload_cache_k_bank_stats_k,
            OFFLOAD_CACHE_COUNTERS,
            stride_offload_cache_counters_n,
            stride_offload_cache_counters_k,
            
            queries,
            
            idx_bsz,
            idx_tsrc,
            idx_head // HEAD_GROUP,
            idx_hid,
            idx_chunk,
            mask_tsrc_active,
            mask_tdst,
            mask_hid,
            
            real_pos_tdst_min,
            model_context_length,
            num_sinks,
            
            USING_EXTEND,
            EXTEND_BACKEND,
            NEED_APPLY_ROPE,
            BLOCK_CHUNK,
            BLOCK_HID,
            True,
        )
        
        scores_right = tl.dot(
            (queries * (tl.sqrt(BLOCK_HID * 1.0) / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(queries.dtype)).to(queries.dtype),
            (keys_right.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(queries.dtype)).to(queries.dtype),
            allow_tf32=True,
            out_dtype=tl.float32,
        ).to(queries.dtype)
        
        if REDUCE == 'max':
            scores_right = tl.where(mask_tdst[:, None], scores_right, float('-inf'))
            scores_right = tl.max(scores_right, axis=0).to(scores_right.dtype)
        elif REDUCE == 'mean':
            scores_right = tl.where(mask_tdst[:, None], scores_right, float('0'))
            scores_right = tl.sum(scores_right, axis=0).to(scores_right.dtype)
            scores_right = (scores_right / tl.sum(mask_tdst.to(tl.float32))).to(scores_right.dtype)
        else:
            raise Exception()
        scores_right = tl.where(mask_tsrc_active, scores_right, float('-inf')).to(scores_right.dtype)
        
        mask_left_win = scores_left > scores_right
        idx_tsrc_left = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                idx_tsrc_left,
                idx_tsrc_center,
            ),
            idx_tsrc_left
        )
        
        idx_tsrc_right = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                idx_tsrc_center,
                idx_tsrc_right,
            ),
            idx_tsrc_right
        )
        
        scores = tl.maximum(
            scores,
            tl.where(
                mask_tsrc_active,
                tl.where(
                    mask_left_win,
                    scores_left,
                    scores_right,
                ),
                scores,
            )
        )
    
    # idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2
    # idx_tsrc_left = idx_tsrc_center - TERMINATE_SIZE // 2
    # idx_tsrc_right = idx_tsrc_left + TERMINATE_SIZE
    
    tl.store(
        INDICES_LEFT +\
            idx_bsz * stride_indices_left_bsz +\
            idx_bdst_scan * stride_indices_left_bdst +\
            idx_head * stride_indices_left_head +\
            idx_chunk * stride_indices_left_chunk,
        value=idx_tsrc_left,
        mask=mask_chunk,
    )
    
    tl.store(
        INDICES_RIGHT +\
            idx_bsz * stride_indices_right_bsz +\
            idx_bdst_scan * stride_indices_right_bdst +\
            idx_head * stride_indices_right_head +\
            idx_chunk * stride_indices_right_chunk,
        value=idx_tsrc_right,
        mask=mask_chunk,
    )
    
    tl.store(
        OUT_SCORES +\
            idx_bsz * stride_out_scores_bsz +\
            idx_bdst_scan * stride_out_scores_bdst +\
            idx_head * stride_out_scores_head +\
            idx_chunk * stride_out_scores_chunk,
        value=scores,
        mask=mask_chunk,
    )

@triton.jit
def calculate_chunk_score(
    Q, 
    stride_q_bsz, 
    stride_q_tdst, 
    stride_q_head, 
    stride_q_hid,
    K, 
    stride_k_bsz, 
    stride_k_tsrc, 
    stride_k_head_kv, 
    stride_k_hid,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    
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
    
    # offload cache args template
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_METHOD: tl.constexpr,
    OFFLOAD_CACHE_BUDGET: tl.constexpr,
    OFFLOAD_CACHE_KV_HEAD: tl.constexpr,
    OFFLOAD_CACHE_K_TABLES,
    stride_offload_cache_k_tables_n,
    stride_offload_cache_k_tables_t,
    OFFLOAD_CACHE_K_BANKS,
    stride_offload_cache_k_banks_n,
    stride_offload_cache_k_banks_page,
    stride_offload_cache_k_banks_offset,
    stride_offload_cache_k_banks_hid,
    OFFLOAD_CACHE_K_BANK_STATS,
    stride_offload_cache_k_bank_stats_n,
    stride_offload_cache_k_bank_stats_page,
    stride_offload_cache_k_bank_stats_k,
    OFFLOAD_CACHE_COUNTERS,
    stride_offload_cache_counters_n,
    stride_offload_cache_counters_k,
    
    INDICES_LEFT,
    stride_indices_left_bsz,
    stride_indices_left_bdst,
    stride_indices_left_head,
    stride_indices_left_chunk,
    
    INDICES_RIGHT,
    stride_indices_right_bsz,
    stride_indices_right_bdst,
    stride_indices_right_head,
    stride_indices_right_chunk,
    
    OUT_SCORES,
    stride_out_scores_bsz,
    stride_out_scores_bdst,
    stride_out_scores_head,
    stride_out_scores_chunk,
    
    model_context_length,
    sliding_window_size,
    num_sinks,
    max_chunk_size,
    
    TDST,
    BDST,
    BDST_SCAN,
    N_HEAD,
    N_CHUNK,
    HEAD_GROUP,
    
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    SCAN_STRIDE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    REDUCE: tl.constexpr = 'max',
):
    pid = tl.program_id(0).to(tl.int64)
    
    # idx_chunk = pid % N_CHUNK
    # pid = pid // N_CHUNK
    idx_head = pid % N_HEAD
    pid = pid // N_HEAD
    idx_bdst_scan = pid % BDST_SCAN
    pid = pid // BDST_SCAN
    idx_bsz = pid
    
    tl.static_assert(
        (NEED_APPLY_ROPE and USING_EXTEND) or\
        (not (NEED_APPLY_ROPE or USING_EXTEND))
    )
    
    idx_tdst = idx_bdst_scan * SCAN_STRIDE * BLOCK_SIZE_Q + (BDST * BLOCK_SIZE_Q - BDST_SCAN * SCAN_STRIDE * BLOCK_SIZE_Q) + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
    mask_tdst = (idx_tdst < TDST) & (idx_tdst >= 0)
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < BLOCK_HID
    
    pos_tdst = tl.load(
        POS +\
            idx_bsz * stride_pos_bsz +\
            idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst, 999999999))
    pos_tdst_max = tl.max(pos_tdst)
    
    # real_pos_tdst_min = idx_bdst * BLOCK_SIZE_Q + TSRC - TDST
    # real_pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst, 99999999999))
    
    # pos_tdst_min = (real_pos_tdst_min - sliding_window_size - num_sinks).to(tl.int32)
    # pos_tdst_min = tl.maximum(pos_tdst_min, 0)
    
    queries = tl.load(
        Q + \
            idx_bsz * stride_q_bsz +\
            idx_tdst[:, None] * stride_q_tdst +\
            idx_head * stride_q_head +\
            idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0
    )
    
    if NEED_APPLY_ROPE and USING_EXTEND:
        if EXTEND_BACKEND == 'dynamic_extend':
            new_tdst = pos_tdst
        elif EXTEND_BACKEND == 'self_extend':
            new_tdst = pos_tdst
        elif EXTEND_BACKEND == 'streaming':
            new_tdst = tl.minimum(pos_tdst, N_CHUNK + sliding_window_size)
        elif EXTEND_BACKEND == 'relative':
            new_tdst = pos_tdst * 0 + sliding_window_size
        else:
            raise Exception()
        
        queries_rot = tl.load(
            Q +\
                idx_bsz * stride_q_bsz +\
                idx_tdst[:, None] * stride_q_tdst +\
                idx_head * stride_q_head +\
                ((idx_hid + BLOCK_HID // 2) % BLOCK_HID)[None, :] * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0,
        )
        
        cos_new = tl.load(
            COS +\
                new_tdst[:, None].to(tl.int64) * stride_cos_t +\
                (idx_hid % (BLOCK_HID // 2))[None, :] * stride_cos_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)
        sin_new = tl.load(
            SIN +\
                new_tdst[:, None].to(tl.int64) * stride_sin_t +\
                (idx_hid % (BLOCK_HID // 2))[None, :] * stride_sin_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)
        
        queries_rot = queries_rot * (((idx_hid + BLOCK_HID // 2)[None, :] < BLOCK_HID) * (-2) + 1).to(queries_rot.dtype)
        
        queries = (queries * cos_new + queries_rot * sin_new).to(queries.dtype)
    
    for idx_chunk_start in range(0, N_CHUNK, BLOCK_CHUNK):
    # for idx_chunk in range(tl.cdiv(pos_tdst_max, max_chunk_size)):
        idx_chunk = tl.arange(0, BLOCK_CHUNK) + idx_chunk_start
        mask_chunk = idx_chunk < N_CHUNK
        idx_tsrc_left = tl.load(
            INDICES_LEFT +\
                idx_bsz * stride_indices_left_bsz +\
                idx_bdst_scan * stride_indices_left_bdst +\
                idx_head * stride_indices_left_head +\
                idx_chunk * stride_indices_left_chunk,
            mask=mask_chunk,
            other=987654321,
        ).to(tl.int64)
        
        idx_tsrc_right = tl.load(
            INDICES_RIGHT +\
                idx_bsz * stride_indices_right_bsz +\
                idx_bdst_scan * stride_indices_right_bdst +\
                idx_head * stride_indices_right_head +\
                idx_chunk * stride_indices_right_chunk,
            mask=mask_chunk,
            other=987654321,
        ).to(tl.int64)
        
        if tl.min(idx_tsrc_left) <= pos_tdst_max:
            idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2
            idx_tsrc_left = tl.maximum(0, idx_tsrc_center - BLOCK_SIZE_K // 2)
            idx_tsrc = idx_tsrc_left[:, None] + tl.arange(0, BLOCK_SIZE_K // BLOCK_STRIDE_K)[None, :] * BLOCK_STRIDE_K
            idx_tsrc = tl.ravel(idx_tsrc)
            mask_tsrc = idx_tsrc <= (tl.max(pos_tdst) - sliding_window_size)
            
            keys = load_keys_with_rope(
                K, 
                stride_k_bsz, 
                stride_k_tsrc, 
                stride_k_head_kv, 
                stride_k_hid,
                
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                
                # paged attention args template
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
                
                # offload cache args template
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_METHOD,
                OFFLOAD_CACHE_BUDGET,
                OFFLOAD_CACHE_KV_HEAD,
                OFFLOAD_CACHE_K_TABLES,
                stride_offload_cache_k_tables_n,
                stride_offload_cache_k_tables_t,
                OFFLOAD_CACHE_K_BANKS,
                stride_offload_cache_k_banks_n,
                stride_offload_cache_k_banks_page,
                stride_offload_cache_k_banks_offset,
                stride_offload_cache_k_banks_hid,
                OFFLOAD_CACHE_K_BANK_STATS,
                stride_offload_cache_k_bank_stats_n,
                stride_offload_cache_k_bank_stats_page,
                stride_offload_cache_k_bank_stats_k,
                OFFLOAD_CACHE_COUNTERS,
                stride_offload_cache_counters_n,
                stride_offload_cache_counters_k,
                
                queries,
                
                idx_bsz,
                idx_tsrc,
                idx_head // HEAD_GROUP,
                idx_hid,
                idx_chunk,
                mask_tsrc,
                mask_tdst,
                mask_hid,
                
                pos_tdst_min,
                model_context_length,
                num_sinks,
                
                USING_EXTEND,
                EXTEND_BACKEND,
                NEED_APPLY_ROPE,
                BLOCK_SIZE_K,
                BLOCK_HID,
                True,
            )
            
            scores = tl.dot(
                (queries * (tl.sqrt(BLOCK_HID * 1.0) / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(queries.dtype)).to(queries.dtype),
                (keys.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(queries.dtype)).to(queries.dtype),
                allow_tf32=True,
                out_dtype=tl.float32,
            ).to(queries.dtype)
            
            if REDUCE == 'max':
                scores_reduced = tl.where(
                    mask_tdst[:, None] & mask_tsrc[None, :],
                    scores,
                    -32000.0
                )
                scores_reduced = tl.reshape(
                    scores_reduced,
                    BLOCK_SIZE_Q // BLOCK_STRIDE_Q,
                    BLOCK_CHUNK,
                    BLOCK_SIZE_K // BLOCK_STRIDE_K,
                )
                scores_reduced = tl.max(
                    scores_reduced,
                    axis=0
                )
                scores_reduced = tl.max(
                    scores_reduced,
                    axis=-1
                )
            # elif REDUCE == 'mean':
            #     scores_reduced = tl.sum(tl.where(
            #         mask_tdst[:, None] & mask_tsrc[None, :],
            #         scores,
            #         0
            #     )) / tl.sum((mask_tdst[:, None] & mask_tsrc[None, :]).to(tl.int32))
            else:
                raise Exception()
            
            tl.store(
                OUT_SCORES +\
                    idx_bsz * stride_out_scores_bsz +\
                    idx_bdst_scan * stride_out_scores_bdst +\
                    idx_head * stride_out_scores_head +\
                    idx_chunk * stride_out_scores_chunk,
                value=scores_reduced,
                mask=mask_chunk,
            )
        else:
            tl.store(
                OUT_SCORES +\
                    idx_bsz * stride_out_scores_bsz +\
                    idx_bdst_scan * stride_out_scores_bdst +\
                    idx_head * stride_out_scores_head +\
                    idx_chunk * stride_out_scores_chunk,
                value=-32000.0,
                mask=mask_chunk,
            )

def dual_stage_quadratic_hip_attention(
    q: Tensor, 
    k: Optional[Tensor], 
    v: Optional[Tensor], 
    args: HiPAttentionArgs,
    second_stage_k: int = 1024,
    stages: List[Stage] = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=32,
            stage_stride=1,
            stage_k=32768,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=8,
            stage_stride=1,
            stage_k=8192,
        ),
    ],
    model_context_length = 131072,
    
    # kernel args,
    mask_only = False,
    block_sparse_block_size_q: Optional[int] = 64,
    
    # scan_stride: int = 1,
    # scan_block_stride_q: int = -1,
    scan_early_terminate: int = 1,
    stage_early_terminate: int = 1,
    # scan_extend_backend: str = 'dynamic_extend',
    scan_extend_backend: str = 'relative',
    sa_extend_backend: str = 'streaming',
    low_percent: float = 0.0,
    low_k_ratio: float = 1.0,
    dim_to_lower: Literal['head', 'seq'] = 1,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
    q_mask: Optional[torch.Tensor] = None,
    k_mask: Optional[torch.Tensor] = None,
    idx_pca_hid_q: Optional[torch.Tensor] = None,
    idx_pca_hid_k: Optional[torch.Tensor] = None,
):
    DEBUG_HEAD = -1
    global DEBUG
    
    if q_mask is None:
        q_mask = q
    if k_mask is None:
        k_mask = k
    
    BLOCK_HID = q_mask.shape[-1]
    
    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        BSZ, TSRC, HEAD_KV, HID = k.shape
        assert v.shape == k.shape
        MAX_TSRC = TSRC
    else:
        MAX_TSRC = args.k_cache.shape[0] * args.k_cache.shape[1]
        MAX_TSRC = int(os.getenv('EXTEND_LEN', '128')) * 1024
        HEAD_KV = args.k_cache.shape[-2]
        TSRC = MAX_TSRC
        # print('asdf', args.k_cache.shape, MAX_TSRC, HEAD_KV, q.shape)
    
    assert len(stages) > 0
    STAGE_STRIDE = stages[0].stage_stride
    BLOCK_SIZE_Q = stages[0].stage_block_size_q
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    BDST_SCAN = triton.cdiv(BDST, STAGE_STRIDE)
    BLOCK_CHUNK = args.block_size_k
    chunk_size = args.mask_k
    chunk_count = triton.cdiv(max(0, MAX_TSRC - args.sink_token_size - args.sliding_window_size), chunk_size)
    
    args = args.clone()
    args.sliding_window_size = max(0, args.sliding_window_size - args.mask_k)
    
    if torch.cuda.is_current_stream_capturing() or args.position_ids is not None:
        assert args.position_ids is not None
        position_ids = args.position_ids
    else:
        position_ids = (torch.arange(0, TDST, device=q.device) + (TSRC - TDST))[None, :].expand(BSZ, TDST)
    assert position_ids.shape == (BSZ, TDST), position_ids.shape
    
    if cached_metadata is None:
        scan_require_recalculate_score = True and (scan_early_terminate > 1)
        stage_require_recalculate_score = True and (stage_early_terminate > 1)
        
        indices_left = torch.zeros(
            (BSZ, BDST_SCAN, HEAD, chunk_count), 
            device=q.device,
            dtype=torch.int64
        )

        indices_left[:, :, :, :] = (
            torch.floor(
                torch.arange(0, chunk_count, device=q.device, dtype=torch.float64) * chunk_size + args.sink_token_size
            ).to(indices_left.dtype)
        )[None, None, None, :]
        indices_right = indices_left + chunk_size
        indices_right.clamp_max_(MAX_TSRC - args.sliding_window_size)
        
        out_scores = torch.full(
            (BSZ, BDST_SCAN, HEAD, triton.next_power_of_2(chunk_count)), 
            device=q.device,
            dtype=q.dtype,
            fill_value=-32000.0
        )
        
        # print(q.shape, k.shape, args.rope_cos.shape, args.rope_sin.shape, TDST, TSRC)
        
        # print('neeeed rope', args.need_apply_rope)
        
        # print('8f8fh', indices_left.shape, chunk_size, chunk_count, MAX_TSRC, args.sink_token_size, args.sliding_window_size)
        
        # pre_device = torch.cuda.current_device()
        # torch.cuda.set_device(q.device)
        # grid = (
        #     BSZ *\
        #     triton.cdiv(chunk_count, BLOCK_CHUNK) *\
        #     BDST_SCAN *\
        #     HEAD,
        # )
        # chunk_controllable_sampling_mask_cuda[grid](
        #     q_pca, *q_pca.stride(),
        #     k_pca, *args.safe_stride(k_pca, 4),
        #     position_ids, *position_ids.stride(),
            
        #     *args.args_paged_kv_cache(),
        #     *args.args_offload_cache(True),
            
        #     indices_left, *indices_left.stride(),
        #     indices_right, *indices_right.stride(),
        #     out_scores, *out_scores.stride(),
        #     args.rope_cos, *args.safe_stride(args.rope_cos, 2),
        #     args.rope_sin, *args.safe_stride(args.rope_sin, 2),
            
        #     chunk_count,
        #     MAX_TSRC,
        #     TDST,
        #     HEAD,
        #     args.sliding_window_size,
        #     args.sink_token_size,
        #     # model_context_length if (not scan_extend_backend == 'streaming') else 0,
        #     model_context_length,
            
        #     BLOCK_HID=BLOCK_HID,
        #     BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        #     STRIDE_Q=scan_block_stride_q if scan_block_stride_q > 0 else args.block_stride_q,
        #     BLOCK_CHUNK=BLOCK_CHUNK,
        #     HEAD_GROUP=HEAD // HEAD_KV,
        #     USING_EXTEND=args.using_extend,
        #     EXTEND_BACKEND=scan_extend_backend,
        #     # EXTEND_BACKEND='relative',
        #     NEED_APPLY_ROPE=args.need_apply_rope,
        #     TERMINATE_SIZE=scan_early_terminate,
        #     SCAN_STRIDE=scan_stride,
            
        #     num_warps=4,
        #     num_stages=2,
        # )
        
        # if scan_require_recalculate_score:
        #     grid = (
        #         BSZ * BDST_SCAN * HEAD,
        #     )
        #     calculate_chunk_score[grid](
        #         q_pca, *q_pca.stride(),
        #         k_pca, *args.safe_stride(k_pca, 4),
        #         position_ids, *position_ids.stride(),
        #         args.rope_cos, *args.safe_stride(args.rope_cos, 2),
        #         args.rope_sin, *args.safe_stride(args.rope_sin, 2),
                
        #         *args.args_paged_kv_cache(),
        #         *args.args_offload_cache(True),
                
        #         indices_left, *indices_left.stride(),
        #         indices_right, *indices_right.stride(),
        #         out_scores, *out_scores.stride(),
                
        #         # model_context_length if (not scan_extend_backend == 'streaming') else 0,
        #         model_context_length,
        #         args.sliding_window_size,
        #         args.sink_token_size,
        #         chunk_size,
                
        #         TDST,
        #         BDST,
        #         BDST_SCAN,
        #         HEAD,
        #         chunk_count,
        #         HEAD // HEAD_KV,
                
        #         USING_EXTEND=args.using_extend,
        #         NEED_APPLY_ROPE=args.need_apply_rope,
        #         EXTEND_BACKEND=scan_extend_backend,
        #         BLOCK_HID=BLOCK_HID,
        #         BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        #         BLOCK_STRIDE_Q=args.block_stride_q,
        #         BLOCK_SIZE_K=scan_early_terminate,
        #         BLOCK_STRIDE_K=args.block_stride_k,
        #         SCAN_STRIDE=scan_stride,
                
        #         # num_warps=2,
        #     )
        
        # torch.cuda.set_device(pre_device)
        
        # if (len(stages) > 0):
        #     first_stage_bk = stages[0].stage_k // chunk_size
        #     out_scores = out_scores[..., :indices_left.shape[-1]]
        #     _, t_indices = out_scores.topk(k=min(out_scores.shape[-1], first_stage_bk), dim=-1, sorted=False)
        #     indices_left = indices_left.gather(dim=-1, index=t_indices[..., :indices_left.shape[-1]])
        #     indices_right = indices_right.gather(dim=-1, index=t_indices[..., :indices_right.shape[-1]])
        # else:
        #     out_scores = out_scores[..., :indices_left.shape[-1]]
        #     _, t_indices = out_scores.sort(dim=-1, descending=True, stable=False)
        #     indices_left = indices_left.gather(dim=-1, index=t_indices[..., :indices_left.shape[-1]])
        #     indices_right = indices_right.gather(dim=-1, index=t_indices[..., :indices_right.shape[-1]])
        
        # print('fif88', indices_left.shape, chunk_size)
        
        # STAGE_STRIDE = STAGE_STRIDE
        
        for i_stage, stage_info in enumerate(stages):
            # if stage_chunk_size > chunk_size: continue
            # if stage_k > TSRC: continue
            
            stage_block_stride_q = stage_info.stage_block_stride_q
            stage_chunk_size = stage_info.stage_chunk_size
            stage_k = stage_info.stage_k
            is_quadratic = i_stage == 0
            
            if i_stage > 0:
                assert (stage_k % chunk_size) == 0, f'{stage_k} % {chunk_size}'
                indices_left = indices_left[..., :stage_k // chunk_size]
                require_align = stage_info.require_realign_index
                if require_align:
                    indices_left = ((indices_left - args.sink_token_size) // chunk_size * chunk_size + args.sink_token_size)
                    indices_right = (indices_left + chunk_size)
                else:
                    indices_right = indices_right[..., :stage_k // chunk_size]
                out_scores = out_scores[..., :stage_k // chunk_size]
                if stage_info.require_reset_score:
                    out_scores.fill_(-32000.0)
                
                indices_left, t_indices = indices_left.sort(dim=-1)
                indices_right = indices_right.gather(dim=-1, index=t_indices)
                out_scores = out_scores.gather(dim=-1, index=t_indices)
                
                if BLOCK_SIZE_Q != stage_info.stage_block_size_q:
                    assert stage_info.stage_block_size_q > 0
                    assert BLOCK_SIZE_Q > stage_info.stage_block_size_q
                    assert (BLOCK_SIZE_Q % stage_info.stage_block_size_q) == 0
                    
                    num_split = BLOCK_SIZE_Q // stage_info.stage_block_size_q
                    BLOCK_SIZE_Q = stage_info.stage_block_size_q
                    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
                    BDST_SCAN = triton.cdiv(BDST, STAGE_STRIDE)
                    
                    indices_left = indices_left.repeat_interleave(num_split, 1)[:, -BDST:].contiguous()
                    indices_right = indices_right.repeat_interleave(num_split, 1)[:, -BDST:].contiguous()
                    out_scores = out_scores.repeat_interleave(num_split, 1)[:, -BDST:].contiguous()
                
                if STAGE_STRIDE != stage_info.stage_stride:
                    assert stage_info.stage_stride < STAGE_STRIDE
                    assert STAGE_STRIDE > 0
                    indices_left = indices_left.repeat_interleave(STAGE_STRIDE // stage_info.stage_stride, 1)[:, -BDST:].contiguous()
                    indices_right = indices_right.repeat_interleave(STAGE_STRIDE // stage_info.stage_stride, 1)[:, -BDST:].contiguous()
                    out_scores = out_scores.repeat_interleave(STAGE_STRIDE // stage_info.stage_stride, 1)[:, -BDST:].contiguous()
                    STAGE_STRIDE = stage_info.stage_stride
                
                # if DEBUG and DEBUG_RENDER and (not torch.cuda.is_current_stream_capturing()) and (BDST > 10) and (i_stage == 1):
                #     out_indices_cpu = indices_left.cpu().numpy()
                #     debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK)))
                #     render_plot_sampled(out_indices_cpu, debug, DEBUG_HEAD, BLOCK_CHUNK, chunk_count, TDST, args.sink_token_size)
                #     cv2.imwrite('dummy_sampled.png', debug * 255)
                #     print('saved dummy_sampled.png')
                
                assert (chunk_size % stage_chunk_size) == 0
                splits = chunk_size // stage_chunk_size
                chunk_sizes = ((indices_right - indices_left).float() / splits).clamp_min_(0)
                indices_left = indices_left[..., None] + (torch.arange(0, splits, device=q_mask.device)[None, None, None, None, :] * chunk_sizes[..., None]).floor().long()
                indices_left = indices_left.flatten(-2, -1)
                indices_right = indices_right[..., None] - (((splits - 1) - torch.arange(0, splits, device=q.device)[None, None, None, None, :]) * chunk_sizes[..., None]).floor().long()
                indices_right = indices_right.flatten(-2, -1)
                out_scores = out_scores.repeat_interleave(splits, -1)
            else:
                assert stage_info.stage_k is None, 'first stage always quadratic'
                assert isinstance(stage_info, ScanStage), 'frist stage always scan'
                STAGE_STRIDE = stage_info.stage_stride
            
            chunk_size = stage_chunk_size
            chunk_count = indices_left.shape[-1]
            BLOCK_CHUNK = max(16, triton.next_power_of_2(min(chunk_count, BLOCK_CHUNK)))
            
            pre_device = torch.cuda.current_device()
            torch.cuda.set_device(q.device)
            
            if isinstance(stage_info, ScanStage):
                extend_backend = scan_extend_backend\
                    if stage_info.stage_extend_backend is None else \
                        stage_info.stage_extend_backend
                
                grid = (
                    BSZ *\
                    triton.cdiv(chunk_count, BLOCK_CHUNK) *\
                    triton.cdiv(triton.cdiv(TDST, BLOCK_SIZE_Q), STAGE_STRIDE) *\
                    HEAD,
                )
                chunk_controllable_sampling_mask_cuda[grid](
                    q_mask, *q_mask.stride(),
                    k_mask, *args.safe_stride(k_mask, 4),
                    position_ids, *position_ids.stride(),
                
                    *args.args_paged_kv_cache(),
                    *args.args_offload_cache(True),
                    
                    indices_left, *indices_left.stride(),
                    indices_right, *indices_right.stride(),
                    out_scores, *out_scores.stride(),
                    args.rope_cos, *args.safe_stride(args.rope_cos, 2),
                    args.rope_sin, *args.safe_stride(args.rope_sin, 2),
                    
                    chunk_count,
                    MAX_TSRC,
                    TDST,
                    HEAD,
                    args.sliding_window_size,
                    args.sink_token_size,
                    # model_context_length if (not scan_extend_backend == 'streaming') else 0,
                    model_context_length,
                    
                    BLOCK_HID=BLOCK_HID,
                    BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                    STRIDE_Q=stage_block_stride_q,
                    BLOCK_CHUNK=BLOCK_CHUNK,
                    HEAD_GROUP=HEAD // HEAD_KV,
                    USING_EXTEND=args.using_extend,
                    EXTEND_BACKEND=extend_backend,
                    NEED_APPLY_ROPE=args.need_apply_rope,
                    TERMINATE_SIZE=stage_early_terminate,
                    SCAN_STRIDE=STAGE_STRIDE,
                )
                
                # if stage_require_recalculate_score:
                #     assert STAGE_STRIDE == 1
                #     grid = (
                #         BSZ * BDST * HEAD, # SCAN_STRIDE = 1
                #     )
                #     calculate_chunk_score[grid](
                #         q_mask, *q_mask.stride(),
                #         k_mask, *args.safe_stride(k_mask, 4),
                #         position_ids, *position_ids.stride(),
                #         args.rope_cos, *args.safe_stride(args.rope_cos, 2),
                #         args.rope_sin, *args.safe_stride(args.rope_sin, 2),
                        
                #         *args.args_paged_kv_cache(),
                #         *args.args_offload_cache(True),
                        
                #         indices_left, *indices_left.stride(),
                #         indices_right, *indices_right.stride(),
                #         out_scores, *out_scores.stride(),
                        
                #         # model_context_length if (not scan_extend_backend == 'streaming') else 0,
                #         model_context_length,
                #         args.sliding_window_size,
                #         args.sink_token_size,
                #         chunk_size,
                        
                #         TDST,
                #         BDST,
                #         BDST, #SCAN STRIDE == 1
                #         HEAD,
                #         chunk_count,
                #         HEAD // HEAD_KV,
                        
                #         USING_EXTEND=args.using_extend,
                #         NEED_APPLY_ROPE=args.need_apply_rope,
                #         EXTEND_BACKEND=extend_backend,
                #         BLOCK_HID=BLOCK_HID,
                #         BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                #         BLOCK_STRIDE_Q=stage_block_stride_q,
                #         BLOCK_SIZE_K=stage_early_terminate,
                #         BLOCK_STRIDE_K=args.block_stride_k,
                #         SCAN_STRIDE=1,
                #     )
            elif isinstance(stage_info, EvalScoreStage):
                extend_backend = scan_extend_backend \
                    if stage_info.stage_extend_backend is None else\
                        stage_info.stage_extend_backend
                
                grid = (
                    BSZ * 
                    triton.cdiv(BDST, stage_info.stage_stride) * 
                    HEAD, # SCAN_STRIDE = 1
                )                
                calculate_chunk_score[grid](
                    q_mask, *q_mask.stride(),
                    k_mask, *args.safe_stride(k_mask, 4),
                    position_ids, *position_ids.stride(),
                    args.rope_cos, *args.safe_stride(args.rope_cos, 2),
                    args.rope_sin, *args.safe_stride(args.rope_sin, 2),
                    
                    *args.args_paged_kv_cache(),
                    *args.args_offload_cache(True),
                    
                    indices_left, *indices_left.stride(),
                    indices_right, *indices_right.stride(),
                    out_scores, *out_scores.stride(),
                    
                    # model_context_length if (not scan_extend_backend == 'streaming') else 0,
                    model_context_length,
                    args.sliding_window_size,
                    args.sink_token_size,
                    chunk_size,
                    
                    TDST,
                    BDST,
                    triton.cdiv(BDST, stage_info.stage_stride), #SCAN STRIDE == 1
                    HEAD,
                    chunk_count,
                    HEAD // HEAD_KV,
                    
                    USING_EXTEND=args.using_extend,
                    NEED_APPLY_ROPE=args.need_apply_rope,
                    EXTEND_BACKEND=extend_backend,
                    BLOCK_HID=BLOCK_HID,
                    BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                    BLOCK_STRIDE_Q=stage_block_stride_q,
                    BLOCK_SIZE_K=stage_early_terminate,
                    BLOCK_STRIDE_K=args.block_stride_k,
                    SCAN_STRIDE=stage_info.stage_stride,
                    BLOCK_CHUNK=stage_info.block_chunk,
                )                
            elif isinstance(stage_info, EnsembleScoreStage):
                raise Exception()
            elif isinstance(stage_info, NopStage):
                pass
            else:
                raise Exception()

            torch.cuda.set_device(pre_device)

            if stage_info.require_post_sort:
                _, t_indices = out_scores[..., :indices_left.shape[-1]].sort(dim=-1, descending=True, stable=False)
                indices_left = indices_left.gather(dim=-1, index=t_indices)
                indices_right = indices_right.gather(dim=-1, index=t_indices)
            
            if DEBUG and DEBUG_RENDER and not torch.cuda.is_current_stream_capturing():
                if (i_stage + 1) < len(stages):
                    next_stage_k = stages[i_stage + 1].stage_k
                else:
                    next_stage_k = second_stage_k
                out_indices_cpu = indices_left.repeat_interleave(STAGE_STRIDE, 1)[:, -BDST:].contiguous().cpu().numpy()
                debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q)))
                render_plot_dynamic(out_indices_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q, next_stage_k, chunk_size)
                cv2.imwrite(f'dummy_sampled_stage_{i_stage}.png', debug * 255)
                print(f'saved dummy_sampled_stage_{i_stage}.png')
        
        if (STAGE_STRIDE > 1):
            indices_left = indices_left.repeat_interleave(STAGE_STRIDE, 1)[:, -BDST:].contiguous()
            indices_right = indices_right.repeat_interleave(STAGE_STRIDE, 1)[:, -BDST:].contiguous()
            out_scores = out_scores.repeat_interleave(STAGE_STRIDE, 1)[:, -BDST:].contiguous()
        
        assert (second_stage_k % chunk_size) == 0
        if DEBUG:
            print('indices_left', indices_left[0, -1])
            print('out_scores', out_scores[0, -1], second_stage_k, indices_left.shape, chunk_size)
        indices = indices_left[..., :second_stage_k // chunk_size] // chunk_size * chunk_size
        
        if DEBUG and DEBUG_RENDER and not torch.cuda.is_current_stream_capturing() and (BDST > 10):
            out_indices_cpu = indices.cpu().numpy()
            debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q)))
            render_plot(out_indices_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q)
            cv2.imwrite('dummy_sampled_final.png', debug * 255)
            print('saved dummy_sampled_final.png')
        
        args = args.clone()
        args.block_size_q = stages[-1].stage_block_size_q
        block_sparse_block_size_q = min(block_sparse_block_size_q, args.block_size_q)
        args.sliding_window_size += args.mask_k
        args.block_size_k = chunk_size
        args.mask_k = second_stage_k
        args.using_extend = args.using_extend and True
        
        # print('ff', indices.shape)
        indices = indices.permute(0, 2, 1, 3).flatten(0, 1)
        
        indices, t_sort_1 = indices.sort(dim=-1)
        indices = indices // args.block_size_k * args.block_size_k
        
        unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
        indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
        indices, t_sort_2 = indices.sort(dim=-1)
        active_mask = indices < (position_ids[:, ::args.block_size_q, None].repeat_interleave(HEAD, 0) + args.block_size_q)
        ks = active_mask.int().sum(-1)
        ks_count = ks.unsqueeze(-1)
        ks_start_end = torch.zeros((ks.shape[0], ks.shape[1], 2), dtype=torch.int32, device=q.device)
        ks_start_end[:, :, -1] = ks
        
        if (low_percent > 0) and (low_k_ratio < 1):
            scores = out_scores[..., :second_stage_k // chunk_size].permute(0, 2, 1, 3).flatten(0, 1)
            scores = scores.gather(dim=-1, index=t_sort_1)
            scores = scores.gather(dim=-1, index=t_sort_2)
            scores = torch.where(active_mask, scores, -32000.0)
            
            masked_scores = torch.where(scores > -16000.0, scores, 0)
            # masked_scores = torch.softmax(scores, dim=-1)
            scores_std, scores_mean = torch.std_mean(masked_scores, dim=-1)
            
            # TODO: TEST SENSITIVITY
            
            if dim_to_lower == 'head':
                dim_to_lower = 0
                values_to_sort = (scores_std).mean(dim=1)
            elif dim_to_lower == 'seq':
                dim_to_lower = 1
                values_to_sort = (scores_std)
            else:
                raise Exception()
            
            _, lowk = values_to_sort.topk(k=int(scores_mean.shape[dim_to_lower] * low_percent), dim=dim_to_lower, largest=False, sorted=False)
            # print(lowk[:, -1])
            if lowk.ndim == 2:
                lowk = lowk[:, :, None].expand(-1, -1, scores.shape[-1])
            if lowk.ndim == 1:
                lowk = lowk[:, None, None].expand(-1, scores.shape[-2], scores.shape[-1])
            _, t_sort_score = torch.topk(scores.gather(dim=dim_to_lower, index=lowk), dim=-1, k=int(scores.shape[-1] * (1 - low_k_ratio)), largest=False)
            # print(t_sort_score.shape)
            N, BDST = scores_mean.shape
            indices.scatter_(
                dim=dim_to_lower,
                index=lowk,
                src=indices\
                    .gather(dim=dim_to_lower, index=lowk)\
                    .scatter(dim=-1, index=t_sort_score, value=987654321),
            )
            indices, t_sort_2 = indices.sort(dim=-1)
            active_mask = indices < (position_ids[:, ::args.block_size_q, None].repeat_interleave(HEAD, 0) + args.block_size_q)
            # print(indices[1, -1, :])
            # print(active_mask[1, -1, :])
            ks = active_mask.int().sum(-1)
            ks_count = ks.unsqueeze(-1)
            ks_start_end = torch.zeros((ks.shape[0], ks.shape[1], 2), dtype=torch.int32, device=q.device)
            ks_start_end[:, :, -1] = ks
        
            if DEBUG and DEBUG_RENDER and not torch.cuda.is_current_stream_capturing() and (BDST > 10):
                indices_cpu = indices.cpu().numpy()
                ks_cpu = ks.cpu().numpy()
                debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q)))
                render_plot_ks(indices_cpu, ks_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q)
                cv2.imwrite('dummy_sampled_final_lowk.png', debug * 255)
                print('saved dummy_sampled_final_lowk.png', DEBUG_HEAD)
                
                print(ks[:, -1])
                
                plt.clf()
                plt.plot(scores_std[:3, :].float().cpu().numpy().T)
                # plt.ylim(0, 0.01)
                plt.savefig('dummy_stat_std.png')
                plt.clf()
                plt.plot(scores_mean[:3, :].float().cpu().numpy().T)
                plt.savefig('dummy_stat_mean.png')
                plt.clf()
                plt.plot(ks[DEBUG_HEAD, :].float().cpu().numpy())
                plt.savefig('dummy_stat_ks.png')
        
        if DEBUG and DEBUG_RENDER and not torch.cuda.is_current_stream_capturing() and (BDST > 10):
            try:
                input('>>>')
            except EOFError:
                time.sleep(1)
                pass
        
        if  (
                (block_sparse_block_size_q is not None) and\
                (triton.cdiv(TDST, block_sparse_block_size_q) != triton.cdiv(TDST, args.block_size_q))
            ):
            assert (BLOCK_SIZE_Q % block_sparse_block_size_q) == 0
            indices = indices.repeat_interleave(BLOCK_SIZE_Q // block_sparse_block_size_q, 1)
            ks = ks.repeat_interleave(BLOCK_SIZE_Q // block_sparse_block_size_q, 1)
            ks_count = ks_count.repeat_interleave(BLOCK_SIZE_Q // block_sparse_block_size_q, 1)
            ks_start_end = ks_start_end.repeat_interleave(BLOCK_SIZE_Q // block_sparse_block_size_q, 1)
            args.block_size_q = block_sparse_block_size_q
        
        if mask_only:
            return None, None
    else:
        args = args.clone()
        args.sliding_window_size += args.mask_k
        args.block_size_k = stages[-1].stage_chunk_size
        args.mask_k = second_stage_k
        args.using_extend = args.using_extend and True
        
        assert cached_metadata is not None
        indices = cached_metadata.indices
        ks = cached_metadata.indices
        ks_count = cached_metadata.ks_count
        ks_start_end = cached_metadata.ks_start_end
    
    args.block_size_q = min(args.block_size_q, triton.next_power_of_2(TDST))
    
    context = block_sparse_attention(
        q=q, 
        k=k, 
        v=v,
        seq_lens=position_ids + 1,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args,
        EXTEND_BACKEND=sa_extend_backend, # streaming works way much better in Gemma2, than dynamic_extend
        # model_context_length=model_context_length if (not sa_extend_backend == 'streaming') else 0,
        model_context_length=model_context_length,
    )
    
    if DEBUG:
        print('context', context[0, :, DEBUG_HEAD, :], context.shape)
        print('indices', indices[0 + DEBUG_HEAD, -1], indices.shape)
        print('ks', ks[0 + DEBUG_HEAD, -1], ks.shape)
    
    return context, HiPAttentionOutputMetadata(
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        key_access_log=None,
        key_access_count=None,
        block_access_log=None,
        block_access_score=None,
        block_access_count=None,
    )

def main_debug():
    global DEBUG
    
    seq_len = int(os.getenv('SEQ_LEN', '131072'))
    seq_dups = int(os.getenv('DUPS', '1'))
    block_size = int(os.getenv('BLOCK_SIZE', '64'))
    num_samples = int(os.getenv('NUM_SAMPLES', '100'))
    batch_size = int(os.getenv('BATCH_SIZE', '1'))
    mask_only = int(os.getenv('MASK_ONLY', '0')) == '1'
    k_group_size = int(os.getenv('K_GROUP_SIZE', '1'))
    
    assert seq_dups > 0
    
    q, k, v, out, cos, sin = load_checkouts(
        idx=0, 
        window=40, 
        seq_len=seq_len, 
        return_cos_sin=True, 
        derope=True,
        dtype=torch.bfloat16
    )
    HEAD = q.shape[0]
    HEAD_KV = k.shape[0]
    seq_len = seq_len * seq_dups
    
    q = q.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    k = k.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)#.to(torch.float8_e5m2)
    v = v.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)#.to(torch.float8_e5m2)
    if cos is not None:
        cos = cos.repeat(seq_dups, 1)#.to(torch.float8_e5m2)
        sin = sin.repeat(seq_dups, 1)#.to(torch.float8_e5m2)
    
    q_mask = q
    k_mask = k
    idx_pca_hid_q = None
    idx_pca_hid_k = None
    
    # q_pca = q[...,:32].contiguous()
    # k_pca = k[...,:32].contiguous()
    
    def pca(q, k, hid=32):
        import einx
        
        KV_HEAD_GROUP = q.shape[2] // k.shape[2]
        q_ori = q
        q = q.view(q.shape[0], q.shape[1], q.shape[2] // KV_HEAD_GROUP, KV_HEAD_GROUP, q.shape[3]).permute(0, 3, 1, 2, 4).flatten(0, 1)
        
        t = einx.rearrange('n t h d -> h (n t) d', q).float()
        _, _, proj = torch.linalg.svd(t, full_matrices=False)
        proj = proj.to(q.dtype) # type: torch.Tensor
        
        q = einx.dot('n t h d1, h d1 d2 -> n t h d2', q, proj)
        k = einx.dot('n t h d1, h d1 d2 -> n t h d2', k, proj)
        
        x_colsum = q.flatten(0, 1).abs().mean(dim=0, keepdim=False)
        y_colsum = k.flatten(0, 1).abs().mean(dim=0, keepdim=False)
        colsum = x_colsum + y_colsum
        
        _, topk_indices = colsum.topk(dim=-1, k=hid)
        idx_hid_keys = topk_indices.sort(dim=-1).values
        idx_hid_queries = idx_hid_keys.repeat_interleave(KV_HEAD_GROUP, 0)
        
        debug = np.zeros((idx_hid_queries.shape[0], q.shape[-1]), dtype=np.uint8)
        for i in range(idx_hid_queries.shape[0]):
            for j in range(idx_hid_queries.shape[1]):
                debug[i, idx_hid_queries[i, j]] = 255
        cv2.imwrite('dummy_idx_pca.png', debug)
        
        assert idx_hid_keys.ndim == 2
        assert idx_hid_keys.shape == (k.shape[2], hid), idx_hid_keys.shape
        q = q_ori.gather(index=idx_hid_queries[None, None, :, :].expand(*q_ori.shape[:-1], -1), dim=-1)
        k = k.gather(index=idx_hid_keys[None, None, :, :].expand(*k.shape[:-1], -1), dim=-1)
        
        return q, k, idx_hid_queries, idx_hid_keys
    # q_pca, k_pca, idx_pca_hid_q, idx_pca_hid_k = pca(q, k)
    
    k_mask = k
    _N, _T, _H, _D = k.shape
    tk = k_mask.view(_N, _T // k_group_size, k_group_size, _H, _D)
    k_mask = (tk.min(dim=2, keepdim=True).values + tk.max(dim=2, keepdim=True).values).expand(_N, _T // k_group_size, k_group_size, _H, _D).contiguous().view(*k.shape)
    
    if batch_size > 1:
        q = q[:, :1, :, :].contiguous()
        q = q.expand(batch_size, -1, -1, -1)
        k = k.expand(batch_size, -1, -1, -1)
        v = v.expand(batch_size, -1, -1, -1)
        q_mask = q
        k_mask = k
    
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    
    print(q.shape, k.shape, v.shape, q_mask.shape, k_mask.shape)
    
    print('-' * 20)
    
    is_decode = q.shape[1] == 1
    
    preset = os.getenv('HIP_PRESET', 'mid')
    config_stage = {
        'high': [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=64,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=16,
                stage_k=65536,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=1,
                stage_k=16384,
                stage_stride=1,
            ),
        ],
        'mid': [
            ScanStage(
                stage_block_size_q=256,
                stage_block_stride_q=4,
                stage_chunk_size=16,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=128,
                stage_block_stride_q=2,
                stage_chunk_size=4,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=1,
                stage_k=8192,
                stage_stride=1,
            ),
        ],
        'low': [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=256,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=32,
                stage_k=32768,
                stage_stride=1,
            ),
        ]
    }[preset]
    config_second_k = {
        'high': 4096,
        'mid': 2048,
        'low': 2048,
    }[preset]
    config_sa_extend_backend = {
        'high': 'streaming', 
        'mid': 'streaming',
        'low': 'streaming',
    }[preset]
    
    dual_stage_kwargs = dict(
        q=q,
        k=k,
        v=v,
        q_mask=q_mask,
        k_mask=k_mask,
        idx_pca_hid_q=idx_pca_hid_q,
        idx_pca_hid_k=idx_pca_hid_k,
        args=HiPAttentionArgs(
            # deprecated
            # mask_k=config_mask_k,
            # block_size_q=64,
            # block_stride_q=config_bsq,
            block_size_k=64, # BLOCK_CHUNK
            block_stride_k=1,
            sliding_window_size=1024 if not is_decode else 1024,
            sink_token_size=256 if not is_decode else 256,
            # position_ids=position_ids,
            
            using_extend=True,
            rope_cos=cos,
            rope_sin=sin,
            need_apply_rope=True,
        ),
        second_stage_k = config_second_k,
        stages=config_stage,
        block_sparse_block_size_q=block_size,
        model_context_length=131072,
        # scan_early_terminate=1,
        # stage_early_terminate=1,
        scan_extend_backend='relative',
        sa_extend_backend=config_sa_extend_backend,
        stage_early_terminate=k_group_size,
        mask_only=mask_only,
    )
    
    hip_1k_kwargs = dict(
        q=q,
        k=k,
        v=v,
        args=HiPAttentionArgs(
            mask_k=1024,
            block_size_q=64,
            block_stride_q=2,
            block_size_k=2,
            block_stride_k=1,
        ),
        mask_only=mask_only,
    )
    
    hip_512_kwargs = dict(
        q=q,
        k=k,
        v=v,
        args=HiPAttentionArgs(
            mask_k=512,
            block_size_q=64,
            block_stride_q=2,
            block_size_k=2,
            block_stride_k=1,
        ),
        mask_only=mask_only,
    )
    
    refresh_interval = 8 if is_decode else 2
    
    metadata = None
    for i in range(min(num_samples, 24)):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        if i==0: DEBUG = os.getenv('DEBUG', '0') == '1'
        
        # print(cos.shape)
        # print(sin.shape)
        
        _, metadata = dual_stage_quadratic_hip_attention(
            **dual_stage_kwargs,
            cached_metadata=metadata
        )
        
        if ((i + 1) % refresh_interval) == 0:
            metadata = None
        
        if i==0: DEBUG = False
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    print('-' * 20)
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    dual_stage_kwargs['args'].using_extend = False
    dual_stage_kwargs['args'].need_apply_rope = False
    
    metadata = None
    for i in range(min(num_samples, 24)):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        if i==0: DEBUG = os.getenv('DEBUG', '0') == '1'
        
        # print(cos.shape)
        # print(sin.shape)
        
        context, metadata = dual_stage_quadratic_hip_attention(
            **dual_stage_kwargs,
            cached_metadata=metadata
        )
        
        if ((i + 1) % refresh_interval) == 0:
            metadata = None
        
        if i==0: DEBUG = False
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    print('-' * 20)
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    metadata = None
    for i in range(min(num_samples, 24)):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
         
        start.record()
        context, metadata = hip_attention(
            **hip_1k_kwargs,
            previous_metadata=metadata,
        )
        end.record()
        
        if ((i + 1) % (8 if is_decode else 1)) == 0:
            metadata = None
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    print('-' * 20)
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    
    metadata = None
    for i in range(min(num_samples, 24)):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
         
        start.record()
        context, metadata = hip_attention(
            **hip_512_kwargs,
            previous_metadata=metadata,
        )
        end.record()
        
        if ((i + 1) % (8 if is_decode else 1)) == 0:
            metadata = None
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    print('-' * 20)
    
    for i in range(min(num_samples, 5)):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        if q.shape[1] == 1:
            flash_attn_with_kvcache(
                q, k, v, causal=True,
            )
        else:
            flash_attn_func(
                q, k, v, causal=True
            )
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))

if __name__ == '__main__':
    main_debug()