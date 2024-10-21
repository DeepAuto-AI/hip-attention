import math
import os

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
import torch
from torch import Tensor
from typing import List, Dict, Optional, Tuple
import triton
import triton.language as tl
import numpy as np
import cv2

DEBUG = (os.getenv('HIP_DEBUG', '0') == '1')

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
    mask_tsrc_active,
    mask_tdst,
    
    real_pos_tdst_min,
    model_context_length,
    num_sinks,
    
    USING_EXTEND,
    NEED_APPLY_ROPE,
    BLOCK_CHUNK,
    BLOCK_HID,
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
        
        mask_tsrc_active[None, :],
        
        BLOCK_CHUNK,
    ).to(queries.dtype)
    
    if USING_EXTEND:
        real_pos_tdst_max = tl.sum(mask_tdst.to(tl.int32)) + real_pos_tdst_min
        tsrc_extend = tl.maximum(0, real_pos_tdst_max - model_context_length)
        if NEED_APPLY_ROPE or (tsrc_extend >= 0):
            old_tsrc = idx_tsrc
            
            # new_tsrc = (old_tsrc * (model_context_length / real_pos_tdst_min)).to(tl.int32)
            
            # new_tsrc = idx_chunk + 4
            
            window = model_context_length // 2
            
            new_tsrc = tl.where(
                idx_tsrc >= (real_pos_tdst_max - window),
                idx_tsrc,
                tl.where(
                    real_pos_tdst_max <= model_context_length,
                    idx_tsrc,
                    (idx_tsrc * ((model_context_length - window) / (real_pos_tdst_max - window))).to(tl.int32) + (real_pos_tdst_min - model_context_length)
                )
            )
            new_tsrc = tl.maximum(0, new_tsrc)
            # new_tsrc = tl.minimum(model_context_length, new_tsrc)
            # new_tsrc = idx_tsrc
            
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
                tl.debug_barrier()
                cos_new = tl.load(
                    COS +\
                        new_tsrc[None, :].to(tl.int64) * stride_cos_t +\
                        idx_hid[:, None] * stride_cos_hid,
                    mask=mask_tsrc_active[None, :],
                    other=0,
                ).to(keys_left.dtype)
                tl.debug_barrier()
                
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
                tl.debug_barrier()
                
                # TODO: multiply -right
                keys_left_rot = tl.where(
                    (idx_hid + BLOCK_HID // 2)[:, None] < BLOCK_HID,
                    -keys_left_rot,
                    keys_left_rot
                )
                tl.debug_barrier()
                
                sin_new = tl.load(
                    SIN +\
                        new_tsrc[None, :].to(tl.int64) * stride_sin_t +\
                        idx_hid[:, None] * stride_sin_hid,
                    mask=mask_tsrc_active[None, :],
                    other=0,
                ).to(keys_left.dtype)
                tl.debug_barrier()
                
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
    REDUCE: tl.constexpr = 'mean',
    USING_EXTEND: tl.constexpr = False,
    NEED_APPLY_ROPE: tl.constexpr = False,
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)
    BCHUNK = tl.cdiv(CHUNK_COUNT, BLOCK_CHUNK)
    
    pid = tl.program_id(0).to(tl.int64)
    
    idx_head = pid % HEAD
    pid = pid // HEAD
    idx_bdst = pid % BDST
    pid = pid // BDST
    idx_bchunk = pid % BCHUNK
    pid = pid // BCHUNK
    idx_bsz = pid
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // STRIDE_Q) * STRIDE_Q
    mask_tdst = idx_tdst < TDST
    idx_hid = tl.arange(0, BLOCK_HID)
    
    pos_tdst = tl.load(
        POS +\
            idx_bsz * stride_pos_bsz +\
            idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    
    # real_pos_tdst_min = idx_bdst * BLOCK_SIZE_Q + TSRC - TDST
    real_pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst, 99999999999))
    
    pos_tdst_min = (real_pos_tdst_min - sliding_window_size - num_sinks).to(tl.int32)
    pos_tdst_min = tl.maximum(pos_tdst_min, 0)
    
    idx_chunk = idx_bchunk * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
    mask_chunk = idx_chunk < CHUNK_COUNT
    
    idx_tsrc_left = tl.load(
        INDICES_LEFT +\
            idx_bsz * stride_indices_left_bsz +\
            idx_bdst * stride_indices_left_bdst +\
            idx_head * stride_indices_left_head +\
            idx_chunk * stride_indices_left_chunk,
        mask=mask_chunk,
        other=MAX_TSRC,
    ).to(tl.int32)
    
    idx_tsrc_right = tl.load(
        INDICES_RIGHT +\
            idx_bsz * stride_indices_right_bsz +\
            idx_bdst * stride_indices_right_bdst +\
            idx_head * stride_indices_right_head +\
            idx_chunk * stride_indices_right_chunk,
        mask=mask_chunk,
        other=MAX_TSRC,
    ).to(tl.int32)
    
    max_chunk_size = tl.ceil(MAX_TSRC / CHUNK_COUNT).to(tl.float32)
    
    
    scores = tl.zeros((BLOCK_CHUNK,), dtype=tl.float32) - 32000.0
    
    queries = tl.load(
        Q + \
            idx_bsz * stride_q_bsz +\
            idx_tdst[:, None] * stride_q_tdst +\
            idx_head * stride_q_head +\
            idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0
    )
    
    if USING_EXTEND:
        if NEED_APPLY_ROPE or (real_pos_tdst_min >= model_context_length):
            old_tdst = pos_tdst
            # new_tdst = tl.minimum(pos_tdst, model_context_length - 1)
            # new_tdst = tl.where(mask_tdst, new_tdst, old_tdst)
            # new_tdst = old_tdst // 16
            
            # new_tdst = tl.maximum(idx_tdst, CHUNK_COUNT + 4)
            new_tdst = pos_tdst
            
            queries = adjust_rope(
                queries,
                old_tdst,
                new_tdst,
                mask_tdst,
                idx_hid,
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                BLOCK_SIZE_Q // STRIDE_Q, 
                BLOCK_HID,
                NEED_APPLY_ROPE,
            ).to(queries.dtype)
            queries = (queries * mask_tdst[:, None]).to(queries.dtype)
        else:
            if NEED_APPLY_ROPE:
                queries = adjust_rope(
                    queries,
                    pos_tdst,
                    pos_tdst,
                    mask_tdst,
                    idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    BLOCK_SIZE_Q // STRIDE_Q, 
                    BLOCK_HID,
                    NEED_APPLY_ROPE,
                ).to(queries.dtype)
    
    while (max_chunk_size > 1):
        max_chunk_size /= 2.0
        mask_tsrc_active = mask_chunk & (idx_tsrc_left < idx_tsrc_right) & (idx_tsrc_left <= pos_tdst_min)
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
            mask_tsrc_active,
            mask_tdst,
            
            real_pos_tdst_min,
            model_context_length,
            num_sinks,
            
            USING_EXTEND,
            NEED_APPLY_ROPE,
            BLOCK_CHUNK,
            BLOCK_HID,
        )
            
        scores_left = tl.dot(
            (queries * (tl.sqrt(BLOCK_HID * 1.0) / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0)))).to(queries.dtype),
            (keys_left.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0)))).to(queries.dtype),
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
            mask_tsrc_active,
            mask_tdst,
            
            real_pos_tdst_min,
            model_context_length,
            num_sinks,
            
            USING_EXTEND,
            NEED_APPLY_ROPE,
            BLOCK_CHUNK,
            BLOCK_HID,
        )
        
        scores_right = tl.dot(
            (queries * (tl.sqrt(BLOCK_HID * 1.0) / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0)))).to(queries.dtype),
            (keys_right.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0)))).to(queries.dtype),
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
        
        scores = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                scores_left,
                scores_right,
            ),
            scores,
        )
    
    tl.store(
        INDICES_LEFT +\
            idx_bsz * stride_indices_left_bsz +\
            idx_bdst * stride_indices_left_bdst +\
            idx_head * stride_indices_left_head +\
            idx_chunk * stride_indices_left_chunk,
        value=idx_tsrc_left,
        mask=mask_chunk,
    )
    
    tl.store(
        INDICES_RIGHT +\
            idx_bsz * stride_indices_right_bsz +\
            idx_bdst * stride_indices_right_bdst +\
            idx_head * stride_indices_right_head +\
            idx_chunk * stride_indices_right_chunk,
        value=idx_tsrc_right,
        mask=mask_chunk,
    )
    
    tl.store(
        OUT_SCORES +\
            idx_bsz * stride_out_scores_bsz +\
            idx_bdst * stride_out_scores_bdst +\
            idx_head * stride_out_scores_head +\
            idx_chunk * stride_out_scores_chunk,
        value=scores,
        mask=mask_chunk,
    )

def dual_stage_quadratic_hip_attention(
    q: Tensor, 
    k: Optional[Tensor], 
    v: Optional[Tensor], 
    args: HiPAttentionArgs,
    second_stage_k: int = 1024,
    stages = [
        (256, 8192),
        (128, 4096),
        (64, 2048),
    ],
    model_context_length = 16384,
    
    # kernel args,
    mask_only = False,
    block_sparse_block_size_q: Optional[int] = 32,
):
    DEBUG_HEAD = -1
    global DEBUG
    
    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        BSZ, TSRC, HEAD_KV, HID = k.shape
        assert v.shape == k.shape
        MAX_TSRC = TSRC
    else:
        MAX_TSRC = args.k_cache.shape[0] * args.k_cache.shape[1]
        HEAD_KV = args.k_cache.shape[-2]
        TSRC = MAX_TSRC
        # print('asdf', args.k_cache.shape, MAX_TSRC, HEAD_KV, q.shape)
    
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
    
    BLOCK_CHUNK = args.block_size_k
    BLOCK_SIZE_Q = args.block_size_q
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    
    indices_left = torch.zeros(
        (BSZ, BDST, HEAD, chunk_count), 
        device=q.device,
        dtype=torch.int64
    )
    indices_right = torch.zeros(
        (BSZ, BDST, HEAD, chunk_count), 
        device=q.device,
        dtype=torch.int64
    )
    
    indices_left[:, :, :, :] = (
        torch.floor(
            torch.arange(0, chunk_count, device=q.device, dtype=torch.float64) * chunk_size + args.sink_token_size
        ).to(indices_left.dtype)
    )[None, None, None, :]
    indices_right[:, :, :, :] = indices_left + chunk_size
    indices_right.clamp_max_(MAX_TSRC - args.sliding_window_size)
    
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    out_scores = torch.full(
        (BSZ, BDST, HEAD, triton.next_power_of_2(chunk_count)), 
        device=q.device,
        dtype=q.dtype,
        fill_value=-32000.0
    )
    
    # print(q.shape, k.shape, args.rope_cos.shape, args.rope_sin.shape, TDST, TSRC)
    
    # print('neeeed rope', args.need_apply_rope)
    
    pre_device = torch.cuda.current_device()
    torch.cuda.set_device(q.device)
    grid = (BSZ * triton.cdiv(chunk_count, BLOCK_CHUNK) * BDST * HEAD,)
    chunk_controllable_sampling_mask_cuda[grid](
        q, *q.stride(),
        k, *args.safe_stride(k, 4),
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
        args.sliding_window_size - max(BLOCK_SIZE_Q, BLOCK_CHUNK),
        args.sink_token_size,
        model_context_length,
        
        BLOCK_HID=q.shape[-1],
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        STRIDE_Q=args.block_stride_q,
        BLOCK_CHUNK=BLOCK_CHUNK,
        HEAD_GROUP=HEAD // HEAD_KV,
        USING_EXTEND=args.using_extend,
        NEED_APPLY_ROPE=args.need_apply_rope,
    )
    torch.cuda.set_device(pre_device)
    
    out_scores[..., indices_left.shape[-1]:] = float('-inf')
    _, t_indices = out_scores.sort(dim=-1, descending=True, stable=False)
    indices_left = indices_left.gather(dim=-1, index=t_indices[..., :indices_left.shape[-1]])
    indices_right = indices_right.gather(dim=-1, index=t_indices[..., :indices_right.shape[-1]])
    
    if DEBUG and not torch.cuda.is_current_stream_capturing() and (BDST > 10):
        out_indices_cpu = indices_left.cpu()
        debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK)))
        for i in range(out_indices_cpu.shape[1]):
            t_chunk_size = triton.cdiv(TDST, chunk_count * BLOCK_CHUNK)
            # print(i, t_chunk_size)
            for j in range(max(
                0,
                math.ceil(out_indices_cpu.shape[-1] * (stages[0][1] / TDST))
            )):
                if j >= out_indices_cpu.shape[-1]: continue
                t = (out_indices_cpu[0, i, DEBUG_HEAD, j] - args.sink_token_size) // BLOCK_CHUNK + args.sink_token_size // BLOCK_CHUNK
                t = t // t_chunk_size * t_chunk_size
                debug[i, t:t+t_chunk_size] = 1
        cv2.imwrite('dummy_sampled.png', debug * 255)
        print('saved dummy_sampled.png')
        
        debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK)))
        for i in range(out_indices_cpu.shape[1]):
            t_chunk_size = triton.cdiv(TDST, chunk_count * BLOCK_CHUNK)
            # print(i, t_chunk_size)
            for j in range(max(
                0,
                math.ceil(out_indices_cpu.shape[-1] * (second_stage_k / TDST))
            )):
                if j >= out_indices_cpu.shape[-1]: continue
                t = (out_indices_cpu[0, i, DEBUG_HEAD, j] - args.sink_token_size) // BLOCK_CHUNK + args.sink_token_size // BLOCK_CHUNK
                t = t // t_chunk_size * t_chunk_size
                debug[i, t:t+t_chunk_size] = 1
        cv2.imwrite('dummy_sampled_cut.png', debug * 255)
        print('saved dummy_sampled_cut.png')
    
    for i_stage, (stage_chunk_size, stage_k) in enumerate(stages):
        # if stage_chunk_size > chunk_size: continue
        # if stage_k > TSRC: continue
        
        assert (stage_k % chunk_size) == 0
        indices_left = indices_left[..., :stage_k // chunk_size]
        indices_left = ((indices_left - args.sink_token_size) // chunk_size * chunk_size + args.sink_token_size)
        indices_right = (indices_left + chunk_size)
        out_scores = out_scores[..., :stage_k // chunk_size]
        
        indices_left, t_indices = indices_left.sort(dim=-1)
        indices_right = indices_right.gather(dim=-1, index=t_indices)
        out_scores = out_scores.gather(dim=-1, index=t_indices)
        
        assert (chunk_size % stage_chunk_size) == 0
        splits = chunk_size // stage_chunk_size
        chunk_sizes = ((indices_right - indices_left).float() / splits).clamp_min_(0)
        indices_left = indices_left[..., None] + (torch.arange(0, splits, device=q.device)[None, None, None, None, :] * chunk_sizes[..., None]).floor().long()
        indices_left = indices_left.flatten(-2, -1)
        indices_right = indices_right[..., None] - (((splits - 1) - torch.arange(0, splits, device=q.device)[None, None, None, None, :]) * chunk_sizes[..., None]).floor().long()
        indices_right = indices_right.flatten(-2, -1)
        out_scores = out_scores.repeat_interleave(splits, -1)
        
        chunk_size = stage_chunk_size
        chunk_count = indices_left.shape[-1]
        BLOCK_CHUNK = max(16, triton.next_power_of_2(min(chunk_count, BLOCK_CHUNK)))
        
        pre_device = torch.cuda.current_device()
        torch.cuda.set_device(q.device)
        grid = (BSZ * triton.cdiv(chunk_count, BLOCK_CHUNK) * triton.cdiv(TDST, BLOCK_SIZE_Q) * HEAD,)
        chunk_controllable_sampling_mask_cuda[grid](
            q, *q.stride(),
            k, *args.safe_stride(k, 4),
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
            args.sliding_window_size - max(BLOCK_SIZE_Q, BLOCK_CHUNK),
            args.sink_token_size,
            model_context_length,
            
            BLOCK_HID=q.shape[-1],
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            STRIDE_Q=args.block_stride_q,
            BLOCK_CHUNK=BLOCK_CHUNK,
            HEAD_GROUP=HEAD // HEAD_KV,
            USING_EXTEND=args.using_extend,
            NEED_APPLY_ROPE=args.need_apply_rope,
        )
        torch.cuda.set_device(pre_device)
        
        _, t_indices = out_scores.sort(dim=-1, descending=True, stable=False)
        indices_left = indices_left.gather(dim=-1, index=t_indices)
        indices_right = indices_right.gather(dim=-1, index=t_indices)
        
        if DEBUG and not torch.cuda.is_current_stream_capturing():
            out_indices_cpu = indices_left.cpu()
            debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q)))
            for i in range(out_indices_cpu.shape[1]):
                for j in range(math.ceil(stage_k / chunk_size)):
                    if j >= out_indices_cpu.shape[-1]: continue
                    t = out_indices_cpu[0, i, 7, j] // BLOCK_SIZE_Q
                    debug[i, t:t+triton.cdiv(chunk_size, BLOCK_SIZE_Q)] = 1
            cv2.imwrite(f'dummy_sampled_stage_{i_stage}.png', debug * 255)
            print(f'saved dummy_sampled_stage_{i_stage}.png')
    
    assert (second_stage_k % chunk_size) == 0
    indices = indices_left[..., :second_stage_k // chunk_size] // chunk_size * chunk_size
    
    if DEBUG and not torch.cuda.is_current_stream_capturing() and (BDST > 10):
        out_indices_cpu = indices.cpu()
        debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q)))
        for i in range(out_indices_cpu.shape[1]):
            for j in range(indices.shape[-1]):
                if j >= out_indices_cpu.shape[-1]: continue
                t = out_indices_cpu[0, i, DEBUG_HEAD, j] // BLOCK_SIZE_Q
                debug[i, t:t+1] = 1
        cv2.imwrite('dummy_sampled_final.png', debug * 255)
        print('saved dummy_sampled_final.png')
        try:
            input('>>>')
        except EOFError:
            pass
    
    args = args.clone()
    args.sliding_window_size += args.mask_k
    args.block_size_k = chunk_size
    args.mask_k = second_stage_k
    args.using_extend = args.using_extend and True
    
    # print('ff', indices.shape)
    indices = indices.permute(0, 2, 1, 3).flatten(0, 1)
    
    indices, _ = indices.sort(dim=-1)
    indices = indices // args.block_size_k * args.block_size_k
    
    unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
    indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
    indices = indices.sort(dim=-1).values
    active_mask = indices < (position_ids[:, ::args.block_size_q, None].repeat_interleave(HEAD, 0) + args.block_size_q)
    ks = active_mask.int().sum(-1)
    ks_count = ks.unsqueeze(-1)
    ks_start_end = torch.zeros((ks.shape[0], ks.shape[1], 2), dtype=torch.int32, device=q.device)
    ks_start_end[:, :, -1] = ks
    
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
        EXTEND_BACKEND='streaming', # streaming works way much better in Gemma2, than dynamic_extend
        model_context_length=model_context_length,
    )
    
    if DEBUG:
        print('context', context[0, :, DEBUG_HEAD, :])
        print('indices', indices[0, -1])
    
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
    
    seq_len = 131072
    seq_dups = int(os.getenv('DUPS', '1'))
    mask_only = False
    
    assert seq_dups > 0
    
    q, k, v, out, cos, sin = load_checkouts(
        idx=0, 
        window=40, 
        seq_len=seq_len, 
        return_cos_sin=True, 
        dtype=torch.bfloat16
    )
    HEAD = q.shape[0]
    HEAD_KV = k.shape[0]
    seq_len = seq_len * seq_dups
    
    q = q.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    k = k.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    v = v.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    if cos is not None:
        cos = cos.repeat(seq_dups, 1)
        sin = sin.repeat(seq_dups, 1)
    
    from flash_attn import flash_attn_func
    
    print(q.shape, k.shape, v.shape)
    
    print('-' * 20)
    
    for i in range(10):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        if i==0: DEBUG = os.getenv('DEBUG', '0') == '1'
        
        # print(cos.shape)
        # print(sin.shape)
        
        dual_stage_quadratic_hip_attention(
            q, k, v,
            args=HiPAttentionArgs(
                mask_k=256,
                block_size_q=64,
                block_stride_q=4,
                block_size_k=64, # BLOCK_CHUNK
                sliding_window_size=1024,
                sink_token_size=256,
                # position_ids=position_ids,
                
                using_extend=True,
                rope_cos=cos,
                rope_sin=sin,
                need_apply_rope=True,
            ),
            second_stage_k=2048,
            stages=[
                (64, 8192),
            ],
            block_sparse_block_size_q=64,
            model_context_length=65536,
            mask_only=mask_only,
        )
        
        if i==0: DEBUG = False
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    print('-' * 20)
    
    for i in range(10):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        if i==0: DEBUG = os.getenv('DEBUG', '0') == '1'
        
        # print(cos.shape)
        # print(sin.shape)
        
        dual_stage_quadratic_hip_attention(
            q, k, v,
            args=HiPAttentionArgs(
                mask_k=256,
                block_size_q=64,
                block_stride_q=4,
                block_size_k=64, # BLOCK_CHUNK
                sliding_window_size=1024,
                sink_token_size=256,
                # position_ids=position_ids,
                
                using_extend=False,
                rope_cos=cos,
                rope_sin=sin,
            ),
            second_stage_k=2048,
            stages=[
                (64, 8192),
            ],
            block_sparse_block_size_q=64,
            model_context_length=65536,
            mask_only=mask_only,
        )
        
        if i==0: DEBUG = False
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    print('-' * 20)
    
    for i in range(10):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        hip_attention(
            q, k, v,
            args=HiPAttentionArgs(
                mask_k=512,
                block_size_q=64,
                block_stride_q=2,
                block_size_k=2,
                block_stride_k=1,
            ),
            mask_only=mask_only,
        )
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    print('-' * 20)
    
    for i in range(3):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        flash_attn_func(
            q, k, v, causal=True
        )
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))

if __name__ == '__main__':
    main_debug()