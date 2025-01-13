from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import os
import torch
from torch import Tensor
import triton
import triton.language as tl

from hip.models.hip_attention.gen3.uvm_gpu_cache import load_tokens
from hip.models.hip_attention.gen3.attention_metadata import safe_stride
from hip.models.hip_attention.gen3.attention_extend_bsa import block_sparse_attention_cuda_step

if TYPE_CHECKING:
    from hip.models.hip_attention.gen3.attention_metadata import HiPAttentionArgs

DEFAULT_EXTEND_BACKEND: tl.constexpr = 'streaming'
MAX_INT: tl.constexpr = 2_147_483_647


@triton.jit
def _fwd_kernel_stage1(
    Q, stride_q_bsz, stride_q_tdst, stride_q_head, stride_q_hid,
    K, stride_k_bsz, stride_k_tsrc, stride_k_head, stride_k_hid,
    V, stride_v_bsz, stride_v_tsrc, stride_v_head, stride_v_hid,
    B_Seqlen, stride_pos_bsz, stride_pos_tdst,

    INDICES,
    stride_indices_b, stride_indices_bdst, stride_indices_bk,

    KS_START_END,
    stride_ks_start_end_b, stride_ks_start_end_bdst, stride_ks_start_end_g,

    ATTN_LOGITS,
    stride_attn_logits_bsz, stride_attn_logits_head, stride_attn_logits_kv_split, stride_attn_logits_hid,

    q_head_num: tl.constexpr,
    BK: tl.constexpr,
    MAX_TDST,
    MAX_TSRC,
    kv_group_num: tl.constexpr,

    sliding_window_size: tl.constexpr,
    sink_token_size: tl.constexpr,
    LOGIT_SOFTCAP: tl.constexpr,

    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    model_context_length,

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

    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_KV_PACKED: tl.constexpr,
    GPU_BANK_COUNT,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
    stride_offload_cache_gpu_global_metadata_k,
    stride_offload_cache_gpu_global_metadata_pad,
    OFFLOAD_CACHE_GPU_BANK,
    stride_offload_cache_gpu_bank_token,
    stride_offload_cache_gpu_bank_hid,
    OFFLOAD_CACHE_GPU_METADATA,
    stride_offload_cache_gpu_metadata_token,
    stride_offload_cache_gpu_metadata_k,
    OFFLOAD_CACHE_GPU_TABLE,
    stride_offload_cache_gpu_table_head_kv,
    stride_offload_cache_gpu_table_token,
    strdie_offload_cache_gpu_table_k,

    ACCESS_COUNTER,
    stride_access_counter_bsz,
    stride_access_counter_head_kv,
    stride_access_counter_tsrc,
    CACHE_MISS_COUNTER,
    stride_cache_miss_counter_bsz,
    stride_cache_miss_counter_head_kv,
    stride_cache_miss_counter_tsrc,

    TDST_NEXT_POWER_OF_2,

    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    Lk: tl.constexpr,  # hidden dim of key
    Lv: tl.constexpr,  # hidden dim of value

    # autotuning parameters
    BLOCK_BK: tl.constexpr,  # = BLOCK_N / BLOCK_SIZE_K
    NUM_SPARSE_KV_SPLITS: tl.constexpr,
    NUM_SINK_KV_SPLITS: tl.constexpr,
    NUM_SLIDING_KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    UPDATE_CACHE: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)
    sink_split_kv_id = split_kv_id - NUM_SPARSE_KV_SPLITS
    sliding_split_kv_id = split_kv_id - NUM_SPARSE_KV_SPLITS - NUM_SINK_KV_SPLITS

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch * stride_pos_bsz + 0 * stride_pos_tdst)
    # cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    offs_q = (
        cur_batch * stride_q_bsz
        + 0 * stride_q_tdst
        + cur_head[:, None] * stride_q_head
        + offs_d[None, :] * stride_q_hid
    )
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)  # [BLOCK_H, BLOCK_DMODEL]
    if q.dtype == tl.float8e5:
        q = q.to(tl.float16)

    if USING_EXTEND and NEED_APPLY_ROPE:
        rope_tdst = cur_batch_seq_len - 1

        queries_rot = tl.load(
            Q
            + cur_batch * stride_q_bsz
            + 0 * stride_q_tdst
            + cur_head[:, None] * stride_q_head
            + ((offs_d[None, :] + Lk // 2) % Lk) * stride_q_hid,
            mask=(mask_h[:, None]) & (mask_d[None, :]),
            other=0.0
        )  # [BLOCK_H, BLOCK_DMODEL]
        if queries_rot.dtype == tl.float8e5:
            queries_rot = queries_rot.to(tl.float16)

        cos_new = tl.load(
            COS
            + rope_tdst.to(tl.int64) * stride_cos_t
            + (offs_d[None, :] % (Lk // 2)) * stride_cos_hid,
            mask=mask_d[None, :],
            other=0.0
        ).to(q.dtype)  # [1, BLOCK_DMODEL]
        sin_new = tl.load(
            SIN
            + rope_tdst.to(tl.int64) * stride_sin_t
            + (offs_d[None, :] % (Lk // 2)) * stride_sin_hid,
            mask=mask_d[None, :],
            other=0.0
        ).to(q.dtype)  # [1, BLOCK_DMODEL]

        queries_rot = queries_rot * (((offs_d[None, :] + Lk // 2) < Lk) * (-2) + 1).to(q.dtype)

        q = (q * cos_new + queries_rot * sin_new).to(q.dtype)

    # Start and end indices to the `indices` tensor
    range_start = tl.load(
        KS_START_END
        + cur_batch * stride_ks_start_end_b
        + 0 * stride_ks_start_end_bdst
        + 0 * stride_ks_start_end_g
    )
    range_end = tl.load(
        KS_START_END
        + cur_batch * stride_ks_start_end_b
        + 0 * stride_ks_start_end_bdst
        + 1 * stride_ks_start_end_g
    )
    if BK <= 0:
        range_start = 0
        range_end = 0

    kv_blocks_per_split = tl.cdiv(BK, NUM_SPARSE_KV_SPLITS)
    split_kv_block_start = kv_blocks_per_split * split_kv_id
    split_kv_block_end = tl.minimum(split_kv_block_start + kv_blocks_per_split, BK)

    e_max = tl.full([BLOCK_H, 1], -float("inf"), dtype=tl.float32)  # m_i
    e_sum = tl.full([BLOCK_H, 1], 1.0, dtype=tl.float32)  # l_i
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_block_end > split_kv_block_start:
        for i_bk in range(split_kv_block_start, split_kv_block_end, BLOCK_BK):
            idx_bk = i_bk + tl.arange(0, BLOCK_BK)  # [BLOCK_BK]
            mask_bk = (range_start <= idx_bk) & (idx_bk < tl.minimum(range_start + BK, range_end))  # [BLOCK_BK]

            if (range_start <= i_bk + BLOCK_BK) & (i_bk < range_end):
                idx_tsrc_start = tl.load(
                    INDICES
                    + cur_batch * stride_indices_b
                    + 0 * stride_indices_bdst
                    + idx_bk * stride_indices_bk,
                    mask=mask_bk,
                )  # [BLOCK_BK]
                idx_tsrc_start = tl.where(mask_bk, idx_tsrc_start, MAX_TSRC + 1)
                idx_tsrc = idx_tsrc_start[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :]
                idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc_from_bk = mask_bk[:, None] & tl.full((1, BLOCK_SIZE_K), 1, dtype=tl.int1)
                mask_tsrc_from_bk = tl.reshape(mask_tsrc_from_bk, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc = ((MAX_TSRC * 0) <= idx_tsrc) & (idx_tsrc < (MAX_TSRC * 1)) & mask_tsrc_from_bk
                idx_tsrc = idx_tsrc % MAX_TSRC  # [BLOCK_BK * BLOCK_SIZE_K]
                mask_tsrc = (sink_token_size <= idx_tsrc) & (idx_tsrc < cur_batch_seq_len) & mask_tsrc

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

                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,

                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,

                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,

                    cur_batch,
                    idx_tsrc[None, :],
                    cur_kv_head,
                    offs_d[:, None],
                    mask_tsrc[None, :],

                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    Lk,
                    
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )

                if USING_EXTEND and NEED_APPLY_ROPE:
                    keys_rot = load_tokens(
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

                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,

                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,

                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,

                        cur_batch,
                        idx_tsrc[None, :],
                        cur_kv_head,
                        ((offs_d[:, None] + Lk // 2) % Lk),
                        mask_tsrc[None, :],

                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        Lk,
                        
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_rot = None

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

                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    True,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,

                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,

                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,

                    cur_batch,
                    idx_tsrc[:, None],
                    cur_kv_head,
                    offs_dv[None, :],
                    mask_tsrc[:, None],

                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    Lv,
                    
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    
                    V_CACHE=K_CACHE,
                    stride_v_cache_page=stride_k_cache_page,
                    stride_v_cache_offset=stride_k_cache_offset,
                    stride_v_cache_kv_head=stride_k_cache_kv_head,
                    stride_v_cache_hid=stride_k_cache_hid,
                )

                acc, e_sum, e_max = block_sparse_attention_cuda_step(
                    q,  # FIXME: q is [BLOCK_H, BLOCK_DMODEL]: the first axis is head, not time
                    keys,
                    keys_rot,
                    values,

                    idx_tsrc, mask_tsrc,
                    tl.zeros([1], dtype=tl.int32),
                    tl.full((1,), 1, dtype=tl.int1),

                    acc, e_sum, e_max,

                    sliding_window_size,
                    sink_token_size,
                    (range_end - range_start) * BLOCK_SIZE_K,  # mask_k
                    True,
                    False,
                    LOGIT_SOFTCAP,

                    USING_EXTEND,
                    NEED_APPLY_ROPE,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    model_context_length,

                    idx_bk + sink_token_size // BLOCK_SIZE_K,
                    cur_batch_seq_len,
                    offs_d,
                    IS_CAUSAL,
                    Lk,
                    BLOCK_SIZE_Q,
                    BLOCK_BK * BLOCK_SIZE_K,
                    BLOCK_SIZE_K,

                    EXTEND_BACKEND=EXTEND_BACKEND,
                )
            else:
                pass

    # process sink tokens
    sink_tokens_per_split = tl.cdiv(sink_token_size, NUM_SINK_KV_SPLITS)
    split_sink_start = sink_tokens_per_split * sink_split_kv_id
    split_sink_end = tl.minimum(split_sink_start + sink_tokens_per_split, sink_token_size)
    if (((sink_token_size > 0 and 0 <= sink_split_kv_id) and sink_split_kv_id < NUM_SINK_KV_SPLITS)
        and split_sink_end > split_sink_start):
        for i_tsrc in range(split_sink_start, split_sink_end, BLOCK_BK * BLOCK_SIZE_K):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < tl.minimum(MAX_TSRC, sink_token_size)

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

                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                False,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,

                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,

                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,

                cur_batch,
                idx_tsrc[None, :],
                cur_kv_head,
                offs_d[:, None],
                mask_tsrc[None, :],

                q_head_num // kv_group_num,
                BLOCK_SIZE_K,
                Lk,
                
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                
                V_CACHE=V_CACHE,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_offset=stride_v_cache_offset,
                stride_v_cache_kv_head=stride_v_cache_kv_head,
                stride_v_cache_hid=stride_v_cache_hid,
            )

            if USING_EXTEND and NEED_APPLY_ROPE:
                keys_rot = load_tokens(
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

                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,

                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,

                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,

                    cur_batch,
                    idx_tsrc[None, :],
                    cur_kv_head,
                    ((offs_d[:, None] + Lk // 2) % Lk),
                    mask_tsrc[None, :],

                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    Lk,
                    
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )
            else:
                keys_rot = None

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

                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                True,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,

                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,

                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,

                cur_batch,
                idx_tsrc[:, None],
                cur_kv_head,
                offs_dv[None, :],
                mask_tsrc[:, None],

                q_head_num // kv_group_num,
                BLOCK_SIZE_K,
                Lv,
                
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                
                V_CACHE=K_CACHE,
                stride_v_cache_page=stride_k_cache_page,
                stride_v_cache_offset=stride_k_cache_offset,
                stride_v_cache_kv_head=stride_k_cache_kv_head,
                stride_v_cache_hid=stride_k_cache_hid,
            )

            acc, e_sum, e_max = block_sparse_attention_cuda_step(
                q,
                keys,
                keys_rot,
                values,

                idx_tsrc, mask_tsrc,
                tl.zeros([1], dtype=tl.int32),
                tl.full((1,), 1, dtype=tl.int1),

                acc, e_sum, e_max,

                sliding_window_size,
                sink_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                True,
                True,
                LOGIT_SOFTCAP,

                USING_EXTEND,
                NEED_APPLY_ROPE,
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                model_context_length,

                tl.arange(0, BLOCK_BK) + i_tsrc // BLOCK_SIZE_K,
                cur_batch_seq_len,
                offs_d,
                IS_CAUSAL,
                Lk,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,

                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    # process sliding window
    i_tsrc_range_start = tl.maximum(0, cur_batch_seq_len - sliding_window_size - BLOCK_SIZE_Q)
    sliding_tokens_per_split = tl.cdiv(cur_batch_seq_len - i_tsrc_range_start, NUM_SLIDING_KV_SPLITS)
    split_sliding_start = i_tsrc_range_start + sliding_tokens_per_split * sliding_split_kv_id
    split_sliding_end = tl.minimum(split_sliding_start + sliding_tokens_per_split, cur_batch_seq_len)
    if (((sliding_window_size > 0 and 0 <= sliding_split_kv_id) and sliding_split_kv_id < NUM_SLIDING_KV_SPLITS)
        and split_sliding_end > split_sliding_start):
        for i_tsrc in range(split_sliding_start, split_sliding_end, BLOCK_BK * BLOCK_SIZE_K):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = (0 <= idx_tsrc) & (idx_tsrc < cur_batch_seq_len)

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

                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                False,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,

                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,

                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,

                cur_batch,
                idx_tsrc[None, :],
                cur_kv_head,
                offs_d[:, None],
                mask_tsrc[None, :],

                q_head_num // kv_group_num,
                BLOCK_SIZE_K,
                Lk,
                
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                
                V_CACHE=V_CACHE,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_offset=stride_v_cache_offset,
                stride_v_cache_kv_head=stride_v_cache_kv_head,
                stride_v_cache_hid=stride_v_cache_hid,
            )

            if USING_EXTEND and NEED_APPLY_ROPE:
                keys_rot = load_tokens(
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

                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,

                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,

                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,

                    cur_batch,
                    idx_tsrc[None, :],
                    cur_kv_head,
                    ((offs_d[:, None] + Lk // 2) % Lk),
                    mask_tsrc[None, :],

                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    Lk,
                    
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )
            else:
                keys_rot = None

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

                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                True,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,

                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,

                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,

                cur_batch,
                idx_tsrc[:, None],
                cur_kv_head,
                offs_dv[None, :],
                mask_tsrc[:, None],

                q_head_num // kv_group_num,
                BLOCK_SIZE_K,
                Lv,
                
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                
                V_CACHE=K_CACHE,
                stride_v_cache_page=stride_k_cache_page,
                stride_v_cache_offset=stride_k_cache_offset,
                stride_v_cache_kv_head=stride_k_cache_kv_head,
                stride_v_cache_hid=stride_k_cache_hid,
            )

            idx_bk = (
                tl.arange(0, BLOCK_BK) +
                (i_tsrc - i_tsrc_range_start) // BLOCK_SIZE_K +
                (cur_batch_seq_len - 1 - sliding_window_size) // BLOCK_SIZE_K
            )
            acc, e_sum, e_max = block_sparse_attention_cuda_step(
                q,  # [BLOCK_H, BLOCK_DMODEL]
                keys,  # [BLOCK_DMODEL, BLOCK_BK * BLOCK_SIZE_K]
                keys_rot,
                values,

                idx_tsrc, mask_tsrc,
                tl.zeros([1], dtype=tl.int32),
                tl.full((1,), 1, dtype=tl.int1),

                acc, e_sum, e_max,

                sliding_window_size,
                sink_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                False,
                False,
                LOGIT_SOFTCAP,

                USING_EXTEND,
                NEED_APPLY_ROPE,
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                model_context_length,

                idx_bk,
                cur_batch_seq_len,
                offs_d,
                IS_CAUSAL,
                Lk,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,

                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    e_sum = tl.where(e_sum < 1e-20, 1e-20, e_sum)

    # Store results
    offs_mid_o = (
        cur_batch * stride_attn_logits_bsz
        + cur_head[:, None] * stride_attn_logits_head
        + split_kv_id * stride_attn_logits_kv_split
        + offs_dv[None, :] * stride_attn_logits_hid
    )
    tl.store(
        ATTN_LOGITS + offs_mid_o,
        acc / e_sum,
        mask=(mask_h[:, None]) & (mask_dv[None, :]),
    )

    offs_mid_o_1 = (
        cur_batch * stride_attn_logits_bsz
        + cur_head * stride_attn_logits_head
        + split_kv_id * stride_attn_logits_kv_split
        + Lv * stride_attn_logits_hid
    )
    tl.store(
        ATTN_LOGITS + offs_mid_o_1[:, None],
        e_max + tl.math.log2(e_sum),
        mask=mask_h[:, None],
    )


def decode_block_sparse_attention_stage1(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    
    seq_lens: Tensor,
    indices: Tensor,
    ks_start_end: Tensor,
    
    args: HiPAttentionArgs,
    
    head_num: int, 
    BK: int, 
    MAX_TDST: int, 
    MAX_TSRC: int, 
    kv_group_num: int,
    model_context_length: int,
    HID: int, 
    BLOCK_BK: int, 
    
    extend_backend: str,
    access_counter: Tensor,
    cache_miss_counter: Tensor,
    offload_update_cache: bool,
):
    batch = q.shape[0]
    BLOCK_H = 16

    NUM_SPARSE_KV_SPLITS = min(16, triton.cdiv(args.second_stage_k, 128))  # TODO: apply from server args
    NUM_SINK_KV_SPLITS = min(4, triton.cdiv(args.sink_token_size, 128))
    NUM_SLIDING_KV_SPLITS = min(8, triton.cdiv(args.sliding_window_size, 128))

    NUM_TOTAL_KV_SPLITS = (
        NUM_SPARSE_KV_SPLITS
        + NUM_SINK_KV_SPLITS
        + NUM_SLIDING_KV_SPLITS
    )
    temp_attn_logits = torch.zeros(
        # NOTE: make address space 32byte bank aligned
        (batch, head_num, NUM_TOTAL_KV_SPLITS, HID + 32//4),
        dtype=torch.float32, device=q.device
    )

    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_TOTAL_KV_SPLITS,
    )

    BLOCK_DMODEL = triton.next_power_of_2(HID)
    BLOCK_DV = triton.next_power_of_2(HID)

    _fwd_kernel_stage1[grid](
        q, *safe_stride(q, 4),
        k, *safe_stride(k, 4),
        v, *safe_stride(v, 4),
        seq_lens, *safe_stride(seq_lens, 2),

        indices, *safe_stride(indices, 3),

        ks_start_end, *safe_stride(ks_start_end, 3),

        temp_attn_logits, *safe_stride(temp_attn_logits, 4),

        head_num, 
        BK, 
        MAX_TDST, 
        MAX_TSRC, 
        kv_group_num,

        args.sliding_window_size,
        args.sink_token_size,
        args.logit_softcap,

        *args.args_extend(),
        model_context_length,
        *args.args_paged_kv_cache(),
        *args.args_offload_cache(is_masking=False),

        access_counter, *safe_stride(access_counter, 3),
        cache_miss_counter, *safe_stride(cache_miss_counter, 3),

        triton.next_power_of_2(MAX_TDST),

        args.is_causal,
        args.block_size_q,
        args.block_size_k,
        Lk=HID, Lv=HID,

        BLOCK_BK=BLOCK_BK,
        NUM_SPARSE_KV_SPLITS=NUM_SPARSE_KV_SPLITS,
        NUM_SINK_KV_SPLITS=NUM_SINK_KV_SPLITS,
        NUM_SLIDING_KV_SPLITS=NUM_SLIDING_KV_SPLITS,
        BLOCK_H=BLOCK_H,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        EXTEND_BACKEND=extend_backend,
        UPDATE_CACHE=offload_update_cache,
    )

    return temp_attn_logits, NUM_TOTAL_KV_SPLITS


@triton.jit
def _fwd_kernel_stage2(
    ATTN_LOGITS,
    stride_attn_logits_bsz, 
    stride_attn_logits_head, 
    stride_attn_logits_kv_split, 
    stride_attn_logits_hid,

    O, 
    stride_o_bsz, 
    stride_o_tdst, 
    stride_o_head, 
    stride_o_hid,
    
    B_SEQ_LEN, 
    stride_pos_bsz, 
    stride_pos_tdst,

    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(
        B_SEQ_LEN 
        + cur_batch * stride_pos_bsz 
        + 0 * stride_pos_tdst
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = (
        cur_batch * stride_attn_logits_bsz
        + cur_head * stride_attn_logits_head
        + offs_d * stride_attn_logits_hid
    )
    offs_logic = (
        cur_batch * stride_attn_logits_bsz
        + cur_head * stride_attn_logits_head
        + Lv * stride_attn_logits_hid
    )

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                ATTN_LOGITS 
                + offs_v 
                + split_kv_id * stride_attn_logits_kv_split,
                mask=mask_d,
                other=0.0
            )
            tlogic = tl.load(
                ATTN_LOGITS 
                + offs_logic 
                + split_kv_id * stride_attn_logits_kv_split
            )
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.math.exp2(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.math.exp2(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    e_sum = tl.where(e_sum < 1e-20, 1e-20, e_sum)

    tl.store(
        O
        + cur_batch * stride_o_bsz
        + 0 * stride_o_tdst
        + cur_head * stride_o_head
        + offs_d * stride_o_hid,
        value=acc / e_sum,
        mask=mask_d,
    )

def decode_block_sparse_attention_stage2(
    logits,
    q,
    o,
    v_buffer,
    b_seq_len,
    num_total_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[2]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_total_kv_splits

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits, *safe_stride(logits, 4),
        o, *safe_stride(o, 4),
        b_seq_len, *safe_stride(b_seq_len, 2),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )

def decode_block_sparse_attention_impl(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    
    seq_lens: Tensor,
    indices: Tensor,
    ks_start_end: Tensor,
    
    context: Tensor,
    
    args: HiPAttentionArgs,
    
    HEAD: int, 
    BK: int, 
    MAX_TDST: int, 
    MAX_TSRC: int, 
    KV_HEAD_REPEAT: int,
    
    model_context_length: int,
    
    HID: int, 
    BLOCK_BK: int, 
    
    extend_backend: str,
    access_counter: Tensor,
    cache_miss_counter: Tensor,
    offload_update_cache: bool,
):
    """
    FlashDecode block sparse attention.
    :param q: (BSZ, TDST, HEAD, HID)
    :param seq_lens: (BSZ, TDST)
    :param indices: (BSZ, TDST, BK)
    :param ks_start_end: (BSZ, BSRC, 2)
    :param context: (BSZ, TDST, HEAD, HID)
    """
    
    attn_logits, NUM_TOTAL_KV_SPLITS = decode_block_sparse_attention_stage1(
        q, k, v,
        
        seq_lens=seq_lens,
        indices=indices,
        ks_start_end=ks_start_end,
        
        args=args,
        
        head_num=HEAD, 
        BK=BK,
        MAX_TDST=MAX_TDST, 
        MAX_TSRC=MAX_TSRC,
        kv_group_num=KV_HEAD_REPEAT,
        model_context_length=model_context_length,
        HID=HID, 
        BLOCK_BK=BLOCK_BK,
        
        extend_backend=extend_backend,
        access_counter=access_counter,
        cache_miss_counter=cache_miss_counter,
        offload_update_cache=offload_update_cache,
    )
    
    decode_block_sparse_attention_stage2(
        attn_logits, 
        q, 
        context, 
        (
            args.v_cache 
            if args.v_cache is not None else 
            args.offload_cache.v_uvm.bank_cpu
        ), 
        seq_lens, 
        NUM_TOTAL_KV_SPLITS
    )
    
    return attn_logits

def decode_block_sparse_attention(
    q: Tensor,  # [1, 1 (TDST), 32 (Q_HEAD), 128]
    k: Optional[Tensor],  # None
    v: Optional[Tensor],  # None
    seq_lens: Tensor,  # [1, 1 (TDST)], tensor([34089])

    indices: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST), 512]
    ks: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST)]
    ks_count: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST), 1]
    ks_start_end: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST), 2]

    args: HiPAttentionArgs,
    # args.block_table: [1 (BSZ), 196612]
    # args.cache_seq_lens: [1 (BSZ)], tensor([34089])
    # args.k_cache: [109527 (NUM_PAGE), 1 (PAGE_SIZE), 8 (KV_HEAD), 128 (Lk)]
    # args.v_cache: [109527 (NUM_PAGE), 1 (PAGE_SIZE), 8 (KV_HEAD), 128 (Lv)]
    # args.position_ids: [1, 1 (TDST)]
    # args.rope_cos: [196608, 128 (Lk)]
    # args.rope_sin: [196608, 128 (Lk)]

    access_counter: Tensor,  # [1, 8, 109527]
    cache_miss_counter: Tensor,  # [1, 8, 109527]

    EXTEND_BACKEND: str = DEFAULT_EXTEND_BACKEND,  # 'streaming'
    model_context_length: int = 131072,  # 131072
    extend_context_length: int = 131072,  # 196608
    offload_update_cache: bool = False,
):
    BSZ, TDST, HEAD, HID = q.shape

    assert TDST == 1, "TDST must be 1 for flashdecode"

    if k is not None:
        _, TSRC, KV_HEAD, _ = k.shape
        MAX_TSRC = TSRC
    else:
        if args.k_cache is not None:
            NUM_PAGE, PAGE_SIZE, KV_HEAD, _ = args.k_cache.shape
        else:
            KV_HEAD = args.offload_cache.k_uvm.bank_cpu.shape[-2]
        MAX_TSRC = extend_context_length
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD == HEAD

    BK = indices.shape[-1]

    context = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    max_block_size = int(os.getenv('SA_BLOCK_SIZE', '16'))
    BLOCK_BK = max_block_size // args.block_size_k
    BLOCK_BK = max(1, min(max_block_size, BLOCK_BK))
    if 'SA_BLOCK_BK' in os.environ:
        BLOCK_BK = int(os.environ['SA_BLOCK_BK'])

    assert BLOCK_BK > 0, BLOCK_BK

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
        if args.k_cache is not None:
            assert args.k_cache.ndim == 4
            assert args.v_cache.ndim == 4
        else:
            assert args.offload_cache.k_uvm.bank_cpu.ndim == 3
            assert args.offload_cache.v_uvm.bank_cpu.ndim == 3
    else:
        raise Exception()
    assert seq_lens.ndim == 2

    pre_device = torch.get_default_device()
    torch.set_default_device(q.device)

    decode_block_sparse_attention_impl(
        q, k, v,
        seq_lens=seq_lens,
        indices=indices,
        ks_start_end=ks_start_end,
        context=context,
        args=args,
        HEAD=HEAD, BK=BK,
        MAX_TDST=TDST, MAX_TSRC=MAX_TSRC,
        KV_HEAD_REPEAT=KV_HEAD_REPEAT,
        model_context_length=model_context_length,
        HID=HID, BLOCK_BK=BLOCK_BK,
        extend_backend=EXTEND_BACKEND,
        access_counter=access_counter,
        cache_miss_counter=cache_miss_counter,
        offload_update_cache=offload_update_cache,
    )

    torch.set_default_device(pre_device)

    return context

def test_correctness(use_cuda_graph=False):
    from hip.models.hip_attention.gen3.attention_extend_bsa import block_sparse_attention
    args = torch.load("../bsa_args_2.pth")

    def run_orig(output=None):
        result = block_sparse_attention(
            args['q'], 
            args['k'], 
            args['v'],
            args['seq_lens'],
            args['indices'],
            args['ks'],
            args['ks_count'],
            args['ks_start_end'],
            args['args'],
            args['access_counter'],
            args['cache_miss_counter'],
            args['EXTEND_BACKEND'],
            args['model_context_length'],
            args['extend_context_length'],
        )
        if output is not None:
            output.copy_(result)
        return result

    def run_flash(output=None):
        result = decode_block_sparse_attention(
            args['q'], 
            args['k'], 
            args['v'],
            args['seq_lens'],
            args['indices'],
            args['ks'],
            args['ks_count'],
            args['ks_start_end'],
            args['args'],
            args['access_counter'],
            args['cache_miss_counter'],
            args['EXTEND_BACKEND'],
            args['model_context_length'],
            args['extend_context_length'],
        )
        if output is not None:
            output.copy_(result)
        return result

    if use_cuda_graph:
        gt_output = torch.zeros_like(args['q'])

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run_orig(gt_output)
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g_orig = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_orig):
            run_orig(gt_output)

        gt_output.zero_()
        g_orig.replay()

        output = torch.zeros_like(args['q'])

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run_flash(output)
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g_flash = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_flash):
            run_flash(output)

        output.zero_()
        g_flash.replay()

    else:
        gt_output = run_orig()
        output = run_flash()

    print('context diff', (output - gt_output).abs().mean() / gt_output.abs().mean())


@torch.no_grad()
def benchmark(use_cuda_graph=False):
    from hip.models.hip_attention.gen3.attention_extend_bsa import block_sparse_attention
    args = torch.load("../bsa_args_2.pth")

    def run_orig():
        return block_sparse_attention(
            args['q'], args['k'], args['v'],
            args['seq_lens'],
            args['indices'],
            args['ks'],
            args['ks_count'],
            args['ks_start_end'],
            args['args'],
            args['access_counter'],
            args['cache_miss_counter'],
            args['EXTEND_BACKEND'],
            args['model_context_length'],
            args['extend_context_length'],
        )

    def run_flash():
        return decode_block_sparse_attention(
            args['q'], args['k'], args['v'],
            args['seq_lens'],
            args['indices'],
            args['ks'],
            args['ks_count'],
            args['ks_start_end'],
            args['args'],
            args['access_counter'],
            args['cache_miss_counter'],
            args['EXTEND_BACKEND'],
            args['model_context_length'],
            args['extend_context_length'],
        )

    if use_cuda_graph:
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run_orig()
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g_orig = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_orig):
            run_orig()

        run_orig = (lambda: g_orig.replay())

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    start_mem = torch.cuda.max_memory_allocated(0)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    run_orig()
    end_time.record()
    torch.cuda.synchronize()
    peak_mem_orig = torch.cuda.max_memory_allocated(0) - start_mem
    elapsed_orig = start_time.elapsed_time(end_time)

    if use_cuda_graph:
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run_flash()
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g_flash = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_flash):
            run_flash()

        run_flash = (lambda: g_flash.replay())

    else:
        # Warmup
        for _ in range(2):
            run_flash()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    start_mem = torch.cuda.max_memory_allocated(0)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    run_flash()
    end_time.record()
    torch.cuda.synchronize()
    peak_mem_flash = torch.cuda.max_memory_allocated(0) - start_mem
    elapsed_flash = start_time.elapsed_time(end_time)

    print(f"Time: orig={elapsed_orig:.2f}ms, flash={elapsed_flash:.2f}ms")
    print(f"Peak memory: orig={peak_mem_orig / 10 ** 6:.3f}MB, flash={peak_mem_flash / 10 ** 6:.3f}MB")


if __name__ == "__main__":
    test_correctness(use_cuda_graph=False)
    benchmark(use_cuda_graph=False)
