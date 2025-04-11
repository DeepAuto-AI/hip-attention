from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention_cuda_step
from hip_attn.v1_2.attention_metadata import safe_stride
from hip_attn.v1_2.uvm_gpu_cache import load_tokens

if TYPE_CHECKING:
    from hip_attn.v1_2.attention_metadata import HiPAttentionArgs

DEFAULT_EXTEND_BACKEND: tl.constexpr = "streaming"
MAX_INT: tl.constexpr = 2_147_483_647


@triton.jit
def load_queries(
    cur_batch,
    cur_head,
    offs_d,
    mask_h,
    mask_d,
    cur_batch_seq_len,
    Lk: tl.constexpr,
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
):
    offs_q = (
        cur_batch.to(tl.int64) * stride_q_bsz
        + 0 * stride_q_tdst
        + cur_head[:, None] * stride_q_head
        + offs_d[None, :] * stride_q_hid
    )
    q = tl.load(
        Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0
    )  # [BLOCK_H, BLOCK_DMODEL]
    if q.dtype == tl.float8e5:
        q = q.to(tl.float16)

    if USING_EXTEND and NEED_APPLY_ROPE:
        ROPE_DIM = rope_range_end - rope_range_begin

        idx_rope_range = offs_d - rope_range_begin
        rope_mask = (rope_range_begin <= offs_d) & (offs_d < rope_range_end)
        if rope_is_neox_style:
            rope_rot_idx = tl.where(
                rope_mask,
                (offs_d - rope_range_begin + ROPE_DIM // 2) % ROPE_DIM
                + rope_range_begin,
                offs_d,
            )
            cos_sin_idx = idx_rope_range % (ROPE_DIM // 2)
            rope_mult = ((idx_rope_range + ROPE_DIM // 2 < ROPE_DIM) * (-2) + 1).to(
                q.dtype
            )
        else:
            flip = tl.where(idx_rope_range & 1 == 0, 1, -1)
            rope_rot_idx = tl.where(
                rope_mask,
                idx_rope_range + flip + rope_range_begin,
                offs_d,
            )
            cos_sin_idx = idx_rope_range // 2
            rope_mult = ((idx_rope_range % 2 == 0) * (-2) + 1).to(q.dtype)

        rope_tdst = cur_batch_seq_len - 1

        queries_rot = tl.load(
            Q
            + cur_batch.to(tl.int64) * stride_q_bsz
            + 0 * stride_q_tdst
            + cur_head[:, None] * stride_q_head
            + rope_rot_idx[None, :] * stride_q_hid,
            mask=(mask_h[:, None]) & (mask_d[None, :] & rope_mask[None, :]),
            other=0.0,
        )  # [BLOCK_H, BLOCK_DMODEL]
        if queries_rot.dtype == tl.float8e5:
            queries_rot = queries_rot.to(tl.float16)

        cos_new = tl.load(
            COS
            + rope_tdst.to(tl.int64) * stride_cos_t
            + cos_sin_idx[None, :] * stride_cos_hid,
            mask=mask_d[None, :] & rope_mask[None, :],
            other=0.0,
        ).to(
            q.dtype
        )  # [1, BLOCK_DMODEL]
        sin_new = tl.load(
            SIN
            + rope_tdst.to(tl.int64) * stride_sin_t
            + cos_sin_idx[None, :] * stride_sin_hid,
            mask=mask_d[None, :] & rope_mask[None, :],
            other=0.0,
        ).to(
            q.dtype
        )  # [1, BLOCK_DMODEL]

        queries_rot *= rope_mult[None, :]

        q = tl.where(
            rope_mask[None, :],
            (q * cos_new + queries_rot * sin_new).to(q.dtype),
            q,
        )

    return q


@triton.jit
def _fwd_kernel_stage1(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    V,
    stride_v_bsz,
    stride_v_tsrc,
    stride_v_head,
    stride_v_hid,
    B_Seqlen,
    stride_pos_bsz,
    stride_pos_tdst,
    INDICES,  # Warning: first dim is a flattened axis of (batch, q_head)
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    KS_START_END,  # Warning: first dim is a flattened axis of (batch, q_head)
    stride_ks_start_end_b,
    stride_ks_start_end_bdst,
    stride_ks_start_end_g,
    ATTN_LOGITS,
    stride_attn_logits_bsz,
    stride_attn_logits_head,
    stride_attn_logits_kv_split,
    stride_attn_logits_hid,
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
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
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
    BLOCK_DMODEL_0: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    UPDATE_CACHE: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head_id = tl.program_id(1).to(tl.int64)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2).to(tl.int64)
    sink_split_kv_id = split_kv_id - NUM_SPARSE_KV_SPLITS
    sliding_split_kv_id = split_kv_id - NUM_SPARSE_KV_SPLITS - NUM_SINK_KV_SPLITS

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head_begin = cur_head_id * VALID_BLOCK_H
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    # FIXME: current implementation is incorrect across heads
    cur_flattened_batch = cur_batch * q_head_num + cur_head_begin  # [BLOCK_H]

    ROPE_DIM = rope_range_end - rope_range_begin

    BLOCK_DMODEL_1: tl.constexpr = Lk - BLOCK_DMODEL_0

    offs_d_0 = tl.arange(0, BLOCK_DMODEL_0)
    mask_d_0 = offs_d_0 < Lk
    rope_mask_0 = (rope_range_begin <= offs_d_0) & (offs_d_0 < rope_range_end)
    idx_rope_range_q0 = offs_d_0 - rope_range_begin
    if rope_is_neox_style:
        rope_rot_idx_0 = tl.where(
            rope_mask_0,
            (idx_rope_range_q0 + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin,
            offs_d_0,
        )
    else:
        flip = tl.where(idx_rope_range_q0 % 2 == 0, 1, -1)
        rope_rot_idx_0 = tl.where(
            rope_mask_0,
            idx_rope_range_q0 + flip + rope_range_begin,
            offs_d_0,
        )

    if BLOCK_DMODEL_1 > 0:
        offs_d_1 = BLOCK_DMODEL_0 + tl.arange(0, BLOCK_DMODEL_1)
        mask_d_1 = offs_d_1 < Lk
        rope_mask_1 = (rope_range_begin <= offs_d_1) & (offs_d_1 < rope_range_end)
        idx_rope_range_q1 = offs_d_1 - rope_range_begin
        if rope_is_neox_style:
            rope_rot_idx_1 = tl.where(
                rope_mask_1,
                (idx_rope_range_q1 + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin,
                offs_d_1,
            )
        else:
            flip = tl.where(idx_rope_range_q1 % 2 == 0, 1, -1)
            rope_rot_idx_1 = tl.where(
                rope_mask_1,
                idx_rope_range_q1 + flip + rope_range_begin,
                offs_d_1,
            )
    else:
        offs_d_1 = None
        mask_d_1 = None

    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < Lv

    cur_batch_seq_len = tl.load(
        B_Seqlen + cur_batch.to(tl.int64) * stride_pos_bsz + 0 * stride_pos_tdst
    )
    # cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    q_0 = load_queries(
        cur_batch,
        cur_head,
        offs_d_0,
        mask_h,
        mask_d_0,
        cur_batch_seq_len,
        Lk,
        Q,
        stride_q_bsz,
        stride_q_tdst,
        stride_q_head,
        stride_q_hid,
        COS,
        stride_cos_t,
        stride_cos_hid,
        SIN,
        stride_sin_t,
        stride_sin_hid,
        rope_range_begin,
        rope_range_end,
        rope_is_neox_style,
        USING_EXTEND and (rope_range_begin < BLOCK_DMODEL_0),
        NEED_APPLY_ROPE,
    )

    if BLOCK_DMODEL_1 > 0:
        q_1 = load_queries(
            cur_batch,
            cur_head,
            offs_d_1,
            mask_h,
            mask_d_1,
            cur_batch_seq_len,
            Lk,
            Q,
            stride_q_bsz,
            stride_q_tdst,
            stride_q_head,
            stride_q_hid,
            COS,
            stride_cos_t,
            stride_cos_hid,
            SIN,
            stride_sin_t,
            stride_sin_hid,
            rope_range_begin,
            rope_range_end,
            rope_is_neox_style,
            USING_EXTEND,
            NEED_APPLY_ROPE,
        )
    else:
        q_1 = None

    # Start and end indices to the `indices` tensor
    range_start = tl.load(
        KS_START_END
        + cur_flattened_batch.to(tl.int64) * stride_ks_start_end_b
        + 0 * stride_ks_start_end_bdst
        + 0 * stride_ks_start_end_g,
        mask=cur_head_begin < q_head_num,
        other=0,
    )
    range_end = tl.load(
        KS_START_END
        + cur_flattened_batch.to(tl.int64) * stride_ks_start_end_b
        + 0 * stride_ks_start_end_bdst
        + 1 * stride_ks_start_end_g,
        mask=cur_head_begin < q_head_num,
        other=0,
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

    if (split_kv_block_end > split_kv_block_start) and True:
        for i_bk in range(split_kv_block_start, split_kv_block_end, BLOCK_BK):
            idx_bk = i_bk + tl.arange(0, BLOCK_BK)  # [BLOCK_BK]
            mask_bk = (range_start <= idx_bk) & (
                idx_bk < tl.minimum(range_start + BK, range_end)
            )  # [BLOCK_BK]

            if (range_start <= i_bk + BLOCK_BK) & (i_bk < range_end):
                idx_tsrc_start = tl.load(
                    INDICES
                    + cur_flattened_batch.to(tl.int64) * stride_indices_b
                    + 0 * stride_indices_bdst
                    + idx_bk * stride_indices_bk,
                    mask=mask_bk & (cur_head_begin < q_head_num),
                    other=0,
                )  # [BLOCK_BK]
                idx_tsrc_start = tl.where(mask_bk, idx_tsrc_start, MAX_TSRC + 1)
                idx_tsrc = idx_tsrc_start[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :]
                idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc_from_bk = mask_bk[:, None] & tl.full(
                    (1, BLOCK_SIZE_K), 1, dtype=tl.int1
                )
                mask_tsrc_from_bk = tl.reshape(
                    mask_tsrc_from_bk, (BLOCK_BK * BLOCK_SIZE_K)
                )
                mask_tsrc = (
                    ((MAX_TSRC * 0) <= idx_tsrc)
                    & (idx_tsrc < (MAX_TSRC * 1))
                    & mask_tsrc_from_bk
                )
                idx_tsrc = idx_tsrc % MAX_TSRC  # [BLOCK_BK * BLOCK_SIZE_K]
                mask_tsrc = (
                    (sink_token_size <= idx_tsrc)
                    & (idx_tsrc < cur_batch_seq_len)
                    & mask_tsrc
                )

                keys_0 = load_tokens(
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
                    offs_d_0[:, None],
                    mask_tsrc[None, :],
                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    BLOCK_DMODEL_0,
                    Lk,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )

                if BLOCK_DMODEL_1 > 0:
                    keys_1 = load_tokens(
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
                        offs_d_1[:, None],
                        mask_tsrc[None, :],
                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        BLOCK_DMODEL_1,
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
                    keys_1 = None

                if USING_EXTEND and NEED_APPLY_ROPE:
                    if rope_range_begin < BLOCK_DMODEL_0:
                        keys_rot_0 = load_tokens(
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
                            rope_rot_idx_0[:, None],
                            mask_tsrc[None, :],
                            q_head_num // kv_group_num,
                            BLOCK_SIZE_K,
                            BLOCK_DMODEL_0,
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
                        keys_rot_0 = None

                    if BLOCK_DMODEL_1 > 0:
                        keys_rot_1 = load_tokens(
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
                            rope_rot_idx_1[:, None],
                            mask_tsrc[None, :],
                            q_head_num // kv_group_num,
                            BLOCK_SIZE_K,
                            BLOCK_DMODEL_1,
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
                        keys_rot_1 = None

                else:
                    keys_rot_0 = None
                    keys_rot_1 = None

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
                    BLOCK_DV,
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
                    q_0,  # FIXME: q is [BLOCK_H, BLOCK_DMODEL]: the first axis is head, not time
                    q_1,
                    keys_0,
                    keys_1,
                    keys_rot_0,
                    keys_rot_1,
                    values,
                    idx_tsrc,
                    mask_tsrc,
                    tl.zeros([1], dtype=tl.int32),
                    tl.full((1,), 1, dtype=tl.int1),
                    acc,
                    e_sum,
                    e_max,
                    sliding_window_size,
                    sink_token_size,
                    (range_end - range_start) * BLOCK_SIZE_K,  # mask_k
                    True,
                    False,
                    LOGIT_SOFTCAP,
                    USING_EXTEND,
                    NEED_APPLY_ROPE,
                    COS,
                    stride_cos_t,
                    stride_cos_hid,
                    SIN,
                    stride_sin_t,
                    stride_sin_hid,
                    rope_range_begin,
                    rope_range_end,
                    rope_is_neox_style,
                    model_context_length,
                    idx_bk + sink_token_size // BLOCK_SIZE_K,
                    cur_batch_seq_len,
                    offs_d_0,
                    offs_d_1,
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
    split_sink_end = tl.minimum(
        split_sink_start + sink_tokens_per_split, sink_token_size
    )
    if (
        (sink_token_size > 0)
        & (0 <= sink_split_kv_id)
        & (sink_split_kv_id < NUM_SINK_KV_SPLITS)
        & (split_sink_end > split_sink_start)
    ) and True:
        for i_tsrc in range(split_sink_start, split_sink_end, BLOCK_BK * BLOCK_SIZE_K):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < tl.minimum(cur_batch_seq_len, split_sink_end)

            keys_0 = load_tokens(
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
                offs_d_0[:, None],
                mask_tsrc[None, :],
                q_head_num // kv_group_num,
                BLOCK_SIZE_K,
                BLOCK_DMODEL_0,
                Lk,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=V_CACHE,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_offset=stride_v_cache_offset,
                stride_v_cache_kv_head=stride_v_cache_kv_head,
                stride_v_cache_hid=stride_v_cache_hid,
            )

            if BLOCK_DMODEL_1 > 0:
                keys_1 = load_tokens(
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
                    offs_d_1[:, None],
                    mask_tsrc[None, :],
                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    BLOCK_DMODEL_1,
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
                keys_1 = None

            if USING_EXTEND and NEED_APPLY_ROPE:
                if rope_range_begin < BLOCK_DMODEL_0:
                    keys_rot_0 = load_tokens(
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
                        rope_rot_idx_0[:, None],
                        mask_tsrc[None, :],
                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        BLOCK_DMODEL_0,
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
                    keys_rot_0 = None

                if BLOCK_DMODEL_1 > 0:
                    keys_rot_1 = load_tokens(
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
                        rope_rot_idx_1[:, None],
                        mask_tsrc[None, :],
                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        BLOCK_DMODEL_1,
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
                    keys_rot_1 = None
            else:
                keys_rot_0 = None
                keys_rot_1 = None

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
                BLOCK_DV,
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
                q_0,
                q_1,
                keys_0,
                keys_1,
                keys_rot_0,
                keys_rot_1,
                values,
                idx_tsrc,
                mask_tsrc,
                tl.zeros([1], dtype=tl.int32),
                tl.full((1,), 1, dtype=tl.int1),
                acc,
                e_sum,
                e_max,
                sliding_window_size,
                sink_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                True,
                True,
                LOGIT_SOFTCAP,
                USING_EXTEND,
                NEED_APPLY_ROPE,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
                model_context_length,
                tl.arange(0, BLOCK_BK) + i_tsrc // BLOCK_SIZE_K,
                cur_batch_seq_len,
                offs_d_0,
                offs_d_1,
                IS_CAUSAL,
                Lk,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,
                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    # process sliding window
    i_tsrc_range_start = tl.maximum(
        0, cur_batch_seq_len - sliding_window_size - BLOCK_SIZE_Q
    )
    sliding_tokens_per_split = tl.cdiv(
        cur_batch_seq_len - i_tsrc_range_start, NUM_SLIDING_KV_SPLITS
    )
    split_sliding_start = (
        i_tsrc_range_start + sliding_tokens_per_split * sliding_split_kv_id
    )
    split_sliding_end = tl.minimum(
        split_sliding_start + sliding_tokens_per_split, cur_batch_seq_len
    )
    if (
        (sliding_window_size > 0)
        & (0 <= sliding_split_kv_id)
        & (sliding_split_kv_id < NUM_SLIDING_KV_SPLITS)
        & (split_sliding_end > split_sliding_start)
    ) and True:
        for i_tsrc in range(
            split_sliding_start, split_sliding_end, BLOCK_BK * BLOCK_SIZE_K
        ):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = (0 <= idx_tsrc) & (idx_tsrc < split_sliding_end)

            # idx_n = idx_b * G + idx_group
            keys_0 = load_tokens(
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
                offs_d_0[:, None],
                mask_tsrc[None, :],
                q_head_num // kv_group_num,
                BLOCK_SIZE_K,
                BLOCK_DMODEL_0,
                Lk,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=V_CACHE,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_offset=stride_v_cache_offset,
                stride_v_cache_kv_head=stride_v_cache_kv_head,
                stride_v_cache_hid=stride_v_cache_hid,
            )

            if BLOCK_DMODEL_1 > 0:
                keys_1 = load_tokens(
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
                    offs_d_1[:, None],
                    mask_tsrc[None, :],
                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    BLOCK_DMODEL_1,
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
                keys_1 = None

            if USING_EXTEND and NEED_APPLY_ROPE:
                if rope_range_begin < BLOCK_DMODEL_0:
                    keys_rot_0 = load_tokens(
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
                        rope_rot_idx_0[:, None],
                        mask_tsrc[None, :],
                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        BLOCK_DMODEL_0,
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
                    keys_rot_0 = None

                if BLOCK_DMODEL_1 > 0:
                    keys_rot_1 = load_tokens(
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
                        rope_rot_idx_1[:, None],
                        mask_tsrc[None, :],
                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        BLOCK_DMODEL_1,
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
                    keys_rot_1 = None

            else:
                keys_rot_0 = None
                keys_rot_1 = None

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
                BLOCK_DV,
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
                tl.arange(0, BLOCK_BK)
                + (i_tsrc - i_tsrc_range_start) // BLOCK_SIZE_K
                + (cur_batch_seq_len - 1 - sliding_window_size) // BLOCK_SIZE_K
            )
            acc, e_sum, e_max = block_sparse_attention_cuda_step(
                q_0,  # [BLOCK_H, BLOCK_DMODEL]
                q_1,  # [BLOCK_DMODEL, BLOCK_BK * BLOCK_SIZE_K]
                keys_0,
                keys_1,
                keys_rot_0,
                keys_rot_1,
                values,
                idx_tsrc,
                mask_tsrc,
                tl.zeros([1], dtype=tl.int32),
                tl.full((1,), 1, dtype=tl.int1),
                acc,
                e_sum,
                e_max,
                sliding_window_size,
                sink_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                False,
                False,
                LOGIT_SOFTCAP,
                USING_EXTEND,
                NEED_APPLY_ROPE,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
                model_context_length,
                idx_bk,
                cur_batch_seq_len,
                offs_d_0,
                offs_d_1,
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
        cur_batch.to(tl.int64) * stride_attn_logits_bsz
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
        cur_batch.to(tl.int64) * stride_attn_logits_bsz
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
    HID_V: int,
    BLOCK_BK: int,
    extend_backend: str,
    access_counter: Tensor,
    cache_miss_counter: Tensor,
    offload_update_cache: bool,
):
    batch = q.shape[0]
    BLOCK_H = 16

    total_tokens = args.second_stage_k + args.sink_token_size + args.sliding_window_size
    token_chunk = triton.cdiv(total_tokens, 8)
    NUM_SPARSE_KV_SPLITS = min(
        12, triton.cdiv(args.second_stage_k, token_chunk)
    )  # TODO: apply from server args
    NUM_SINK_KV_SPLITS = min(12, triton.cdiv(args.sink_token_size, token_chunk))
    NUM_SLIDING_KV_SPLITS = min(12, triton.cdiv(args.sliding_window_size, token_chunk))

    NUM_TOTAL_KV_SPLITS = (
        NUM_SPARSE_KV_SPLITS + NUM_SINK_KV_SPLITS + NUM_SLIDING_KV_SPLITS
    )
    temp_attn_logits = torch.zeros(
        (batch, head_num, NUM_TOTAL_KV_SPLITS, HID + 32 // 4),
        dtype=torch.float32,
        device=q.device,
    )

    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_TOTAL_KV_SPLITS,
    )

    if args.rope_range[0] == 0 and args.rope_range[1] == HID:
        BLOCK_DMODEL = triton.next_power_of_2(HID)
    else:
        assert triton.next_power_of_2(args.rope_range[0]) == args.rope_range[0]
        assert args.rope_range[1] == HID
        BLOCK_DMODEL = args.rope_range[0]

    BLOCK_DV = triton.next_power_of_2(HID_V)

    _fwd_kernel_stage1[grid](
        q,
        *safe_stride(q, 4),
        k,
        *safe_stride(k, 4),
        v,
        *safe_stride(v, 4),
        seq_lens,
        *safe_stride(seq_lens, 2),
        indices,
        *safe_stride(indices, 3),
        ks_start_end,
        *safe_stride(ks_start_end, 3),
        temp_attn_logits,
        *safe_stride(temp_attn_logits, 4),
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
        access_counter,
        *safe_stride(access_counter, 3),
        cache_miss_counter,
        *safe_stride(cache_miss_counter, 3),
        triton.next_power_of_2(MAX_TDST),
        args.is_causal,
        args.block_size_q,
        args.block_size_k,
        Lk=HID,
        Lv=HID_V,
        BLOCK_BK=BLOCK_BK,
        NUM_SPARSE_KV_SPLITS=NUM_SPARSE_KV_SPLITS,
        NUM_SINK_KV_SPLITS=NUM_SINK_KV_SPLITS,
        NUM_SLIDING_KV_SPLITS=NUM_SLIDING_KV_SPLITS,
        BLOCK_H=BLOCK_H,
        BLOCK_DMODEL_0=BLOCK_DMODEL,
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
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)

    cur_batch_seq_len = tl.load(
        B_SEQ_LEN + cur_batch.to(tl.int64) * stride_pos_bsz + 0 * stride_pos_tdst
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
        tv = tl.load(
            ATTN_LOGITS
            + offs_v.to(tl.int64)
            + split_kv_id * stride_attn_logits_kv_split,
            mask=mask_d,
            other=0.0,
        )
        tlogic = tl.load(
            ATTN_LOGITS
            + offs_logic.to(tl.int64)
            + split_kv_id * stride_attn_logits_kv_split
        )
        n_e_max = tl.maximum(tlogic, e_max)

        n_e_max_valid = n_e_max > -1e50
        old_scale = tl.math.exp2(e_max - n_e_max)
        exp_logic = tl.math.exp2(tlogic - n_e_max)
        acc = tl.where(n_e_max_valid, acc * old_scale + exp_logic * tv, acc)

        e_sum = tl.where(n_e_max_valid, e_sum * old_scale + exp_logic, e_sum)
        e_max = n_e_max

    e_sum = tl.where(e_sum < 1e-20, 1e-20, e_sum)

    tl.store(
        O
        + cur_batch.to(tl.int64) * stride_o_bsz
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
    b_seq_len,
    num_total_kv_splits,
    HID_V: int,
):
    batch, head_num = q.shape[0], q.shape[2]
    Lv = HID_V
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_total_kv_splits

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        *safe_stride(logits, 4),
        o,
        *safe_stride(o, 4),
        b_seq_len,
        *safe_stride(b_seq_len, 2),
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
    HID_V: int,
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
        q,
        k,
        v,
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
        HID_V=HID_V,
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
        seq_lens,
        NUM_TOTAL_KV_SPLITS,
        HID_V,
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
        HID_V = v.shape[-1]
    else:
        if args.k_cache is not None:
            NUM_PAGE, PAGE_SIZE, KV_HEAD, _ = args.k_cache.shape
            HID_V = args.v_cache.shape[-1]
        else:
            KV_HEAD = args.offload_cache.k_uvm.bank_cpu.shape[-2]
            HID_V = args.offload_cache.v_uvm.bank_cpu.shape[-1]
        MAX_TSRC = extend_context_length
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD == HEAD
    HID_V = args.v_hidden_dim if args.v_hidden_dim is not None else HID_V

    BK = indices.shape[-1]

    context = torch.empty((BSZ, TDST, HEAD, HID_V), dtype=q.dtype, device=q.device)

    max_block_size = int(os.getenv("SA_BLOCK_SIZE", "16"))
    BLOCK_BK = max_block_size // args.block_size_k
    BLOCK_BK = max(1, min(max_block_size, BLOCK_BK))
    if "SA_BLOCK_BK" in os.environ:
        BLOCK_BK = int(os.environ["SA_BLOCK_BK"])

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
        q,
        k,
        v,
        seq_lens=seq_lens,
        indices=indices,
        ks_start_end=ks_start_end,
        context=context,
        args=args,
        HEAD=HEAD,
        BK=BK,
        MAX_TDST=TDST,
        MAX_TSRC=MAX_TSRC,
        KV_HEAD_REPEAT=KV_HEAD_REPEAT,
        model_context_length=model_context_length,
        HID=HID,
        HID_V=HID_V,
        BLOCK_BK=BLOCK_BK,
        extend_backend=EXTEND_BACKEND,
        access_counter=access_counter,
        cache_miss_counter=cache_miss_counter,
        offload_update_cache=offload_update_cache,
    )

    torch.set_default_device(pre_device)

    return context
