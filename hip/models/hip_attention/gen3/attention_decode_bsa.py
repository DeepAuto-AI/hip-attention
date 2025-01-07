from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import os
import torch
from torch import Tensor
import triton
import triton.language as tl

from hip.models.hip_attention.gen3.uvm_gpu_cache import load_tokens
from hip.models.hip_attention.gen3.attention_metadata import safe_stride

if TYPE_CHECKING:
    from hip.models.hip_attention.gen3.attention_metadata import HiPAttentionArgs

DEFAULT_EXTEND_BACKEND: tl.constexpr = 'streaming'
MAX_INT: tl.constexpr = 2_147_483_647


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
    mask_t: tl.tensor,
    idx_hid: tl.tensor,

    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,

    T: tl.constexpr,
    HID: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
):
    if not NEED_APPLY_ROPE:
        mask_t = mask_t & (old_t != 0)

        cos_old = tl.load(
            COS +
            old_t[:, None].to(tl.int64) * stride_cos_t +
            idx_hid[None, :] * stride_cos_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0,
        )
        sin_old = tl.load(
            SIN +
            old_t[:, None].to(tl.int64) * stride_sin_t +
            idx_hid[None, :] * stride_sin_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0,
        )

        cos_new = tl.load(
            COS +
            new_t[:, None].to(tl.int64) * stride_cos_t +
            idx_hid[None, :] * stride_cos_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0,
        )
        sin_new = tl.load(
            SIN +
            new_t[:, None].to(tl.int64) * stride_sin_t +
            idx_hid[None, :] * stride_sin_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0,
        )

        tokens_adjusted = de_rope(
            tokens.to(tl.float32),
            cos_old.to(tl.float32),
            sin_old.to(tl.float32),
            T, HID
        )
        tokens_adjusted = apply_rope(
            tokens_adjusted.to(tl.float32),
            cos_new.to(tl.float32),
            sin_new.to(tl.float32),
            T, HID
        )

        tokens = tl.where(mask_t[:, None], tokens_adjusted.to(tokens.dtype), tokens)

        return tokens
    else:
        cos_new = tl.load(
            COS +
            new_t[:, None].to(tl.int64) * stride_cos_t +
            idx_hid[None, :] * stride_cos_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0.0,
        )
        sin_new = tl.load(
            SIN +
            new_t[:, None].to(tl.int64) * stride_sin_t +
            idx_hid[None, :] * stride_sin_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0.0,
        )

        tokens = apply_rope(
            tokens.to(tl.float32),
            cos_new.to(tl.float32),
            sin_new.to(tl.float32),
            T, HID
        ).to(tokens.dtype)

        return tokens


@triton.jit
def safe_dot(
    input,
    other,
    allow_tf32: tl.constexpr = None,
    out_dtype: tl.constexpr = tl.float32
):
    N: tl.constexpr = input.shape[0]
    K: tl.constexpr = input.shape[1]
    M: tl.constexpr = other.shape[1]
    no_tensorcores: tl.constexpr = ((N < 16 or M < 16) or K < 16)
    if no_tensorcores:
        result = tl.sum(input[:, :, None] * other[None, :, :], axis=1).to(out_dtype)
    else:
        result = tl.dot(
            input,
            other,
            allow_tf32=allow_tf32,
            out_dtype=out_dtype,
        )
    return result


@triton.jit
def block_sparse_attention_cuda_step(
    # QKV
    queries,  # [BLOCK_H, HID] TODO: different
    keys,  # [BLOCK_BK * BLOCK_SIZE_K, HID]
    keys_rot,  # [BLOCK_BK * BLOCK_SIZE_K, HID]
    values,  # [BLOCK_BK * BLOCK_SIZE_K, HID]

    # indices
    idx_tsrc, mask_tsrc,  # [BLOCK_BK * BLOCK_SIZE_K]
    idx_tdst, mask_tdst,  # [1] TODO: different

    # rolling value
    acc, l_i, m_i,  # [BLOCK_H, HID], [BLOCK_H, 1], [BLOCK_H, 1]

    sliding_window_size,
    sink_token_size,
    mask_k,
    EXCLUDE_SLIDING_WINDOW: tl.constexpr,
    HAS_FIRST_TOKEN: tl.constexpr,
    LOGIT_SOFTCAP: tl.constexpr,

    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    model_context_length,

    idx_bk,  # [BLOCK_BK]
    pos_tdst,  # [] TODO: different
    idx_hid,  # [HID]
    IS_CAUSAL: tl.constexpr,
    HID: tl.constexpr,
    BLOCK_TQ,
    BLOCK_TK,
    BLOCK_SIZE_K: tl.constexpr,

    EXTEND_BACKEND: tl.constexpr = DEFAULT_EXTEND_BACKEND,
):
    if USING_EXTEND:
        if EXTEND_BACKEND == 'self_extend':
            raise Exception()

        elif (EXTEND_BACKEND == 'streaming') or (EXTEND_BACKEND == 'dynamic_extend'):
            pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst - 1, 987654321))
            if not NEED_APPLY_ROPE:
                if ((pos_tdst_min >= model_context_length) and EXCLUDE_SLIDING_WINDOW) and True:
                    assert COS is not None
                    assert SIN is not None

                    if HAS_FIRST_TOKEN:
                        old_tdst = (pos_tdst - 1)
                        new_tdst = tl.minimum(old_tdst, sliding_window_size + mask_k + sink_token_size - 1)

                        queries_adjusted = adjust_rope(
                            queries,
                            old_tdst,
                            new_tdst,
                            mask_tdst,
                            idx_hid,
                            COS, stride_cos_t, stride_cos_hid,
                            SIN, stride_sin_t, stride_sin_hid,
                            BLOCK_TQ,
                            HID,
                            NEED_APPLY_ROPE,
                        )

                        keys_adjusted = keys
                    else:
                        old_tsrc = idx_tsrc
                        new_tsrc = tl.ravel(
                            (idx_bk * BLOCK_SIZE_K)[:, None]
                            + tl.arange(0, BLOCK_SIZE_K)[None, :]
                        )
                        new_tsrc = tl.maximum(
                            0,
                            new_tsrc + pos_tdst_min - sliding_window_size - sink_token_size - mask_k - BLOCK_TQ + 1
                        )

                        keys_adjusted = keys.trans(1, 0)
                        keys_adjusted = adjust_rope(
                            keys_adjusted.to(queries.dtype),
                            old_tsrc,
                            new_tsrc,
                            mask_tsrc,
                            idx_hid,
                            COS, stride_cos_t, stride_cos_hid,
                            SIN, stride_sin_t, stride_sin_hid,
                            BLOCK_TK,
                            HID,
                            NEED_APPLY_ROPE,
                        )
                        keys_adjusted = tl.trans(keys_adjusted, 1, 0)

                        queries_adjusted = queries
                else:
                    if NEED_APPLY_ROPE:
                        queries = adjust_rope(
                            queries.to(tl.float32),
                            pos_tdst - 1,
                            pos_tdst - 1,
                            mask_tdst,
                            idx_hid,
                            COS, stride_cos_t, stride_cos_hid,
                            SIN, stride_sin_t, stride_sin_hid,
                            BLOCK_TQ,
                            HID,
                            True,
                        ).to(queries.dtype)
                        queries_adjusted = (queries * mask_tdst[:, None]).to(queries.dtype)

                        keys = tl.trans(
                            adjust_rope(
                                tl.trans(keys.to(tl.float32), 1, 0),
                                idx_tsrc,
                                idx_tsrc,
                                mask_tsrc,
                                idx_hid,
                                COS, stride_cos_t, stride_cos_hid,
                                SIN, stride_sin_t, stride_sin_hid,
                                BLOCK_TK,
                                HID,
                                True,
                            ),
                            1, 0
                        ).to(keys.dtype)
                        keys_adjusted = (keys * mask_tsrc[None, :]).to(keys.dtype)
            else:
                tl.static_assert(NEED_APPLY_ROPE)
                tl.static_assert(USING_EXTEND)

                if EXCLUDE_SLIDING_WINDOW:
                    pos_tdst_max = pos_tdst_min + tl.sum(mask_tdst.to(tl.int32))

                    if EXTEND_BACKEND == 'streaming':
                        # streaming
                        new_tsrc = tl.ravel((idx_bk * BLOCK_SIZE_K)[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :])
                        new_tsrc = tl.maximum(
                            0,
                            new_tsrc + pos_tdst_min - sliding_window_size - sink_token_size - mask_k + 1
                        )
                    elif EXTEND_BACKEND == 'dynamic_extend':
                        # dynamic extend
                        window = model_context_length // 4

                        new_tsrc = tl.where(
                            (idx_tsrc >= (pos_tdst_max - window)) | (pos_tdst_max <= model_context_length),
                            idx_tsrc,
                            ((idx_tsrc + window - pos_tdst_min)
                             * ((model_context_length - window) / (pos_tdst_min - window))
                             ).to(tl.int32) + pos_tdst_min - window
                        )
                        new_tsrc = tl.maximum(pos_tdst_max - model_context_length, new_tsrc)
                    else:
                        raise Exception()
                else:
                    new_tsrc = idx_tsrc

                keys = keys.to(queries.dtype)
                keys_rot = keys_rot.to(queries.dtype)

                cos_new = tl.load(
                    COS +
                    new_tsrc[None, :].to(tl.int64) * stride_cos_t +
                    (tl.arange(0, HID) % (HID // 2))[:, None] * stride_cos_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                ).to(keys.dtype)
                sin_new = tl.load(
                    SIN +
                    new_tsrc[None, :].to(tl.int64) * stride_sin_t +
                    (tl.arange(0, HID) % (HID // 2))[:, None] * stride_sin_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                ).to(keys.dtype)

                if EXCLUDE_SLIDING_WINDOW:
                    if EXTEND_BACKEND == 'dynamic_extend':
                        streaming_tsrc = tl.ravel(
                            (idx_bk * BLOCK_SIZE_K)[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :])
                        streaming_tsrc = tl.maximum(
                            0,
                            streaming_tsrc + pos_tdst_min - sliding_window_size - sink_token_size - mask_k + 1
                        )

                        cos_zero = tl.load(
                            COS +
                            streaming_tsrc[None, :].to(tl.int64) * stride_cos_t +
                            (tl.arange(0, HID) % (HID // 2))[:, None] * stride_cos_hid,
                        ).to(keys.dtype)
                        sin_zero = tl.load(
                            SIN +
                            streaming_tsrc[None, :].to(tl.int64) * stride_sin_t +
                            (tl.arange(0, HID) % (HID // 2))[:, None] * stride_sin_hid,
                        ).to(keys.dtype)

                        cos_new = (cos_zero * 0.75 + cos_new * 0.25).to(cos_new.dtype)
                        sin_new = (sin_zero * 0.75 + sin_new * 0.25).to(sin_new.dtype)

                keys_rot = keys_rot * (((idx_hid + HID // 2)[:, None] < HID) * (-2) + 1).to(keys_rot.dtype)

                keys_adjusted = (keys * cos_new + keys_rot * sin_new).to(keys.dtype)

                queries_adjusted = queries

            qk = safe_dot(
                queries_adjusted * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0))).to(queries.dtype),
                keys_adjusted * (1 / tl.sqrt(tl.sqrt(HID * 1.0))).to(queries.dtype),
                out_dtype=tl.float32,
                allow_tf32=True,
            ).to(tl.float32)

            if LOGIT_SOFTCAP is not None:
                qk = tl.extra.cuda.libdevice.tanh(qk / LOGIT_SOFTCAP) * LOGIT_SOFTCAP

            qk = qk * 1.44269504

        elif EXTEND_BACKEND == 'dynamic_extend':
            assert COS is not None
            assert SIN is not None

            pos_tdst_min = tl.min(tl.where(mask_tdst, tl.maximum(0, pos_tdst - 1), 987654321)) + tl.sum(
                mask_tdst.to(tl.int32))
            if (pos_tdst_min >= model_context_length) and EXCLUDE_SLIDING_WINDOW:
                old_tdst = (pos_tdst - 1)
                new_tdst = tl.minimum(model_context_length - 1, old_tdst)

                queries = adjust_rope(
                    queries,
                    old_tdst,
                    new_tdst,
                    mask_tdst & (old_tdst != 0),
                    idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    BLOCK_TQ,
                    HID,
                    NEED_APPLY_ROPE,
                ).to(queries.dtype)
                queries = (queries * mask_tdst[:, None]).to(queries.dtype)

                if not HAS_FIRST_TOKEN:
                    old_tsrc = idx_tsrc
                    new_tsrc = tl.where(
                        (old_tsrc - pos_tdst_min + model_context_length - 1) > (model_context_length // 2),
                        old_tsrc - pos_tdst_min + model_context_length - 1,
                        ((old_tsrc - sink_token_size)
                         * ((model_context_length // 2) / (pos_tdst_min - model_context_length // 2))
                         ).to(tl.int32) + sink_token_size
                    )

                    keys_adjusted = keys.trans(1, 0)
                    keys_adjusted = adjust_rope(
                        keys_adjusted,
                        old_tsrc,
                        new_tsrc,
                        mask_tsrc & (old_tsrc != 0),
                        idx_hid,
                        COS, stride_cos_t, stride_cos_hid,
                        SIN, stride_sin_t, stride_sin_hid,
                        BLOCK_TK,
                        HID,
                        NEED_APPLY_ROPE,
                    ).to(keys.dtype)
                    keys_adjusted = tl.trans(keys_adjusted, 1, 0).to(keys.dtype)
                else:
                    keys_adjusted = keys
            else:
                keys_adjusted = keys

            qk = safe_dot(
                (queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(queries.dtype),
                (keys_adjusted.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(queries.dtype),
                out_dtype=tl.float32,
                allow_tf32=True,
            ).to(tl.float32)

            if LOGIT_SOFTCAP is not None:
                qk = tl.extra.cuda.libdevice.tanh(qk / LOGIT_SOFTCAP) * LOGIT_SOFTCAP
            qk = qk * 1.44269504
        else:
            raise Exception()
    else:
        qk = safe_dot(
            (queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(queries.dtype),
            (keys.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(queries.dtype),
            out_dtype=tl.float32,
            allow_tf32=True,
        ).to(tl.float32)
        if LOGIT_SOFTCAP is not None:
            qk = tl.extra.cuda.libdevice.tanh(qk / LOGIT_SOFTCAP) * LOGIT_SOFTCAP
        qk = qk * 1.44269504

    if IS_CAUSAL:
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
    else:
        qk_mask = (
            (~(mask_tdst[:, None] & mask_tsrc[None, :]))
        )

    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1)[:, None])
    qk = qk - m_ij
    # [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]
    p = tl.math.exp2(qk)

    p = tl.where(qk_mask, 0, p)

    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    l_ij = tl.sum(p, axis=1)

    # -- update m_i and l_i
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = (l_i * alpha + l_ij[:, None]).to(l_i.dtype)

    # -- update output accumulator --
    acc = acc * alpha.to(acc.dtype)

    # update acc
    acc += safe_dot(
        p.to(queries.dtype),
        values.to(queries.dtype),
        out_dtype=tl.float32,
        allow_tf32=True,
    ).to(acc.dtype)

    # update m_i and l_i
    m_i = m_ij.to(m_i.dtype)

    return acc, l_i, m_i


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
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
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
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
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
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
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
                    True,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
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
                False,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
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
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
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
                True,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
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

                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                False,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
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
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
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
                True,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
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

    e_sum = tl.where(e_sum == 0.0, 1e-20, e_sum)

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
    head_num: int, BK: int, MAX_TDST: int, MAX_TSRC: int, kv_group_num: int,
    model_context_length: int,
    HID: int, BLOCK_BK: int, extend_backend: str,
    access_counter: Tensor,
    cache_miss_counter: Tensor,
):
    batch = q.shape[0]
    BLOCK_H = 16

    NUM_SPARSE_KV_SPLITS = 8  # TODO: apply from server args
    NUM_SINK_KV_SPLITS = 1
    NUM_SLIDING_KV_SPLITS = 1

    NUM_TOTAL_KV_SPLITS = (
        NUM_SPARSE_KV_SPLITS
        + NUM_SINK_KV_SPLITS
        + NUM_SLIDING_KV_SPLITS
    )
    temp_attn_logits = torch.zeros(
        (batch, head_num, NUM_TOTAL_KV_SPLITS, HID + 1),
        dtype=q.dtype, device=q.device
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

        head_num, BK, MAX_TDST, MAX_TSRC, kv_group_num,

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
    )

    return temp_attn_logits, NUM_TOTAL_KV_SPLITS


@triton.jit
def _fwd_kernel_stage2(
    ATTN_LOGITS,
    stride_attn_logits_bsz, stride_attn_logits_head, stride_attn_logits_kv_split, stride_attn_logits_hid,

    O, stride_o_bsz, stride_o_tdst, stride_o_head, stride_o_hid,
    B_Seqlen, stride_pos_bsz, stride_pos_tdst,

    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch * stride_pos_bsz + 0 * stride_pos_tdst)

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
                ATTN_LOGITS + offs_v + split_kv_id * stride_attn_logits_kv_split,
                mask=mask_d,
                other=0.0
            )
            tlogic = tl.load(ATTN_LOGITS + offs_logic + split_kv_id * stride_attn_logits_kv_split)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.math.exp2(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.math.exp2(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O
        + cur_batch * stride_o_bsz
        + 0 * stride_o_tdst
        + cur_head * stride_o_head
        + offs_d * stride_o_hid,
        acc / e_sum,
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
    HEAD: int, BK: int, MAX_TDST: int, MAX_TSRC: int, KV_HEAD_REPEAT: int,
    model_context_length: int,
    HID: int, BLOCK_BK: int, extend_backend: str,
    access_counter: Tensor,
    cache_miss_counter: Tensor,
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
        head_num=HEAD, BK=BK,
        MAX_TDST=MAX_TDST, MAX_TSRC=MAX_TSRC,
        kv_group_num=KV_HEAD_REPEAT,
        model_context_length=model_context_length,
        HID=HID, BLOCK_BK=BLOCK_BK,
        extend_backend=extend_backend,
        access_counter=access_counter,
        cache_miss_counter=cache_miss_counter,
    )
    decode_block_sparse_attention_stage2(
        attn_logits, q, context, args.v_cache, seq_lens, NUM_TOTAL_KV_SPLITS
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

    max_block_size = int(os.getenv('SA_BLOCK_SIZE', '32'))
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

    attn_logits = decode_block_sparse_attention_impl(
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
    )

    torch.set_default_device(pre_device)

    return context


def test_correctness():
    from hip.models.hip_attention.gen3.attention_extend_bsa import block_sparse_attention
    args = torch.load("../bsa_args_2.pth", weights_only=True)
    gt_output = block_sparse_attention(
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
    output = decode_block_sparse_attention(
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
    print('context diff', (output - gt_output).abs().mean() / gt_output.abs().mean())


if __name__ == "__main__":
    test_correctness()
