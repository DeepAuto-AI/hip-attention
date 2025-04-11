import os
import warnings
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton import cdiv as cdiv_python

from hip_attn.utils.rope import adjust_rope
from hip_attn.v1_2.attention_metadata import HiPAttentionArgs, safe_stride
from hip_attn.v1_2.uvm_gpu_cache import load_tokens

DEFAULT_EXTEND_BACKEND: tl.constexpr = "streaming"


@triton.jit
def apply_rope_to_keys(
    queries,
    keys,
    keys_rot,
    # indices
    idx_tsrc,
    mask_tsrc,
    mask_tdst,
    pos_tdst,
    idx_bk,
    idx_hid,
    # configs
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
    sink_token_size,
    mask_k,
    sliding_window_size,
    HID: tl.constexpr,
    BLOCK_TQ: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    HAS_FIRST_TOKEN: tl.constexpr,
    EXCLUDE_SLIDING_WINDOW: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
):
    if EXTEND_BACKEND == "self_extend":
        raise Exception()
    elif (
        (EXTEND_BACKEND == "streaming")
        | (EXTEND_BACKEND == "dynamic_extend")
        | (EXTEND_BACKEND == "infllm")
    ):
        pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst - 1, 987654321))
        if not NEED_APPLY_ROPE:
            if (
                (pos_tdst_min >= model_context_length) and EXCLUDE_SLIDING_WINDOW
            ) and True:
                assert COS is not None
                assert SIN is not None

                if HAS_FIRST_TOKEN:
                    old_tdst = pos_tdst - 1
                    new_tdst = tl.minimum(
                        old_tdst, sliding_window_size + mask_k + sink_token_size - 1
                    )

                    queries_adjusted = adjust_rope(
                        queries,
                        old_tdst,
                        new_tdst,
                        mask_tdst,
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_TQ,
                        HID,
                        idx_hid.shape[0],
                        NEED_APPLY_ROPE,
                        rope_range_begin,
                        rope_range_end,
                        rope_is_neox_style,
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
                        new_tsrc
                        + pos_tdst_min
                        - sliding_window_size
                        - sink_token_size
                        - mask_k
                        - BLOCK_TQ
                        + 1,
                    )

                    keys_adjusted = keys.trans(1, 0)
                    keys_adjusted = adjust_rope(
                        keys_adjusted.to(queries.dtype),
                        old_tsrc,
                        new_tsrc,
                        mask_tsrc,
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_TK,
                        HID,
                        idx_hid.shape[0],
                        NEED_APPLY_ROPE,
                        rope_range_begin,
                        rope_range_end,
                        rope_is_neox_style,
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
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_TQ,
                        HID,
                        idx_hid.shape[0],
                        True,
                        rope_range_begin,
                        rope_range_end,
                        rope_is_neox_style,
                    ).to(queries.dtype)
                    queries_adjusted = (queries * mask_tdst[:, None]).to(queries.dtype)

                    keys = tl.trans(
                        adjust_rope(
                            tl.trans(keys.to(tl.float32), 1, 0),
                            idx_tsrc,
                            idx_tsrc,
                            mask_tsrc,
                            idx_hid,
                            COS,
                            stride_cos_t,
                            stride_cos_hid,
                            SIN,
                            stride_sin_t,
                            stride_sin_hid,
                            BLOCK_TK,
                            HID,
                            idx_hid.shape[0],
                            True,
                            rope_range_begin,
                            rope_range_end,
                            rope_is_neox_style,
                        ),
                        1,
                        0,
                    ).to(keys.dtype)
                    keys_adjusted = (keys * mask_tsrc[None, :]).to(keys.dtype)

        else:
            tl.static_assert(NEED_APPLY_ROPE)
            tl.static_assert(USING_EXTEND)

            ROPE_DIM = rope_range_end - rope_range_begin

            idx_rope_range = idx_hid - rope_range_begin
            rope_mask = (rope_range_begin <= idx_hid) & (idx_hid < rope_range_end)
            if rope_is_neox_style:
                cos_sin_idx = idx_rope_range % (ROPE_DIM // 2)
                rope_mult = ((idx_rope_range + ROPE_DIM // 2 < ROPE_DIM) * (-2) + 1).to(
                    queries.dtype
                )
            else:
                cos_sin_idx = idx_rope_range // 2
                rope_mult = ((idx_rope_range % 2 == 0) * (-2) + 1).to(queries.dtype)

            if EXCLUDE_SLIDING_WINDOW:
                pos_tdst_max = pos_tdst_min + tl.sum(mask_tdst.to(tl.int32))

                if EXTEND_BACKEND == "streaming":
                    # streaming
                    new_tsrc = tl.ravel(
                        (idx_bk * BLOCK_SIZE_K)[:, None]
                        + tl.arange(0, BLOCK_SIZE_K)[None, :]
                    )
                    new_tsrc = tl.maximum(
                        0,
                        new_tsrc
                        + pos_tdst_min
                        - sliding_window_size
                        - sink_token_size
                        - mask_k
                        + 1,
                    )
                elif EXTEND_BACKEND == "dynamic_extend":
                    # dynamic extend
                    window = model_context_length // 4

                    new_tsrc = tl.where(
                        (idx_tsrc >= (pos_tdst_max - window))
                        | (pos_tdst_max <= model_context_length),
                        idx_tsrc,
                        (
                            (idx_tsrc + window - pos_tdst_min)
                            * (
                                (model_context_length - window)
                                / (pos_tdst_min - window)
                            )
                        ).to(tl.int32)
                        + pos_tdst_min
                        - window,
                    )
                    new_tsrc = tl.maximum(pos_tdst_max - model_context_length, new_tsrc)
                elif EXTEND_BACKEND == "infllm":
                    new_tsrc = tl.ravel(
                        (idx_bk * BLOCK_SIZE_K)[:, None]
                        + tl.arange(0, BLOCK_SIZE_K)[None, :]
                    )
                    new_tsrc = tl.maximum(
                        0, new_tsrc * 0 + pos_tdst_min - sliding_window_size
                    )
                else:
                    raise Exception()
            else:
                new_tsrc = idx_tsrc

            keys = keys.to(queries.dtype)
            keys_rot = keys_rot.to(queries.dtype)

            cos_new = tl.load(
                COS
                + new_tsrc[None, :].to(tl.int64) * stride_cos_t
                + cos_sin_idx[:, None] * stride_cos_hid,
                mask=mask_tsrc[None, :] & rope_mask[:, None],
                other=0.0,
            ).to(keys.dtype)
            sin_new = tl.load(
                SIN
                + new_tsrc[None, :].to(tl.int64) * stride_sin_t
                + cos_sin_idx[:, None] * stride_sin_hid,
                mask=mask_tsrc[None, :] & rope_mask[:, None],
                other=0.0,
            ).to(keys.dtype)

            if EXCLUDE_SLIDING_WINDOW:
                if EXTEND_BACKEND == "dynamic_extend":
                    streaming_tsrc = tl.ravel(
                        (idx_bk * BLOCK_SIZE_K)[:, None]
                        + tl.arange(0, BLOCK_SIZE_K)[None, :]
                    )
                    streaming_tsrc = tl.maximum(
                        0,
                        streaming_tsrc
                        + pos_tdst_min
                        - sliding_window_size
                        - sink_token_size
                        - mask_k
                        + 1,
                    )

                    cos_zero = tl.load(
                        COS
                        + streaming_tsrc[None, :].to(tl.int64) * stride_cos_t
                        + cos_sin_idx[:, None] * stride_cos_hid,
                        mask=rope_mask[:, None],
                        # mask=mask_tsrc[None, :],
                        other=0.0,
                    ).to(keys.dtype)
                    sin_zero = tl.load(
                        SIN
                        + streaming_tsrc[None, :].to(tl.int64) * stride_sin_t
                        + cos_sin_idx[:, None] * stride_sin_hid,
                        mask=rope_mask[:, None],
                        # mask=mask_tsrc[None, :],
                        other=0.0,
                    ).to(keys.dtype)

                    cos_new = (cos_zero * 0.75 + cos_new * 0.25).to(cos_new.dtype)
                    sin_new = (sin_zero * 0.75 + sin_new * 0.25).to(sin_new.dtype)

            keys_rot *= rope_mult[:, None]

            keys_adjusted = tl.where(
                rope_mask[:, None],
                (keys * cos_new + keys_rot * sin_new).to(keys.dtype),
                keys,
            )

            queries_adjusted = queries

    else:
        raise Exception()

    return queries_adjusted, keys_adjusted


@triton.jit
def block_sparse_attention_cuda_step(
    # QKV
    queries_0,
    queries_1,
    keys_0,
    keys_1,
    keys_rot_0,
    keys_rot_1,
    values,
    # indices
    idx_tsrc,
    mask_tsrc,
    idx_tdst,
    mask_tdst,
    # rolling value
    acc,
    l_i,
    m_i,
    # TDST,
    # TSRC,
    sliding_window_size,
    sink_token_size,
    mask_k,
    EXCLUDE_SLIDING_WINDOW: tl.constexpr,
    HAS_FIRST_TOKEN: tl.constexpr,
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
    idx_bk,
    pos_tdst,
    idx_hid_q0,
    idx_hid_q1,
    IS_CAUSAL: tl.constexpr,
    HID: tl.constexpr,
    BLOCK_TQ,
    BLOCK_TK,
    BLOCK_SIZE_K: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr = DEFAULT_EXTEND_BACKEND,
):
    HID_BLOCK_0: tl.constexpr = queries_0.shape[1]
    HID_BLOCK_1: tl.constexpr = queries_1.shape[1] if queries_1 is not None else 0

    if USING_EXTEND:
        if rope_range_begin < HID_BLOCK_0:
            queries_0, keys_0 = apply_rope_to_keys(
                queries_0,
                keys_0,
                keys_rot_0,
                idx_tsrc,
                mask_tsrc,
                mask_tdst,
                pos_tdst,
                idx_bk,
                idx_hid_q0,
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
                sink_token_size,
                mask_k,
                sliding_window_size,
                HID,
                BLOCK_TQ,
                BLOCK_TK,
                BLOCK_SIZE_K,
                USING_EXTEND,
                HAS_FIRST_TOKEN,
                EXCLUDE_SLIDING_WINDOW,
                NEED_APPLY_ROPE,
                EXTEND_BACKEND,
            )

        if HID_BLOCK_1 > 0:
            queries_1, keys_1 = apply_rope_to_keys(
                queries_1,
                keys_1,
                keys_rot_1,
                idx_tsrc,
                mask_tsrc,
                mask_tdst,
                pos_tdst,
                idx_bk,
                idx_hid_q1,
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
                sink_token_size,
                mask_k,
                sliding_window_size,
                HID,
                BLOCK_TQ,
                BLOCK_TK,
                BLOCK_SIZE_K,
                USING_EXTEND,
                HAS_FIRST_TOKEN,
                EXCLUDE_SLIDING_WINDOW,
                NEED_APPLY_ROPE,
                EXTEND_BACKEND,
            )

    q_dtype = queries_0.dtype

    cq = tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0))
    ck = 1 / tl.sqrt(tl.sqrt(HID * 1.0))

    qk = tl.dot(
        (queries_0 * cq).to(q_dtype),
        (keys_0.to(q_dtype) * ck).to(q_dtype),
        out_dtype=tl.float32,
        allow_tf32=True,
    ).to(tl.float32)

    if HID_BLOCK_1 > 0:
        qk += tl.dot(
            (queries_1 * cq).to(q_dtype),
            (keys_1.to(q_dtype) * ck).to(q_dtype),
            out_dtype=tl.float32,
            allow_tf32=True,
        ).to(tl.float32)

    if LOGIT_SOFTCAP is not None:
        qk = tl.extra.cuda.libdevice.tanh(qk / LOGIT_SOFTCAP) * LOGIT_SOFTCAP
    qk = qk * 1.44269504

    if IS_CAUSAL:
        if EXCLUDE_SLIDING_WINDOW:
            qk_mask = (
                ((pos_tdst - 1)[:, None] < idx_tsrc[None, :])
                | ((pos_tdst - 1)[:, None] < (idx_tsrc + sliding_window_size)[None, :])
                | (~(mask_tdst[:, None] & mask_tsrc[None, :]))
            )
        else:
            qk_mask = (
                ((pos_tdst - 1)[:, None] < idx_tsrc[None, :])
                | ((pos_tdst - 1)[:, None] >= (idx_tsrc + sliding_window_size)[None, :])
                | (~(mask_tdst[:, None] & mask_tsrc[None, :]))
            )
    else:
        qk_mask = ~(mask_tdst[:, None] & mask_tsrc[None, :])

    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    qk = tl.where(qk_mask, tl.full(qk.shape, float("-inf"), qk.dtype), qk)
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1)[:, None])

    qk = qk - m_ij
    # [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]
    p = tl.math.exp2(qk)

    p = tl.where(qk_mask, 0, p)

    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    l_ij = tl.sum(p, axis=1)

    # -- update m_i and l_i
    l_valid = m_ij > -1e50
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = tl.where(
        l_valid,
        (l_i * alpha + l_ij[:, None]).to(l_i.dtype),
        l_i,
    )

    # -- update output accumulator --
    acc = tl.where(
        l_valid,
        acc * alpha.to(acc.dtype)
        + tl.dot(
            p.to(q_dtype),
            values.to(q_dtype),
            out_dtype=tl.float32,
            allow_tf32=True,
        ).to(acc.dtype),
        acc,
    )

    # update m_i and l_i
    m_i = tl.where(l_valid, m_ij.to(m_i.dtype), m_i)

    return acc, l_i, m_i


def get_block_sparse_attention_configs():
    autotune_disabled = os.getenv("HIP_DISABLE_AUTOTUNE", "1") == "1"
    if autotune_disabled:
        device_name = torch.cuda.get_device_name()
        defaults = {
            "NVIDIA A100-SXM4-80GB": dict(
                num_warps=4,
                num_stages=2,
                maxnreg=256,
            ),
        }.get(device_name, dict(num_warps=4, num_stages=2))
        return [triton.Config({}, **defaults)]
    if os.getenv("HIP_DISABLE_AUTOTUNE_WARNINGS", "0") == "0":
        warnings.warn(
            "triton autotuning is activated. this should be disabled for faster startup. if you want set HIP_DISABLE_AUTOTUNE=1"
        )

    NUM_WARPS = [4]  # workaround for triton bug
    if triton.__version__ >= "3.2.0":
        NUM_WARPS.append(8)

    configs = []
    # for block_bk in [4, 8, 16, 32]:
    # for block_bk in [16, 32,]:
    for max_nreg in [128, 256, 512]:
        for num_warps in NUM_WARPS:
            for num_stages in [1, 2, 4]:
                configs.append(
                    triton.Config(
                        {}, num_warps=num_warps, num_stages=num_stages, maxnreg=max_nreg
                    )
                )
    return configs


# def perf_model_block_sparse_attention(**kwargs):
#     block_bk = kwargs['BLOCK_BK']
#     block_k = kwargs['BLOCK_SIZE_K']
#     assert block_k <= 64, 'this will not good idea'
#     if ((block_bk * block_k) <= 64) and ((block_bk * block_k) >= 32):
#         return 0
#     return 999999999 # run might fails


@triton.jit
def apply_rope_to_queries(
    queries,
    pos_tdst,
    idx_hid,
    idx_bsz,
    idx_tdst,
    mask_tdst,
    idx_head,
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
):
    rope_tdst = pos_tdst - 1

    ROPE_DIM = rope_range_end - rope_range_begin

    idx_rope_range = idx_hid - rope_range_begin
    rope_mask = (rope_range_begin <= idx_hid) & (idx_hid < rope_range_end)
    if rope_is_neox_style:
        rope_rot_idx = tl.where(
            rope_mask,
            (idx_rope_range + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin,
            idx_hid,
        )
        cos_sin_idx = idx_rope_range % (ROPE_DIM // 2)
        rope_mult = ((idx_rope_range + ROPE_DIM // 2 < ROPE_DIM) * (-2) + 1).to(
            queries.dtype
        )
    else:
        flip = tl.where(idx_rope_range & 1 == 0, 1, -1)
        rope_rot_idx = tl.where(
            rope_mask,
            idx_rope_range + flip + rope_range_begin,
            idx_hid,
        )
        cos_sin_idx = idx_rope_range // 2
        rope_mult = ((idx_rope_range % 2 == 0) * (-2) + 1).to(queries.dtype)

    queries_rot = tl.load(
        Q
        + idx_bsz * stride_q_bsz
        + idx_tdst[:, None] * stride_q_tdst
        + idx_head * stride_q_head
        + rope_rot_idx[None, :] * stride_q_hid,
        mask=mask_tdst[:, None] & rope_mask[None, :],
        other=0.0,
    )
    if queries_rot.dtype == tl.float8e5:
        queries_rot = queries_rot.to(tl.float16)

    cos_new = tl.load(
        COS
        + rope_tdst[:, None].to(tl.int64) * stride_cos_t
        + cos_sin_idx[None, :] * stride_cos_hid,
        mask=mask_tdst[:, None] & rope_mask[None, :],
        other=0.0,
    ).to(queries.dtype)
    sin_new = tl.load(
        SIN
        + rope_tdst[:, None].to(tl.int64) * stride_sin_t
        + cos_sin_idx[None, :] * stride_sin_hid,
        mask=mask_tdst[:, None] & rope_mask[None, :],
        other=0.0,
    ).to(queries.dtype)

    queries_rot *= rope_mult[None, :]

    queries = tl.where(
        rope_mask[None, :],
        (queries * cos_new + queries_rot * sin_new).to(queries.dtype),
        queries,
    )

    return queries


@triton.autotune(
    configs=get_block_sparse_attention_configs(),
    key=[
        "BLOCK_SIZE_K",
        "BLOCK_SIZE_Q",
        "HID",
        "TDST_NEXT_POWER_OF_2",
    ],
    # prune_configs_by={
    #     'perf_model': perf_model_block_sparse_attention,
    #     'top_k': 24,
    # }
)
@triton.jit
def block_sparse_attention_cuda(
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
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    INDICES,
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    KS_START_END,
    stride_ks_start_end_b,
    stride_ks_start_end_bdst,
    stride_ks_start_end_g,
    CONTEXT,
    stride_context_bsz,
    stride_context_tdst,
    stride_context_head,
    stride_context_hid,
    HEAD: tl.constexpr,
    BK: tl.constexpr,
    MAX_TDST,
    MAX_TSRC,
    KV_HEAD_REPEAT: tl.constexpr,
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
    GPU_BANK_COUNT: int,
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
    HID_BLOCK_0: tl.constexpr,
    HID: tl.constexpr,
    HID_BLOCK_V: tl.constexpr,
    HID_V: tl.constexpr,
    # autotuning parameters
    BLOCK_BK: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    UPDATE_CACHE: tl.constexpr,
):
    G: tl.constexpr = 1

    pid_bsz = tl.program_id(2).to(tl.int64)
    pid_bdst = tl.program_id(1).to(tl.int64)
    pid_head = tl.program_id(0).to(tl.int64) % HEAD
    pid_v = tl.program_id(0).to(tl.int64) // HEAD
    dim_v_offset = pid_v * HID_BLOCK_V

    idx_bsz = pid_bsz.to(tl.int64)
    idx_head = pid_head
    idx_n = idx_bsz * HEAD + idx_head
    idx_b = idx_n
    idx_g = 0

    idx_bdst = pid_bdst
    if BLOCK_SIZE_Q < 16:
        idx_tdst = BLOCK_SIZE_Q * idx_bdst + tl.arange(0, 16)
        mask_tdst = (idx_tdst < MAX_TDST) & (tl.arange(0, 16) < BLOCK_SIZE_Q)
    else:
        idx_tdst = BLOCK_SIZE_Q * idx_bdst + tl.arange(0, BLOCK_SIZE_Q)
        mask_tdst = idx_tdst < MAX_TDST
    if IS_CAUSAL:
        pos_tdst = tl.load(
            POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
            mask=mask_tdst,
            other=0,
        )
    else:
        pos_tdst = tl.where(
            mask_tdst, tl.full((BLOCK_SIZE_Q,), value=MAX_TSRC, dtype=tl.int64), 0
        )

    ROPE_DIM = rope_range_end - rope_range_begin

    HID_BLOCK_1: tl.constexpr = HID - HID_BLOCK_0

    idx_hid_q0 = tl.arange(0, HID_BLOCK_0)
    rope_mask_0 = (rope_range_begin <= idx_hid_q0) & (idx_hid_q0 < rope_range_end)
    idx_rope_range_q0 = idx_hid_q0 - rope_range_begin
    if rope_is_neox_style:
        rope_rot_idx_0 = tl.where(
            rope_mask_0,
            (idx_rope_range_q0 + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin,
            idx_hid_q0,
        )
    else:
        flip = tl.where(idx_rope_range_q0 % 2 == 0, 1, -1)
        rope_rot_idx_0 = tl.where(
            rope_mask_0,
            idx_rope_range_q0 + flip + rope_range_begin,
            idx_hid_q0,
        )

    if HID_BLOCK_1 > 0:
        idx_hid_q1 = HID_BLOCK_0 + tl.arange(0, HID_BLOCK_1)
        rope_mask_1 = (rope_range_begin <= idx_hid_q1) & (idx_hid_q1 < rope_range_end)
        idx_rope_range_q1 = idx_hid_q1 - rope_range_begin
        if rope_is_neox_style:
            rope_rot_idx_1 = tl.where(
                rope_mask_1,
                (idx_hid_q1 - rope_range_begin + ROPE_DIM // 2) % ROPE_DIM
                + rope_range_begin,
                idx_hid_q1,
            )
        else:
            flip = tl.where(idx_rope_range_q1 % 2 == 0, 1, -1)
            rope_rot_idx_1 = tl.where(
                rope_mask_1,
                idx_rope_range_q1 + flip + rope_range_begin,
                idx_hid_q1,
            )
    else:
        idx_hid_q1 = None
        rope_rot_idx_1 = None

    idx_hid_v = dim_v_offset + tl.arange(0, HID_BLOCK_V)

    if BLOCK_SIZE_Q < 16:
        acc = tl.zeros((16, HID_BLOCK_V), dtype=tl.float32)
        m_i = tl.full((16, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.full((16, 1), 1.0, dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_Q, HID_BLOCK_V), dtype=tl.float32)
        m_i = tl.full((BLOCK_SIZE_Q, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_SIZE_Q, 1), 1.0, dtype=tl.float32)

    range_start = tl.load(
        KS_START_END
        + idx_b * stride_ks_start_end_b
        + idx_bdst * stride_ks_start_end_bdst
        + idx_g * stride_ks_start_end_g
    )
    range_end = tl.load(
        KS_START_END
        + idx_b * stride_ks_start_end_b
        + idx_bdst * stride_ks_start_end_bdst
        + (idx_g + 1) * stride_ks_start_end_g
    )
    if BK <= 0:
        range_start = 0
        range_end = 0

    queries_0 = tl.load(
        Q
        + idx_bsz * stride_q_bsz
        + idx_tdst[:, None] * stride_q_tdst
        + idx_head * stride_q_head
        + idx_hid_q0[None, :] * stride_q_hid,
        mask=mask_tdst[:, None] & (idx_hid_q0[None, :] < HID),
        other=0.0,
    )
    if queries_0.dtype == tl.float8e5:
        queries_0 = queries_0.to(tl.float16)

    if HID_BLOCK_1 > 0:
        queries_1 = tl.load(
            Q
            + idx_bsz * stride_q_bsz
            + idx_tdst[:, None] * stride_q_tdst
            + idx_head * stride_q_head
            + idx_hid_q1[None, :] * stride_q_hid,
            mask=mask_tdst[:, None] & (idx_hid_q1[None, :] < HID),
            other=0.0,
        )
        if queries_1.dtype == tl.float8e5:
            queries_1 = queries_1.to(tl.float16)
    else:
        queries_1 = None

    if USING_EXTEND and NEED_APPLY_ROPE:
        if rope_range_begin < HID_BLOCK_0:
            queries_0 = apply_rope_to_queries(
                queries_0,
                pos_tdst,
                idx_hid_q0,
                idx_bsz,
                idx_tdst,
                mask_tdst,
                idx_head,
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
            )

        if HID_BLOCK_1 > 0:
            queries_1 = apply_rope_to_queries(
                queries_1,
                pos_tdst,
                idx_hid_q1,
                idx_bsz,
                idx_tdst,
                mask_tdst,
                idx_head,
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
            )

    if (BK > 0) and True:
        for i_bk in range(range_start, range_start + (BK * G), BLOCK_BK):
            idx_bk = i_bk + tl.arange(0, BLOCK_BK)
            mask_bk = (idx_bk < (range_start + BK * G)) & (idx_bk < range_end)

            if i_bk < range_end:
                idx_tsrc_start = tl.load(
                    INDICES
                    + idx_b * stride_indices_b
                    + idx_bdst * stride_indices_bdst
                    + idx_bk * stride_indices_bk,
                    mask=mask_bk,
                )
                idx_tsrc_start = tl.where(mask_bk, idx_tsrc_start, MAX_TSRC * G + 1)
                idx_tsrc = idx_tsrc_start[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :]
                idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc_from_bk = mask_bk[:, None] & tl.full(
                    (1, BLOCK_SIZE_K), 1, dtype=tl.int1
                )
                mask_tsrc_from_bk = tl.reshape(
                    mask_tsrc_from_bk, (BLOCK_BK * BLOCK_SIZE_K)
                )
                mask_tsrc = (
                    (idx_tsrc < (MAX_TSRC * (idx_g + 1)))
                    & (idx_tsrc >= (MAX_TSRC * idx_g))
                    & mask_tsrc_from_bk
                )
                idx_tsrc = idx_tsrc % MAX_TSRC
                mask_tsrc = (
                    mask_tsrc
                    & (idx_tsrc < tl.max(pos_tdst))
                    & (idx_tsrc >= sink_token_size)
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
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid_q0[:, None],
                    mask_tsrc[None, :],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
                    HID_BLOCK_0,
                    HID,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )

                if HID_BLOCK_1 > 0:
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
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        idx_hid_q1[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
                        HID_BLOCK_1,
                        HID,
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
                    if rope_range_begin < HID_BLOCK_0:
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
                            idx_bsz,
                            idx_tsrc[None, :],
                            idx_head // KV_HEAD_REPEAT,
                            rope_rot_idx_0[:, None],
                            mask_tsrc[None, :],
                            HEAD // KV_HEAD_REPEAT,
                            BLOCK_BK * BLOCK_SIZE_K,
                            HID_BLOCK_0,
                            HID,
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

                    if HID_BLOCK_1 > 0:
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
                            idx_bsz,
                            idx_tsrc[None, :],
                            idx_head // KV_HEAD_REPEAT,
                            rope_rot_idx_1[:, None],
                            mask_tsrc[None, :],
                            HEAD // KV_HEAD_REPEAT,
                            BLOCK_BK * BLOCK_SIZE_K,
                            HID_BLOCK_1,
                            HID,
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
                    idx_bsz,
                    idx_tsrc[:, None],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid_v[None, :],
                    mask_tsrc[:, None],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
                    HID_BLOCK_V,
                    HID_V,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=K_CACHE,
                    stride_v_cache_page=stride_k_cache_page,
                    stride_v_cache_offset=stride_k_cache_offset,
                    stride_v_cache_kv_head=stride_k_cache_kv_head,
                    stride_v_cache_hid=stride_k_cache_hid,
                )

                acc, l_i, m_i = block_sparse_attention_cuda_step(
                    queries_0,
                    queries_1,
                    keys_0,
                    keys_1,
                    keys_rot_0,
                    keys_rot_1,
                    values,
                    idx_tsrc,
                    mask_tsrc,
                    idx_tdst,
                    mask_tdst,
                    acc,
                    l_i,
                    m_i,
                    sliding_window_size,
                    sink_token_size,
                    (range_end - range_start) * BLOCK_SIZE_K,
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
                    pos_tdst,
                    idx_hid_q0,
                    idx_hid_q1,
                    IS_CAUSAL,
                    HID,
                    BLOCK_SIZE_Q,
                    BLOCK_BK * BLOCK_SIZE_K,
                    BLOCK_SIZE_K,
                    EXTEND_BACKEND=EXTEND_BACKEND,
                )
            else:
                pass

    if (sink_token_size > 0) and True:
        CURR_TSRC = tl.max(pos_tdst)
        for i_tsrc in range(0, sink_token_size, BLOCK_BK * BLOCK_SIZE_K):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < tl.minimum(CURR_TSRC, sink_token_size)

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
                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid_q0[:, None],
                mask_tsrc[None, :],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID_BLOCK_0,
                HID,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=V_CACHE,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_offset=stride_v_cache_offset,
                stride_v_cache_kv_head=stride_v_cache_kv_head,
                stride_v_cache_hid=stride_v_cache_hid,
            )

            if HID_BLOCK_1 > 0:
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
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid_q1[:, None],
                    mask_tsrc[None, :],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
                    HID_BLOCK_1,
                    HID,
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
                if rope_range_begin < HID_BLOCK_0:
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
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        rope_rot_idx_0[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
                        HID_BLOCK_0,
                        HID,
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

                if HID_BLOCK_1 > 0:
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
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        rope_rot_idx_1[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
                        HID_BLOCK_1,
                        HID,
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
                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid_v[None, :],
                mask_tsrc[:, None],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID_BLOCK_V,
                HID_V,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=K_CACHE,
                stride_v_cache_page=stride_k_cache_page,
                stride_v_cache_offset=stride_k_cache_offset,
                stride_v_cache_kv_head=stride_k_cache_kv_head,
                stride_v_cache_hid=stride_k_cache_hid,
            )

            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries_0,
                queries_1,
                keys_0,
                keys_1,
                keys_rot_0,
                keys_rot_1,
                values,
                idx_tsrc,
                mask_tsrc,
                idx_tdst,
                mask_tdst,
                acc,
                l_i,
                m_i,
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
                pos_tdst,
                idx_hid_q0,
                idx_hid_q1,
                IS_CAUSAL,
                HID,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,
                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    if sliding_window_size > 0:
        CURR_TSRC = tl.max(pos_tdst)
        # CURR_TSRC = (idx_bdst + 1) * BLOCK_SIZE_Q + MAX_TSRC - MAX_TDST
        i_tsrc_range_start = tl.maximum(
            0, CURR_TSRC - sliding_window_size - BLOCK_SIZE_Q
        )
        TSRC_RANGE_STEP: tl.constexpr = BLOCK_BK * BLOCK_SIZE_K
        for i_tsrc in range(i_tsrc_range_start, CURR_TSRC, TSRC_RANGE_STEP):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < CURR_TSRC

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
                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid_q0[:, None],
                mask_tsrc[None, :],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID_BLOCK_0,
                HID,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=V_CACHE,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_offset=stride_v_cache_offset,
                stride_v_cache_kv_head=stride_v_cache_kv_head,
                stride_v_cache_hid=stride_v_cache_hid,
            )

            if HID_BLOCK_1 > 0:
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
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid_q1[:, None],
                    mask_tsrc[None, :],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
                    HID_BLOCK_1,
                    HID,
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
                if rope_range_begin < HID_BLOCK_0:
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
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        rope_rot_idx_0[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
                        HID_BLOCK_0,
                        HID,
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

                if HID_BLOCK_1 > 0:
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
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        rope_rot_idx_1[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
                        HID_BLOCK_1,
                        HID,
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
                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid_v[None, :],
                mask_tsrc[:, None],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID_BLOCK_V,
                HID_V,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=K_CACHE,
                stride_v_cache_page=stride_k_cache_page,
                stride_v_cache_offset=stride_k_cache_offset,
                stride_v_cache_kv_head=stride_k_cache_kv_head,
                stride_v_cache_hid=stride_k_cache_hid,
            )

            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries_0,
                queries_1,
                keys_0,
                keys_1,
                keys_rot_0,
                keys_rot_1,
                values,
                idx_tsrc,
                mask_tsrc,
                idx_tdst,
                mask_tdst,
                acc,
                l_i,
                m_i,
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
                # tl.arange(0, BLOCK_BK) +\
                #     (range_end - range_start) +\
                #     (sink_token_size // BLOCK_SIZE_K) +\
                #     (i_tsrc-i_tsrc_range_start) // BLOCK_SIZE_K,
                tl.arange(0, BLOCK_BK)
                + (i_tsrc - i_tsrc_range_start) // BLOCK_SIZE_K
                + (
                    tl.max(pos_tdst * mask_tdst)
                    - tl.sum(mask_tdst.to(tl.int32))
                    - sliding_window_size
                )
                // BLOCK_SIZE_K,
                pos_tdst,
                idx_hid_q0,
                idx_hid_q1,
                IS_CAUSAL,
                HID,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,
                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / (tl.where(l_i == 0.0, 1e-20, l_i))

    tl.store(
        CONTEXT
        + idx_bsz * stride_context_bsz
        + idx_tdst[:, None] * stride_context_tdst
        + idx_head * stride_context_head
        + idx_hid_v[None, :] * stride_context_hid,
        mask=mask_tdst[:, None] & (idx_hid_v < HID_V),
        value=acc.to(CONTEXT.type.element_ty),
        # eviction_policy='evict_first',
        # cache_modifier='.cs', # TODO: uncomment this
        # value = l_i
    )


def block_sparse_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    seq_lens: Tensor,
    indices: Tensor,
    ks: Tensor,
    ks_count: Tensor,
    ks_start_end: Tensor,
    args: "HiPAttentionArgs",
    access_counter: Tensor,
    cache_miss_counter: Tensor,
    EXTEND_BACKEND: str = DEFAULT_EXTEND_BACKEND,
    model_context_length: int = 131072,
    extend_context_length: int = 131072,
    offload_update_cache: bool = False,
):
    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        _, TSRC, KV_HEAD, _ = k.shape
        BSRC = cdiv_python(TSRC, args.block_size_k)
        MAX_TSRC = TSRC
        MAX_BSRC = BSRC
        HID_V = v.shape[-1]
    else:
        if args.k_cache is not None:
            NUM_PAGE, PAGE_SIZE, KV_HEAD, _ = args.k_cache.shape
            HID_V = args.v_cache.shape[-1]
        else:
            KV_HEAD = args.offload_cache.k_uvm.bank_cpu.shape[-2]
            HID_V = args.offload_cache.v_uvm.bank_cpu.shape[-1]
        TSRC = None
        BSRC = None
        # MAX_TSRC = NUM_PAGE * PAGE_SIZE
        MAX_TSRC = extend_context_length
        MAX_BSRC = cdiv_python(MAX_TSRC, args.block_size_k)
    HID_V = args.v_hidden_dim if args.v_hidden_dim is not None else HID_V
    N = BSZ * HEAD
    # assert q.shape == k.shape
    BDST = cdiv_python(TDST, args.block_size_q)
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD == HEAD

    B = N
    assert B == N
    BK = indices.shape[-1]  # cdiv_python(args.mask_k, args.block_size_k)

    context = torch.empty((BSZ, TDST, HEAD, HID_V), dtype=q.dtype, device=q.device)

    # BLOCK_BK = 64 // block_size_k
    # if block_size_k > 4:
    #     BLOCK_BK = 128 // block_size_k
    # elif block_size_k > 8:
    #     BLOCK_BK = 256 // block_size_k
    # BLOCK_BK = 64 // args.block_size_k

    max_block_size = int(os.getenv("SA_BLOCK_SIZE", "32"))
    BLOCK_BK = max_block_size // args.block_size_k
    BLOCK_BK = max(1, min(max_block_size, BLOCK_BK))
    if "SA_BLOCK_BK" in os.environ:
        BLOCK_BK = int(os.environ["SA_BLOCK_BK"])

    assert BLOCK_BK > 0, BLOCK_BK

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
        if args.k_cache is not None:
            assert args.k_cache.ndim == 4
            assert args.v_cache.ndim == 4
        else:
            assert args.offload_cache.k_uvm.bank_cpu.ndim == 3
            assert args.offload_cache.v_uvm.bank_cpu.ndim == 3
    else:
        raise Exception()
    assert seq_lens.ndim == 2

    if args.rope_range[0] == 0 and args.rope_range[1] == HID:
        HID_BLOCK = triton.next_power_of_2(HID)
    else:
        assert triton.next_power_of_2(args.rope_range[0]) == args.rope_range[0]
        assert args.rope_range[1] == HID
        HID_BLOCK = args.rope_range[0]

    HID_BLOCK_V = triton.next_power_of_2(min(HID_V, 256))
    NUM_HID_V_BLOCKS = triton.cdiv(HID_V, HID_BLOCK_V)

    grid = (HEAD * NUM_HID_V_BLOCKS, BDST, BSZ)
    pre_device = torch.get_default_device()
    torch.set_default_device(q.device)

    # print(indices.shape, indices[0, -1], ks_start_end[0, -1])
    # if indices.shape[1] == 1:
    #     input()

    if os.getenv("HIP_VERBOSE", "0") == "1":
        print(
            f"{HEAD=}",
            f"{BK=}",
            f"{KV_HEAD_REPEAT=}",
            f"{args.sliding_window_size=}",
            f"{args.sink_token_size=}",
            f"{args.logit_softcap=}",
            f"{args.using_extend=}",
            f"{args.need_apply_rope=}",
            f"{args.rope_range[0]=}",
            f"{args.rope_range[1]=}",
            f"{args.using_paged_cache=}",
            f"{args.k_cache.shape[1]=}",
            f"{args.is_causal=}",
            f"{args.block_size_q=}",
            f"{args.block_size_k=}",
            f"{HID_BLOCK=}",
            f"{HID=}",
            f"{HID_BLOCK_V=}",
            f"{HID_V=}",
            f"{BLOCK_BK=}",
            f"{EXTEND_BACKEND=}",
            f"{offload_update_cache=}",
            sep=", ",
        )
    block_sparse_attention_cuda[grid](
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
        context,
        *safe_stride(context, 4),
        HEAD,
        BK,
        TDST,
        MAX_TSRC,
        KV_HEAD_REPEAT,
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
        triton.next_power_of_2(TDST),
        args.is_causal,
        args.block_size_q,
        args.block_size_k,
        HID_BLOCK,
        HID,
        HID_BLOCK_V,
        HID_V,
        # 2,
        BLOCK_BK=BLOCK_BK,
        EXTEND_BACKEND=EXTEND_BACKEND,
        UPDATE_CACHE=offload_update_cache,
        # num_warps=4,
        # num_stages=2 if not using_extend else 1,
    )
    torch.set_default_device(pre_device)

    if (
        (os.getenv("HIP_CUMSUM", "0") == "1")
        and isinstance(v, Tensor)
        and q.shape[1] > 1
    ):
        v_cumsum = (
            v.cumsum(dim=1)
            / torch.arange(1, v.shape[1] + 1, device=v.device)[None, :, None, None]
        )
        a = torch.arange(1, v.shape[1] + 1, device=v.device)[None, :, None]
        b = (
            ks.repeat_interleave(args.block_size_q, 1)[:, : v.shape[1]]
            .view(BSZ, HEAD, -1)
            .permute(0, 2, 1)
            * args.block_size_k
        )
        scaler = ((a - b) / a).clamp_min(0)[:, :, :, None].pow(2) * 0.05
        context = (
            context * (1 - scaler)
            + v_cumsum.repeat_interleave(HEAD // KV_HEAD, dim=2) * scaler
        )

    return context
