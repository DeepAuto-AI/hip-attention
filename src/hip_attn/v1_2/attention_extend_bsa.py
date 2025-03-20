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
def block_sparse_attention_cuda_step(
    # QKV
    queries,
    keys,
    keys_rot,
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
    model_context_length,
    idx_bk,
    pos_tdst,
    idx_hid,
    IS_CAUSAL: tl.constexpr,
    HID: tl.constexpr,
    BLOCK_TQ,
    BLOCK_TK,
    BLOCK_SIZE_K: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr = DEFAULT_EXTEND_BACKEND,
):
    if USING_EXTEND:
        if EXTEND_BACKEND == "self_extend":
            raise Exception()
            # assert COS is not None
            # assert SIN is not None

            # # dynamic_group_size = tl.maximum(1.0, tl.math.ceil(tl.max(pos_tdst / 8192))).to(tl.int32)
            # dynamic_group_size = extend_group_size

            # old_tsrc = idx_tsrc
            # mask_tsrc_window = idx_tsrc >= (tl.min(tl.where(mask_tdst, (pos_tdst - 1), 987654321)) - extend_window_size)
            # new_tsrc = tl.where(
            #     mask_tsrc_window,
            #     old_tsrc,
            #     old_tsrc // dynamic_group_size
            # )

            # keys = keys.trans(1, 0)
            # keys = adjust_rope(
            #     keys,
            #     old_tsrc,
            #     new_tsrc,
            #     mask_tsrc,
            #     idx_hid,
            #     COS, stride_cos_t, stride_cos_hid,
            #     SIN, stride_sin_t, stride_sin_hid,
            #     BLOCK_TK,
            #     HID,
            #     NEED_APPLY_ROPE,
            # )
            # keys = tl.trans(keys, 1, 0)
            # keys = keys * mask_tsrc[None, :]

            # old_tdst = (pos_tdst - 1)
            # new_tdst = old_tdst // dynamic_group_size

            # queries_grouped = adjust_rope(
            #     queries,
            #     old_tdst,
            #     new_tdst,
            #     mask_tdst,
            #     idx_hid,
            #     COS, stride_cos_t, stride_cos_hid,
            #     SIN, stride_sin_t, stride_sin_hid,
            #     BLOCK_TQ,
            #     HID,
            #     NEED_APPLY_ROPE,
            # )
            # queries_grouped = queries_grouped * mask_tdst[:, None]

            # t_window = tl.dot(
            #     (queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(queries.dtype),
            #     (keys.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(queries.dtype),
            #     allow_tf32=True,
            # )
            # t_grouped = tl.dot(
            #     (queries_grouped * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(queries_grouped.dtype),
            #     (keys.to(queries_grouped.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(queries_grouped.dtype),
            #     allow_tf32=True,
            # )
            # qk = tl.where(
            #     mask_tsrc_window[None, :],
            #     t_window,
            #     t_grouped,
            # ).to(tl.float32) * 1.44269504
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
                            COS,
                            stride_cos_t,
                            stride_cos_hid,
                            SIN,
                            stride_sin_t,
                            stride_sin_hid,
                            BLOCK_TQ,
                            HID,
                            True,
                        ).to(queries.dtype)
                        queries_adjusted = (queries * mask_tdst[:, None]).to(
                            queries.dtype
                        )

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
                                True,
                            ),
                            1,
                            0,
                        ).to(keys.dtype)
                        keys_adjusted = (keys * mask_tsrc[None, :]).to(keys.dtype)
            else:
                tl.static_assert(NEED_APPLY_ROPE)
                tl.static_assert(USING_EXTEND)
                # tl.static_assert(not EXCLUDE_SLIDING_WINDOW)

                if EXCLUDE_SLIDING_WINDOW:
                    # new_tsrc = tl.ravel((idx_bk * BLOCK_SIZE_K)[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :])
                    # # new_tsrc = tl.minimum(new_tsrc, sliding_window_size + sink_token_size + mask_k - 1)
                    # # new_tsrc = tl.minimum(new_tsrc, pos_tdst_min + tl.sum(mask_tdst.to(tl.int32)) - 1)
                    # new_tsrc = tl.maximum(0, new_tsrc)

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
                        new_tsrc = tl.maximum(
                            pos_tdst_max - model_context_length, new_tsrc
                        )
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
                    + (tl.arange(0, HID) % (HID // 2))[:, None] * stride_cos_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                ).to(keys.dtype)
                sin_new = tl.load(
                    SIN
                    + new_tsrc[None, :].to(tl.int64) * stride_sin_t
                    + (tl.arange(0, HID) % (HID // 2))[:, None] * stride_sin_hid,
                    mask=mask_tsrc[None, :],
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
                            + (tl.arange(0, HID) % (HID // 2))[:, None]
                            * stride_cos_hid,
                            # mask=mask_tsrc[None, :],
                            # other=0.0,
                        ).to(keys.dtype)
                        sin_zero = tl.load(
                            SIN
                            + streaming_tsrc[None, :].to(tl.int64) * stride_sin_t
                            + (tl.arange(0, HID) % (HID // 2))[:, None]
                            * stride_sin_hid,
                            # mask=mask_tsrc[None, :],
                            # other=0.0,
                        ).to(keys.dtype)

                        cos_new = (cos_zero * 0.75 + cos_new * 0.25).to(cos_new.dtype)
                        sin_new = (sin_zero * 0.75 + sin_new * 0.25).to(sin_new.dtype)

                # rope_theta = 500000.0
                # inv_freqs = ((tl.arange(0, HID) * 2) % HID)
                # 1.0 / tl.extra.cuda.libdevice.fast_powf(
                #     rope_theta,
                #     ((tl.arange(0, HID) * 2) % HID).to(tl.float32) / HID
                # )
                # freqs = new_tsrc[None, :].to(tl.float32) * inv_freqs[:, None]
                # cos_new = tl.extra.cuda.libdevice.fast_cosf(freqs)
                # sin_new = tl.extra.cuda.libdevice.fast_sinf(freqs)

                # keys_rot = tl.where(
                #     (idx_hid + HID // 2)[:, None] < HID,
                #     -keys_rot,
                #     keys_rot
                # )
                keys_rot = keys_rot * (
                    ((idx_hid + HID // 2)[:, None] < HID) * (-2) + 1
                ).to(keys_rot.dtype)

                keys_adjusted = (keys * cos_new + keys_rot * sin_new).to(keys.dtype)

                queries_adjusted = queries

            qk = tl.dot(
                queries_adjusted
                * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0))).to(queries.dtype),
                keys_adjusted * (1 / tl.sqrt(tl.sqrt(HID * 1.0))).to(queries.dtype),
                out_dtype=tl.float32,
                allow_tf32=True,
            ).to(tl.float32)
            if LOGIT_SOFTCAP is not None:
                qk = tl.extra.cuda.libdevice.tanh(qk / LOGIT_SOFTCAP) * LOGIT_SOFTCAP
            qk = qk * 1.44269504
        elif EXTEND_BACKEND == "dynamic_extend":
            assert COS is not None
            assert SIN is not None

            pos_tdst_min = tl.min(
                tl.where(mask_tdst, tl.maximum(0, pos_tdst - 1), 987654321)
            ) + tl.sum(mask_tdst.to(tl.int32))
            if (pos_tdst_min >= model_context_length) and EXCLUDE_SLIDING_WINDOW:
                old_tdst = pos_tdst - 1
                new_tdst = tl.minimum(model_context_length - 1, old_tdst)
                # new_tdst = old_tdst // 16

                queries = adjust_rope(
                    queries,
                    old_tdst,
                    new_tdst,
                    mask_tdst & (old_tdst != 0),
                    idx_hid,
                    COS,
                    stride_cos_t,
                    stride_cos_hid,
                    SIN,
                    stride_sin_t,
                    stride_sin_hid,
                    BLOCK_TQ,
                    HID,
                    NEED_APPLY_ROPE,
                ).to(queries.dtype)
                queries = (queries * mask_tdst[:, None]).to(queries.dtype)

                if not HAS_FIRST_TOKEN:
                    old_tsrc = idx_tsrc

                    # src_scale = (
                    #     (pos_tdst_min + 1 - sink_token_size - sliding_window_size) /\
                    #     (model_context_length - sink_token_size - sliding_window_size)
                    # )
                    # new_tsrc = (
                    #     (old_tsrc - sink_token_size) / src_scale + sink_token_size
                    # ).to(tl.int32)

                    # new_tsrc = tl.where(
                    #     (old_tsrc - pos_tdst_min + model_context_length - 1) > (model_context_length // 2),
                    #     old_tsrc - pos_tdst_min + model_context_length - 1,
                    #     ((old_tsrc - sink_token_size) * ((model_context_length // 2 - 1 - sink_token_size - sliding_window_size) / (pos_tdst_min - sink_token_size - sliding_window_size))).to(tl.int32) + sink_token_size
                    # )

                    new_tsrc = tl.where(
                        (old_tsrc - pos_tdst_min + model_context_length - 1)
                        > (model_context_length // 2),
                        old_tsrc - pos_tdst_min + model_context_length - 1,
                        (
                            (old_tsrc - sink_token_size)
                            * (
                                (model_context_length // 2)
                                / (pos_tdst_min - model_context_length // 2)
                            )
                        ).to(tl.int32)
                        + sink_token_size,
                        # ((old_tsrc - num_sinks) * (model_context_length / real_pos_tdst_min)).to(tl.int32) + num_sinks
                    )

                    # new_tsrc = old_tsrc // 16

                    keys_adjusted = keys.trans(1, 0)
                    keys_adjusted = adjust_rope(
                        keys_adjusted,
                        old_tsrc,
                        new_tsrc,
                        mask_tsrc & (old_tsrc != 0),
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_TK,
                        HID,
                        NEED_APPLY_ROPE,
                    ).to(keys.dtype)
                    keys_adjusted = tl.trans(keys_adjusted, 1, 0).to(keys.dtype)
                    # keys_adjusted = (keys_adjusted * mask_tsrc[None, :]).to(keys.dtype)
                else:
                    keys_adjusted = keys
            else:
                keys_adjusted = keys

            qk = tl.dot(
                (queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                    queries.dtype
                ),
                (
                    keys_adjusted.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))
                ).to(queries.dtype),
                out_dtype=tl.float32,
                allow_tf32=True,
            ).to(tl.float32)

            if LOGIT_SOFTCAP is not None:
                qk = tl.extra.cuda.libdevice.tanh(qk / LOGIT_SOFTCAP) * LOGIT_SOFTCAP
            qk = qk * 1.44269504
        else:
            raise Exception()
    else:
        qk = tl.dot(
            (queries * (tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                queries.dtype
            ),
            (keys.to(queries.dtype) * (1 / tl.sqrt(tl.sqrt(HID * 1.0)))).to(
                queries.dtype
            ),
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
    qk_ = tl.where(qk_mask, tl.full(qk.shape, float("-inf"), qk.dtype), qk)
    m_ij = tl.maximum(m_i, tl.max(qk_, axis=1)[:, None])

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
            p.to(queries.dtype),
            values.to(queries.dtype),
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
    configs = []
    # for block_bk in [4, 8, 16, 32]:
    # for block_bk in [16, 32,]:
    for max_nreg in [128, 256, 512]:
        for num_warps in [4, 8]:
            for num_stages in [2, 4]:
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
    HID: tl.constexpr,
    # autotuning parameters
    BLOCK_BK: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    UPDATE_CACHE: tl.constexpr,
):
    G: tl.constexpr = 1

    pid_bsz = tl.program_id(2).to(tl.int64)
    pid_bdst = tl.program_id(1).to(tl.int64)
    pid_head = tl.program_id(0).to(tl.int64)

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

    idx_hid = tl.arange(0, HID)

    if BLOCK_SIZE_Q < 16:
        acc = tl.zeros((16, HID), dtype=tl.float32)
        m_i = tl.full((16, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.full((16, 1), 1.0, dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_Q, HID), dtype=tl.float32)
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

    queries = tl.load(
        Q
        + idx_bsz * stride_q_bsz
        + idx_tdst[:, None] * stride_q_tdst
        + idx_head * stride_q_head
        + idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0.0,
        # cache_modifier='.cg',
        # eviction_policy='evict_last',
        # volatile=True,
    )
    if queries.dtype == tl.float8e5:
        queries = queries.to(tl.float16)

    if USING_EXTEND and NEED_APPLY_ROPE:
        rope_tdst = pos_tdst - 1
        # rope_tdst = tl.minimum(
        #     rope_tdst,
        #     sliding_window_size + sink_token_size + (range_end - range_start) * BLOCK_SIZE_K - 1
        # )

        # queries = adjust_rope(
        #     queries,
        #     rope_tdst,
        #     rope_tdst,
        #     mask_tdst,
        #     idx_hid,
        #     COS, stride_cos_t, stride_cos_hid,
        #     SIN, stride_sin_t, stride_sin_hid,
        #     BLOCK_SIZE_Q,
        #     HID,
        #     True,
        # ).to(queries.dtype)

        queries_rot = tl.load(
            Q
            + idx_bsz * stride_q_bsz
            + idx_tdst[:, None] * stride_q_tdst
            + idx_head * stride_q_head
            + ((idx_hid + HID // 2) % HID)[None, :] * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0.0,
            # cache_modifier='.cg',
            # eviction_policy='evict_last',
            # volatile=True,
        )
        if queries_rot.dtype == tl.float8e5:
            queries_rot = queries_rot.to(tl.float16)

        cos_new = tl.load(
            COS
            + rope_tdst[:, None].to(tl.int64) * stride_cos_t
            + (idx_hid % (HID // 2))[None, :] * stride_cos_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)
        sin_new = tl.load(
            SIN
            + rope_tdst[:, None].to(tl.int64) * stride_sin_t
            + (idx_hid % (HID // 2))[None, :] * stride_sin_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)

        # queries_rot = tl.where(
        #     (idx_hid + HID // 2)[None, :] < HID,
        #     -queries_rot,
        #     queries_rot
        # )
        queries_rot = queries_rot * (
            ((idx_hid + HID // 2)[None, :] < HID) * (-2) + 1
        ).to(queries_rot.dtype)

        queries = (queries * cos_new + queries_rot * sin_new).to(queries.dtype)

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
                    # other=(MAX_TSRC + 1) * G,
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
                # mask_tsrc = True
                # mask_tsrc = idx_tsrc > 0
                # idx_group = idx_tsrc // MAX_TSRC

                # min_tsrc = tl.min(idx_tsrc)

                # if min_tsrc <= tl.max(idx_tdst):
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
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid[:, None],
                    mask_tsrc[None, :],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
                    HID,
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
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        ((idx_hid + HID // 2) % HID)[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
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
                    idx_bsz,
                    idx_tsrc[:, None],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid[None, :],
                    mask_tsrc[:, None],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
                    HID,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=K_CACHE,
                    stride_v_cache_page=stride_k_cache_page,
                    stride_v_cache_offset=stride_k_cache_offset,
                    stride_v_cache_kv_head=stride_k_cache_kv_head,
                    stride_v_cache_hid=stride_k_cache_hid,
                )

                acc, l_i, m_i = block_sparse_attention_cuda_step(
                    queries,
                    keys,
                    keys_rot,
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
                    model_context_length,
                    idx_bk + sink_token_size // BLOCK_SIZE_K,
                    pos_tdst,
                    idx_hid,
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
                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[:, None],
                mask_tsrc[None, :],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID,
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
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    ((idx_hid + HID // 2) % HID)[:, None],
                    mask_tsrc[None, :],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
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
                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[None, :],
                mask_tsrc[:, None],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=K_CACHE,
                stride_v_cache_page=stride_k_cache_page,
                stride_v_cache_offset=stride_k_cache_offset,
                stride_v_cache_kv_head=stride_k_cache_kv_head,
                stride_v_cache_hid=stride_k_cache_hid,
            )

            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries,
                keys,
                keys_rot,
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
                model_context_length,
                tl.arange(0, BLOCK_BK) + i_tsrc // BLOCK_SIZE_K,
                pos_tdst,
                idx_hid,
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
                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[:, None],
                mask_tsrc[None, :],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID,
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
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    ((idx_hid + HID // 2) % HID)[:, None],
                    mask_tsrc[None, :],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
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
                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid[None, :],
                mask_tsrc[:, None],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=K_CACHE,
                stride_v_cache_page=stride_k_cache_page,
                stride_v_cache_offset=stride_k_cache_offset,
                stride_v_cache_kv_head=stride_k_cache_kv_head,
                stride_v_cache_hid=stride_k_cache_hid,
            )

            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries,
                keys,
                keys_rot,
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
                idx_hid,
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
        + idx_hid[None, :] * stride_context_hid,
        mask=mask_tdst[:, None],
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
    else:
        if args.k_cache is not None:
            NUM_PAGE, PAGE_SIZE, KV_HEAD, _ = args.k_cache.shape
        else:
            KV_HEAD = args.offload_cache.k_uvm.bank_cpu.shape[-2]
        TSRC = None
        BSRC = None
        # MAX_TSRC = NUM_PAGE * PAGE_SIZE
        MAX_TSRC = extend_context_length
        MAX_BSRC = cdiv_python(MAX_TSRC, args.block_size_k)
    N = BSZ * HEAD
    # assert q.shape == k.shape
    BDST = cdiv_python(TDST, args.block_size_q)
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD == HEAD

    B = N
    assert B == N
    BK = indices.shape[-1]  # cdiv_python(args.mask_k, args.block_size_k)

    context = torch.empty(q.shape, dtype=q.dtype, device=q.device)

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

    grid = (HEAD, BDST, BSZ)
    pre_device = torch.get_default_device()
    torch.set_default_device(q.device)

    # print(indices.shape, indices[0, -1], ks_start_end[0, -1])
    # if indices.shape[1] == 1:
    #     input()

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
        HID,
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
