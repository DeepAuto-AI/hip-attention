import math
import os
import time
import warnings
from typing import Optional

import cv2
import numba
import numba.cuda
import numpy as np
import torch
import triton
import triton.language as tl
from matplotlib import pyplot as plt
from torch import Tensor

from hip_attn.utils.rope import adjust_rope
from hip_attn.v1_2.attention_decode_bsa import decode_block_sparse_attention
from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention
from hip_attn.v1_2.attention_metadata import (
    EnsembleScoreStage,
    EvalScoreStage,
    HiPAttentionArgs,
    HiPAttentionCacheAccessStatistics,
    HiPAttentionOutputMetadata,
    HiPAttentionStageInputCache,
    NopStage,
    ScanStage,
    safe_stride,
)
from hip_attn.v1_2.uvm_gpu_cache import load_tokens

_NUM_STREAMING_MULTIPROCESSOR = None


def num_streaming_multiprocessor():
    global _NUM_STREAMING_MULTIPROCESSOR
    if _NUM_STREAMING_MULTIPROCESSOR is None:
        _NUM_STREAMING_MULTIPROCESSOR = (
            numba.cuda.get_current_device().MULTIPROCESSOR_COUNT
        )
    return _NUM_STREAMING_MULTIPROCESSOR


@numba.njit(parallel=True)
def render_plot(out_indices_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q):
    for i in numba.prange(out_indices_cpu.shape[1]):
        for j in range(out_indices_cpu.shape[-1]):
            # if j >= out_indices_cpu.shape[-1]: continue
            t = out_indices_cpu[0, i, DEBUG_HEAD, j] // BLOCK_SIZE_Q
            debug[i, t : t + 1] = 1


@numba.njit(parallel=True)
def render_plot_dynamic(
    out_indices_cpu,
    debug,
    DEBUG_HEAD,
    BLOCK_SIZE_Q,
    stage_k,
    chunk_size,
    causal_mask=False,
    sliding_window_size=0,
):
    for i in numba.prange(out_indices_cpu.shape[1]):
        for j in range(math.ceil(stage_k / chunk_size)):
            if j >= out_indices_cpu.shape[-1]:
                continue
            t = out_indices_cpu[0, i, DEBUG_HEAD, j] // BLOCK_SIZE_Q
            if causal_mask and ((t + sliding_window_size // BLOCK_SIZE_Q) >= i):
                continue
            tt = t + math.ceil(chunk_size / BLOCK_SIZE_Q)
            if causal_mask:
                tt = min(tt, i + 1)
            debug[i, t:tt] = 1


@numba.njit(parallel=True)
def render_plot_sampled(
    out_indices_cpu,
    debug,
    DEBUG_HEAD,
    BLOCK_CHUNK,
    chunk_count,
    TDST,
    sink_token_size,
):
    for i in numba.prange(out_indices_cpu.shape[1]):
        t_chunk_size = math.ceil(TDST / chunk_count * BLOCK_CHUNK)
        # print(i, t_chunk_size)
        for j in range(max(0, out_indices_cpu.shape[-1])):
            if j >= out_indices_cpu.shape[-1]:
                continue
            t = (
                out_indices_cpu[0, i, DEBUG_HEAD, j] - sink_token_size
            ) // BLOCK_CHUNK + sink_token_size // BLOCK_CHUNK
            t = t // t_chunk_size * t_chunk_size
            debug[i, t : t + t_chunk_size] = 1


@numba.njit(parallel=True)
def render_plot_ks(indices, ks, debug, DEBUG_HEAD, BLOCK_SIZE_Q):
    for i in numba.prange(indices.shape[1]):
        k = ks[DEBUG_HEAD, i]
        for j in range(indices.shape[-1]):
            if j >= k:
                continue
            t = indices[DEBUG_HEAD, i, j] // BLOCK_SIZE_Q
            debug[i, t : t + 1] = 1


DEBUG = os.getenv("HIP_DEBUG", "0") == "1"
DEBUG_RENDER = os.getenv("HIP_DEBUG_RENDER", "1") == "1"


@triton.jit
def load_keys_with_rope(
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head_kv,
    stride_k_hid,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
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
    HEAD_KV,
    UPDATE_CACHE,
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
        idx_head_kv,
        idx_hid[:, None],
        mask_tsrc_active[None, :],  # & mask_hid[:, None],
        # mask_tsrc_active[None, :] & mask_hid[:, None],
        HEAD_KV,
        BLOCK_CHUNK,
        BLOCK_HID,
        UPDATE_CACHE=UPDATE_CACHE,
    ).to(queries.dtype)

    if USING_EXTEND:
        real_pos_tdst_max = tl.sum(mask_tdst.to(tl.int32)) + real_pos_tdst_min
        tsrc_extend = tl.maximum(0, real_pos_tdst_max - model_context_length)
        if NEED_APPLY_ROPE or (tsrc_extend >= 0):
            old_tsrc = idx_tsrc

            if EXTEND_BACKEND == "dynamic_extend":
                window = model_context_length // 4

                new_tsrc = tl.where(
                    (idx_tsrc >= (real_pos_tdst_max - window))
                    | (real_pos_tdst_max <= model_context_length),
                    idx_tsrc,
                    # idx_tsrc * 0 + real_pos_tdst_max,
                    (
                        (idx_tsrc.to(tl.float32) - (real_pos_tdst_min - window))
                        * (
                            (model_context_length - window)
                            / (real_pos_tdst_min - window)
                        ).to(tl.float32)
                    ).to(tl.int32)
                    + (real_pos_tdst_min - window),
                )
                # new_tsrc = idx_tsrc * 0 + real_pos_tdst_max
                new_tsrc = tl.maximum(
                    real_pos_tdst_max - model_context_length, new_tsrc
                )
            elif EXTEND_BACKEND == "self_extend":
                window = 8192
                group_size = 16

                new_tsrc = tl.where(
                    idx_tsrc >= (real_pos_tdst_max - window),
                    idx_tsrc,
                    tl.where(
                        real_pos_tdst_max <= model_context_length,
                        idx_tsrc,
                        (idx_tsrc - real_pos_tdst_min) // group_size
                        + real_pos_tdst_min,
                    ),
                )
                new_tsrc = tl.maximum(0, new_tsrc)
            elif EXTEND_BACKEND == "relative":
                new_tsrc = idx_chunk * 0
                if IS_RIGHT:
                    new_tsrc += 1
            elif EXTEND_BACKEND == "infllm":
                new_tsrc = idx_chunk * 0
            elif EXTEND_BACKEND == "streaming":
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
                    COS,
                    stride_cos_t,
                    stride_cos_hid,
                    SIN,
                    stride_sin_t,
                    stride_sin_hid,
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
                    idx_head_kv,
                    ((idx_hid + BLOCK_HID // 2) % BLOCK_HID)[:, None],
                    mask_tsrc_active[None, :],
                    HEAD_KV,
                    BLOCK_CHUNK,
                    BLOCK_HID,
                    # NOTE: in previous load, the fetch should be succesfully done.
                    UPDATE_CACHE=UPDATE_CACHE,
                ).to(queries.dtype)

                # TODO: multiply -right
                # keys_left_rot = tl.where(
                #     (idx_hid + BLOCK_HID // 2)[:, None] < BLOCK_HID,
                #     -keys_left_rot,
                #     keys_left_rot
                # )
                keys_left_rot = keys_left_rot * (
                    ((idx_hid + BLOCK_HID // 2)[:, None] < BLOCK_HID) * (-2) + 1
                ).to(keys_left_rot.dtype)

                cos_new = tl.load(
                    COS
                    + new_tsrc[None, :].to(tl.int64) * stride_cos_t
                    + (idx_hid % (BLOCK_HID // 2))[:, None] * stride_cos_hid,
                    mask=mask_tsrc_active[None, :],
                    other=0.0,
                ).to(keys_left.dtype)
                sin_new = tl.load(
                    SIN
                    + new_tsrc[None, :].to(tl.int64) * stride_sin_t
                    + (idx_hid % (BLOCK_HID // 2))[:, None] * stride_sin_hid,
                    mask=mask_tsrc_active[None, :],
                    other=0.0,
                ).to(keys_left.dtype)

                keys_left = keys_left * cos_new + keys_left_rot * sin_new

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
    MASK_ACCESS_COUNTER,
    stride_mask_access_counter_bsz,
    stride_mask_access_counter_head_kv,
    stride_mask_access_counter_tsrc,
    MASK_CACHE_MISS_COUNTER,
    stride_mask_cache_miss_counter_bsz,
    stride_mask_cache_miss_counter_head_kv,
    stride_mask_cache_miss_counter_tsrc,
    CHUNK_COUNT: int,
    MAX_TSRC: int,
    TDST: int,
    HEAD: int,
    sliding_window_size: int,
    num_sinks: int,
    model_context_length: int,
    group_jobs: int,
    total_jobs: int,
    BLOCK_HID: tl.constexpr = 128,
    BLOCK_SIZE_Q: tl.constexpr = 32,
    STRIDE_Q: tl.constexpr = 1,
    BLOCK_CHUNK: tl.constexpr = 32,
    HEAD_GROUP: tl.constexpr = 4,
    REDUCE: tl.constexpr = "max",
    USING_EXTEND: tl.constexpr = False,
    EXTEND_BACKEND: tl.constexpr = "relative",
    NEED_APPLY_ROPE: tl.constexpr = False,
    TERMINATE_SIZE: tl.constexpr = 1,
    SCAN_STRIDE: tl.constexpr = 1,
    UPDATE_CACHE: tl.constexpr = True,
    ORACLE_MAXIMUM: tl.constexpr = False,
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)
    BDST_SCAN = tl.cdiv(BDST, SCAN_STRIDE)
    BCHUNK = tl.cdiv(CHUNK_COUNT, BLOCK_CHUNK)

    pid_group = tl.program_id(0).to(tl.int64)

    for i in range(group_jobs):
        pid = pid_group * group_jobs + i
        if pid < total_jobs:
            idx_bdst_scan = pid % BDST_SCAN
            pid = pid // BDST_SCAN
            idx_bchunk = pid % BCHUNK
            pid = pid // BCHUNK
            idx_head = pid % HEAD
            pid = pid // HEAD
            idx_bsz = pid

            # idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // STRIDE_Q) * STRIDE_Q
            # mask_tdst = idx_tdst < TDST
            if BLOCK_SIZE_Q // STRIDE_Q < 16:
                idx_tdst = (
                    (BDST - 1)
                    - (BDST_SCAN - 1) * SCAN_STRIDE
                    + idx_bdst_scan * SCAN_STRIDE
                ) * BLOCK_SIZE_Q + tl.arange(0, 16) * STRIDE_Q
                mask_tdst = (
                    (idx_tdst < TDST)
                    & (idx_tdst >= 0)
                    & (tl.arange(0, 16) < (BLOCK_SIZE_Q // STRIDE_Q))
                )
            else:
                idx_tdst = (
                    (BDST - 1)
                    - (BDST_SCAN - 1) * SCAN_STRIDE
                    + idx_bdst_scan * SCAN_STRIDE
                ) * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // STRIDE_Q) * STRIDE_Q
                mask_tdst = (idx_tdst < TDST) & (idx_tdst >= 0)
            idx_hid = tl.arange(0, BLOCK_HID)
            mask_hid = idx_hid < BLOCK_HID  # (tl.arange(0, BLOCK_HID) % 4) == 0

            pos_tdst = tl.load(
                POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
                mask=mask_tdst,
                other=0,
            )

            # real_pos_tdst_min = idx_bdst * BLOCK_SIZE_Q + TSRC - TDST
            real_pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst, 999999999))
            real_pos_tdst_min = tl.where(
                tl.sum(mask_tdst.to(tl.int32)) > 0, real_pos_tdst_min, -1
            )

            if real_pos_tdst_min >= 0:
                pos_tdst_min = (real_pos_tdst_min - sliding_window_size).to(tl.int32)
                pos_tdst_min = tl.maximum(pos_tdst_min, 0)

                idx_chunk = idx_bchunk * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
                mask_chunk = idx_chunk < CHUNK_COUNT

                idx_tsrc_left = tl.load(
                    INDICES_LEFT
                    + idx_bsz * stride_indices_left_bsz
                    + idx_bdst_scan * stride_indices_left_bdst
                    + idx_head * stride_indices_left_head
                    + idx_chunk * stride_indices_left_chunk,
                    mask=mask_chunk,
                    other=MAX_TSRC,
                ).to(tl.int32)

                idx_tsrc_right = tl.load(
                    INDICES_RIGHT
                    + idx_bsz * stride_indices_right_bsz
                    + idx_bdst_scan * stride_indices_right_bdst
                    + idx_head * stride_indices_right_head
                    + idx_chunk * stride_indices_right_chunk,
                    mask=mask_chunk,
                    other=MAX_TSRC,
                ).to(tl.int32)

                if (real_pos_tdst_min + BLOCK_SIZE_Q * SCAN_STRIDE) >= tl.min(
                    idx_tsrc_left
                ):
                    # mask_chunk = mask_chunk & (idx_tsrc_left < (real_pos_tdst_min - sliding_window_size + BLOCK_SIZE_Q))

                    # max_chunk_size = tl.ceil(MAX_TSRC / CHUNK_COUNT).to(tl.float32)
                    max_chunk_size = tl.max(idx_tsrc_right - idx_tsrc_left).to(
                        tl.float32
                    )

                    scores = tl.zeros((BLOCK_CHUNK,), dtype=tl.float32) - 32000.0

                    queries_sum = tl.zeros(
                        (BLOCK_SIZE_Q // STRIDE_Q, BLOCK_HID), dtype=tl.float32
                    )
                    queries_counter = tl.zeros(
                        (BLOCK_SIZE_Q // STRIDE_Q,), dtype=tl.int32
                    )
                    tl.static_assert(BLOCK_SIZE_Q // STRIDE_Q > 0)

                    for i_offset in tl.static_range(STRIDE_Q):
                        idx_tdst_iter = idx_tdst + i_offset
                        mask_tdst_iter = mask_tdst & (idx_tdst_iter < TDST)
                        queries_iter = tl.load(
                            Q
                            + idx_bsz * stride_q_bsz
                            + idx_tdst_iter[:, None] * stride_q_tdst
                            + idx_head * stride_q_head
                            + idx_hid[None, :] * stride_q_hid,
                            mask=mask_tdst_iter[:, None] & mask_hid[None, :],
                            other=0.0,
                        )
                        if queries_iter.dtype == tl.float8e5:
                            queries_iter = queries_iter.to(tl.float16)

                        if USING_EXTEND:
                            if NEED_APPLY_ROPE or (
                                real_pos_tdst_min >= model_context_length
                            ):
                                old_tdst = pos_tdst
                                if EXTEND_BACKEND == "dynamic_extend":
                                    new_tdst = pos_tdst
                                elif EXTEND_BACKEND == "self_extend":
                                    new_tdst = pos_tdst
                                elif EXTEND_BACKEND == "relative":
                                    new_tdst = pos_tdst * 0 + 1 + sliding_window_size
                                elif EXTEND_BACKEND == "infllm":
                                    new_tdst = pos_tdst * 0 + sliding_window_size
                                elif EXTEND_BACKEND == "streaming":
                                    # streaming
                                    new_tdst = tl.minimum(
                                        pos_tdst, CHUNK_COUNT + sliding_window_size
                                    )
                                else:
                                    raise Exception()

                                if NEED_APPLY_ROPE:
                                    queries_rot = tl.load(
                                        Q
                                        + idx_bsz * stride_q_bsz
                                        + idx_tdst_iter[:, None] * stride_q_tdst
                                        + idx_head * stride_q_head
                                        + ((idx_hid + BLOCK_HID // 2) % BLOCK_HID)[
                                            None, :
                                        ]
                                        * stride_q_hid,
                                        mask=mask_tdst_iter[:, None],
                                        other=0.0,
                                        # cache_modifier='.cg',
                                        # eviction_policy='evict_last',
                                        # volatile=True,
                                    )
                                    if queries_rot.dtype == tl.float8e5:
                                        queries_rot = queries_rot.to(tl.float16)

                                    cos_new = tl.load(
                                        COS
                                        + new_tdst[:, None].to(tl.int64) * stride_cos_t
                                        + (idx_hid % (BLOCK_HID // 2))[None, :]
                                        * stride_cos_hid,
                                        mask=mask_tdst_iter[:, None],
                                        other=0.0,
                                    ).to(queries_iter.dtype)
                                    sin_new = tl.load(
                                        SIN
                                        + new_tdst[:, None].to(tl.int64) * stride_sin_t
                                        + (idx_hid % (BLOCK_HID // 2))[None, :]
                                        * stride_sin_hid,
                                        mask=mask_tdst_iter[:, None],
                                        other=0.0,
                                    ).to(queries_iter.dtype)

                                    # queries_rot = tl.where(
                                    #     (idx_hid + BLOCK_HID // 2)[None, :] < BLOCK_HID,
                                    #     -queries_rot,
                                    #     queries_rot
                                    # )
                                    queries_rot = queries_rot * (
                                        (
                                            (idx_hid + BLOCK_HID // 2)[None, :]
                                            < BLOCK_HID
                                        )
                                        * (-2)
                                        + 1
                                    ).to(queries_rot.dtype)

                                    queries_iter = (
                                        queries_iter * cos_new + queries_rot * sin_new
                                    ).to(queries_iter.dtype)
                                else:
                                    queries_iter = adjust_rope(
                                        queries,
                                        old_tdst,
                                        new_tdst,
                                        mask_tdst_iter,
                                        idx_hid,
                                        COS,
                                        stride_cos_t,
                                        stride_cos_hid,
                                        SIN,
                                        stride_sin_t,
                                        stride_sin_hid,
                                        BLOCK_SIZE_Q // STRIDE_Q,
                                        BLOCK_HID,
                                        NEED_APPLY_ROPE,
                                    ).to(queries_iter.dtype)
                                    queries_iter = (
                                        queries_iter * mask_tdst_iter[:, None]
                                    ).to(queries_iter.dtype)

                        queries_sum += queries_iter
                        queries_counter += mask_tdst_iter.to(tl.int32)

                    queries = (
                        queries_sum / (queries_counter[:, None] + 1e-12)
                    ) * mask_tdst[:, None]
                    if Q.dtype.element_ty != tl.float8e5:
                        queries = queries.to(Q.dtype.element_ty)
                    else:
                        queries = queries.to(tl.float16)

                    while max_chunk_size >= TERMINATE_SIZE:
                        max_chunk_size /= 2.0
                        mask_tsrc_active = (
                            mask_chunk
                            & (idx_tsrc_left < idx_tsrc_right)
                            & (idx_tsrc_left <= pos_tdst_min)
                            & (idx_tsrc_left >= 0)
                        )
                        # mask_tsrc_active = mask_tsrc_active & (idx_tsrc_left < (real_pos_tdst_min - sliding_window_size + BLOCK_SIZE_Q))
                        idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2

                        if ORACLE_MAXIMUM:
                            scores_left = tl.full(
                                (idx_tdst.shape[0], BLOCK_CHUNK),
                                dtype=tl.float32,
                                value=float("-inf"),
                            )
                            for tt in range(0, max_chunk_size.to(tl.int32)):
                                idx_tsrc = idx_tsrc_left + tt
                                keys_left = load_keys_with_rope(
                                    K,
                                    stride_k_bsz,
                                    stride_k_tsrc,
                                    stride_k_head_kv,
                                    stride_k_hid,
                                    COS,
                                    stride_cos_t,
                                    stride_cos_hid,
                                    SIN,
                                    stride_sin_t,
                                    stride_sin_hid,
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
                                    USING_OFFLOAD_CACHE,
                                    OFFLOAD_CACHE_KV_PACKED,
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
                                    MASK_ACCESS_COUNTER,
                                    stride_mask_access_counter_bsz,
                                    stride_mask_access_counter_head_kv,
                                    stride_mask_access_counter_tsrc,
                                    MASK_CACHE_MISS_COUNTER,
                                    stride_mask_cache_miss_counter_bsz,
                                    stride_mask_cache_miss_counter_head_kv,
                                    stride_mask_cache_miss_counter_tsrc,
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
                                    HEAD // HEAD_GROUP,
                                    UPDATE_CACHE,
                                )

                                t_scores_left = tl.dot(
                                    (
                                        queries
                                        * (
                                            tl.sqrt(BLOCK_HID * 1.0)
                                            / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))
                                        ).to(queries.dtype)
                                    ).to(queries.dtype),
                                    (
                                        keys_left.to(queries.dtype)
                                        * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(
                                            queries.dtype
                                        )
                                    ).to(queries.dtype),
                                    allow_tf32=True,
                                    out_dtype=tl.float32,
                                ).to(queries.dtype)

                                scores_left = tl.maximum(scores_left, t_scores_left)
                        else:
                            idx_tsrc = (idx_tsrc_left + idx_tsrc_center) // 2
                            keys_left = load_keys_with_rope(
                                K,
                                stride_k_bsz,
                                stride_k_tsrc,
                                stride_k_head_kv,
                                stride_k_hid,
                                COS,
                                stride_cos_t,
                                stride_cos_hid,
                                SIN,
                                stride_sin_t,
                                stride_sin_hid,
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
                                USING_OFFLOAD_CACHE,
                                OFFLOAD_CACHE_KV_PACKED,
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
                                MASK_ACCESS_COUNTER,
                                stride_mask_access_counter_bsz,
                                stride_mask_access_counter_head_kv,
                                stride_mask_access_counter_tsrc,
                                MASK_CACHE_MISS_COUNTER,
                                stride_mask_cache_miss_counter_bsz,
                                stride_mask_cache_miss_counter_head_kv,
                                stride_mask_cache_miss_counter_tsrc,
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
                                HEAD // HEAD_GROUP,
                                UPDATE_CACHE,
                            )

                            scores_left = tl.dot(
                                (
                                    queries
                                    * (
                                        tl.sqrt(BLOCK_HID * 1.0)
                                        / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))
                                    ).to(queries.dtype)
                                ).to(queries.dtype),
                                (
                                    keys_left.to(queries.dtype)
                                    * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(
                                        queries.dtype
                                    )
                                ).to(queries.dtype),
                                allow_tf32=True,
                                out_dtype=tl.float32,
                            ).to(queries.dtype)

                        if REDUCE == "max":
                            scores_left = tl.where(
                                mask_tdst[:, None], scores_left, float("-inf")
                            )
                            scores_left = tl.max(scores_left, axis=0).to(
                                scores_left.dtype
                            )
                        elif REDUCE == "mean":
                            scores_left = tl.where(
                                mask_tdst[:, None], scores_left, float("0")
                            )
                            scores_left = tl.sum(scores_left, axis=0).to(
                                scores_left.dtype
                            )
                            scores_left = (
                                scores_left / tl.sum(mask_tdst.to(tl.float32))
                            ).to(scores_left.dtype)
                        else:
                            raise Exception()
                        scores_left = tl.where(
                            mask_tsrc_active, scores_left, float("-inf")
                        ).to(scores_left.dtype)

                        if ORACLE_MAXIMUM:
                            scores_right = tl.full(
                                (idx_tdst.shape[0], BLOCK_CHUNK),
                                dtype=tl.float32,
                                value=float("-inf"),
                            )
                            for tt in range(0, max_chunk_size.to(tl.int32)):
                                idx_tsrc = idx_tsrc_center + tt
                                keys_right = load_keys_with_rope(
                                    K,
                                    stride_k_bsz,
                                    stride_k_tsrc,
                                    stride_k_head_kv,
                                    stride_k_hid,
                                    COS,
                                    stride_cos_t,
                                    stride_cos_hid,
                                    SIN,
                                    stride_sin_t,
                                    stride_sin_hid,
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
                                    USING_OFFLOAD_CACHE,
                                    OFFLOAD_CACHE_KV_PACKED,
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
                                    MASK_ACCESS_COUNTER,
                                    stride_mask_access_counter_bsz,
                                    stride_mask_access_counter_head_kv,
                                    stride_mask_access_counter_tsrc,
                                    MASK_CACHE_MISS_COUNTER,
                                    stride_mask_cache_miss_counter_bsz,
                                    stride_mask_cache_miss_counter_head_kv,
                                    stride_mask_cache_miss_counter_tsrc,
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
                                    HEAD // HEAD_GROUP,
                                    UPDATE_CACHE,
                                )

                                t_scores_right = tl.dot(
                                    (
                                        queries
                                        * (
                                            tl.sqrt(BLOCK_HID * 1.0)
                                            / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))
                                        ).to(queries.dtype)
                                    ).to(queries.dtype),
                                    (
                                        keys_right.to(queries.dtype)
                                        * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(
                                            queries.dtype
                                        )
                                    ).to(queries.dtype),
                                    allow_tf32=True,
                                    out_dtype=tl.float32,
                                ).to(queries.dtype)

                                scores_right = tl.maximum(scores_right, t_scores_right)
                        else:
                            idx_tsrc = (idx_tsrc_center + idx_tsrc_right) // 2
                            keys_right = load_keys_with_rope(
                                K,
                                stride_k_bsz,
                                stride_k_tsrc,
                                stride_k_head_kv,
                                stride_k_hid,
                                COS,
                                stride_cos_t,
                                stride_cos_hid,
                                SIN,
                                stride_sin_t,
                                stride_sin_hid,
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
                                USING_OFFLOAD_CACHE,
                                OFFLOAD_CACHE_KV_PACKED,
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
                                MASK_ACCESS_COUNTER,
                                stride_mask_access_counter_bsz,
                                stride_mask_access_counter_head_kv,
                                stride_mask_access_counter_tsrc,
                                MASK_CACHE_MISS_COUNTER,
                                stride_mask_cache_miss_counter_bsz,
                                stride_mask_cache_miss_counter_head_kv,
                                stride_mask_cache_miss_counter_tsrc,
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
                                HEAD // HEAD_GROUP,
                                UPDATE_CACHE,
                            )

                            scores_right = tl.dot(
                                (
                                    queries
                                    * (
                                        tl.sqrt(BLOCK_HID * 1.0)
                                        / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))
                                    ).to(queries.dtype)
                                ).to(queries.dtype),
                                (
                                    keys_right.to(queries.dtype)
                                    * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(
                                        queries.dtype
                                    )
                                ).to(queries.dtype),
                                allow_tf32=True,
                                out_dtype=tl.float32,
                            ).to(queries.dtype)

                        if REDUCE == "max":
                            scores_right = tl.where(
                                mask_tdst[:, None], scores_right, float("-inf")
                            )
                            scores_right = tl.max(scores_right, axis=0).to(
                                scores_right.dtype
                            )
                        elif REDUCE == "mean":
                            scores_right = tl.where(
                                mask_tdst[:, None], scores_right, float("0")
                            )
                            scores_right = tl.sum(scores_right, axis=0).to(
                                scores_right.dtype
                            )
                            scores_right = (
                                scores_right / tl.sum(mask_tdst.to(tl.float32))
                            ).to(scores_right.dtype)
                        else:
                            raise Exception()
                        scores_right = tl.where(
                            mask_tsrc_active, scores_right, float("-inf")
                        ).to(scores_right.dtype)

                        mask_left_win = scores_left > scores_right
                        idx_tsrc_left = tl.where(
                            mask_tsrc_active,
                            tl.where(
                                mask_left_win,
                                idx_tsrc_left,
                                idx_tsrc_center,
                            ),
                            idx_tsrc_left,
                        )

                        idx_tsrc_right = tl.where(
                            mask_tsrc_active,
                            tl.where(
                                mask_left_win,
                                idx_tsrc_center,
                                idx_tsrc_right,
                            ),
                            idx_tsrc_right,
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
                            ),
                        )

                    # idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2
                    # idx_tsrc_left = idx_tsrc_center - TERMINATE_SIZE // 2
                    # idx_tsrc_right = idx_tsrc_left + TERMINATE_SIZE

                    tl.store(
                        INDICES_LEFT
                        + idx_bsz * stride_indices_left_bsz
                        + idx_bdst_scan * stride_indices_left_bdst
                        + idx_head * stride_indices_left_head
                        + idx_chunk * stride_indices_left_chunk,
                        value=idx_tsrc_left,
                        mask=mask_chunk,
                    )

                    tl.store(
                        INDICES_RIGHT
                        + idx_bsz * stride_indices_right_bsz
                        + idx_bdst_scan * stride_indices_right_bdst
                        + idx_head * stride_indices_right_head
                        + idx_chunk * stride_indices_right_chunk,
                        value=idx_tsrc_right,
                        mask=mask_chunk,
                    )

                    tl.store(
                        OUT_SCORES
                        + idx_bsz * stride_out_scores_bsz
                        + idx_bdst_scan * stride_out_scores_bdst
                        + idx_head * stride_out_scores_head
                        + idx_chunk * stride_out_scores_chunk,
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
    REDUCE: tl.constexpr = "max",
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
        (NEED_APPLY_ROPE and USING_EXTEND) or (not (NEED_APPLY_ROPE or USING_EXTEND))
    )

    idx_tdst = (
        idx_bdst_scan * SCAN_STRIDE * BLOCK_SIZE_Q
        + (BDST * BLOCK_SIZE_Q - BDST_SCAN * SCAN_STRIDE * BLOCK_SIZE_Q)
        + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
    )
    mask_tdst = (idx_tdst < TDST) & (idx_tdst >= 0)
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < BLOCK_HID

    pos_tdst = tl.load(
        POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
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
        Q
        + idx_bsz * stride_q_bsz
        + idx_tdst[:, None] * stride_q_tdst
        + idx_head * stride_q_head
        + idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0,
    )

    if NEED_APPLY_ROPE and USING_EXTEND:
        if EXTEND_BACKEND == "dynamic_extend":
            new_tdst = pos_tdst
        elif EXTEND_BACKEND == "self_extend":
            new_tdst = pos_tdst
        elif EXTEND_BACKEND == "streaming":
            new_tdst = tl.minimum(pos_tdst, N_CHUNK + sliding_window_size)
        elif EXTEND_BACKEND == "relative":
            new_tdst = pos_tdst * 0 + sliding_window_size
        else:
            raise Exception()

        queries_rot = tl.load(
            Q
            + idx_bsz * stride_q_bsz
            + idx_tdst[:, None] * stride_q_tdst
            + idx_head * stride_q_head
            + ((idx_hid + BLOCK_HID // 2) % BLOCK_HID)[None, :] * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0,
        )

        cos_new = tl.load(
            COS
            + new_tdst[:, None].to(tl.int64) * stride_cos_t
            + (idx_hid % (BLOCK_HID // 2))[None, :] * stride_cos_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)
        sin_new = tl.load(
            SIN
            + new_tdst[:, None].to(tl.int64) * stride_sin_t
            + (idx_hid % (BLOCK_HID // 2))[None, :] * stride_sin_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)

        queries_rot = queries_rot * (
            ((idx_hid + BLOCK_HID // 2)[None, :] < BLOCK_HID) * (-2) + 1
        ).to(queries_rot.dtype)

        queries = (queries * cos_new + queries_rot * sin_new).to(queries.dtype)

    for idx_chunk_start in range(0, N_CHUNK, BLOCK_CHUNK):
        # for idx_chunk in range(tl.cdiv(pos_tdst_max, max_chunk_size)):
        idx_chunk = tl.arange(0, BLOCK_CHUNK) + idx_chunk_start
        mask_chunk = idx_chunk < N_CHUNK
        idx_tsrc_left = tl.load(
            INDICES_LEFT
            + idx_bsz * stride_indices_left_bsz
            + idx_bdst_scan * stride_indices_left_bdst
            + idx_head * stride_indices_left_head
            + idx_chunk * stride_indices_left_chunk,
            mask=mask_chunk,
            other=987654321,
        ).to(tl.int64)

        idx_tsrc_right = tl.load(
            INDICES_RIGHT
            + idx_bsz * stride_indices_right_bsz
            + idx_bdst_scan * stride_indices_right_bdst
            + idx_head * stride_indices_right_head
            + idx_chunk * stride_indices_right_chunk,
            mask=mask_chunk,
            other=987654321,
        ).to(tl.int64)

        if tl.min(idx_tsrc_left) <= pos_tdst_max:
            idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2
            idx_tsrc_left = tl.maximum(0, idx_tsrc_center - BLOCK_SIZE_K // 2)
            idx_tsrc = (
                idx_tsrc_left[:, None]
                + tl.arange(0, BLOCK_SIZE_K // BLOCK_STRIDE_K)[None, :] * BLOCK_STRIDE_K
            )
            idx_tsrc = tl.ravel(idx_tsrc)
            mask_tsrc = idx_tsrc <= (tl.max(pos_tdst) - sliding_window_size)

            keys = load_keys_with_rope(
                K,
                stride_k_bsz,
                stride_k_tsrc,
                stride_k_head_kv,
                stride_k_hid,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
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
                HEAD // HEAD_GROUP,
                UPDATE_CACHE,
            )

            scores = tl.dot(
                (
                    queries
                    * (tl.sqrt(BLOCK_HID * 1.0) / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(
                        queries.dtype
                    )
                ).to(queries.dtype),
                (
                    keys.to(queries.dtype)
                    * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(queries.dtype)
                ).to(queries.dtype),
                allow_tf32=True,
                out_dtype=tl.float32,
            ).to(queries.dtype)

            if REDUCE == "max":
                scores_reduced = tl.where(
                    mask_tdst[:, None] & mask_tsrc[None, :], scores, -32000.0
                )
                scores_reduced = tl.reshape(
                    scores_reduced,
                    BLOCK_SIZE_Q // BLOCK_STRIDE_Q,
                    BLOCK_CHUNK,
                    BLOCK_SIZE_K // BLOCK_STRIDE_K,
                )
                scores_reduced = tl.max(scores_reduced, axis=0)
                scores_reduced = tl.max(scores_reduced, axis=-1)
            # elif REDUCE == 'mean':
            #     scores_reduced = tl.sum(tl.where(
            #         mask_tdst[:, None] & mask_tsrc[None, :],
            #         scores,
            #         0
            #     )) / tl.sum((mask_tdst[:, None] & mask_tsrc[None, :]).to(tl.int32))
            else:
                raise Exception()

            tl.store(
                OUT_SCORES
                + idx_bsz * stride_out_scores_bsz
                + idx_bdst_scan * stride_out_scores_bdst
                + idx_head * stride_out_scores_head
                + idx_chunk * stride_out_scores_chunk,
                value=scores_reduced,
                mask=mask_chunk,
            )
        else:
            tl.store(
                OUT_SCORES
                + idx_bsz * stride_out_scores_bsz
                + idx_bdst_scan * stride_out_scores_bdst
                + idx_head * stride_out_scores_head
                + idx_chunk * stride_out_scores_chunk,
                value=-32000.0,
                mask=mask_chunk,
            )


@triton.jit
def compute_v_cos(
    V,
    stride_v_bsz,
    stride_v_tsrc,
    stride_v_head_kv,
    stride_v_hid,
    INDICES,
    stride_indices_bsz,
    stride_indices_bdst,
    stride_indices_head,
    strdie_indices_k,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    OUT_SCORES,
    stride_out_scores_bsz,
    stride_out_scores_bdst,
    stride_out_scores_head,
    stride_out_scores_k,
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
    TDST,
    TSRC,
    HEAD,
    KS,
    HEAD_GROUP: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    pid = tl.program_id(0)

    idx_head = pid % HEAD
    pid = pid // HEAD

    idx_bk = pid % tl.cdiv(KS, GROUP_K)
    idx_k = idx_bk * GROUP_K + tl.arange(0, GROUP_K)
    mask_k = idx_k < KS
    pid = pid // tl.cdiv(KS, GROUP_K)

    idx_bdst = pid % tl.cdiv(TDST, BLOCK_SIZE_Q)
    idx_tdst = (
        idx_bdst * BLOCK_SIZE_Q
        + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
    )
    mask_tdst = idx_tdst < TDST
    idx_bsz = pid // tl.cdiv(TDST, BLOCK_SIZE_Q)

    idx_hid = tl.arange(0, BLOCK_HID)

    pos_tdst = tl.load(
        POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    mask_tdst = mask_tdst & (pos_tdst < TSRC)
    seq_len = tl.max(pos_tdst)  # + 1

    indices = tl.load(
        INDICES
        + idx_bsz * stride_indices_bsz
        + idx_bdst * stride_indices_bdst
        + idx_head * stride_indices_head
        + idx_k * strdie_indices_k,
        mask=mask_k,
        other=seq_len + 2 * BLOCK_SIZE_K,
    )
    indices = indices // BLOCK_SIZE_K * BLOCK_SIZE_K

    idx_tsrc = tl.ravel(indices[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :])
    mask_tsrc = (idx_tsrc < seq_len) & (idx_tsrc >= 0)

    # values_tdst = tl.load(
    #     V +\
    #         idx_bsz * stride_v_bsz+\
    #         idx_tdst[:, None] * stride_v_tsrc+\
    #         (idx_head // HEAD_GROUP) * stride_v_head_kv +\
    #         idx_hid[None, :] * strdie_v_hid,
    #     mask=mask_tdst[:, None],
    #     other=0,
    # )

    # tl.static_assert(not USING_OFFLOAD_CACHE)
    values_tdst = load_tokens(
        V,
        stride_v_bsz,
        stride_v_tsrc,
        stride_v_head_kv,
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
        pos_tdst[:, None],
        idx_head // HEAD_GROUP,
        idx_hid[None, :],
        mask_tdst[:, None],
        HEAD // HEAD_GROUP,
        BLOCK_SIZE_Q // BLOCK_STRIDE_Q,
        BLOCK_HID,
    ).to(tl.bfloat16)

    # values_tdst = (
    #     tl.sum(values_tdst, axis=0) /\
    #     tl.sum(mask_tdst.to(tl.int32))
    # )

    # values_tsrc = tl.load(
    #     V +\
    #         idx_bsz * stride_v_bsz +\
    #         idx_tsrc[:, None] * stride_v_tsrc +\
    #         (idx_head // HEAD_GROUP) * stride_v_head_kv +\
    #         idx_hid[None, :] * strdie_v_hid,
    #     mask=mask_tsrc[:, None],
    #     other=0,
    # )

    values_tsrc = load_tokens(
        V,
        stride_v_bsz,
        stride_v_tsrc,
        stride_v_head_kv,
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
        idx_head // HEAD_GROUP,
        idx_hid[None, :],
        mask_tsrc[:, None],
        HEAD // HEAD_GROUP,
        GROUP_K * BLOCK_SIZE_K,
        BLOCK_HID,
    ).to(tl.bfloat16)

    # values_tsrc = (
    #     tl.sum(tl.reshape(values_tsrc, [GROUP_K, BLOCK_SIZE_K, BLOCK_HID]), axis=1) /\
    #     tl.sum(tl.reshape(mask_tsrc.to(tl.int32), [GROUP_K, BLOCK_SIZE_K, 1]), axis=1)
    # )

    values_tdst_norm = tl.sqrt(
        tl.sum(values_tdst.to(tl.float32) * values_tdst.to(tl.float32), axis=-1)
    )
    values_tsrc_norm = tl.sqrt(
        tl.sum(values_tsrc.to(tl.float32) * values_tsrc.to(tl.float32), axis=-1)
    )

    normalized_values_tdst = values_tdst
    normalized_values_tsrc = values_tsrc
    normalized_values_tdst = values_tdst / tl.maximum(values_tdst_norm[:, None], 1e-20)
    normalized_values_tsrc = values_tsrc / tl.maximum(values_tsrc_norm[:, None], 1e-20)

    # -
    # cos_sim_scores = tl.sum(normalized_values_tdst[None, :] * normalized_values_tsrc, axis=-1)
    cos_sim_scores = tl.dot(
        normalized_values_tdst, tl.trans(normalized_values_tsrc, 1, 0)
    )
    # cos_sim_scores = ((cos_sim_scores + 1) * 0.5).to(tl.float32)
    cos_sim_scores = cos_sim_scores  # * cos_sim_scores * cos_sim_scores

    scores = tl.reshape(
        cos_sim_scores, (BLOCK_SIZE_Q // BLOCK_STRIDE_Q, GROUP_K, BLOCK_SIZE_K)
    )
    # scores = tl.reshape(values_tsrc_norm, (GROUP_K, BLOCK_SIZE_K))
    mask_scores = tl.reshape(
        mask_tdst[:, None] & mask_tsrc[None, :],
        (BLOCK_SIZE_Q // BLOCK_STRIDE_Q, GROUP_K, BLOCK_SIZE_K),
    )
    scores = scores * mask_scores

    # reduce-mean
    mask_scores = tl.sum(mask_scores.to(scores.dtype), axis=-1)
    scores = tl.sum(scores, axis=-1)
    mask_scores = tl.sum(mask_scores.to(scores.dtype), axis=0)
    scores = tl.sum(scores, axis=0) / tl.maximum(mask_scores, 1e-20)
    # -

    # scores = tl.sum(values_tdst[None, :] * values_tsrc, axis=1)

    # reduce max
    # scores = tl.max(tl.max(scores, axis=-1), axis=0)

    # norm reduce-mean
    # scores = tl.reshape(values_tsrc_norm, (GROUP_K, BLOCK_SIZE_K))
    # scores = tl.sum(scores, axis=-1) / tl.maximum(tl.sum(tl.reshape(mask_tsrc, (GROUP_K, BLOCK_SIZE_K)), axis=-1), 1e-20)

    # scores = tl.sum(values_tdst[None, :] * values_tsrc)

    tl.store(
        OUT_SCORES
        + idx_bsz * stride_out_scores_bsz
        + idx_bdst * stride_out_scores_bdst
        + idx_head * stride_out_scores_head
        + idx_k * stride_out_scores_k,
        value=scores,
        mask=mask_k,
    )


def dual_stage_quadratic_hip_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    args: HiPAttentionArgs,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
):
    DEBUG_HEAD = -1
    global DEBUG

    # if (q.shape[1] == 1) and (not args.disable_flashdecode):
    #     pass
    # else:
    #     # FIXME: just for dev
    #     k = args.gather_k_from_paged_cache(chunk_size=args.stages[0].stage_chunk_size)
    #     v = args.gather_v_from_paged_cache(chunk_size=args.stages[0].stage_chunk_size)

    if args.q_mask is None:
        q_bsa = q
    else:
        q_bsa = q
        q = args.q_mask
    if args.k_mask is None:
        k_mask = k
    else:
        k_mask = args.k_mask

    k_mask_original = k_mask

    BLOCK_HID = q.shape[-1]

    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        BSZ, TSRC, HEAD_KV, HID = k.shape
        assert v.shape == k.shape
        MAX_TSRC = TSRC
    else:
        # MAX_TSRC = args.k_cache.shape[0] * args.k_cache.shape[1]
        # MAX_TSRC = int(os.getenv('EXTEND_LEN', '128')) * 1024
        MAX_TSRC = args.extend_context_length
        if args.k_cache is not None:
            HEAD_KV = args.k_cache.shape[-2]
        else:
            HEAD_KV = args.offload_cache.k_uvm.bank_cpu.shape[-2]
        TSRC = MAX_TSRC

    assert len(args.stages) > 0
    STAGE_STRIDE = args.stages[0].stage_stride
    BLOCK_SIZE_Q = args.stages[0].stage_block_size_q
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    BDST_SCAN = triton.cdiv(BDST, STAGE_STRIDE)
    BLOCK_CHUNK = args.block_size_k
    chunk_size = args.stages[0].stage_chunk_size
    chunk_count = triton.cdiv(
        max(0, MAX_TSRC - args.sink_token_size - args.sliding_window_size), chunk_size
    )

    args = args.clone()
    args.mask_k = args.stages[0].stage_chunk_size
    original_sliding_window_size = args.sliding_window_size
    args.sliding_window_size = max(0, args.sliding_window_size - args.mask_k)

    if torch.cuda.is_current_stream_capturing() or args.position_ids is not None:
        assert args.position_ids is not None
        position_ids = args.position_ids
    else:
        position_ids = (torch.arange(0, TDST, device=q.device) + (TSRC - TDST))[
            None, :
        ].expand(BSZ, TDST)
    assert position_ids.shape == (BSZ, TDST), position_ids.shape

    if args.using_paged_cache:
        MAX_PAGE = args.paged_cache_page_count
    else:
        MAX_PAGE = MAX_TSRC

    if args.require_cache_statistics:
        mask_access_counter = torch.zeros(
            (BSZ, HEAD_KV, MAX_PAGE), dtype=torch.int32, device=q.device
        )
        mask_cache_miss_counter = torch.zeros(
            (BSZ, HEAD_KV, MAX_PAGE), dtype=torch.int32, device=q.device
        )
        sa_access_counter = torch.zeros(
            (BSZ, HEAD_KV, MAX_PAGE), dtype=torch.int32, device=q.device
        )
        sa_cache_miss_counter = torch.zeros(
            (BSZ, HEAD_KV, MAX_PAGE), dtype=torch.int32, device=q.device
        )
    else:
        sa_cache_miss_counter = sa_access_counter = mask_cache_miss_counter = (
            mask_access_counter
        ) = None

    stage_caches = (
        []
        if (cached_metadata is None) or (cached_metadata.stage_caches is None)
        else cached_metadata.stage_caches
    )
    if not args.require_stage_caches:
        stage_caches = None

    if (cached_metadata is None) or (cached_metadata.indices is None):
        # loop carrying variables: indices_left, indices_right, out_scores
        if (
            (cached_metadata is None)
            or (cached_metadata.stage_caches is None)
            or (stage_caches is None)
        ):
            indices_left = torch.zeros(
                (BSZ, BDST_SCAN, HEAD, chunk_count), device=q.device, dtype=torch.int64
            )

            indices_left[:, :, :, :] = (
                torch.floor(
                    torch.arange(0, chunk_count, device=q.device, dtype=torch.float64)
                    * chunk_size
                    + args.sink_token_size
                ).to(indices_left.dtype)
            )[None, None, None, :]
            indices_right = indices_left + chunk_size
            indices_right.clamp_max_(MAX_TSRC - args.sliding_window_size)

            out_scores = torch.full(
                (BSZ, BDST_SCAN, HEAD, triton.next_power_of_2(chunk_count)),
                device=q.device,
                dtype=torch.float32,
                fill_value=-32000.0,
            )
        else:
            assert cached_metadata is not None
            assert cached_metadata.stage_caches is not None
            assert len(stage_caches) <= len(args.stages)

            last_stage_cache = stage_caches[-1]

            indices_left = last_stage_cache.indices_left.clone()
            indices_right = last_stage_cache.indices_right.clone()
            out_scores = last_stage_cache.out_scores.clone()

        for i_stage, stage_info in enumerate(args.stages):
            # if stage_chunk_size > chunk_size: continue
            # if stage_k > TSRC: continue

            stage_block_stride_q = stage_info.stage_block_stride_q
            stage_chunk_size = stage_info.stage_chunk_size
            stage_k = stage_info.stage_k

            if i_stage < (len(stage_caches if stage_caches is not None else []) - 1):
                # print('stage cached pass', i_stage)
                continue
            elif i_stage == (len(stage_caches if stage_caches is not None else []) - 1):
                # print('last cached stage', i_stage)
                pass
            elif i_stage > 0:
                assert (stage_k % chunk_size) == 0, f"{stage_k} % {chunk_size}"
                indices_left = indices_left[..., : stage_k // chunk_size]
                require_align = stage_info.require_realign_index
                if require_align:
                    indices_left = (
                        indices_left - args.sink_token_size
                    ) // chunk_size * chunk_size + args.sink_token_size
                    indices_right = indices_left + chunk_size
                else:
                    indices_right = indices_right[..., : stage_k // chunk_size]
                out_scores = out_scores[..., : stage_k // chunk_size]
                # NOTE: revert this
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

                    indices_left = indices_left.repeat_interleave(num_split, 1)[
                        :, -BDST:
                    ].contiguous()
                    indices_right = indices_right.repeat_interleave(num_split, 1)[
                        :, -BDST:
                    ].contiguous()
                    out_scores = out_scores.repeat_interleave(num_split, 1)[
                        :, -BDST:
                    ].contiguous()

                if STAGE_STRIDE != stage_info.stage_stride:
                    assert stage_info.stage_stride < STAGE_STRIDE
                    assert STAGE_STRIDE > 0
                    indices_left = indices_left.repeat_interleave(
                        STAGE_STRIDE // stage_info.stage_stride, 1
                    )[:, -BDST:].contiguous()
                    indices_right = indices_right.repeat_interleave(
                        STAGE_STRIDE // stage_info.stage_stride, 1
                    )[:, -BDST:].contiguous()
                    out_scores = out_scores.repeat_interleave(
                        STAGE_STRIDE // stage_info.stage_stride, 1
                    )[:, -BDST:].contiguous()
                    STAGE_STRIDE = stage_info.stage_stride

                # if DEBUG and DEBUG_RENDER and (not torch.cuda.is_current_stream_capturing()) and (BDST > 10) and (i_stage == 1):
                #     out_indices_cpu = indices_left.cpu().numpy()
                #     debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK)))
                #     render_plot_sampled(out_indices_cpu, debug, DEBUG_HEAD, BLOCK_CHUNK, chunk_count, TDST, args.sink_token_size)
                #     cv2.imwrite('dummy_sampled.png', debug * 255)
                #     print('saved dummy_sampled.png')

                assert (chunk_size % stage_chunk_size) == 0
                splits = chunk_size // stage_chunk_size
                chunk_sizes = (
                    (indices_right - indices_left).float() / splits
                ).clamp_min_(0)
                indices_left = (
                    indices_left[..., None]
                    + (
                        torch.arange(0, splits, device=q.device)[
                            None, None, None, None, :
                        ]
                        * chunk_sizes[..., None]
                    )
                    .floor()
                    .long()
                )
                indices_left = indices_left.flatten(-2, -1)
                indices_right = (
                    indices_right[..., None]
                    - (
                        (
                            (splits - 1)
                            - torch.arange(0, splits, device=q.device)[
                                None, None, None, None, :
                            ]
                        )
                        * chunk_sizes[..., None]
                    )
                    .floor()
                    .long()
                )
                indices_right = indices_right.flatten(-2, -1)
                out_scores = out_scores.repeat_interleave(splits, -1)
            else:
                assert stage_info.stage_k is None, "first stage always quadratic"
                assert isinstance(
                    stage_info, ScanStage
                ), f"frist stage always scan {stage_info}"
                STAGE_STRIDE = stage_info.stage_stride

            if (stage_caches is not None) and (i_stage >= len(stage_caches)):
                if i_stage == 0:
                    # NOTE: do not cache first stage input, because it is meaning less.
                    stage_caches.append(
                        HiPAttentionStageInputCache(
                            indices_left=None,
                            indices_right=None,
                            out_scores=None,
                        )
                    )
                else:
                    stage_caches.append(
                        HiPAttentionStageInputCache(
                            indices_left=indices_left.clone(),
                            indices_right=indices_right.clone(),
                            out_scores=out_scores.clone(),
                        )
                    )

            chunk_size = stage_chunk_size
            chunk_count = indices_left.shape[-1]
            BLOCK_CHUNK = max(16, triton.next_power_of_2(min(chunk_count, BLOCK_CHUNK)))

            pre_device = torch.cuda.current_device()
            torch.cuda.set_device(q.device)

            if isinstance(stage_info, ScanStage):
                extend_backend = (
                    args.scan_extend_backend
                    if stage_info.stage_extend_backend is None
                    else stage_info.stage_extend_backend
                )

                if not (args.online_update_cache and (args.offload_cache is not None)):
                    grid = (
                        BSZ
                        * triton.cdiv(chunk_count, BLOCK_CHUNK)
                        * triton.cdiv(triton.cdiv(TDST, BLOCK_SIZE_Q), STAGE_STRIDE)
                        * HEAD,
                    )
                    njobs = grid[0]
                    group_jobs = 1
                else:
                    njobs = (
                        BSZ
                        * triton.cdiv(chunk_count, BLOCK_CHUNK)
                        * triton.cdiv(triton.cdiv(TDST, BLOCK_SIZE_Q), STAGE_STRIDE)
                        * HEAD
                    )
                    sm_count = num_streaming_multiprocessor()
                    group_jobs = triton.cdiv(njobs, sm_count)
                    grid = (min(sm_count, njobs),)
                # if args.offload_cache is not None:
                #     print('before masking')
                #     args.offload_cache.mask_k_cache._verify_cache()

                # B T H D
                # if k_mask_original is not None:
                #     B, T, H, D = k.shape
                #     wind_size = args.stages[i_stage + 1].stage_chunk_size // 2 - 1 if (i_stage + 1) < len(args.stages) else 0
                #     if wind_size > 0:
                #         k_max = torch.nn.functional.max_pool1d(k_mask_original.permute(0, 2, 3, 1).reshape(-1, 1, T), kernel_size=wind_size*2 + 1, padding=wind_size, stride=1)
                #         k_min = -torch.nn.functional.max_pool1d((-k_mask_original).permute(0, 2, 3, 1).reshape(-1, 1, T), kernel_size=wind_size*2 + 1, padding=wind_size, stride=1)
                #         k_mask = ((k_min + k_max) / 2).view(B, H, D, T).permute(0, 3, 1, 2).contiguous()
                #         del k_max, k_min
                #     else:
                #         k_mask = k_mask_original

                assert q.shape[1] <= BDST * BLOCK_SIZE_Q
                if (
                    os.getenv("HIP_DEBUG_TOPKMEAN", "0") == "1"
                    and (i_stage == 0)
                    and (BDST > 1)
                    and ((q.shape[1] % BLOCK_SIZE_Q) == 0)
                    and (args.position_ids.shape[0] == 1)
                ):
                    debug_topk_window = int(os.getenv("HIP_DEBUG_TOPK_WINDOW", "8"))
                    k_dense = args.gather_k_from_paged_cache(
                        chunk_size=chunk_size, disable_gqa=True, gqa_q=q
                    )
                    scores = torch.matmul(
                        q.permute(0, 2, 1, 3), k_dense.permute(0, 2, 3, 1)
                    )[:, args.sink_token_size : -args.sliding_window_size, :, :]
                    mask = (
                        args.position_ids[0][:, None]
                        >= (
                            args.sink_token_size
                            + torch.arange(
                                0, k_dense.shape[1], dtype=q.dtype, device=q.device
                            )
                        )[None, :]
                    )[None, None, :, :]
                    scores = torch.where(mask, scores, -32000.0)
                    scores = scores.view(
                        scores.shape[0],
                        scores.shape[1],
                        scores.shape[2] // BLOCK_SIZE_Q,
                        BLOCK_SIZE_Q,
                        scores.shape[3] // chunk_size,
                        chunk_size,
                    )
                    scores = torch.amax(scores, dim=3)
                    topk_scores, _ = torch.topk(scores, dim=-1, k=debug_topk_window)
                    scores = topk_scores.mean(dim=-1)
                    scores = scores.permute(0, 2, 1, 3)
                    out_scores[:, :, :, : scores.shape[-1]] = scores
                elif (
                    os.getenv("HIP_DEBUG_SOFTMAXMEAN", "0") == "1"
                    and (i_stage == 0)
                    and (BDST > 1)
                    and ((q.shape[1] % BLOCK_SIZE_Q) == 0)
                    and (args.position_ids.shape[0] == 1)
                ):

                    def rotate_half(vec):
                        # assert len(vec.shape) == 1
                        out = torch.zeros_like(vec)
                        x1 = vec[..., : vec.shape[-1] // 2]
                        x2 = vec[..., vec.shape[-1] // 2 :]
                        out[..., : vec.shape[-1] // 2] = -x2
                        out[..., vec.shape[-1] // 2 :] = x1
                        return out

                    def apply_rope(vec, cos, sin):
                        vec_rope = (vec * cos) + (rotate_half(vec) * sin)
                        return vec_rope

                    k_dense = args.gather_k_from_paged_cache(
                        chunk_size=chunk_size, disable_gqa=True, gqa_q=q
                    )[:, args.sink_token_size : -args.sliding_window_size, :, :]
                    k_dense = apply_rope(
                        k_dense,
                        args.rope_cos[None, 0 : 0 + 1, None, :],
                        args.rope_sin[None, 0 : 0 + 1, None, :],
                    )
                    q_dense = apply_rope(
                        q,
                        args.rope_cos[None, 1024 : 1024 + 1, None, :],
                        args.rope_sin[None, 1024 : 1024 + 1, None, :],
                    )
                    scores = torch.matmul(
                        q_dense.permute(0, 2, 1, 3), k_dense.permute(0, 2, 3, 1)
                    )
                    mask = (
                        args.position_ids[0][:, None]
                        >= (
                            args.sink_token_size
                            + torch.arange(
                                0,
                                k_dense.shape[1],
                                dtype=q_dense.dtype,
                                device=q_dense.device,
                            )
                        )[None, :]
                    )[None, None, :, :]
                    scores = torch.where(mask, scores, -32000.0).float()
                    scores = scores.softmax(dim=-1)
                    scores = torch.where(mask, scores, -32000.0)
                    scores = scores.view(
                        scores.shape[0],
                        scores.shape[1],
                        scores.shape[2] // BLOCK_SIZE_Q,
                        BLOCK_SIZE_Q,
                        scores.shape[3] // chunk_size,
                        chunk_size,
                    )
                    scores = (
                        scores.permute(0, 1, 2, 4, 3, 5)
                        .contiguous()
                        .view(
                            scores.shape[0],
                            scores.shape[1],
                            scores.shape[2],
                            scores.shape[4],
                            -1,
                        )
                    )
                    mask = scores > -30000.0
                    scores = (scores * mask).sum(dim=-1) / (
                        mask.float().sum(dim=-1) + 1e-12
                    )
                    scores.masked_fill_(mask.float().sum(dim=-1) == 0, -32000.0)
                    scores = scores.permute(0, 2, 1, 3)
                    # print(scores[0,:,0,:])
                    out_scores[:, :, :, : scores.shape[-1]] = scores
                elif (
                    os.getenv("HIP_DEBUG_FLATTENMEAN", "0") == "1"
                    and (i_stage == 0)
                    and (BDST > 1)
                    and ((q.shape[1] % BLOCK_SIZE_Q) == 0)
                    and (args.position_ids.shape[0] == 1)
                ):

                    def rotate_half(vec):
                        # assert len(vec.shape) == 1
                        out = torch.zeros_like(vec)
                        x1 = vec[..., : vec.shape[-1] // 2]
                        x2 = vec[..., vec.shape[-1] // 2 :]
                        out[..., : vec.shape[-1] // 2] = -x2
                        out[..., vec.shape[-1] // 2 :] = x1
                        return out

                    def apply_rope(vec, cos, sin):
                        vec_rope = (vec * cos) + (rotate_half(vec) * sin)
                        return vec_rope

                    k_dense = args.gather_k_from_paged_cache(
                        chunk_size=chunk_size, disable_gqa=True, gqa_q=q
                    )[:, args.sink_token_size : -args.sliding_window_size, :, :]
                    k_dense = apply_rope(
                        k_dense,
                        args.rope_cos[None, 0 : 0 + 1, None, :],
                        args.rope_sin[None, 0 : 0 + 1, None, :],
                    )
                    q_dense = apply_rope(
                        q,
                        args.rope_cos[None, 1024 : 1024 + 1, None, :],
                        args.rope_sin[None, 1024 : 1024 + 1, None, :],
                    )
                    scores = torch.matmul(
                        q_dense.permute(0, 2, 1, 3), k_dense.permute(0, 2, 3, 1)
                    )
                    mask = (
                        args.position_ids[0][:, None]
                        >= (
                            args.sink_token_size
                            + torch.arange(
                                0,
                                k_dense.shape[1],
                                dtype=q_dense.dtype,
                                device=q_dense.device,
                            )
                        )[None, :]
                    )[None, None, :, :]
                    scores = torch.where(mask, scores, -32000.0)
                    scores = scores.view(
                        scores.shape[0],
                        scores.shape[1],
                        scores.shape[2] // BLOCK_SIZE_Q,
                        BLOCK_SIZE_Q,
                        scores.shape[3] // chunk_size,
                        chunk_size,
                    )
                    scores = (
                        scores.permute(0, 1, 2, 4, 3, 5)
                        .contiguous()
                        .view(
                            scores.shape[0],
                            scores.shape[1],
                            scores.shape[2],
                            scores.shape[4],
                            -1,
                        )
                    )
                    mask = scores > -30000.0
                    scores = (scores * mask).sum(dim=-1) / (
                        mask.float().sum(dim=-1) + 1e-12
                    )
                    scores.masked_fill_(mask.float().sum(dim=-1) == 0, -32000.0)
                    scores = scores.permute(0, 2, 1, 3)
                    # print(scores[0,:,0,:])
                    out_scores[:, :, :, : scores.shape[-1]] = scores
                elif (
                    os.getenv("HIP_DEBUG_FLATTENTOPKMEAN", "0") == "1"
                    and (i_stage == 0)
                    and (BDST > 1)
                    and ((q.shape[1] % BLOCK_SIZE_Q) == 0)
                    and (args.position_ids.shape[0] == 1)
                ):

                    def rotate_half(vec):
                        # assert len(vec.shape) == 1
                        out = torch.zeros_like(vec)
                        x1 = vec[..., : vec.shape[-1] // 2]
                        x2 = vec[..., vec.shape[-1] // 2 :]
                        out[..., : vec.shape[-1] // 2] = -x2
                        out[..., vec.shape[-1] // 2 :] = x1
                        return out

                    def apply_rope(vec, cos, sin):
                        vec_rope = (vec * cos) + (rotate_half(vec) * sin)
                        return vec_rope

                    debug_topk_window = int(os.getenv("HIP_DEBUG_TOPK_WINDOW", "8"))
                    k_dense = args.gather_k_from_paged_cache(
                        chunk_size=chunk_size, disable_gqa=True, gqa_q=q
                    )[:, args.sink_token_size : -args.sliding_window_size, :, :]
                    k_dense = apply_rope(
                        k_dense,
                        args.rope_cos[None, 0 : 0 + 1, None, :],
                        args.rope_sin[None, 0 : 0 + 1, None, :],
                    )
                    q_dense = apply_rope(
                        q,
                        args.rope_cos[None, 1024 : 1024 + 1, None, :],
                        args.rope_sin[None, 1024 : 1024 + 1, None, :],
                    )
                    scores = torch.matmul(
                        q_dense.permute(0, 2, 1, 3), k_dense.permute(0, 2, 3, 1)
                    )
                    mask = (
                        args.position_ids[0][:, None]
                        >= (
                            args.sink_token_size
                            + torch.arange(
                                0,
                                k_dense.shape[1],
                                dtype=q_dense.dtype,
                                device=q_dense.device,
                            )
                        )[None, :]
                    )[None, None, :, :]
                    scores = torch.where(mask, scores, -32000.0)
                    scores = scores.view(
                        scores.shape[0],
                        scores.shape[1],
                        scores.shape[2] // BLOCK_SIZE_Q,
                        BLOCK_SIZE_Q,
                        scores.shape[3] // chunk_size,
                        chunk_size,
                    )
                    scores = (
                        scores.permute(0, 1, 2, 4, 3, 5)
                        .contiguous()
                        .view(
                            scores.shape[0],
                            scores.shape[1],
                            scores.shape[2],
                            scores.shape[4],
                            -1,
                        )
                    )
                    topk_scores, _ = torch.topk(scores, dim=-1, k=debug_topk_window)
                    topk_scores_mask = topk_scores > -30000.0
                    scores = (topk_scores * topk_scores_mask).sum(dim=-1) / (
                        topk_scores_mask.float().sum(dim=-1) + 1e-12
                    )
                    scores = torch.where(
                        topk_scores_mask.int().sum(dim=-1) != 0, scores, -32000.0
                    )
                    scores = scores.permute(0, 2, 1, 3)
                    out_scores[:, :, :, : scores.shape[-1]] = scores
                else:
                    chunk_controllable_sampling_mask_cuda[grid](
                        q,
                        *q.stride(),
                        k_mask,
                        *safe_stride(k_mask, 4),
                        position_ids,
                        *position_ids.stride(),
                        *args.args_paged_kv_cache(disable_cache=k_mask is not None),
                        *args.args_offload_cache(
                            True, disable_cache=k_mask is not None
                        ),
                        indices_left,
                        *indices_left.stride(),
                        indices_right,
                        *indices_right.stride(),
                        out_scores,
                        *out_scores.stride(),
                        args.rope_cos,
                        *safe_stride(args.rope_cos, 2),
                        args.rope_sin,
                        *safe_stride(args.rope_sin, 2),
                        mask_access_counter,
                        *safe_stride(mask_access_counter, 3),
                        mask_cache_miss_counter,
                        *safe_stride(mask_cache_miss_counter, 3),
                        chunk_count,
                        MAX_TSRC,
                        q.shape[1],
                        HEAD,
                        args.sliding_window_size,
                        args.sink_token_size,
                        # model_context_length if (not scan_extend_backend == 'streaming') else 0,
                        args.model_context_length,
                        group_jobs,
                        njobs,
                        BLOCK_HID=BLOCK_HID,
                        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                        STRIDE_Q=stage_block_stride_q,
                        BLOCK_CHUNK=BLOCK_CHUNK,
                        HEAD_GROUP=HEAD // HEAD_KV,
                        USING_EXTEND=args.using_extend,
                        EXTEND_BACKEND=extend_backend,
                        NEED_APPLY_ROPE=args.need_apply_rope,
                        TERMINATE_SIZE=args.stage_early_terminate,
                        SCAN_STRIDE=STAGE_STRIDE,
                        UPDATE_CACHE=args.online_update_cache,
                        ORACLE_MAXIMUM=False,  # NOTE: seems has bug... but why?
                    )

                # TODO: OPTIMIZE THIS. Add head unified version of HiP.
                if os.getenv("HIP_HEAD_REDUCE", "1") == "1":
                    ori_shape = out_scores.shape
                    # out_scores = out_scores.softmax(dim=2) # NOTE: not good idea
                    out_scores, _ = torch.max(out_scores, keepdim=True, dim=2)
                    out_scores = torch.broadcast_to(out_scores, ori_shape).contiguous()

                if args.offload_cache is not None:
                    # print('after masking')
                    args.offload_cache.mask_k_cache.verify_cache()
            elif isinstance(stage_info, EvalScoreStage):
                raise Exception()  # TODO: handle new args
                extend_backend = (
                    args.scan_extend_backend
                    if stage_info.stage_extend_backend is None
                    else stage_info.stage_extend_backend
                )

                grid = (
                    BSZ
                    * triton.cdiv(BDST, stage_info.stage_stride)
                    * HEAD,  # SCAN_STRIDE = 1
                )
                calculate_chunk_score[grid](
                    q_mask,
                    *q_mask.stride(),
                    k_mask,
                    *safe_stride(k_mask, 4),
                    position_ids,
                    *position_ids.stride(),
                    args.rope_cos,
                    *safe_stride(args.rope_cos, 2),
                    args.rope_sin,
                    *safe_stride(args.rope_sin, 2),
                    *args.args_paged_kv_cache(),
                    *args.args_offload_cache(True),
                    indices_left,
                    *indices_left.stride(),
                    indices_right,
                    *indices_right.stride(),
                    out_scores,
                    *out_scores.stride(),
                    # model_context_length if (not scan_extend_backend == 'streaming') else 0,
                    args.model_context_length,
                    args.sliding_window_size,
                    args.sink_token_size,
                    chunk_size,
                    TDST,
                    BDST,
                    triton.cdiv(BDST, stage_info.stage_stride),  # SCAN STRIDE == 1
                    HEAD,
                    chunk_count,
                    HEAD // HEAD_KV,
                    USING_EXTEND=args.using_extend,
                    NEED_APPLY_ROPE=args.need_apply_rope,
                    EXTEND_BACKEND=extend_backend,
                    BLOCK_HID=BLOCK_HID,
                    BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                    BLOCK_STRIDE_Q=stage_block_stride_q,
                    BLOCK_SIZE_K=args.stage_early_terminate,
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
                apply_v_dot = os.getenv("APPLY_V_DOT", "0") == "1"
                # apply_v_dot = apply_v_dot and (i_stage == (len(stages) - 1))
                apply_v_dot = apply_v_dot and (i_stage != 0)
                if apply_v_dot:
                    v_scores = torch.zeros_like(out_scores, dtype=torch.float32)
                    V_BLOCK_SIZE_K = 8
                    V_BLOCK_STRIDE_Q = 1
                    V_BLOCK_STRIDE_K = 1
                    V_GROUP_K = 64 // V_BLOCK_SIZE_K
                    # V_GROUP_K = indices_left.shape[3]
                    grid = (
                        v_scores.shape[0]
                        * v_scores.shape[1]
                        * v_scores.shape[2]
                        * triton.cdiv(indices_left.shape[3], V_GROUP_K),
                    )
                    compute_v_cos[grid](
                        v,
                        *safe_stride(v, 4),
                        indices_left,
                        *indices_left.stride(),
                        position_ids,
                        *position_ids.stride(),
                        v_scores,
                        *v_scores.stride(),
                        *args.args_paged_kv_cache(),
                        *args.args_offload_cache(is_masking=True),
                        sa_access_counter,
                        *safe_stride(sa_access_counter, 3),
                        sa_cache_miss_counter,
                        *safe_stride(sa_cache_miss_counter, 3),
                        TDST,
                        MAX_TSRC,
                        HEAD,
                        indices_left.shape[3],
                        HEAD_GROUP=HEAD // HEAD_KV,
                        GROUP_K=V_GROUP_K,
                        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                        BLOCK_SIZE_K=V_BLOCK_SIZE_K,
                        BLOCK_STRIDE_Q=V_BLOCK_STRIDE_Q,
                        BLOCK_STRIDE_K=V_BLOCK_STRIDE_K,
                        BLOCK_HID=q.shape[-1],
                    )

                    if out_scores.dtype != torch.float32:
                        out_scores = out_scores.to(torch.float32)
                    out_scores = (
                        out_scores - out_scores.min(dim=-1, keepdim=True).values
                    )

                    # print(indices_left[0, -1, DEBUG_HEAD, :])
                    # print(out_scores[0, -1, DEBUG_HEAD, :])
                    # print(v_scores[0, -1, DEBUG_HEAD, :])

                    if DEBUG and DEBUG_RENDER:
                        img = v_scores[0, :, DEBUG_HEAD, :].cpu().float().numpy()
                        plt.clf()
                        plt.imshow(img)
                        plt.colorbar()
                        plt.savefig("dummy_v_scores.png")

                    # out_scores = torch.where(
                    #     torch.isnan(v_scores),
                    #     out_scores,
                    #     out_scores * v_scores
                    # )

                    # out_scores = out_scores * v_scores

                    out_scores = out_scores + v_scores

                if i_stage < (len(args.stages) - 1):
                    # print(indices_left.shape, (stages[i_stage + 1].stage_k // stages[i_stage + 1].stage_chunk_size))
                    next_stage_k = (
                        args.stages[i_stage + 1].stage_k
                        // args.stages[i_stage].stage_chunk_size
                    )
                    next_stage_k = min(next_stage_k, indices_left.shape[-1])
                    _, t_indices = out_scores[..., : indices_left.shape[-1]].topk(
                        k=next_stage_k,
                        dim=-1,
                        sorted=False,
                        largest=True,
                    )
                else:
                    _, t_indices = out_scores[..., : indices_left.shape[-1]].sort(
                        dim=-1, descending=True, stable=False
                    )
                indices_left = indices_left.gather(dim=-1, index=t_indices)
                indices_right = indices_right.gather(dim=-1, index=t_indices)

            if DEBUG and DEBUG_RENDER and not torch.cuda.is_current_stream_capturing():
                if (i_stage + 1) < len(args.stages):
                    next_stage_k = args.stages[i_stage + 1].stage_k
                else:
                    next_stage_k = args.second_stage_k
                out_indices_cpu = (
                    indices_left.repeat_interleave(STAGE_STRIDE, 1)[:, -BDST:]
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                debug = np.zeros(
                    (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q))
                )
                render_plot_dynamic(
                    out_indices_cpu,
                    debug,
                    DEBUG_HEAD,
                    BLOCK_SIZE_Q,
                    next_stage_k,
                    chunk_size,
                    causal_mask=True,
                    sliding_window_size=args.sliding_window_size,
                )
                cv2.imwrite(f"dummy_sampled_stage_{i_stage}.png", debug * 255)
                # print(f'saved dummy_sampled_stage_{i_stage}.png')

        if STAGE_STRIDE > 1:
            indices_left = indices_left.repeat_interleave(STAGE_STRIDE, 1)[
                :, -BDST:
            ].contiguous()
            indices_right = indices_right.repeat_interleave(STAGE_STRIDE, 1)[
                :, -BDST:
            ].contiguous()
            out_scores = out_scores.repeat_interleave(STAGE_STRIDE, 1)[
                :, -BDST:
            ].contiguous()

        assert (args.second_stage_k % chunk_size) == 0
        # if DEBUG:
        #     print('indices_left', indices_left[0, -1])
        #     print('out_scores', out_scores[0, -1], args.second_stage_k, indices_left.shape, chunk_size)
        indices = (
            indices_left[..., : args.second_stage_k // chunk_size]
            // chunk_size
            * chunk_size
        )

        # NOTE: union head masks
        if os.getenv("HIP_DEBUG_UNION_HEAD", "0") == "1":
            assert os.getenv("HIP_HEAD_REDUCE", "1") == "0"
            # args.disable_flashdecode = True
            # B BDST H CHUNK
            indices = indices.flatten(-2, -1).unsqueeze(-2).repeat(1, 1, HEAD, 1)

        # NOTE: sampled indices might be delayed
        if os.getenv("HIP_DEBUG_ADD_DELAY_WINDOW", "0") == "1":
            delayed_indices = [
                indices,
            ]
            delay_window = 64
            for i_delay in range(0, delay_window, chunk_size):
                delayed_indices.append(indices - i_delay - chunk_size)
            # print(indices.shape)
            indices = torch.cat(delayed_indices, dim=-1)
            # print(indices.shape)

        # NOTE: performing SnapKV
        if (os.getenv("HIP_DEBUG_SNAP_KV", "0") == "1") and (BDST > 1):
            is_paged = False
            if k_mask_original is None:
                is_paged = True
                k_mask = args.gather_k_from_paged_cache(chunk_size=chunk_size)
            else:
                k_mask = k_mask_original
            scores = torch.matmul(
                q.permute(0, 2, 1, 3)[:, :, -128:, :],
                k_mask.permute(0, 2, 3, 1).repeat(1, HEAD // HEAD_KV, 1, 1),
            )
            # if is_paged:
            tsrcs = torch.arange(0, scores.shape[-1], device=q.device)
            tsrc_mask = tsrcs[None, :] > args.position_ids[:, -1, None]
            scores = scores.masked_fill_(tsrc_mask[:, None, None, :], float("-inf"))
            scores = scores.amax(dim=-2)  # B H TSRC
            snap_window = 127
            scores = torch.nn.functional.max_pool1d(
                scores, kernel_size=snap_window * 2 + 1, stride=1, padding=snap_window
            )
            scores = scores.view(scores.shape[0], scores.shape[1], -1, chunk_size)
            scores = scores.amax(dim=-1)
            # print(scores.shape)
            _, snap_indices = scores.topk(
                k=min(scores.shape[-1], 131072 // chunk_size), dim=-1
            )
            snap_indices = snap_indices * chunk_size
            snap_indices = snap_indices.unsqueeze(1).expand(
                snap_indices.shape[0],
                indices.shape[1],
                snap_indices.shape[1],
                snap_indices.shape[2],
            )
            indices = torch.concat([indices, snap_indices], dim=-1)
            if is_paged:
                k_mask = None

        # NOTE: add sliding window indices
        if args.sliding_window_indices is not None:
            sw_indices = (args.sliding_window_indices // chunk_size) * chunk_size
            assert position_ids.shape == (BSZ, TDST), position_ids.shape
            assert sw_indices.shape[0] == HEAD, sw_indices.shape
            args.disable_flashdecode = True
            warnings.warn("Flash Decode is disabled due to experimental feature")
            sw_indices = (
                position_ids[:, ::BLOCK_SIZE_Q, None, None]
                + sw_indices[None, None, :, :]
            )
            sw_indices = (sw_indices // chunk_size) * chunk_size
            sw_indices.clamp_min_(0)
            indices = torch.concat([indices, sw_indices], dim=-1)

        # NOTE: adding important Ks
        if (os.getenv("HIP_DEBUG_IMPORTANT_K", "0") == "1") and (BDST > 1):
            k_seq = args.gather_k_from_paged_cache(chunk_size=chunk_size)
            k_bos = k_seq[:, :1, :, :].contiguous().permute(0, 2, 1, 3)
            k_seq = k_seq.permute(0, 2, 1, 3)
            k_seq = k_seq / k_seq.square().sum(dim=-1, keepdim=True).sqrt()
            k_bos = k_bos / k_bos.square().sum(dim=-1, keepdim=True).sqrt()
            scores = torch.matmul(k_bos, k_seq.permute(0, 1, 3, 2)).squeeze(2)  # B H T
            tsrcs = torch.arange(0, scores.shape[-1], device=q.device)
            tsrc_mask = (
                tsrcs[None, :] + original_sliding_window_size
            ) > args.position_ids[:, -1, None]
            scores.masked_fill_(tsrc_mask[:, None, :], float("-inf"))
            scores[:, :, : args.sink_token_size].fill_(float("-inf"))
            scores = scores.view(scores.shape[0], scores.shape[1], -1, chunk_size)
            # scores = scores.amax(dim=1, keepdim=True)
            scores = scores.amax(dim=-1)
            _, important_indices = torch.topk(scores, k=8192 // chunk_size, dim=-1)
            important_indices = (
                important_indices.repeat_interleave(
                    HEAD // important_indices.shape[1], 1
                )
                * chunk_size
            )
            important_indices = important_indices.unsqueeze(1).expand(
                important_indices.shape[0],
                indices.shape[1],
                important_indices.shape[1],
                important_indices.shape[2],
            )
            indices = torch.concat([indices, important_indices], dim=-1)

        if (
            DEBUG
            and DEBUG_RENDER
            and not torch.cuda.is_current_stream_capturing()
            and (BDST > 10)
        ):
            out_indices_cpu = indices.cpu().numpy()
            debug = np.zeros(
                (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q))
            )
            render_plot(out_indices_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q)
            cv2.imwrite("dummy_sampled_final.png", debug * 255)
            # print('saved dummy_sampled_final.png')

        args = args.clone()
        args.block_size_q = args.stages[-1].stage_block_size_q
        block_sparse_block_size_q = min(
            args.block_sparse_block_size_q, args.block_size_q
        )
        args.sliding_window_size += args.mask_k
        args.block_size_k = chunk_size
        args.mask_k = args.second_stage_k
        args.using_extend = args.using_extend and True

        # NOTE: convert format and taking unique in indices
        indices = indices.permute(0, 2, 1, 3).flatten(0, 1)

        indices, t_sort_1 = indices.sort(dim=-1)
        indices = indices // args.block_size_k * args.block_size_k

        unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
        indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
        indices, t_sort_2 = indices.sort(dim=-1)
        active_mask = indices < (
            position_ids[:, :: args.block_size_q, None].repeat_interleave(HEAD, 0)
            + args.block_size_q
        )
        ks = active_mask.int().sum(-1)
        ks_count = ks.unsqueeze(-1)
        ks_start_end = torch.zeros(
            (ks.shape[0], ks.shape[1], 2), dtype=torch.int32, device=q.device
        )
        ks_start_end[:, :, -1] = ks

        # print(args.layer_id, round(ks.float().mean().item() * args.block_size_k))

        if (args.low_percent > 0) and (args.low_k_ratio < 1):
            scores = (
                out_scores[..., : args.second_stage_k // chunk_size]
                .permute(0, 2, 1, 3)
                .flatten(0, 1)
            )
            scores = scores.gather(dim=-1, index=t_sort_1)
            scores = scores.gather(dim=-1, index=t_sort_2)
            scores = torch.where(active_mask, scores, -32000.0)

            masked_scores = torch.where(scores > -16000.0, scores, 0)
            # masked_scores = torch.softmax(scores, dim=-1)
            scores_std, scores_mean = torch.std_mean(masked_scores, dim=-1)

            # TODO: TEST SENSITIVITY

            if dim_to_lower == "head":
                dim_to_lower = 0
                values_to_sort = (scores_std).mean(dim=1)
            elif dim_to_lower == "seq":
                dim_to_lower = 1
                values_to_sort = scores_std
            else:
                raise Exception()

            _, lowk = values_to_sort.topk(
                k=int(scores_mean.shape[dim_to_lower] * args.low_percent),
                dim=dim_to_lower,
                largest=False,
                sorted=False,
            )
            # print(lowk[:, -1])
            if lowk.ndim == 2:
                lowk = lowk[:, :, None].expand(-1, -1, scores.shape[-1])
            if lowk.ndim == 1:
                lowk = lowk[:, None, None].expand(
                    -1, scores.shape[-2], scores.shape[-1]
                )
            _, t_sort_score = torch.topk(
                scores.gather(dim=dim_to_lower, index=lowk),
                dim=-1,
                k=int(scores.shape[-1] * (1 - args.low_k_ratio)),
                largest=False,
            )
            # print(t_sort_score.shape)
            N, BDST = scores_mean.shape
            indices.scatter_(
                dim=dim_to_lower,
                index=lowk,
                src=indices.gather(dim=dim_to_lower, index=lowk).scatter(
                    dim=-1, index=t_sort_score, value=987654321
                ),
            )
            indices, t_sort_2 = indices.sort(dim=-1)
            active_mask = indices < (
                position_ids[:, :: args.block_size_q, None].repeat_interleave(HEAD, 0)
                + args.block_size_q
            )
            # print(indices[1, -1, :])
            # print(active_mask[1, -1, :])
            ks = active_mask.int().sum(-1)
            ks_count = ks.unsqueeze(-1)
            ks_start_end = torch.zeros(
                (ks.shape[0], ks.shape[1], 2), dtype=torch.int32, device=q.device
            )
            ks_start_end[:, :, -1] = ks

            if (
                DEBUG
                and DEBUG_RENDER
                and not torch.cuda.is_current_stream_capturing()
                and (BDST > 10)
            ):
                indices_cpu = indices.cpu().numpy()
                ks_cpu = ks.cpu().numpy()
                debug = np.zeros(
                    (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q))
                )
                render_plot_ks(indices_cpu, ks_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q)
                cv2.imwrite("dummy_sampled_final_lowk.png", debug * 255)
                print("saved dummy_sampled_final_lowk.png", DEBUG_HEAD)

                # print(ks[:, -1])

                plt.clf()
                plt.plot(scores_std[:3, :].float().cpu().numpy().T)
                # plt.ylim(0, 0.01)
                plt.savefig("dummy_stat_std.png")
                plt.clf()
                plt.plot(scores_mean[:3, :].float().cpu().numpy().T)
                plt.savefig("dummy_stat_mean.png")
                plt.clf()
                plt.plot(ks[DEBUG_HEAD, :].float().cpu().numpy())
                plt.savefig("dummy_stat_ks.png")

        if (
            DEBUG
            and DEBUG_RENDER
            and not torch.cuda.is_current_stream_capturing()
            and (BDST > 10)
        ):
            try:
                input(">>>")
            except EOFError:
                # time.sleep(1)
                pass

        # NOTE: break-down to fit BSA block size
        if (block_sparse_block_size_q is not None) and (
            triton.cdiv(TDST, block_sparse_block_size_q)
            != triton.cdiv(TDST, args.block_size_q)
        ):
            assert (BLOCK_SIZE_Q % block_sparse_block_size_q) == 0
            indices = indices.repeat_interleave(
                BLOCK_SIZE_Q // block_sparse_block_size_q, 1
            )
            ks = ks.repeat_interleave(BLOCK_SIZE_Q // block_sparse_block_size_q, 1)
            ks_count = ks_count.repeat_interleave(
                BLOCK_SIZE_Q // block_sparse_block_size_q, 1
            )
            ks_start_end = ks_start_end.repeat_interleave(
                BLOCK_SIZE_Q // block_sparse_block_size_q, 1
            )
            args.block_size_q = block_sparse_block_size_q

        if args.mask_only:
            return None, None
    else:
        args = args.clone()
        args.sliding_window_size += args.mask_k
        args.block_size_k = args.stages[-1].stage_chunk_size
        args.mask_k = args.second_stage_k
        args.using_extend = args.using_extend and True

        assert cached_metadata is not None
        indices = cached_metadata.indices.clone()
        ks = cached_metadata.ks.clone()
        ks_count = cached_metadata.ks_count.clone()
        ks_start_end = cached_metadata.ks_start_end.clone()

    args.block_size_q = min(args.block_size_q, triton.next_power_of_2(TDST))

    if args.sliding_window_size == 777:
        args.sliding_window_size = (
            args.model_context_length
            - args.sink_token_size
            - args.second_stage_k
            - args.block_size_q
        )

    block_sparse_attention_backend = block_sparse_attention

    # Use flashdecode
    if (
        (TDST == 1)
        and (not os.environ.get("HIP_DISABLE_FLASHDECODE", "0") == "1")
        and (not args.disable_flashdecode)
    ):
        block_sparse_attention_backend = decode_block_sparse_attention

    context = block_sparse_attention_backend(
        q=q_bsa,
        k=k,
        v=v,
        seq_lens=position_ids[:, -q_bsa.shape[1] :] + 1,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args,
        access_counter=sa_access_counter,
        cache_miss_counter=sa_cache_miss_counter,
        EXTEND_BACKEND=args.sa_extend_backend,  # streaming works way much better in Gemma2, than dynamic_extend
        model_context_length=args.model_context_length,
        extend_context_length=args.extend_context_length,
        offload_update_cache=(cached_metadata is None) and args.online_update_cache,
        # offload_update_cache=args.online_update_cache,
        # offload_update_cache=False,
    )
    if args.offload_cache is not None:
        args.offload_cache.sa_kv_cache.verify_cache()

    # if DEBUG:
    #     print('context', context[0, :, DEBUG_HEAD, :], context.shape)
    #     print('indices', indices[0 + DEBUG_HEAD, -1], indices.shape)
    #     print('ks', ks[0 + DEBUG_HEAD, -1], ks.shape)

    return context, HiPAttentionOutputMetadata(
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        mask_cache_statistics=(
            HiPAttentionCacheAccessStatistics(
                access_counter=mask_access_counter,
                cache_miss_counter=mask_cache_miss_counter,
            )
            if (cached_metadata is None) or (cached_metadata.indices is None)
            else None
        ),
        sa_cache_statistics=HiPAttentionCacheAccessStatistics(
            access_counter=sa_access_counter,
            cache_miss_counter=sa_cache_miss_counter,
        ),
        stage_caches=stage_caches,
    )
