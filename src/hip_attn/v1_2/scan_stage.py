import os
import warnings

import torch
import triton
import triton.language as tl

from hip_attn.utils.rope import adjust_rope
from hip_attn.v1_2.attention_metadata import safe_stride
from hip_attn.v1_2.utils import capture
from hip_attn.v1_2.uvm_gpu_cache import load_tokens


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
    queries_dtype,
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
    BLOCK_HID: tl.constexpr,
    HID_DIM,
    IS_RIGHT,
    HEAD_KV,
    UPDATE_CACHE,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
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
        HID_DIM,
        UPDATE_CACHE=UPDATE_CACHE,
    ).to(queries_dtype)

    if USING_EXTEND:
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
                queries_dtype
            )
        else:
            flip = tl.where(idx_rope_range & 1 == 0, 1, -1)
            rope_rot_idx = tl.where(
                rope_mask,
                idx_rope_range + flip + rope_range_begin,
                idx_hid,
            )
            cos_sin_idx = idx_rope_range // 2
            rope_mult = ((idx_rope_range % 2 == 0) * (-2) + 1).to(queries_dtype)

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
                    HID_DIM,
                    NEED_APPLY_ROPE,
                    rope_range_begin,
                    rope_range_end,
                    rope_is_neox_style,
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
                    rope_rot_idx[:, None],
                    mask_tsrc_active[None, :],
                    HEAD_KV,
                    BLOCK_CHUNK,
                    BLOCK_HID,
                    HID_DIM,
                    # NOTE: in previous load, the fetch should be succesfully done.
                    UPDATE_CACHE=UPDATE_CACHE,
                ).to(queries_dtype)

                # TODO: multiply -right
                # keys_left_rot = tl.where(
                #     (idx_hid + BLOCK_HID // 2)[:, None] < BLOCK_HID,
                #     -keys_left_rot,
                #     keys_left_rot
                # )

                keys_left_rot *= rope_mult[:, None]

                cos_new = tl.load(
                    COS
                    + new_tsrc[None, :].to(tl.int64) * stride_cos_t
                    + cos_sin_idx[:, None] * stride_cos_hid,
                    mask=mask_tsrc_active[None, :] & rope_mask[:, None],
                    other=0.0,
                ).to(keys_left.dtype)
                sin_new = tl.load(
                    SIN
                    + new_tsrc[None, :].to(tl.int64) * stride_sin_t
                    + cos_sin_idx[:, None] * stride_sin_hid,
                    mask=mask_tsrc_active[None, :] & rope_mask[:, None],
                    other=0.0,
                ).to(keys_left.dtype)

                keys_left = tl.where(
                    rope_mask[:, None],
                    keys_left * cos_new + keys_left_rot * sin_new,
                    keys_left,
                )

    return keys_left


@triton.jit
def pool_queries(
    idx_bsz,
    idx_head,
    pos_tdst,
    idx_tdst,
    mask_tdst,
    idx_hid,
    mask_hid,
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
    HID_DIM: int,
    TDST: int,
    CHUNK_COUNT: int,
    real_pos_tdst_min: int,
    model_context_length: int,
    sliding_window_size: int,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    HID_BLOCK: tl.constexpr,
    STRIDE_Q: tl.constexpr,
):
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
        rope_mult = (idx_rope_range + ROPE_DIM // 2 < ROPE_DIM) * (-2) + 1
    else:
        flip = tl.where(idx_rope_range & 1 == 0, 1, -1)
        rope_rot_idx = tl.where(
            rope_mask,
            idx_rope_range + flip + rope_range_begin,
            idx_hid,
        )
        cos_sin_idx = idx_rope_range // 2
        rope_mult = (idx_rope_range % 2 == 0) * (-2) + 1

    queries_sum = tl.zeros((BLOCK_SIZE_Q // STRIDE_Q, HID_BLOCK), dtype=tl.float32)
    queries_counter = tl.zeros((BLOCK_SIZE_Q // STRIDE_Q,), dtype=tl.int32)
    tl.static_assert(BLOCK_SIZE_Q // STRIDE_Q > 0)

    for i_offset in tl.range(0, STRIDE_Q, num_stages=3):
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
            queries_iter = queries_iter.to(tl.bfloat16)

        if USING_EXTEND:
            if NEED_APPLY_ROPE or (real_pos_tdst_min >= model_context_length):
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
                    new_tdst = tl.minimum(pos_tdst, CHUNK_COUNT + sliding_window_size)
                else:
                    raise Exception()

                if NEED_APPLY_ROPE:
                    queries_rot = tl.load(
                        Q
                        + idx_bsz * stride_q_bsz
                        + idx_tdst_iter[:, None] * stride_q_tdst
                        + idx_head * stride_q_head
                        + rope_rot_idx[None, :] * stride_q_hid,
                        mask=mask_tdst_iter[:, None]
                        & rope_mask[None, :]
                        & mask_hid[None, :],
                        other=0.0,
                    )
                    if queries_rot.dtype == tl.float8e5:
                        queries_rot = queries_rot.to(tl.bfloat16)

                    cos_new = tl.load(
                        COS
                        + new_tdst[:, None].to(tl.int64) * stride_cos_t
                        + cos_sin_idx[None, :] * stride_cos_hid,
                        mask=mask_tdst_iter[:, None]
                        & rope_mask[None, :]
                        & mask_hid[None, :],
                        other=0.0,
                    ).to(queries_iter.dtype)
                    sin_new = tl.load(
                        SIN
                        + new_tdst[:, None].to(tl.int64) * stride_sin_t
                        + cos_sin_idx[None, :] * stride_sin_hid,
                        mask=mask_tdst_iter[:, None]
                        & rope_mask[None, :]
                        & mask_hid[None, :],
                        other=0.0,
                    ).to(queries_iter.dtype)

                    queries_rot *= rope_mult[None, :].to(queries_rot.dtype)

                    queries_iter = tl.where(
                        rope_mask[None, :] & mask_hid[None, :],
                        (queries_iter * cos_new + queries_rot * sin_new).to(
                            queries_iter.dtype
                        ),
                        queries_iter,
                    )
                else:
                    raise Exception()
                    queries_iter = adjust_rope(
                        queries_iter,
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
                        HID_BLOCK,
                        HID_DIM,
                        NEED_APPLY_ROPE,
                        rope_range_begin,
                        rope_range_end,
                        rope_is_neox_style,
                    ).to(queries_iter.dtype)
                    queries_iter = (queries_iter * mask_tdst_iter[:, None]).to(
                        queries_iter.dtype
                    )

        queries_sum += queries_iter
        queries_counter += mask_tdst_iter.to(tl.int32)

    queries = (queries_sum / (queries_counter[:, None] + 1e-12)) * mask_tdst[:, None]
    if Q.dtype.element_ty != tl.float8e5:
        queries = queries.to(Q.dtype.element_ty)
    else:
        queries = queries.to(tl.bfloat16)

    return queries


def get_scan_stage_configs():
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
    for LOAD_Q_EACH_TIME in [
        False,
    ]:
        for num_warps in NUM_WARPS:
            for num_stages in [1, 2, 3]:
                configs.append(
                    triton.Config(
                        {"LOAD_Q_EACH_TIME": LOAD_Q_EACH_TIME},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


def get_decode_scan_stage_configs():
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
    for num_warps in NUM_WARPS:
        for num_stages in [1, 2, 3]:
            configs.append(
                triton.Config(
                    {},
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            )
    return configs


@triton.autotune(
    configs=get_decode_scan_stage_configs(),
    key=[
        "BLOCK_SIZE_Q",
        "HID_DIM",
        "USING_PAGES",
    ],
    restore_value=[
        "INDICES_LEFT",
        "INDICES_RIGHT",
    ],
)
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
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
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
    HID_DIM: tl.constexpr,
    HID_BLOCK_0: tl.constexpr,
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
    LOAD_Q_EACH_TIME: tl.constexpr = False,
    COMPUTE_MLA_ROPE: tl.constexpr = False,
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)
    BDST_SCAN = tl.cdiv(BDST, SCAN_STRIDE)
    BCHUNK = tl.cdiv(CHUNK_COUNT, BLOCK_CHUNK)

    pid_group = tl.program_id(0).to(tl.int64)

    for i in range(group_jobs):
        pid = pid_group * group_jobs + i
        if pid < total_jobs:
            idx_head = pid % HEAD
            pid = pid // HEAD
            idx_bdst_scan = pid % BDST_SCAN
            pid = pid // BDST_SCAN
            idx_bchunk = pid % BCHUNK
            pid = pid // BCHUNK
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

            HID_BLOCK_1: tl.constexpr = HID_DIM - HID_BLOCK_0

            idx_hid_q0 = tl.arange(0, HID_BLOCK_0)
            mask_hid_q0 = idx_hid_q0 < HID_DIM

            if HID_BLOCK_1 > 0:
                idx_hid_q1 = HID_BLOCK_0 + tl.arange(0, HID_BLOCK_1)
                mask_hid_q1 = idx_hid_q1 < HID_DIM
            else:
                idx_hid_q1 = None
                mask_hid_q1 = None

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

            if (
                (Q.dtype.element_ty != tl.float8e5)
                & (Q.dtype.element_ty != tl.float8e4nv)
                & (Q.dtype.element_ty != tl.float8e4b8)
                & (Q.dtype.element_ty != tl.float8e4b15)
            ):
                q_dtype = Q.dtype.element_ty
            else:
                q_dtype = tl.bfloat16
            cq = (tl.sqrt(HID_DIM * 1.0) / tl.sqrt(tl.sqrt(HID_DIM * 1.0))).to(q_dtype)
            ck = (1.0 / tl.sqrt(tl.sqrt(HID_DIM * 1.0))).to(q_dtype)

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
                    max_chunk_size = tl.max(idx_tsrc_right - idx_tsrc_left).to(
                        tl.float32
                    )

                    scores = tl.zeros((BLOCK_CHUNK,), dtype=tl.float32) - 32000.0

                    if not LOAD_Q_EACH_TIME:
                        queries_0 = pool_queries(
                            idx_bsz,
                            idx_head,
                            pos_tdst,
                            idx_tdst,
                            mask_tdst,
                            idx_hid_q0,
                            mask_hid_q0,
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
                            HID_DIM,
                            TDST,
                            CHUNK_COUNT,
                            real_pos_tdst_min,
                            model_context_length,
                            sliding_window_size,
                            USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                            NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                            EXTEND_BACKEND,
                            BLOCK_SIZE_Q,
                            HID_BLOCK_0,
                            STRIDE_Q,
                        )

                        if HID_BLOCK_1 > 0:
                            queries_1 = pool_queries(
                                idx_bsz,
                                idx_head,
                                pos_tdst,
                                idx_tdst,
                                mask_tdst,
                                idx_hid_q1,
                                mask_hid_q1,
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
                                HID_DIM,
                                TDST,
                                CHUNK_COUNT,
                                real_pos_tdst_min,
                                model_context_length,
                                sliding_window_size,
                                USING_EXTEND,
                                NEED_APPLY_ROPE,
                                EXTEND_BACKEND,
                                BLOCK_SIZE_Q,
                                HID_BLOCK_1,
                                STRIDE_Q,
                            )
                        else:
                            queries_1 = None

                    matmul_dtype = q_dtype
                    # max_chunk_size
                    # while max_chunk_size >= TERMINATE_SIZE:
                    #     max_chunk_size /= 2.0
                    for _ in tl.range(
                        0,
                        tl.ceil(tl.log2(max_chunk_size / TERMINATE_SIZE)).to(tl.int32),
                        num_stages=1 if USING_EXTEND else 3,
                    ):
                        mask_tsrc_active = (
                            mask_chunk
                            & (idx_tsrc_left < idx_tsrc_right)
                            & (idx_tsrc_left <= pos_tdst_min)
                            & (idx_tsrc_left >= 0)
                        )
                        idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2

                        assert (
                            not ORACLE_MAXIMUM
                        ), "this is deprecated at fad90fea0d37ba88c04e90f2c5597e6800e97e8f"

                        idx_tsrc = (idx_tsrc_left + idx_tsrc_center) // 2

                        if LOAD_Q_EACH_TIME:
                            queries_0 = pool_queries(
                                idx_bsz,
                                idx_head,
                                pos_tdst,
                                idx_tdst,
                                mask_tdst,
                                idx_hid_q0,
                                mask_hid_q0,
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
                                HID_DIM,
                                TDST,
                                CHUNK_COUNT,
                                real_pos_tdst_min,
                                model_context_length,
                                sliding_window_size,
                                USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                                NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                                EXTEND_BACKEND,
                                BLOCK_SIZE_Q,
                                HID_BLOCK_0,
                                STRIDE_Q,
                            )

                        keys_left_0 = load_keys_with_rope(
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
                            q_dtype,
                            idx_bsz,
                            idx_tsrc,
                            idx_head // HEAD_GROUP,
                            idx_hid_q0,
                            idx_chunk,
                            mask_tsrc_active,
                            mask_tdst,
                            mask_hid_q0,
                            real_pos_tdst_min,
                            model_context_length,
                            num_sinks,
                            USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                            EXTEND_BACKEND,
                            NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                            BLOCK_CHUNK,
                            HID_BLOCK_0,
                            HID_DIM,
                            False,
                            HEAD // HEAD_GROUP,
                            UPDATE_CACHE,
                            rope_range_begin,
                            rope_range_end,
                            rope_is_neox_style,
                        )

                        scores_left = tl.dot(
                            (queries_0 * cq).to(matmul_dtype),
                            (keys_left_0.to(q_dtype) * ck).to(matmul_dtype),
                        ).to(scores.dtype)

                        if HID_BLOCK_1 > 0:
                            if LOAD_Q_EACH_TIME:
                                queries_1 = pool_queries(
                                    idx_bsz,
                                    idx_head,
                                    pos_tdst,
                                    idx_tdst,
                                    mask_tdst,
                                    idx_hid_q1,
                                    mask_hid_q1,
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
                                    HID_DIM,
                                    TDST,
                                    CHUNK_COUNT,
                                    real_pos_tdst_min,
                                    model_context_length,
                                    sliding_window_size,
                                    USING_EXTEND,
                                    NEED_APPLY_ROPE,
                                    EXTEND_BACKEND,
                                    BLOCK_SIZE_Q,
                                    HID_BLOCK_1,
                                    STRIDE_Q,
                                )

                            keys_left_1 = load_keys_with_rope(
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
                                q_dtype,
                                idx_bsz,
                                idx_tsrc,
                                idx_head // HEAD_GROUP,
                                idx_hid_q1,
                                idx_chunk,
                                mask_tsrc_active,
                                mask_tdst,
                                mask_hid_q1,
                                real_pos_tdst_min,
                                model_context_length,
                                num_sinks,
                                USING_EXTEND,
                                EXTEND_BACKEND,
                                NEED_APPLY_ROPE,
                                BLOCK_CHUNK,
                                HID_BLOCK_1,
                                HID_DIM,
                                False,
                                HEAD // HEAD_GROUP,
                                UPDATE_CACHE,
                                rope_range_begin,
                                rope_range_end,
                                rope_is_neox_style,
                            )

                            if COMPUTE_MLA_ROPE:
                                scores_left += tl.dot(
                                    (queries_1 * cq).to(matmul_dtype),
                                    (keys_left_1.to(q_dtype) * ck).to(matmul_dtype),
                                ).to(scores.dtype)

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

                        idx_tsrc = (idx_tsrc_center + idx_tsrc_right) // 2

                        if LOAD_Q_EACH_TIME:
                            queries_0 = pool_queries(
                                idx_bsz,
                                idx_head,
                                pos_tdst,
                                idx_tdst,
                                mask_tdst,
                                idx_hid_q0,
                                mask_hid_q0,
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
                                HID_DIM,
                                TDST,
                                CHUNK_COUNT,
                                real_pos_tdst_min,
                                model_context_length,
                                sliding_window_size,
                                USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                                NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                                EXTEND_BACKEND,
                                BLOCK_SIZE_Q,
                                HID_BLOCK_0,
                                STRIDE_Q,
                            )

                        keys_right_0 = load_keys_with_rope(
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
                            q_dtype,
                            idx_bsz,
                            idx_tsrc,
                            idx_head // HEAD_GROUP,
                            idx_hid_q0,
                            idx_chunk,
                            mask_tsrc_active,
                            mask_tdst,
                            mask_hid_q0,
                            real_pos_tdst_min,
                            model_context_length,
                            num_sinks,
                            USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                            EXTEND_BACKEND,
                            NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                            BLOCK_CHUNK,
                            HID_BLOCK_0,
                            HID_DIM,
                            True,
                            HEAD // HEAD_GROUP,
                            UPDATE_CACHE,
                            rope_range_begin,
                            rope_range_end,
                            rope_is_neox_style,
                        )

                        scores_right = tl.dot(
                            (queries_0 * cq).to(matmul_dtype),
                            (keys_right_0.to(q_dtype) * ck).to(matmul_dtype),
                        ).to(scores.dtype)

                        if HID_BLOCK_1 > 0:
                            if LOAD_Q_EACH_TIME:
                                queries_1 = pool_queries(
                                    idx_bsz,
                                    idx_head,
                                    pos_tdst,
                                    idx_tdst,
                                    mask_tdst,
                                    idx_hid_q1,
                                    mask_hid_q1,
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
                                    HID_DIM,
                                    TDST,
                                    CHUNK_COUNT,
                                    real_pos_tdst_min,
                                    model_context_length,
                                    sliding_window_size,
                                    USING_EXTEND,
                                    NEED_APPLY_ROPE,
                                    EXTEND_BACKEND,
                                    BLOCK_SIZE_Q,
                                    HID_BLOCK_1,
                                    STRIDE_Q,
                                )

                            keys_right_1 = load_keys_with_rope(
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
                                q_dtype,
                                idx_bsz,
                                idx_tsrc,
                                idx_head // HEAD_GROUP,
                                idx_hid_q1,
                                idx_chunk,
                                mask_tsrc_active,
                                mask_tdst,
                                mask_hid_q1,
                                real_pos_tdst_min,
                                model_context_length,
                                num_sinks,
                                USING_EXTEND,
                                EXTEND_BACKEND,
                                NEED_APPLY_ROPE,
                                BLOCK_CHUNK,
                                HID_BLOCK_1,
                                HID_DIM,
                                True,
                                HEAD // HEAD_GROUP,
                                UPDATE_CACHE,
                                rope_range_begin,
                                rope_range_end,
                                rope_is_neox_style,
                            )

                            if COMPUTE_MLA_ROPE:
                                scores_right += tl.dot(
                                    (queries_1 * cq).to(matmul_dtype),
                                    (keys_right_1.to(q_dtype) * ck).to(matmul_dtype),
                                ).to(scores.dtype)

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


@triton.autotune(
    configs=get_decode_scan_stage_configs(),
    key=[
        "BLOCK_SIZE_Q",
        "HID_DIM",
        "USING_PAGES",
    ],
    restore_value=[
        "INDICES_LEFT",
        "INDICES_RIGHT",
    ],
)
@triton.jit
def decode_chunk_controllable_sampling_mask_cuda(
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
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
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
    HID_DIM: tl.constexpr,
    HID_BLOCK_0: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr = 32,
    STRIDE_Q: tl.constexpr = 1,
    BLOCK_HEAD: tl.constexpr = 4,
    BLOCK_HEAD_PADDED: tl.constexpr = 16,
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
    COMPUTE_MLA_ROPE: tl.constexpr = False,
):
    BCHUNK = tl.cdiv(CHUNK_COUNT, BLOCK_CHUNK)
    BHEAD = tl.cdiv(HEAD, BLOCK_HEAD)

    # NOTE (BHEAD, TDST, BCHUNK, BSZ)
    pid = tl.program_id(0).to(tl.int64)

    idx_bhead = pid % BHEAD
    idx_head = idx_bhead * BLOCK_HEAD + tl.arange(0, BLOCK_HEAD_PADDED)
    mask_head = (idx_head < HEAD) & (idx_head < (idx_bhead * BLOCK_HEAD + BLOCK_HEAD))
    pid = pid // BHEAD

    idx_tdst = pid % TDST
    idx_bdst = idx_tdst // BLOCK_SIZE_Q
    pid = pid // TDST
    idx_bchunk = pid % BCHUNK
    pid = pid // BCHUNK
    idx_bsz = pid

    HID_BLOCK_1: tl.constexpr = HID_DIM - HID_BLOCK_0

    idx_hid_q0 = tl.arange(0, HID_BLOCK_0)
    mask_hid_q0 = idx_hid_q0 < HID_DIM

    if HID_BLOCK_1 > 0:
        idx_hid_q1 = HID_BLOCK_0 + tl.arange(0, HID_BLOCK_1)
        mask_hid_q1 = idx_hid_q1 < HID_DIM
    else:
        idx_hid_q1 = None
        mask_hid_q1 = None

    pos_tdst = tl.load(POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst)

    if Q.dtype.element_ty != tl.float8e5:
        q_dtype = Q.dtype.element_ty
    else:
        q_dtype = tl.bfloat16
    cq = (tl.sqrt(HID_DIM * 1.0) / tl.sqrt(tl.sqrt(HID_DIM * 1.0))).to(q_dtype)
    ck = (1.0 / tl.sqrt(tl.sqrt(HID_DIM * 1.0))).to(q_dtype)

    if pos_tdst < 0:
        return

    pos_tdst_min = (pos_tdst - sliding_window_size).to(tl.int32)
    pos_tdst_min = tl.maximum(pos_tdst_min, 0)

    idx_chunk = idx_bchunk * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
    mask_chunk = idx_chunk < CHUNK_COUNT

    idx_tsrc_left = tl.load(
        INDICES_LEFT
        + idx_bsz * stride_indices_left_bsz
        + idx_bdst * stride_indices_left_bdst
        + (idx_bhead * BLOCK_HEAD) * stride_indices_left_head
        + idx_chunk * stride_indices_left_chunk,
        mask=mask_chunk,
        other=MAX_TSRC,
    ).to(tl.int32)

    idx_tsrc_right = tl.load(
        INDICES_RIGHT
        + idx_bsz * stride_indices_right_bsz
        + idx_bdst * stride_indices_right_bdst
        + (idx_bhead * BLOCK_HEAD) * stride_indices_right_head
        + idx_chunk * stride_indices_right_chunk,
        mask=mask_chunk,
        other=MAX_TSRC,
    ).to(tl.int32)

    if (pos_tdst + BLOCK_SIZE_Q * SCAN_STRIDE) >= tl.min(idx_tsrc_left):
        max_chunk_size = tl.max(idx_tsrc_right - idx_tsrc_left).to(tl.float32)

        scores = tl.zeros((BLOCK_CHUNK,), dtype=tl.float32) - 32000.0

        queries_0 = pool_queries(
            idx_bsz,
            idx_head[:, None],
            pos_tdst,
            idx_tdst,
            mask_head,
            idx_hid_q0,
            mask_hid_q0,
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
            HID_DIM,
            TDST,
            CHUNK_COUNT,
            pos_tdst,
            model_context_length,
            sliding_window_size,
            USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
            NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
            EXTEND_BACKEND,
            BLOCK_HEAD_PADDED,
            HID_BLOCK_0,
            1,
        )

        if HID_BLOCK_1 > 0:
            queries_1 = pool_queries(
                idx_bsz,
                idx_head[:, None],
                pos_tdst,
                idx_tdst,
                mask_head,
                idx_hid_q1,
                mask_hid_q1,
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
                HID_DIM,
                TDST,
                CHUNK_COUNT,
                pos_tdst,
                model_context_length,
                sliding_window_size,
                USING_EXTEND,
                NEED_APPLY_ROPE,
                EXTEND_BACKEND,
                BLOCK_HEAD_PADDED,
                HID_BLOCK_1,
                1,
            )
        else:
            queries_1 = None

        matmul_dtype = q_dtype
        # max_chunk_size
        # while max_chunk_size >= TERMINATE_SIZE:
        #     max_chunk_size /= 2.0
        for _ in tl.range(
            0,
            tl.ceil(tl.log2(max_chunk_size / TERMINATE_SIZE)).to(tl.int32),
            num_stages=1 if (USING_EXTEND and (EXTEND_BACKEND != "none")) else 3,
        ):
            mask_tsrc_active = (
                mask_chunk
                & (idx_tsrc_left < idx_tsrc_right)
                & (idx_tsrc_left <= pos_tdst_min)
                & (idx_tsrc_left >= 0)
            )
            idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2

            assert (
                not ORACLE_MAXIMUM
            ), "this is deprecated at fad90fea0d37ba88c04e90f2c5597e6800e97e8f"

            idx_tsrc = (idx_tsrc_left + idx_tsrc_center) // 2

            keys_left_0 = load_keys_with_rope(
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
                q_dtype,
                idx_bsz,
                idx_tsrc,
                (idx_bhead * BLOCK_HEAD) // HEAD_GROUP,
                idx_hid_q0,
                idx_chunk,
                mask_tsrc_active,
                True,
                mask_hid_q0,
                pos_tdst,
                model_context_length,
                num_sinks,
                USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                EXTEND_BACKEND,
                NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                BLOCK_CHUNK,
                HID_BLOCK_0,
                HID_DIM,
                False,
                HEAD // HEAD_GROUP,
                UPDATE_CACHE,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
            )

            scores_left = tl.dot(
                (queries_0 * cq).to(matmul_dtype),
                (keys_left_0.to(q_dtype) * ck).to(matmul_dtype),
            ).to(scores.dtype)

            if HID_BLOCK_1 > 0:
                keys_left_1 = load_keys_with_rope(
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
                    q_dtype,
                    idx_bsz,
                    idx_tsrc,
                    (idx_bhead * BLOCK_HEAD) // HEAD_GROUP,
                    idx_hid_q1,
                    idx_chunk,
                    mask_tsrc_active,
                    True,
                    mask_hid_q1,
                    pos_tdst,
                    model_context_length,
                    num_sinks,
                    USING_EXTEND,
                    EXTEND_BACKEND,
                    NEED_APPLY_ROPE,
                    BLOCK_CHUNK,
                    HID_BLOCK_1,
                    HID_DIM,
                    False,
                    HEAD // HEAD_GROUP,
                    UPDATE_CACHE,
                    rope_range_begin,
                    rope_range_end,
                    rope_is_neox_style,
                )

                if COMPUTE_MLA_ROPE:
                    scores_left += tl.dot(
                        (queries_1 * cq).to(matmul_dtype),
                        (keys_left_1.to(q_dtype) * ck).to(matmul_dtype),
                    ).to(scores.dtype)

            if REDUCE == "max":
                scores_left = tl.where(mask_head[:, None], scores_left, float("-inf"))
                scores_left = tl.max(scores_left, axis=0).to(scores_left.dtype)
            elif REDUCE == "mean":
                scores_left = tl.where(mask_head[:, None], scores_left, float("0"))
                scores_left = tl.sum(scores_left, axis=0).to(scores_left.dtype)
                scores_left = (scores_left / tl.sum(mask_head.to(tl.float32))).to(
                    scores_left.dtype
                )
            else:
                raise Exception()
            scores_left = tl.where(mask_tsrc_active, scores_left, float("-inf")).to(
                scores_left.dtype
            )

            idx_tsrc = (idx_tsrc_center + idx_tsrc_right) // 2

            keys_right_0 = load_keys_with_rope(
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
                q_dtype,
                idx_bsz,
                idx_tsrc,
                (idx_bhead * BLOCK_HEAD) // HEAD_GROUP,
                idx_hid_q0,
                idx_chunk,
                mask_tsrc_active,
                True,
                mask_hid_q0,
                pos_tdst,
                model_context_length,
                num_sinks,
                USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                EXTEND_BACKEND,
                NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                BLOCK_CHUNK,
                HID_BLOCK_0,
                HID_DIM,
                True,
                HEAD // HEAD_GROUP,
                UPDATE_CACHE,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
            )

            scores_right = tl.dot(
                (queries_0 * cq).to(matmul_dtype),
                (keys_right_0.to(q_dtype) * ck).to(matmul_dtype),
            ).to(scores.dtype)

            if HID_BLOCK_1 > 0:

                keys_right_1 = load_keys_with_rope(
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
                    q_dtype,
                    idx_bsz,
                    idx_tsrc,
                    (idx_bhead * BLOCK_HEAD) // HEAD_GROUP,
                    idx_hid_q1,
                    idx_chunk,
                    mask_tsrc_active,
                    True,
                    mask_hid_q1,
                    pos_tdst,
                    model_context_length,
                    num_sinks,
                    USING_EXTEND,
                    EXTEND_BACKEND,
                    NEED_APPLY_ROPE,
                    BLOCK_CHUNK,
                    HID_BLOCK_1,
                    HID_DIM,
                    True,
                    HEAD // HEAD_GROUP,
                    UPDATE_CACHE,
                    rope_range_begin,
                    rope_range_end,
                    rope_is_neox_style,
                )

                if COMPUTE_MLA_ROPE:
                    scores_right += tl.dot(
                        (queries_1 * cq).to(matmul_dtype),
                        (keys_right_1.to(q_dtype) * ck).to(matmul_dtype),
                    ).to(scores.dtype)

            if REDUCE == "max":
                scores_right = tl.where(mask_head[:, None], scores_right, float("-inf"))
                scores_right = tl.max(scores_right, axis=0).to(scores_right.dtype)
            elif REDUCE == "mean":
                scores_right = tl.where(mask_head[:, None], scores_right, float("0"))
                scores_right = tl.sum(scores_right, axis=0).to(scores_right.dtype)
                scores_right = (scores_right / tl.sum(mask_head.to(tl.float32))).to(
                    scores_right.dtype
                )
            else:
                raise Exception()
            scores_right = tl.where(mask_tsrc_active, scores_right, float("-inf")).to(
                scores_right.dtype
            )

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
            + idx_bdst * stride_indices_left_bdst
            + idx_head[:, None] * stride_indices_left_head
            + idx_chunk[None, :] * stride_indices_left_chunk,
            value=idx_tsrc_left[None, :],
            mask=mask_chunk[None, :] & mask_head[:, None],
        )

        tl.store(
            INDICES_RIGHT
            + idx_bsz * stride_indices_right_bsz
            + idx_bdst * stride_indices_right_bdst
            + idx_head[:, None] * stride_indices_right_head
            + idx_chunk[None, :] * stride_indices_right_chunk,
            value=idx_tsrc_right[None, :],
            mask=mask_chunk[None, :] & mask_head[:, None],
        )

        tl.store(
            OUT_SCORES
            + idx_bsz * stride_out_scores_bsz
            + idx_bdst * stride_out_scores_bdst
            + idx_head[:, None] * stride_out_scores_head
            + idx_chunk[None, :] * stride_out_scores_chunk,
            value=scores[None, :],
            mask=mask_chunk[None, :] & mask_head[:, None],
        )


_NUM_STREAMING_MULTIPROCESSOR = None


def num_streaming_multiprocessor():
    import numba.cuda

    global _NUM_STREAMING_MULTIPROCESSOR
    if _NUM_STREAMING_MULTIPROCESSOR is None:
        _NUM_STREAMING_MULTIPROCESSOR = (
            numba.cuda.get_current_device().MULTIPROCESSOR_COUNT
        )
    return _NUM_STREAMING_MULTIPROCESSOR


@capture
def chunk_controllable_sampling_mask(
    args,
    chunk_count,
    BLOCK_CHUNK,
    TDST,
    BLOCK_SIZE_Q,
    STAGE_STRIDE,
    HEAD,
    BSZ,
    q,
    k_mask,
    position_ids,
    indices_left,
    indices_right,
    out_scores,
    mask_access_counter,
    mask_cache_miss_counter,
    MAX_TSRC,
    HID,
    HID_BLOCK,
    stage_block_stride_q,
    HEAD_KV,
    extend_backend,
):
    using_online_cache_update = args.online_update_cache and (
        args.offload_cache is not None
    )

    assert q.ndim == 4
    if (q.shape[1] == 1) and not using_online_cache_update:
        HEAD_REPEAT = HEAD // HEAD_KV
        BLOCK_HEAD = HEAD_REPEAT
        assert triton.next_power_of_2(BLOCK_HEAD) == BLOCK_HEAD
        BLOCK_HEAD_PADDED = max(BLOCK_HEAD, 16)

        assert TDST == q.shape[1]

        grid = (
            triton.cdiv(HEAD, BLOCK_HEAD)
            * TDST
            * triton.cdiv(chunk_count, BLOCK_CHUNK)
            * BSZ,
        )
        decode_chunk_controllable_sampling_mask_cuda[grid](
            q,
            *safe_stride(q, 4),
            k_mask,
            *safe_stride(k_mask, 4),
            position_ids,
            *safe_stride(position_ids, 2),
            *args.args_paged_kv_cache(disable_cache=k_mask is not None),
            *args.args_offload_cache(True, disable_cache=k_mask is not None),
            indices_left,
            *safe_stride(indices_left, 4),
            indices_right,
            *safe_stride(indices_right, 4),
            out_scores,
            *safe_stride(out_scores, 4),
            args.rope_cos,
            *safe_stride(args.rope_cos, 2),
            args.rope_sin,
            *safe_stride(args.rope_sin, 2),
            args.rope_range[0],
            args.rope_range[1],
            args.rope_is_neox_style,
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
            HID_DIM=HID,
            HID_BLOCK_0=HID_BLOCK,
            BLOCK_SIZE_Q=1,
            STRIDE_Q=1,
            BLOCK_HEAD=BLOCK_HEAD,
            BLOCK_HEAD_PADDED=BLOCK_HEAD_PADDED,
            BLOCK_CHUNK=BLOCK_CHUNK,
            HEAD_GROUP=HEAD // HEAD_KV,
            USING_EXTEND=args.using_extend and (extend_backend != "none"),
            EXTEND_BACKEND=extend_backend,
            NEED_APPLY_ROPE=args.need_apply_rope and (extend_backend != "none"),
            TERMINATE_SIZE=args.stage_early_terminate,
            SCAN_STRIDE=1,
            UPDATE_CACHE=args.online_update_cache,
            ORACLE_MAXIMUM=False,  # NOTE: seems has bug... but why?
            COMPUTE_MLA_ROPE=os.getenv("HIP_DEBUG_SCAN_COMPUTE_MLA_ROPE", "0") == "1",
        )
    else:
        if not using_online_cache_update:
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

        chunk_controllable_sampling_mask_cuda[grid](
            q,
            *q.stride(),
            k_mask,
            *safe_stride(k_mask, 4),
            position_ids,
            *position_ids.stride(),
            *args.args_paged_kv_cache(disable_cache=k_mask is not None),
            *args.args_offload_cache(True, disable_cache=k_mask is not None),
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
            args.rope_range[0],
            args.rope_range[1],
            args.rope_is_neox_style,
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
            HID_DIM=HID,
            HID_BLOCK_0=HID_BLOCK,
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            STRIDE_Q=stage_block_stride_q,
            BLOCK_CHUNK=BLOCK_CHUNK,
            HEAD_GROUP=HEAD // HEAD_KV,
            USING_EXTEND=args.using_extend and (extend_backend != "none"),
            EXTEND_BACKEND=extend_backend,
            NEED_APPLY_ROPE=args.need_apply_rope and (extend_backend != "none"),
            TERMINATE_SIZE=args.stage_early_terminate,
            SCAN_STRIDE=STAGE_STRIDE,
            UPDATE_CACHE=args.online_update_cache,
            ORACLE_MAXIMUM=False,  # NOTE: seems has bug... but why?
            COMPUTE_MLA_ROPE=os.getenv("HIP_DEBUG_SCAN_COMPUTE_MLA_ROPE", "0") == "1",
        )
