import triton
import triton.language as tl

from hip_attn.v1_2.uvm_gpu_cache import load_tokens


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
    stride_indices_k,
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
    stride_offload_cache_gpu_table_k,
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
        + idx_k * stride_indices_k,
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
    #         idx_hid[None, :] * stride_v_hid,
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
        stride_offload_cache_gpu_table_k,
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
    #         idx_hid[None, :] * stride_v_hid,
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
        stride_offload_cache_gpu_table_k,
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
