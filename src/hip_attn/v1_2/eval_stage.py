import triton
import triton.language as tl

from hip_attn.v1_2.scan_stage import load_keys_with_rope


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
