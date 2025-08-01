import os
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from hip_attn.v1_2.attention_metadata import safe_stride


@triton.jit
def split_half(x: tl.tensor):
    return tl.split(tl.trans(tl.reshape(x, [x.shape[0], 2, x.shape[1] // 2]), 0, 2, 1))


@triton.jit
def merge_half(x: tl.tensor, y: tl.tensor):
    return tl.reshape(
        tl.trans(tl.join(x, y), 0, 2, 1), x.shape[0], x.shape[1] + y.shape[1]
    )


@triton.jit
def de_rope(
    vec: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
):
    c0, ch = split_half(cos.to(tl.float32))
    s0, sh = split_half(sin.to(tl.float32))
    vr0, vrh = split_half(vec.to(tl.float32))

    out0 = (vrh * s0 + vr0 * ch) / (c0 * ch + sh * s0 + 1e-20)
    outh = (out0 * c0 - vr0) / (s0 + 1e-20)

    out = merge_half(out0, outh).to(vec.dtype)
    return out


@triton.jit
def de_rope_load(
    vec: tl.tensor,
    idx_t: tl.tensor,
    mask_t: tl.tensor,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
):
    cos = tl.load(
        COS
        + idx_t[:, None] * stride_cos_t
        + tl.arange(0, vec.shape[1])[None, :] * stride_cos_hid,
        mask=mask_t[:, None],
        other=0,
    )

    sin = tl.load(
        SIN
        + idx_t[:, None] * stride_sin_t
        + tl.arange(0, vec.shape[1])[None, :] * stride_sin_hid,
        mask=mask_t[:, None],
        other=0,
    )

    return de_rope(vec, cos, sin)


configs = [
    triton.Config(
        {
            "BLOCK_CHUNK": BLOCK_CHUNK,
        },
        num_stages=s,
        num_warps=w,
    )
    for BLOCK_CHUNK in [64, 128, 256]
    for s in [
        3,
    ]
    for w in [4, 8]
    # for BM in [128,]
    # for BN in [64,]
    # for s in [3, ]
    # for w in [4, ]
]


def keep(conf):
    BLOCK_CHUNK = conf.kwargs["BLOCK_CHUNK"]
    return True


@triton.autotune(list(filter(keep, configs)), key=["HID"])
@triton.jit
def _compute_scores_landmark_cuda(
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
    K_CACHE,
    stride_k_cache_t,
    stride_k_cache_page,
    stride_k_cache_head_kv,
    stride_k_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_tsrc,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    INDICES_LEFT,
    stride_indices_left_bsz,
    stride_indices_left_bdst,
    stride_indices_left_head,
    stride_indices_left_chunk,
    LANDMARK,
    stride_landmark_bsz,
    stride_landmark_tchunk,
    stride_landmark_head,
    stride_landmark_k,
    SCORES,
    stride_scores_bsz,
    stride_scores_bdst,
    stride_scores_head,
    stride_scores_tchunk,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    HEAD_KV: int,
    HEAD: int,
    TDST: int,
    NUM_CHUNKS: int,
    SLIDING_WINDOW_SIZE: int,
    HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    USING_PAGED_CACHE: tl.constexpr,
    DEROPE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)

    pid = tl.program_id(0).to(tl.int64)

    idx_head = pid % HEAD
    idx_head_kv = idx_head // (HEAD // HEAD_KV)
    pid = pid // HEAD

    idx_bdst = pid % BDST
    pid = pid // BDST

    idx_bsz = pid

    idx_hid = tl.arange(0, HID)

    Q = Q + idx_bsz * stride_q_bsz + idx_head * stride_q_head
    if K is not None:
        K = K + idx_bsz * stride_k_bsz + idx_head_kv * stride_k_head_kv
    if K_CACHE is not None:
        K_CACHE = (
            K_CACHE
            +
            # 0 * stride_k_cache_page +
            idx_head_kv * stride_k_cache_head_kv
        )
        BLOCK_TABLE = BLOCK_TABLE + idx_bsz * stride_block_table_bsz
    INDICES_LEFT = (
        INDICES_LEFT
        + idx_bsz * stride_indices_left_bsz
        + idx_bdst * stride_indices_left_bdst
        + idx_head * stride_indices_left_head
    )
    LANDMARK = (
        LANDMARK + idx_bsz * stride_landmark_bsz + idx_head * stride_landmark_head
    )
    SCORES = (
        SCORES
        + idx_bsz * stride_scores_bsz
        + idx_bdst * stride_scores_bdst
        + idx_head * stride_scores_head
    )

    idx_tdst = (
        tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
        + idx_bdst * BLOCK_SIZE_Q
    )
    mask_tdst = idx_tdst < TDST
    pos_tdst = tl.load(
        POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    pos_tdst_max = tl.max(pos_tdst * mask_tdst)
    seq_len_max = pos_tdst_max + 1 - SLIDING_WINDOW_SIZE

    queries = tl.load(
        Q + idx_tdst[:, None] * stride_q_tdst + idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0,
    )  # .to(tl.float8e5)

    if DEROPE:
        queries = de_rope_load(
            queries,
            pos_tdst,
            mask_tdst,
            COS,
            stride_cos_t,
            stride_cos_hid,
            SIN,
            stride_sin_t,
            stride_sin_hid,
        )

    for i_chunk in range(0, NUM_CHUNKS, BLOCK_CHUNK):
        idx_chunk = tl.arange(0, BLOCK_CHUNK) + i_chunk
        mask_chunk = idx_chunk < NUM_CHUNKS
        idx_k = tl.arange(0, BLOCK_K)
        idx_tsrc_base = tl.load(
            INDICES_LEFT + idx_chunk * stride_indices_left_chunk, mask=mask_chunk
        )
        idx_tchunk = idx_tsrc_base // CHUNK_SIZE
        idx_tsrc_offset = tl.load(
            LANDMARK
            + idx_tchunk[:, None] * stride_landmark_tchunk
            + idx_k[None, :] * stride_landmark_k,
            mask=mask_chunk[:, None],
        )
        idx_tsrc = idx_tsrc_base[:, None] + idx_tsrc_offset
        mask_tsrc = mask_chunk[:, None] & (idx_tsrc < seq_len_max)
        idx_tsrc = tl.reshape(idx_tsrc, BLOCK_CHUNK * BLOCK_K)
        mask_tsrc = tl.reshape(mask_tsrc, BLOCK_CHUNK * BLOCK_K)

        if seq_len_max >= tl.min(tl.where(mask_tsrc, idx_tsrc, 98765431)):
            if not USING_PAGED_CACHE:
                keys = tl.load(
                    K
                    + idx_tsrc[None, :] * stride_k_tsrc
                    + idx_hid[:, None] * stride_k_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                )  # .to(tl.float8e5)
            else:
                block_index = tl.load(
                    BLOCK_TABLE + idx_tsrc * stride_block_table_tsrc,
                    mask=mask_tsrc,
                    other=0,
                )
                keys = tl.load(
                    K_CACHE
                    + block_index[None, :] * stride_k_cache_t
                    + idx_hid[:, None] * stride_k_cache_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                )

            if keys.dtype == tl.float8e5:
                keys = keys.to(tl.bfloat16)

            if DEROPE:
                keys = tl.trans(
                    de_rope_load(
                        tl.trans(keys, 1, 0),
                        idx_tsrc,
                        mask_tsrc,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                    ),
                    1,
                    0,
                )

            scores = tl.dot(
                queries.to(keys.dtype),
                keys,
            )

            scores = tl.where(scores == 0.0, float("-inf"), scores).to(scores.dtype)

            # mask = (
            #     (mask_tdst[:, None] & mask_tsrc[None, :]) &
            #     ((pos_tdst - SLIDING_WINDOW_SIZE)[:, None] >= idx_tsrc[None, :])
            # )
            # scores = tl.where(mask, scores, float('-inf')).to(scores.dtype)
            # scores = tl.where(mask, scores, 0)

            if BLOCK_K > 1:
                scores = tl.reshape(
                    scores, BLOCK_SIZE_Q // BLOCK_STRIDE_Q, BLOCK_CHUNK, BLOCK_K
                )
                scores = tl.max(scores, axis=0)
                scores = tl.max(scores, axis=-1)
            else:
                scores = tl.reshape(scores, BLOCK_SIZE_Q // BLOCK_STRIDE_Q, BLOCK_CHUNK)
                scores = tl.max(scores, axis=0)

            tl.store(
                SCORES + idx_chunk * stride_scores_tchunk,
                value=scores,
                mask=mask_chunk,
            )


from .utils import capture


@capture
def compute_scores_landmark(
    # [BSZ, TDST, HEAD, HID]
    q: Tensor,
    # [BSZ, TSRC, HEAD_KV, HID]
    k: Tensor,
    # [T, 1, HEAD_KV, HID]
    k_cache: Optional[Tensor],
    # [BSZ, MAX_TSRC]
    block_table: Optional[Tensor],
    # [BSZ, TDST]
    position_ids: Tensor,
    # [BSZ, BDST, HEAD, CHUNK_COUNT]
    indices_left: Tensor,
    # [BSZ, TSRC // CHUNK_SIZE, HEAD, K]
    landmarks: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    BLOCK_SIZE_Q: int,
    BLOCK_STRIDE_Q: int,
    CHUNK_SIZE: int,
    SLIDING_WINDOW_SIZE: int,
) -> Tensor:
    # output: [BSZ, BDST, HEAD, CHUNK_COUNT]
    BSZ, TDST, HEAD, HID = q.shape
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    if k is not None:
        _, TSRC, HEAD_KV, _ = k.shape
        assert k.shape == (BSZ, TSRC, HEAD_KV, HID)
    else:
        assert k_cache is not None
        HEAD_KV = k_cache.shape[-2]
    assert position_ids.shape == (BSZ, TDST)
    K = landmarks.shape[-1]
    assert landmarks.shape == (BSZ, landmarks.shape[1], HEAD, K)
    CHUNK_COUNT = indices_left.shape[-1]
    assert indices_left.shape == (BSZ, BDST, HEAD, CHUNK_COUNT)
    if k_cache is not None:
        assert k_cache.shape[2:] == (HEAD_KV, HID)
        assert k_cache.shape[1] == 1

    BLOCK_K = K
    # BLOCK_CHUNK = int(os.getenv('SA_BLOCK_SIZE_LANDMARK', '128')) // BLOCK_K
    # assert BLOCK_CHUNK > 0

    USING_PAGED_CACHE = k_cache is not None
    DEROPE = False

    scores = torch.full(
        (BSZ, BDST, HEAD, CHUNK_COUNT),
        dtype=torch.float32,
        device=q.device,
        fill_value=float("-inf"),
    )

    grid = lambda kwargs: (BSZ * BDST * HEAD,)
    _compute_scores_landmark_cuda[grid](
        q,
        *safe_stride(q, 4),
        k,
        *safe_stride(k, 4),
        k_cache,
        *safe_stride(k_cache, 4),
        block_table,
        *safe_stride(block_table, 2),
        position_ids,
        *safe_stride(position_ids, 2),
        indices_left,
        *safe_stride(indices_left, 4),
        landmarks,
        *safe_stride(landmarks, 4),
        scores,
        *safe_stride(scores, 4),
        cos,
        *safe_stride(cos, 2),
        sin,
        *safe_stride(sin, 2),
        HEAD_KV,
        HEAD,
        TDST,
        CHUNK_COUNT,
        SLIDING_WINDOW_SIZE,
        HID,
        BLOCK_SIZE_Q,
        BLOCK_STRIDE_Q,
        BLOCK_K,
        CHUNK_SIZE,
        USING_PAGED_CACHE,
        DEROPE,
        # BLOCK_CHUNK,
        # num_warps=4,
        # num_stages=3,
    )

    return scores
