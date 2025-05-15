import torch
from torch import Tensor
import triton
import triton.language as tl
from hip_attn.v1_2.attention_metadata import safe_stride

@triton.jit
def _compute_scores_landmark_cuda(
    Q, stride_q_bsz, stride_q_tdst, stride_q_head, stride_q_hid,
    K, stride_k_bsz, stride_k_tsrc, stride_k_head_kv, stride_k_hid,
    POS, stride_pos_bsz, stride_pos_tdst,
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
    
    HEAD_KV: int,
    HEAD: int,
    TDST: int,
    NUM_CHUNKS: int,
    SLIDING_WINDOW_SIZE: int,
    
    HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)
    
    pid = tl.program_id(0)
    idx_head = pid % HEAD
    idx_head_kv = idx_head // (HEAD // HEAD_KV)
    pid = pid // HEAD
    idx_bdst = pid % BDST
    idx_bsz = pid // BDST
    idx_hid = tl.arange(0, HID)
    
    Q = (
        Q + 
        idx_bsz * stride_q_bsz + 
        idx_head * stride_q_head
    )
    K = (
        K + 
        idx_bsz * stride_k_bsz + 
        idx_head_kv * stride_k_head_kv
    )
    INDICES_LEFT = (
        INDICES_LEFT +
        idx_bsz * stride_indices_left_bsz +
        idx_bdst * stride_indices_left_bdst +
        idx_head * stride_indices_left_head
    )
    LANDMARK = (
        LANDMARK + 
        idx_bsz * stride_landmark_bsz + 
        idx_head * stride_landmark_head
    )
    SCORES = (
        SCORES + 
        idx_bsz * stride_scores_bsz + 
        idx_bdst * stride_scores_bdst + 
        idx_head * stride_scores_head
    )
    
    idx_tdst = tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q + idx_bdst * BLOCK_SIZE_Q
    mask_tdst = idx_tdst < TDST
    pos_tdst = tl.load(
        POS +
        idx_bsz * stride_pos_bsz +
        idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    pos_tdst_max = tl.max(pos_tdst * mask_tdst)
    seq_len_max = pos_tdst_max + 1
    
    queries = tl.load(
        Q +
        idx_tdst[:, None] * stride_q_tdst +
        idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0,
    )#.to(tl.float8e5)
    
    for i_chunk in range(0, NUM_CHUNKS, BLOCK_CHUNK):
        idx_chunk = tl.arange(0, BLOCK_CHUNK) + i_chunk
        mask_chunk = idx_chunk < NUM_CHUNKS
        idx_k = tl.arange(0, BLOCK_K)
        idx_tsrc_base = tl.load(
            INDICES_LEFT +
            idx_chunk * stride_indices_left_chunk,
            mask=mask_chunk
        )
        idx_tchunk = idx_tsrc_base // CHUNK_SIZE
        idx_tsrc_offset = tl.load(
            LANDMARK +
            idx_tchunk[:, None] * stride_landmark_tchunk +
            idx_k[None, :] * stride_landmark_k,
            mask=mask_chunk[:, None]
        )
        idx_tsrc = idx_tsrc_base[:, None] + idx_tsrc_offset
        mask_tsrc = mask_chunk[:, None] & (idx_tsrc < seq_len_max)
        idx_tsrc = tl.reshape(idx_tsrc, BLOCK_CHUNK * BLOCK_K)
        mask_tsrc = tl.reshape(mask_tsrc, BLOCK_CHUNK * BLOCK_K)
        
        keys = tl.load(
            K +
            idx_tsrc[None, :] * stride_k_tsrc +
            idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc[None, :],
            other=0,
        )#.to(tl.float8e5)

        scores = tl.dot(
            queries, 
            keys, 
            # out_dtype=tl.float16
        )
        mask = (
            (mask_tdst[:, None] & mask_tsrc[None, :]) &
            ((pos_tdst - SLIDING_WINDOW_SIZE)[:, None] >= idx_tsrc[None, :])
        )
        scores = tl.where(mask, scores, float('-inf'))
        # scores = tl.where(mask, scores, 0)
        
        scores = tl.reshape(scores, BLOCK_SIZE_Q // BLOCK_STRIDE_Q, BLOCK_CHUNK, BLOCK_K)
        scores = tl.max(scores, axis=0)
        scores = tl.max(scores, axis=-1)
        
        tl.store(
            SCORES + idx_chunk * stride_scores_tchunk,
            value=scores,
            mask=mask_chunk,
        )

def compute_scores_landmark(
    # [BSZ, TDST, HEAD, HID]
    q: Tensor,
    # [BSZ, TSRC, HEAD_KV, HID]
    k: Tensor,
    # [BSZ, TDST]
    position_ids: Tensor,
    # [BSZ, BDST, HEAD, CHUNK_COUNT]
    indices_left: Tensor,
    # [BSZ, TSRC // CHUNK_SIZE, HEAD, K]
    landmarks: Tensor,
    
    BLOCK_SIZE_Q: int,
    BLOCK_STRIDE_Q: int,
    CHUNK_SIZE: int,
    SLIDING_WINDOW_SIZE: int,
) -> Tensor:
    # output: [BSZ, BDST, HEAD, CHUNK_COUNT]
    BSZ, TDST, HEAD, HID = q.shape
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    _, TSRC, HEAD_KV, _ = k.shape
    assert k.shape == (BSZ, TSRC, HEAD_KV, HID)
    assert position_ids.shape == (BSZ, TDST)
    K = landmarks.shape[-1]
    assert landmarks.shape == (BSZ, TSRC // CHUNK_SIZE, HEAD, K)
    CHUNK_COUNT = indices_left.shape[-1]
    assert indices_left.shape == (BSZ, BDST, HEAD, CHUNK_COUNT)
    
    BLOCK_K = K
    BLOCK_CHUNK = 128 // BLOCK_K
    assert BLOCK_CHUNK > 0
    
    scores = torch.full(
        (BSZ, BDST, HEAD, CHUNK_COUNT),
        dtype=torch.float32,
        device=q.device,
        fill_value=float('-inf')
    )
    
    grid = (BSZ * BDST * HEAD,)
    _compute_scores_landmark_cuda[grid](
        q, *safe_stride(q, 4),
        k, *safe_stride(k, 4),
        position_ids, *safe_stride(position_ids, 2),
        indices_left, *safe_stride(indices_left, 4),
        landmarks, *safe_stride(landmarks, 4),
        scores, *safe_stride(scores, 4),
        
        HEAD_KV,
        HEAD,
        TDST,
        CHUNK_COUNT,
        SLIDING_WINDOW_SIZE,
        
        HID,
        BLOCK_SIZE_Q,
        BLOCK_STRIDE_Q,
        BLOCK_K,
        BLOCK_CHUNK,
        CHUNK_SIZE,
    )
    
    return scores