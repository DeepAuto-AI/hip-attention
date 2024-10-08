import math
from hip.models.hip_attention.attention2_draft_prefetch import (
    hip_attention, 
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
    load_checkouts,
)
import torch
from torch import Tensor
from typing import List, Dict, Optional, Tuple
import triton
import triton.language as tl
import numpy as np
import cv2

def sampling_hip_attention(
    q: Tensor, 
    k: Optional[Tensor], 
    v: Optional[Tensor], 
    first_stage_args: HiPAttentionArgs,
    second_stage_range_k: 1024,
) -> Tensor:
    BSZ, TDST, HEAD, HID = q.shape
    BSZ, TSRC, HEAD_KV, HID = k.shape
    assert v.shape == k.shape
    assert TDST == TSRC
    
    chunk_count = first_stage_args.mask_k
    chunk_size = TSRC // chunk_count
    assert (chunk_size * chunk_count) == TSRC
    first_stage_chunk_args = first_stage_args.clone()
    first_stage_chunk_args.mask_k = first_stage_chunk_args.block_size_k
    
    k_chunked = k.view(BSZ * chunk_count, chunk_size, HEAD_KV, HID)
    q_chunked = q.expand(BSZ * chunk_count, TDST, HEAD, HID)
    
    position_ids = torch.arange(0, TDST, device=q.device)[None, :].expand(BSZ * chunk_count, TDST)
    position_ids = position_ids - torch.arange(0, TSRC, chunk_size, device=q.device)[:, None]
    position_ids = position_ids.clamp(0, chunk_size - 1)
    first_stage_chunk_args.position_ids = position_ids + 1
    
    _, metadata = hip_attention(
        q=q_chunked, k=k_chunked, v=None, 
        args=first_stage_chunk_args, 
        mask_only=True,
    )

@triton.jit
def sampling_mask_cuda(
    Q, stride_q_bsz, stride_q_tdst, stride_q_head, stride_q_hid,
    K, stride_k_bsz, stride_k_tsrc, stride_k_head_kv, stride_k_hid,
    
    OUT_INDICES, 
    stride_out_indices_bsz, 
    stride_out_indices_bdst, 
    stride_out_indices_head,
    stride_out_indices_chunk,
    
    OUT_SCORES, 
    stride_out_scores_bsz, 
    stride_out_scores_bdst, 
    stride_out_scores_head,
    stride_out_scores_chunk,
    
    CHUNK_COUNT: int,
    TSRC: int,
    TDST: int,
    HEAD: int,
    sliding_window_size: int,
    num_sinks: int,
    
    BLOCK_HID: tl.constexpr = 128,
    BLOCK_SIZE_Q: tl.constexpr = 32,
    STRIDE_Q: tl.constexpr = 1,
    BLOCK_CHUNK: tl.constexpr = 32,
    HEAD_GROUP: tl.constexpr = 4,
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)
    BCHUNK = tl.cdiv(CHUNK_COUNT, BLOCK_CHUNK)
    
    pid = tl.program_id(0).to(tl.int64)
    
    idx_head = pid % HEAD
    pid = pid // HEAD
    idx_bdst = pid % BDST
    pid = pid // BDST
    idx_bchunk = pid % BCHUNK
    pid = pid // BCHUNK
    idx_bsz = pid
    
    # idx_bsz = tl.program_id(3).to(tl.int64)
    # idx_bchunk = tl.program_id(2).to(tl.int64)
    # idx_bdst = tl.program_id(1).to(tl.int64)
    # idx_head = tl.program_id(0).to(tl.int64)
    
    # if ((idx_bchunk + 1) * BLOCK_CHUNK * (TSRC / CHUNK_COUNT)) > (idx_bdst * BLOCK_SIZE_Q + TSRC - TDST):
    #     return
    
    pos_tdst_min = (idx_bdst * BLOCK_SIZE_Q + TSRC - TDST - sliding_window_size - num_sinks).to(tl.float32)
    pos_tdst_min = tl.maximum(pos_tdst_min, 0)
    
    idx_chunk = idx_bchunk * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
    mask_chunk = idx_chunk < CHUNK_COUNT
    
    idx_tsrc_start = tl.floor(idx_chunk.to(tl.float32) * (pos_tdst_min / CHUNK_COUNT)).to(tl.int64) + num_sinks
    idx_tsrc_end = tl.where(
        (idx_chunk + 1) == CHUNK_COUNT,
        pos_tdst_min,
        tl.floor((idx_chunk + 1).to(tl.float32) * (pos_tdst_min / CHUNK_COUNT)).to(tl.int64),
    ).to(tl.int64) + num_sinks
    
    max_chunk_size = tl.ceil(TSRC / CHUNK_COUNT).to(tl.float32)
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // STRIDE_Q) * STRIDE_Q
    mask_tdst = idx_tdst < TDST
    
    idx_hid = tl.arange(0, BLOCK_HID)
    
    queries = tl.load(
        Q + \
            idx_bsz * stride_q_bsz +\
            idx_tdst[:, None] * stride_q_tdst +\
            idx_head * stride_q_head +\
            idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0
    )
    
    idx_tsrc_left = idx_tsrc_start
    idx_tsrc_right = idx_tsrc_end
    scores = tl.zeros((BLOCK_CHUNK,), dtype=tl.float16)
    
    while max_chunk_size > 1:
        max_chunk_size /= 2.0
        mask_tsrc_active = mask_chunk & (idx_tsrc_left < idx_tsrc_right)
        idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2
        idx_tsrc_left = (idx_tsrc_left + idx_tsrc_center) // 2
        idx_tsrc_right = (idx_tsrc_center + idx_tsrc_right) // 2
        
        keys_left = tl.load(
            K +\
                idx_bsz * stride_k_bsz +\
                idx_tsrc_left[None, :] * stride_k_tsrc +\
                (idx_head // HEAD_GROUP) * stride_k_head_kv +\
                idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc_active[None, :],
        )
        scores_left = tl.dot(
            (queries / 1).to(tl.float16),
            (keys_left / 1).to(tl.float16),
            out_dtype=tl.float16,
        )
        scores_left = tl.max(scores_left, axis=0).to(scores_left.dtype)
        
        keys_right = tl.load(
            K +\
                idx_bsz * stride_k_bsz +\
                idx_tsrc_right[None, :] * stride_k_tsrc +\
                (idx_head // HEAD_GROUP) * stride_k_head_kv +\
                idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc_active[None, :],
        )
        scores_right = tl.dot(
            (queries / 1).to(tl.float16),
            (keys_right / 1).to(tl.float16),
            out_dtype=tl.float16,
        )
        scores_right = tl.max(scores_right, axis=0).to(scores_right.dtype)
        
        mask_left_win = scores_left > scores_right
        idx_tsrc_left = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                idx_tsrc_left,
                idx_tsrc_center,
            ),
            idx_tsrc_left
        )
        
        idx_tsrc_right = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                idx_tsrc_center,
                idx_tsrc_right,
            ),
            idx_tsrc_right
        )
        
        scores = tl.where(
            mask_tsrc_active,
            tl.where(
                mask_left_win,
                scores_left,
                scores_right,
            ),
            scores,
        )
    
    tl.store(
        OUT_INDICES +\
            idx_bsz * stride_out_indices_bsz +\
            idx_bdst * stride_out_indices_bdst +\
            idx_head * stride_out_indices_head +\
            (tl.arange(0, BLOCK_CHUNK) + idx_bchunk * BLOCK_CHUNK) * stride_out_indices_chunk,
        value=idx_tsrc_left,
        mask=mask_chunk,
    )
    
    tl.store(
        OUT_SCORES +\
            idx_bsz * stride_out_scores_bsz +\
            idx_bdst * stride_out_scores_bdst +\
            idx_head * stride_out_scores_head +\
            (tl.arange(0, BLOCK_CHUNK) + idx_bchunk * BLOCK_CHUNK) * stride_out_scores_chunk,
        value=scores,
        mask=mask_chunk,
    )

DEBUG = False

def sampling_mask(
    q: Tensor, 
    k: Optional[Tensor], 
    v: Optional[Tensor], 
    first_stage_args: HiPAttentionArgs,
    second_stage_init_k: int,
    second_stage_init_chunk: int,
    second_stage_args: HiPAttentionArgs,
):
    global DEBUG
    
    BSZ, TDST, HEAD, HID = q.shape
    BSZ, TSRC, HEAD_KV, HID = k.shape
    assert v.shape == k.shape
    assert TDST == TSRC
    
    chunk_count = first_stage_args.mask_k
    chunk_size = TSRC // chunk_count
    assert (chunk_size * chunk_count) == TSRC
    
    BLOCK_CHUNK=first_stage_args.block_size_k
    BLOCK_SIZE_Q=first_stage_args.block_size_q
    
    out_indices = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count), 
        device=q.device,
        dtype=torch.int64
    )
    out_scores = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count), 
        device=q.device,
        dtype=torch.float16,
    )
    
    sampling_mask_cuda[(BSZ * triton.cdiv(chunk_count, BLOCK_CHUNK) * triton.cdiv(TDST, BLOCK_SIZE_Q) * HEAD,)](
        q, *q.stride(),
        k, *k.stride(),
        out_indices, *out_indices.stride(),
        out_scores, *out_scores.stride(),
        
        chunk_count,
        TSRC,
        TDST,
        HEAD,
        first_stage_args.sliding_window_size - max(BLOCK_SIZE_Q, BLOCK_CHUNK),
        first_stage_args.sink_token_size,
        
        BLOCK_HID=q.shape[-1],
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        STRIDE_Q=first_stage_args.block_stride_q,
        BLOCK_CHUNK=BLOCK_CHUNK,
        HEAD_GROUP=HEAD // HEAD_KV
    )
    
    _, indices = out_scores.sort(dim=-1, descending=True, stable=False)
    out_indices = out_indices.gather(dim=-1, index=indices)
    
    if DEBUG:
        out_indices = out_indices.cpu()
        debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK)))
        for i in range(out_indices.shape[1]):
            for j in range(max(
                second_stage_init_chunk,
                math.ceil(out_indices.shape[-1] * (second_stage_init_k / ((i + 1) * BLOCK_SIZE_Q)))
            )):
                if j >= out_indices.shape[-1]: continue
                chunk_size = triton.cdiv(((i + 1) * BLOCK_SIZE_Q), chunk_count * BLOCK_CHUNK)
                t = out_indices[0, i, -3, j] // BLOCK_CHUNK
                t = t // chunk_size * chunk_size
                debug[i, t:t+chunk_size] = 1
        cv2.imwrite('dummy.png', debug * 255)
    
    # print(out_indices[0, -2, 0])

def main_debug():
    global DEBUG
    
    seq_len = 131072
    
    q, k, v, out, cos, sin = load_checkouts(
        idx=0, 
        window=40, 
        seq_len=seq_len, 
        return_cos_sin=True, 
        dtype=torch.bfloat16
    )
    HEAD = q.shape[0]
    HEAD_KV = k.shape[0]
    
    q = q.permute(1, 0, 2).contiguous().unsqueeze(0)
    k = k.permute(1, 0, 2).contiguous().unsqueeze(0)
    v = v.permute(1, 0, 2).contiguous().unsqueeze(0)
    
    for i in range(10):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        hip_attention(
            q, k, v,
            args=HiPAttentionArgs(
                mask_k=512,
                block_size_q=64,
                block_stride_q=2,
                block_size_k=2,
                block_stride_k=1,
            ),
            mask_only=True
        )
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    for i in range(10):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        if i==0: DEBUG = True
        sampling_mask(
            q, k, v,
            first_stage_args=HiPAttentionArgs(
                mask_k=64,
                block_size_q=64,
                block_stride_q=2,
                block_size_k=64,
                sliding_window_size=512,
                sink_token_size=512,
            ),
            second_stage_init_chunk=16,
            second_stage_init_k=1024,
            second_stage_args=HiPAttentionArgs(
                mask_k=512,
                block_size_q=64,
                block_stride_q=2,
                block_size_k=2,
                block_stride_k=1,
            )
        )
        if i==0: DEBUG = False
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    # for i in range(10):
    #     start = torch.cuda.Event(True)
    #     end = torch.cuda.Event(True)
        
    #     start.record()
    #     context = sampling_hip_attention(
    #         q, k, v,
    #         first_stage_args=HiPAttentionArgs(
    #             mask_k=128,
    #             block_size_q=64,
    #             block_stride_q=2,
    #             block_size_k=2,
    #             block_stride_k=1,
    #         ),
    #         second_stage_range_k=1024, 
    #     )
    #     end.record()
        
    #     end.synchronize()
    #     print(start.elapsed_time(end))

if __name__ == '__main__':
    main_debug()