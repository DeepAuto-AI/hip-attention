import math
import os

from matplotlib import pyplot as plt
from hip.models.hip_attention.attention2_draft_prefetch import (
    hip_attention, 
    masking_iteration_draft,
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
    load_checkouts,
    block_sparse_attention,
    to_dense,
    adjust_rope,
)
import torch
from torch import Tensor
from typing import List, Dict, Optional, Tuple
import triton
import triton.language as tl
import numpy as np
import cv2

DEBUG = (os.getenv('HIP_DEBUG', '0') == '1')

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
    
    CHUNK_COUNT: int,
    TSRC: int,
    TDST: int,
    HEAD: int,
    sliding_window_size: int,
    num_sinks: int,
    model_context_length: int,
    
    BLOCK_HID: tl.constexpr = 128,
    BLOCK_SIZE_Q: tl.constexpr = 32,
    STRIDE_Q: tl.constexpr = 1,
    BLOCK_CHUNK: tl.constexpr = 32,
    HEAD_GROUP: tl.constexpr = 4,
    REDUCE: tl.constexpr = 'mean',
    USING_EXTEND: tl.constexpr = False,
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
    
    pos_tdst_min = (idx_bdst * BLOCK_SIZE_Q + TSRC - TDST - sliding_window_size - num_sinks).to(tl.int32)
    pos_tdst_min = tl.maximum(pos_tdst_min, 0)
    real_pos_tdst_min = idx_bdst * BLOCK_SIZE_Q + TSRC - TDST
    
    idx_chunk = idx_bchunk * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
    mask_chunk = idx_chunk < CHUNK_COUNT
    
    idx_tsrc_left = tl.load(
        INDICES_LEFT +\
            idx_bsz * stride_indices_left_bsz +\
            idx_bdst * stride_indices_left_bdst +\
            idx_head * stride_indices_left_head +\
            idx_chunk * stride_indices_left_chunk,
        mask=mask_chunk,
        other=TSRC,
    ).to(tl.int32)
    
    idx_tsrc_right = tl.load(
        INDICES_RIGHT +\
            idx_bsz * stride_indices_right_bsz +\
            idx_bdst * stride_indices_right_bdst +\
            idx_head * stride_indices_right_head +\
            idx_chunk * stride_indices_right_chunk,
        mask=mask_chunk,
        other=TSRC,
    ).to(tl.int32)
    
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
    
    if USING_EXTEND:
        if real_pos_tdst_min >= model_context_length:
            old_tdst = idx_tdst + TSRC - TDST
            new_tdst = idx_tdst * 0 + model_context_length - 1
            # new_tdst = tl.where(mask_tdst, new_tdst, old_tdst)
            # new_tdst = old_tdst // 16
            
            queries = adjust_rope(
                queries,
                old_tdst,
                new_tdst,
                mask_tdst,
                idx_hid,
                COS, stride_cos_t, stride_cos_hid,
                SIN, stride_sin_t, stride_sin_hid,
                BLOCK_SIZE_Q // STRIDE_Q, 
                BLOCK_HID,
            ).to(queries.dtype)
            queries = (queries * mask_tdst[:, None]).to(queries.dtype)
    
    scores = tl.zeros((BLOCK_CHUNK,), dtype=tl.float32) - 32000.0
    
    while (max_chunk_size > 1):
        max_chunk_size /= 2.0
        mask_tsrc_active = mask_chunk & (idx_tsrc_left < idx_tsrc_right) & (idx_tsrc_left <= pos_tdst_min)
        idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2
        
        idx_tsrc = (idx_tsrc_left + idx_tsrc_center) // 2
        keys_left = tl.load(
            K +\
                idx_bsz * stride_k_bsz +\
                idx_tsrc[None, :] * stride_k_tsrc +\
                (idx_head // HEAD_GROUP) * stride_k_head_kv +\
                idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc_active[None, :],
            other=0,
        )
        
        if USING_EXTEND:
            if real_pos_tdst_min >= model_context_length:
                old_tsrc = idx_tsrc
                
                new_tsrc = (old_tsrc * (model_context_length / real_pos_tdst_min)).to(tl.int32)
                
                keys_left = keys_left.trans(1, 0)
                keys_left = adjust_rope(
                    keys_left,
                    old_tsrc,
                    new_tsrc,
                    mask_tsrc_active,
                    idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    BLOCK_CHUNK,
                    BLOCK_HID,
                ).to(keys_left.dtype)
                keys_left = tl.trans(keys_left, 1, 0)
                keys_left = (keys_left * mask_tsrc_active[None, :]).to(keys_left.dtype)
        
        scores_left = tl.dot(
            queries,
            keys_left.to(queries.dtype),
            out_dtype=tl.float32,
        ).to(queries.dtype)
        
        if REDUCE == 'max':
            scores_left = tl.where(mask_tdst[:, None], scores_left, float('-inf'))
            scores_left = tl.max(scores_left, axis=0).to(scores_left.dtype)
        elif REDUCE == 'mean':
            scores_left = tl.where(mask_tdst[:, None], scores_left, float('0'))
            scores_left = tl.sum(scores_left, axis=0).to(scores_left.dtype)
            scores_left = (scores_left / tl.sum(mask_tdst.to(tl.float32))).to(scores_left.dtype)
        else:
            raise Exception()
        scores_left = tl.where(mask_tsrc_active, scores_left, float('-inf')).to(scores_left.dtype)
        
        idx_tsrc = (idx_tsrc_center + idx_tsrc_right) // 2
        keys_right = tl.load(
            K +\
                idx_bsz * stride_k_bsz +\
                idx_tsrc[None, :] * stride_k_tsrc +\
                (idx_head // HEAD_GROUP) * stride_k_head_kv +\
                idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc_active[None, :],
            other=0,
        )
        
        if USING_EXTEND:
            if real_pos_tdst_min >= model_context_length:
                old_tsrc = idx_tsrc
                
                new_tsrc = (old_tsrc * (model_context_length / real_pos_tdst_min)).to(tl.int32)
                
                keys_right = keys_right.trans(1, 0)
                keys_right = adjust_rope(
                    keys_right, 
                    old_tsrc, 
                    new_tsrc, 
                    mask_tsrc_active,
                    idx_hid,
                    COS, stride_cos_t, stride_cos_hid,
                    SIN, stride_sin_t, stride_sin_hid,
                    BLOCK_CHUNK, 
                    BLOCK_HID,
                ).to(keys_right.dtype)
                keys_right = tl.trans(keys_right, 1, 0).to(keys_right.dtype)
                keys_right = (keys_right * mask_tsrc_active[None, :]).to(keys_right.dtype)
        
        scores_right = tl.dot(
            queries,
            keys_right.to(queries.dtype),
            out_dtype=tl.float32,
        ).to(queries.dtype)
        
        if REDUCE == 'max':
            scores_right = tl.where(mask_tdst[:, None], scores_right, float('-inf'))
            scores_right = tl.max(scores_right, axis=0).to(scores_right.dtype)
        elif REDUCE == 'mean':
            scores_right = tl.where(mask_tdst[:, None], scores_right, float('0'))
            scores_right = tl.sum(scores_right, axis=0).to(scores_right.dtype)
            scores_right = (scores_right / tl.sum(mask_tdst.to(tl.float32))).to(scores_right.dtype)
        else:
            raise Exception()
        scores_right = tl.where(mask_tsrc_active, scores_right, float('-inf')).to(scores_right.dtype)
        
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
        INDICES_LEFT +\
            idx_bsz * stride_indices_left_bsz +\
            idx_bdst * stride_indices_left_bdst +\
            idx_head * stride_indices_left_head +\
            idx_chunk * stride_indices_left_chunk,
        value=idx_tsrc_left,
        mask=mask_chunk,
    )
    
    tl.store(
        INDICES_RIGHT +\
            idx_bsz * stride_indices_right_bsz +\
            idx_bdst * stride_indices_right_bdst +\
            idx_head * stride_indices_right_head +\
            idx_chunk * stride_indices_right_chunk,
        value=idx_tsrc_right,
        mask=mask_chunk,
    )
    
    tl.store(
        OUT_SCORES +\
            idx_bsz * stride_out_scores_bsz +\
            idx_bdst * stride_out_scores_bdst +\
            idx_head * stride_out_scores_head +\
            idx_chunk * stride_out_scores_chunk,
        value=scores,
        mask=mask_chunk,
    )

def dual_stage_quadratic_hip_attention(
    q: Tensor, 
    k: Optional[Tensor], 
    v: Optional[Tensor], 
    args: HiPAttentionArgs,
    second_stage_k: int = 1024,
    stages = [
        (256, 8192),
        (128, 4096),
        (64, 2048),
    ],
    mask_only = False,
    model_context_length = 16384,
    block_sparse_block_size_q: Optional[int] = 32,
):
    DEBUG_HEAD = -1
    global DEBUG
    
    BSZ, TDST, HEAD, HID = q.shape
    BSZ, TSRC, HEAD_KV, HID = k.shape
    assert v.shape == k.shape
    
    chunk_size = args.mask_k
    chunk_count = triton.cdiv(TSRC - args.sink_token_size - args.sliding_window_size, chunk_size)
    
    MAX_TDST = TDST
    MAX_TSRC = TSRC
    
    args = args.clone()
    args.sliding_window_size = max(0, args.sliding_window_size - args.mask_k)
    
    if args.position_ids is not None:
        position_ids = args.position_ids
    else:
        position_ids = (torch.arange(0, TDST, device=q.device) + (TSRC - TDST))[None, :].expand(BSZ, TDST)
    assert position_ids.shape == (BSZ, TDST), position_ids.shape
    
    BLOCK_CHUNK=args.block_size_k
    BLOCK_SIZE_Q=args.block_size_q
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    
    indices_left = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count), 
        device=q.device,
        dtype=torch.int64
    )
    indices_right = torch.zeros(
        (BSZ, triton.cdiv(TDST, BLOCK_SIZE_Q), HEAD, chunk_count), 
        device=q.device,
        dtype=torch.int64
    )
    
    indices_left[:, :, :, :] = (
        torch.floor(
            torch.arange(0, chunk_count, device=q.device, dtype=torch.float64) * chunk_size + args.sink_token_size
        ).to(indices_left.dtype)
    )[None, None, None, :]
    indices_right[:, :, :, :] = indices_left + chunk_size
    indices_right.clamp_max_(TSRC - args.sliding_window_size)
    
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    out_scores = torch.full(
        (BSZ, BDST, HEAD, triton.next_power_of_2(chunk_count)), 
        device=q.device,
        dtype=q.dtype,
        fill_value=-32000.0
    )
    
    # print(q.shape, k.shape, args.rope_cos.shape, args.rope_sin.shape, TDST, TSRC)
    
    grid = (BSZ * triton.cdiv(chunk_count, BLOCK_CHUNK) * triton.cdiv(TDST, BLOCK_SIZE_Q) * HEAD,)
    chunk_controllable_sampling_mask_cuda[grid](
        q, *q.stride(),
        k, *k.stride(),
        indices_left, *indices_left.stride(),
        indices_right, *indices_right.stride(),
        out_scores, *out_scores.stride(),
        args.rope_cos, *args.safe_stride(args.rope_cos, 2),
        args.rope_sin, *args.safe_stride(args.rope_sin, 2),
        
        chunk_count,
        TSRC,
        TDST,
        HEAD,
        args.sliding_window_size - max(BLOCK_SIZE_Q, BLOCK_CHUNK),
        args.sink_token_size,
        model_context_length,
        
        BLOCK_HID=q.shape[-1],
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        STRIDE_Q=args.block_stride_q,
        BLOCK_CHUNK=BLOCK_CHUNK,
        HEAD_GROUP=HEAD // HEAD_KV,
        USING_EXTEND=args.using_extend,
    )
    
    out_scores[..., indices_left.shape[-1]:] = float('-inf')
    _, t_indices = out_scores.sort(dim=-1, descending=True, stable=False)
    indices_left = indices_left.gather(dim=-1, index=t_indices[..., :indices_left.shape[-1]])
    indices_right = indices_right.gather(dim=-1, index=t_indices[..., :indices_right.shape[-1]])
    
    if DEBUG:
        out_indices_cpu = indices_left.cpu()
        debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK)))
        for i in range(out_indices_cpu.shape[1]):
            t_chunk_size = triton.cdiv(TDST, chunk_count * BLOCK_CHUNK)
            # print(i, t_chunk_size)
            for j in range(max(
                0,
                math.ceil(out_indices_cpu.shape[-1] * (stages[0][1] / TDST))
            )):
                if j >= out_indices_cpu.shape[-1]: continue
                t = (out_indices_cpu[0, i, DEBUG_HEAD, j] - args.sink_token_size) // BLOCK_CHUNK + args.sink_token_size // BLOCK_CHUNK
                t = t // t_chunk_size * t_chunk_size
                debug[i, t:t+t_chunk_size] = 1
        cv2.imwrite('dummy_sampled.png', debug * 255)
        print('saved dummy_sampled.png')
        
        debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_CHUNK)))
        for i in range(out_indices_cpu.shape[1]):
            t_chunk_size = triton.cdiv(TDST, chunk_count * BLOCK_CHUNK)
            # print(i, t_chunk_size)
            for j in range(max(
                0,
                math.ceil(out_indices_cpu.shape[-1] * (second_stage_k / TDST))
            )):
                if j >= out_indices_cpu.shape[-1]: continue
                t = (out_indices_cpu[0, i, DEBUG_HEAD, j] - args.sink_token_size) // BLOCK_CHUNK + args.sink_token_size // BLOCK_CHUNK
                t = t // t_chunk_size * t_chunk_size
                debug[i, t:t+t_chunk_size] = 1
        cv2.imwrite('dummy_sampled_cut.png', debug * 255)
        print('saved dummy_sampled_cut.png')
    
    for i_stage, (stage_chunk_size, stage_k) in enumerate(stages):
        # if stage_chunk_size > chunk_size: continue
        # if stage_k > TSRC: continue
        
        assert (stage_k % chunk_size) == 0
        indices_left = indices_left[..., :stage_k // chunk_size]
        indices_left = ((indices_left - args.sink_token_size) // chunk_size * chunk_size + args.sink_token_size)
        indices_right = (indices_left + chunk_size)
        out_scores = out_scores[..., :stage_k // chunk_size]
        
        indices_left, t_indices = indices_left.sort(dim=-1)
        indices_right = indices_right.gather(dim=-1, index=t_indices)
        out_scores = out_scores.gather(dim=-1, index=t_indices)
        
        assert (chunk_size % stage_chunk_size) == 0
        splits = chunk_size // stage_chunk_size
        chunk_sizes = ((indices_right - indices_left).float() / splits).clamp_min_(0)
        indices_left = indices_left[..., None] + (torch.arange(0, splits, device=q.device)[None, None, None, None, :] * chunk_sizes[..., None]).floor().long()
        indices_left = indices_left.flatten(-2, -1)
        indices_right = indices_right[..., None] - (((splits - 1) - torch.arange(0, splits, device=q.device)[None, None, None, None, :]) * chunk_sizes[..., None]).floor().long()
        indices_right = indices_right.flatten(-2, -1)
        out_scores = out_scores.repeat_interleave(splits, -1)
        
        chunk_size = stage_chunk_size
        chunk_count = indices_left.shape[-1]
        BLOCK_CHUNK = max(16, triton.next_power_of_2(min(chunk_count, BLOCK_CHUNK)))
        
        grid = (BSZ * triton.cdiv(chunk_count, BLOCK_CHUNK) * triton.cdiv(TDST, BLOCK_SIZE_Q) * HEAD,)
        chunk_controllable_sampling_mask_cuda[grid](
            q, *q.stride(),
            k, *k.stride(),
            indices_left, *indices_left.stride(),
            indices_right, *indices_right.stride(),
            out_scores, *out_scores.stride(),
            args.rope_cos, *args.safe_stride(args.rope_cos, 2),
            args.rope_sin, *args.safe_stride(args.rope_sin, 2),
            
            chunk_count,
            TSRC,
            TDST,
            HEAD,
            args.sliding_window_size - max(BLOCK_SIZE_Q, BLOCK_CHUNK),
            args.sink_token_size,
            model_context_length,
            
            BLOCK_HID=q.shape[-1],
            BLOCK_SIZE_Q=BLOCK_SIZE_Q,
            STRIDE_Q=args.block_stride_q,
            BLOCK_CHUNK=BLOCK_CHUNK,
            HEAD_GROUP=HEAD // HEAD_KV,
            USING_EXTEND=args.using_extend,
        )
        
        _, t_indices = out_scores.sort(dim=-1, descending=True, stable=False)
        indices_left = indices_left.gather(dim=-1, index=t_indices)
        indices_right = indices_right.gather(dim=-1, index=t_indices)
        
        if DEBUG:
            out_indices_cpu = indices_left.cpu()
            debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q)))
            for i in range(out_indices_cpu.shape[1]):
                for j in range(math.ceil(stage_k / chunk_size)):
                    if j >= out_indices_cpu.shape[-1]: continue
                    t = out_indices_cpu[0, i, 7, j] // BLOCK_SIZE_Q
                    debug[i, t:t+triton.cdiv(chunk_size, BLOCK_SIZE_Q)] = 1
            cv2.imwrite(f'dummy_sampled_stage_{i_stage}.png', debug * 255)
            print(f'saved dummy_sampled_stage_{i_stage}.png')
    
    assert (second_stage_k % chunk_size) == 0
    indices = indices_left[..., :second_stage_k // chunk_size] // chunk_size * chunk_size
    
    if DEBUG:
        out_indices_cpu = indices.cpu()
        debug = np.zeros((triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q)))
        for i in range(out_indices_cpu.shape[1]):
            for j in range(indices.shape[-1]):
                if j >= out_indices_cpu.shape[-1]: continue
                t = out_indices_cpu[0, i, DEBUG_HEAD, j] // BLOCK_SIZE_Q
                debug[i, t:t+1] = 1
        cv2.imwrite('dummy_sampled_final.png', debug * 255)
        print('saved dummy_sampled_final.png')
        input('>>>')
    
    args = args.clone()
    args.sliding_window_size += args.mask_k
    args.block_size_k = chunk_size
    args.mask_k = second_stage_k
    args.using_extend = args.using_extend and True
    
    indices = indices.permute(0, 2, 1, 3).flatten(0, 1)
    
    indices, _ = indices.sort(dim=-1)
    indices = indices // args.block_size_k * args.block_size_k
    
    unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
    indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
    indices = indices.sort(dim=-1).values
    active_mask = indices < (position_ids[:, ::args.block_size_q, None].repeat_interleave(HEAD, 0) + args.block_size_q)
    ks = active_mask.int().sum(-1)
    ks_count = ks.unsqueeze(-1)
    ks_start_end = torch.zeros((ks.shape[0], ks.shape[1], 2), dtype=torch.int32, device=q.device)
    ks_start_end[:, :, -1] = ks
    
    if  (block_sparse_block_size_q is not None) and (block_sparse_block_size_q != args.block_size_q):
        assert (BLOCK_SIZE_Q % block_sparse_block_size_q) == 0
        indices = indices.repeat_interleave(BLOCK_SIZE_Q // block_sparse_block_size_q, 1)
        ks = ks.repeat_interleave(BLOCK_SIZE_Q // block_sparse_block_size_q, 1)
        ks_count = ks_count.repeat_interleave(BLOCK_SIZE_Q // block_sparse_block_size_q, 1)
        ks_start_end = ks_start_end.repeat_interleave(BLOCK_SIZE_Q // block_sparse_block_size_q, 1)
        args.block_size_q = block_sparse_block_size_q
    
    if mask_only:
        return
    
    context = block_sparse_attention(
        q=q, k=k, v=v,
        seq_lens=position_ids + 1,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args,
        EXTEND_BACKEND='streaming',
        model_context_length=model_context_length,
    )
    
    if DEBUG:
        print(context)
    
    return context

def main_debug():
    global DEBUG
    
    seq_len = 131072
    seq_dups = int(os.getenv('DUPS', '1'))
    using_extend = os.getenv('USING_EXTEND', '1') == '1'
    
    q, k, v, out, cos, sin = load_checkouts(
        idx=0, 
        window=40, 
        seq_len=seq_len, 
        return_cos_sin=using_extend, 
        dtype=torch.bfloat16
    )
    HEAD = q.shape[0]
    HEAD_KV = k.shape[0]
    seq_len = seq_len * seq_dups
    
    q = q.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    k = k.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    v = v.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    if cos is not None:
        cos = cos.repeat(seq_dups, 1)
        sin = sin.repeat(seq_dups, 1)
    
    from flash_attn import flash_attn_func
    
    print('-' * 20)
    
    for i in range(0):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        flash_attn_func(
            q, k, v, causal=True
        )
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    print('-' * 20)
    
    for i in range(0):
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
            mask_only=False,
        )
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    print('-' * 20)
    
    for i in range(10):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        if i==0: DEBUG = os.getenv('DEBUG', '0') == '1'
        
        # print(cos.shape)
        # print(sin.shape)
        
        dual_stage_quadratic_hip_attention(
            q, k, v,
            args=HiPAttentionArgs(
                mask_k=256,
                block_size_q=64,
                block_stride_q=4,
                block_size_k=64, # BLOCK_CHUNK
                sliding_window_size=1024,
                sink_token_size=256,
                # position_ids=position_ids,
                
                using_extend=using_extend,
                rope_cos=cos,
                rope_sin=sin,
            ),
            second_stage_k=2048,
            stages=[
                (64, 8192),
            ],
            block_sparse_block_size_q=32,
            model_context_length=65536,
        )
        
        if i==0: DEBUG = False
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))

if __name__ == '__main__':
    main_debug()