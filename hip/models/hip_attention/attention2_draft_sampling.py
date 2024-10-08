from hip.models.hip_attention.attention2_draft_prefetch import (
    hip_attention, 
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
    load_checkouts,
)
import torch
from torch import Tensor
from typing import List, Dict, Optional, Tuple

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

def main_debug():
    seq_len = 32768
    
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
        )
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))
    
    for i in range(10):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        
        start.record()
        context = sampling_hip_attention(
            q, k, v,
            first_stage_args=HiPAttentionArgs(
                mask_k=128,
                block_size_q=64,
                block_stride_q=2,
                block_size_k=2,
                block_stride_k=1,
            ),
            second_stage_range_k=1024, 
        )
        end.record()
        
        end.synchronize()
        print(start.elapsed_time(end))

if __name__ == '__main__':
    main_debug()