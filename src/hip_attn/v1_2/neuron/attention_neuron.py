import torch
from torch import Tensor
from torch_xla.core import xla_model as xm
import os
from hip_attn.v1_2.attention_metadata import HiPAttentionArgs, ScanStage
from hip_research.utils.load_checkouts import load_checkouts
from typing import Optional, List
import triton
from dataclasses import dataclass

from hip_attn.v1_2.neuron.kernels.scan_stage_neuron import scan_stage_neuron

@dataclass
class ScanStageKernelOutput:
    indices_left: Tensor
    indices_right: Tensor
    chunk_scores: Tensor

def scan_stage(
    q: torch.Tensor,
    k: torch.Tensor,
    
    stage_info: ScanStage,
    
    previous_output: Optional[ScanStageKernelOutput] = None,
) -> ScanStageKernelOutput:
    BSZ, TDST, HEAD, HID = q.shape
    _, TSRC, HEAD_KV, _ = k.shape
    assert k.shape == (BSZ, TSRC, HEAD_KV, HID)
    assert stage_info.stage_stride == 1
    
    BDST = triton.cdiv(TDST, stage_info.stage_block_size_q)
    
    if previous_output is None:
        indices_left = torch.arange(
            0, 
            TSRC, 
            stage_info.stage_chunk_size, 
            device=q.device, 
            dtype=torch.int32,
        )[None, None, None, :]\
            .expand(BSZ, BDST, HEAD, -1)\
                .contiguous()
        indices_right = indices_left + stage_info.stage_chunk_size
        chunk_scores = torch.full_like(
            indices_left,
            dtype=torch.float32,
            device=q.device,
            fill_value=-32000.0,
        )
    else:
        indices_left = previous_output.indices_left
        indices_right = previous_output.indices_right
        chunk_scores = previous_output.chunk_scores
        
    mask = ScanStageKernelOutput(
        indices_left=indices_left,
        indices_right=indices_right,
        chunk_scores=chunk_scores,
    )
    
    N_CHUNK = indices_left.shape[-1]
    BLOCK_CHUNK = 256
    BN_CHUNK = triton.cdiv(N_CHUNK, BLOCK_CHUNK)
    
    position_ids = torch.arange(0, TDST, device=q.device, dtype=torch.int32)[None, :,].expand(BSZ, -1)
    
    max_chunk_size = stage_info.stage_chunk_size
    
    while max_chunk_size >= 1:
        max_chunk_size /= 2
        (
            out_indices_left, 
            out_indices_right, 
            out_chunk_scores
        ) = scan_stage_neuron[BSZ, BDST, HEAD, BN_CHUNK](
            q,
            k,
            mask.indices_left,
            mask.indices_right,
            mask.chunk_scores,
            position_ids,
            
            stage_info.stage_block_size_q,
            BLOCK_CHUNK,
        )
        mask = ScanStageKernelOutput(
            indices_left=out_indices_left,
            indices_right=out_indices_right,
            chunk_scores=out_chunk_scores,
        )
    
    return mask

def hip_masking(
    q: torch.Tensor,
    k: torch.Tensor,
    
    stages: List[ScanStage],
):
    scan_state = None
    previous_stage = None
    for stage in stages:
        if scan_state is not None:
            assert previous_stage.stage_block_size_q == stage.stage_block_size_q
            assert (previous_stage.stage_chunk_size % stage.stage_chunk_size) == 0
            
            num_chunk_div = previous_stage.stage_chunk_size // stage.stage_chunk_size
            
            indices_left = (
                scan_state.indices_left
                // previous_stage.stage_chunk_size 
                * previous_stage.stage_chunk_size
            )
            indices_left = (
                indices_left[:, :, :, :, None] 
                + torch.arange(0, num_chunk_div, dtype=indices_left.dtype)[None, None, None, None, :]
                * stage.stage_chunk_size
            )
            indices_left = indices_left.flatten(-2, -1)
            indices_right = indices_left + stage.stage_chunk_size
            chunk_scores = scan_state.chunk_scores.repeat_interleave(dim=-1, repeats=num_chunk_div)
            
            _, topk_indices = torch.topk(
                chunk_scores, 
                dim=-1, 
                k=stage.stage_k // stage.stage_chunk_size,
                sorted=False,
            )
            indices_left = indices_left.gather(
                dim=-1, index=topk_indices
            )
            indices_right = indices_right.gather(
                dim=-1, index=topk_indices
            )
            chunk_scores = chunk_scores.gather(
                dim=-1, index=topk_indices
            )
            scan_state = ScanStageKernelOutput(
                indices_left=indices_left,
                indices_right=indices_right,
                chunk_scores=chunk_scores,
            )
        
        previous_stage = stage
        
        scan_state = scan_stage(
            q, k, stage, scan_state
        )
    
    return scan_state

def main_debug():
    seq_len = int(os.getenv("SEQ_LEN", "32768"))
    query_seq_dups = int(os.getenv("Q_DUPS", "-1"))
    seq_dups = int(os.getenv("DUPS", "1"))
    if query_seq_dups < 0:
        query_seq_dups = seq_dups

    assert seq_dups > 0
    
    device = xm.xla_device()
    q, k, v, out, cos, sin = load_checkouts(
        idx=0,
        window=40,
        seq_len=seq_len,
        return_cos_sin=True,
        derope=True,
        dtype=torch.bfloat16,
        device=device,
    )
    HEAD = q.shape[0]
    HEAD_KV = k.shape[0]
    seq_len = seq_len * seq_dups

    q = q.repeat(1, query_seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    k = (
        k.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    )
    v = (
        v.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
    )
    if cos is not None:
        cos = cos.repeat(seq_dups, 1)
        sin = sin.repeat(seq_dups, 1)

    print(q.shape, k.shape, v.shape)
    
    output = hip_masking(
        q, k,
        stages=[
            ScanStage(
                stage_block_size_q=128, 
                stage_block_stride_q=4, 
                stage_chunk_size=128,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=128, 
                stage_block_stride_q=4, 
                stage_chunk_size=32,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=128, 
                stage_block_stride_q=1, 
                stage_chunk_size=8,
                stage_k=8192,
                stage_stride=1,
            ),
        ]
    )
    
    print(output)

if __name__ == '__main__':
    main_debug()