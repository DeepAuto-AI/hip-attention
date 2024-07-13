import time
import torch
from torch import Tensor
import triton
import triton.language as tl

@triton.jit
def load_tokens(
    ptr, stride_ptr_n, stride_ptr_t, stride_ptr_hid, 
    idx_n, idx_t, mask_t, HID: tl.constexpr
):
    return tl.load(
        ptr +\
            idx_n * stride_ptr_n +\
            idx_t[:, None] * stride_ptr_t +\
            tl.arange(0, HID)[None, :] * stride_ptr_hid,
        mask = mask_t[:, None]
    )

@triton.jit
def attention_norm_cuda(
    Q, stride_q_n, stride_q_tdst, stride_q_hid,
    K, stride_k_n, stride_k_tsrc, stride_k_hid,
    
    NORM, stride_norm_n, stride_norm_tdst,
    
    TDST, TSRC,
    
    HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    idx_n = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_tdst = tl.arange(0, BLOCK_SIZE_Q) + idx_bdst * BLOCK_SIZE_Q
    mask_tdst = idx_tdst < TDST
    
    q = load_tokens(
        Q, stride_q_n, stride_q_tdst, stride_q_hid, 
        idx_n, idx_tdst, mask_tdst, HID
    )
    
    score_max = tl.full((BLOCK_SIZE_Q, ), dtype=tl.float32, value=float('-inf'))
    for i_tsrc in range(0, TSRC, BLOCK_SIZE_K):
        idx_tsrc = i_tsrc + tl.arange(0, BLOCK_SIZE_K)
        mask_tsrc = idx_tsrc < TSRC
        
        k = load_tokens(
            K, stride_k_n, stride_k_tsrc, stride_k_hid,
            idx_n, idx_tsrc, mask_tsrc, HID,
        )
        
        qk = tl.dot(
            q, k.trans(1, 0),
            allow_tf32=True
        ).to(tl.float32)
        
        qk = tl.where(
            idx_tsrc[None, :] <= idx_tdst[:, None],
            qk, float('-inf')
        )
        
        score_max = tl.maximum(
            score_max,
            tl.max(qk, axis=-1)
        )
    
    exp_score_sum = tl.zeros((BLOCK_SIZE_Q, ), dtype=tl.float32)
    for i_tsrc in range(0, TSRC, BLOCK_SIZE_K):
        idx_tsrc = i_tsrc + tl.arange(0, BLOCK_SIZE_K)
        mask_tsrc = idx_tsrc < TSRC
        
        k = load_tokens(
            K, stride_k_n, stride_k_tsrc, stride_k_hid,
            idx_n, idx_tsrc, mask_tsrc, HID,
        )
        
        qk = tl.dot(
            q, k.trans(1, 0),
            allow_tf32=True
        ).to(tl.float32)
        
        qk = tl.where(
            idx_tsrc[None, :] <= idx_tdst[:, None],
            qk, float('-inf')
        )
        
        qk = qk - score_max[:, None]
        qk = tl.exp(qk)
        exp_score_sum += tl.sum(qk, axis=-1)
    
    norm_sum = tl.zeros((BLOCK_SIZE_Q, ), dtype=tl.float64)
    for i_tsrc in range(0, TSRC, BLOCK_SIZE_K):
        idx_tsrc = i_tsrc + tl.arange(0, BLOCK_SIZE_K)
        mask_tsrc = idx_tsrc < TSRC
        
        k = load_tokens(
            K, stride_k_n, stride_k_tsrc, stride_k_hid,
            idx_n, idx_tsrc, mask_tsrc, HID,
        )
        
        qk = tl.dot(
            q, k.trans(1, 0),
            allow_tf32=True
        ).to(tl.float32)
        
        qk = tl.where(
            idx_tsrc[None, :] <= idx_tdst[:, None],
            qk, float('-inf')
        )
        
        qk = qk - score_max[:, None]
        prob = tl.exp(qk) / tl.maximum(exp_score_sum[:, None], 1e-20)
        norm_sum += tl.sum(prob * prob, axis=-1)
    
    norm = tl.sqrt(norm_sum)
    
    tl.store(
        NORM +\
            idx_n * stride_norm_n +\
            idx_tdst * stride_norm_tdst,
        value=norm,
        mask=mask_tdst,
    )

def attention_norm(
    q: Tensor,
    k: Tensor,
):
    """
    q: fp*[N, TDST, HID]
    k: fp*[N, TSRC, HID]
    
    # return
    norm: fp32[N, TDST]
    """
    assert q.ndim == 3
    assert q.shape == k.shape
    
    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape
    
    norm = torch.zeros((N, TDST), dtype=torch.float32, device=q.device)
    
    BLOCK_SIZE_Q = 32
    BLOCK_SIZE_K = 64
    
    grid = (N, triton.cdiv(TDST, BLOCK_SIZE_Q))
    
    pre_device = torch.get_default_device()
    torch.set_default_device(q.device)
    attention_norm_cuda[grid](
        q, *q.stride(),
        k, *k.stride(),
        norm, *norm.stride(),
        
        TDST, TSRC,
        
        q.shape[-1],
        BLOCK_SIZE_Q, 
        BLOCK_SIZE_K,
        
        num_warps=4,
        num_stages=2,
    )
    torch.set_default_device(pre_device)
    
    return norm

if __name__ == '__main__':
    q = torch.randn((32, 8192, 128), dtype=torch.float16, device=0)
    k = torch.randn((32, 8192, 128), dtype=torch.float16, device=0)
    
    for _ in range(10):
        torch.cuda.synchronize()
        t = time.time()
        truth = q @ k.transpose(-1, -2)
        truth = torch.where(
            torch.arange(0, k.shape[1], device=q.device)[None, None, :] <= torch.arange(0, q.shape[1], device=q.device)[None, :, None],
            truth,
            float('-inf')
        )
        truth = torch.norm(torch.softmax(truth, dim=-1), dim=-1)
        torch.cuda.synchronize()
        elapsed_turth = time.time() - t
    print(truth[-1])
    
    for _ in range(10):
        torch.cuda.synchronize()
        t = time.time()
        norm = attention_norm(q, k)
        torch.cuda.synchronize()
        elapsed_triton = time.time() - t
    print(norm[-1])
    
    print(elapsed_turth * 1000, elapsed_triton * 1000)