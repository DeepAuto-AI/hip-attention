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

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_T': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 128}, num_warps=8, num_stages=2),
        
        triton.Config({'BLOCK_T': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_T': 256}, num_warps=16, num_stages=2),
        
        triton.Config({'BLOCK_T': 64}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_T': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_T': 64}, num_warps=4, num_stages=2),
    ],
    key=[
        'HID',
        'BLOCK_SIZE_Q', 
        'BLOCK_SIZE_K',
    ],
    use_cuda_graph=True,
)
@triton.jit
def attention_norm_cuda(
    Q, stride_q_n, stride_q_tdst, stride_q_hid,
    K, stride_k_n, stride_k_tsrc, stride_k_hid,
    
    NORM, stride_norm_n, stride_norm_bdst,
    
    TDST, TSRC,
    
    HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    
    BLOCK_T: tl.constexpr,
):
    idx_n = tl.program_id(1)
    idx_bdst = tl.program_id(0)
    idx_tdst = tl.arange(0, BLOCK_SIZE_Q) + idx_bdst * BLOCK_SIZE_Q
    mask_tdst = idx_tdst < TDST
    
    q = load_tokens(
        Q, stride_q_n, stride_q_tdst, stride_q_hid, 
        idx_n, idx_tdst, mask_tdst, HID
    )
    
    score_max = tl.full((), dtype=tl.float32, value=float('-inf'))
    for i_tsrc in range(0, TSRC, BLOCK_T):
        idx_tsrc = i_tsrc + tl.arange(0, BLOCK_T)
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
            tl.max(qk)
        )
    
    exp_score_sum = tl.zeros((), dtype=tl.float32)
    for i_tsrc in range(0, TSRC, BLOCK_T):
        idx_tsrc = i_tsrc + tl.arange(0, BLOCK_T)
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
        qk = tl.reshape(qk, (BLOCK_SIZE_Q, BLOCK_T // BLOCK_SIZE_K, BLOCK_SIZE_K))
        qk = tl.max(qk, axis=0)
        qk = tl.max(qk, axis=-1)
        qk = tl.exp(qk)
        exp_score_sum += tl.sum(qk)
    
    norm_sum = tl.zeros((), dtype=tl.float64)
    for i_tsrc in range(0, TSRC, BLOCK_T):
        idx_tsrc = i_tsrc + tl.arange(0, BLOCK_T)
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
        qk = tl.reshape(qk, BLOCK_SIZE_Q, BLOCK_T // BLOCK_SIZE_K, BLOCK_SIZE_K)
        qk = tl.max(qk, axis=0)
        qk = tl.max(qk, axis=-1)
        prob = tl.exp(qk) / tl.maximum(exp_score_sum[:, None], 1e-20)
        norm_sum += tl.sum(prob * prob, axis=-1)
    
    norm = tl.sqrt(norm_sum)
    
    tl.store(
        NORM +\
            idx_n * stride_norm_n +\
            idx_bdst[:] * stride_norm_bdst,
        value=norm,
    )

def attention_norm(
    q: Tensor,
    k: Tensor,
    
    BLOCK_SIZE_Q: int = 32,
    BLOCK_SIZE_K: int = 2,
):
    """
    q: fp*[N, TDST, HID]
    k: fp*[N, TSRC, HID]
    
    # return
    norm: fp32[N, TDST]
    """
    assert q.ndim == 3
    assert q.shape == k.shape
    assert q.device == k.device
    
    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape
    
    norm = torch.zeros(
        (N, triton.cdiv(TDST, BLOCK_SIZE_Q)), 
        dtype=torch.float32, 
        device=q.device
    )
    
    grid = (triton.cdiv(TDST, BLOCK_SIZE_Q), N)
    
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
    )
    torch.set_default_device(pre_device)
    
    return norm

if __name__ == '__main__':
    q = torch.randn((32, 8192, 128), dtype=torch.float16, device=0)
    k = torch.randn((32, 8192, 128), dtype=torch.float16, device=0)
    block_size_q = 32
    block_size_k = 2
    
    num_samples = 50
    
    elapsed_truth = 0
    for i in range(num_samples):
        torch.cuda.synchronize()
        t = time.time()
        truth = q @ k.transpose(-1, -2)
        truth = torch.where(
            torch.arange(0, k.shape[1], device=q.device)[None, None, :] <= torch.arange(0, q.shape[1], device=q.device)[None, :, None],
            truth,
            float('-inf')
        )
        truth = truth.view(
            q.shape[0], 
            q.shape[1] // block_size_q, 
            block_size_q, 
            k.shape[1] // block_size_k, 
            block_size_k
        )
        truth = truth.amax(-1).amax(-2)
        truth = torch.norm(torch.softmax(truth, dim=-1), dim=-1)
        torch.cuda.synchronize()
        if i > 3:
            elapsed_truth += time.time() - t
    
    elapsed_triton = 0
    for i in range(num_samples):
        torch.cuda.synchronize()
        t = time.time()
        norm = attention_norm(q, k)
        torch.cuda.synchronize()
        if i > 3:
            elapsed_triton += time.time() - t
    
    print('torch', truth[-1, -10:].tolist())
    print('triton', norm[-1, -10:].tolist())
    
    elapsed_truth /= (num_samples - 4)
    elapsed_triton /= (num_samples - 4)
    
    print('torch', elapsed_truth * 1000)
    print('triton', elapsed_triton * 1000)