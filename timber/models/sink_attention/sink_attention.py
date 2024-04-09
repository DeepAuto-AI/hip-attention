"""
Streaming-LLM: Triton Implementation
gmlwns2000 @ github
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

@triton.jit
def load_rotary_embedded_vector(
    QK, stride_qk_n, stride_qk_t, stride_qk_hid,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    
    idx_n,
    idx_t_qk,
    idx_t_rope,
    
    HID,
    BLOCK_HID,
):
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    
    idx_hid_rot = (idx_hid + HID // 2) % HID
    mask_hid_rot = mask_hid
    
    vec = tl.load(
        QK +\
            idx_n * stride_qk_n +\
            idx_t_qk * stride_qk_t +\
            idx_hid * stride_qk_hid,
        mask = mask_hid,
        other = 0,
    )
    
    vec_rot = tl.load(
        QK +\
            idx_n * stride_qk_n +\
            idx_t_qk * stride_qk_t +\
            idx_hid_rot * stride_qk_hid,
        mask = mask_hid_rot,
        other = 0,
    )
    vec_rot = tl.where(idx_hid < HID // 2, -vec_rot, vec_rot)
    
    cos = tl.load(
        COS +\
            idx_t_rope * stride_cos_t +\
            idx_hid * stride_cos_hid,
        mask=mask_hid,
        other=0,
    )
    sin = tl.load(
        SIN +\
            idx_t_rope * stride_sin_t +\
            idx_hid * stride_sin_hid,
        mask=mask_hid,
        other=0,
    )
    
    vec_rope = ((vec.to(tl.float32) * cos) + (vec_rot.to(tl.float32) * sin)).to(vec.dtype)
    
    return vec_rope, vec, vec_rot, cos, sin

@triton.jit
def grad_rotary_embedded_vector(
    grad_vec_rope, vec_origin, vec_rot, cos, sin,
    HID, BLOCK_HID,
):
    grad_vec_origin = grad_vec_rope * cos
    idx_vec_origin_hid = tl.arange(0, BLOCK_HID)
    
    grad_vec_rot = grad_vec_rope * sin
    grad_vec_rot = tl.where(idx_vec_origin_hid < HID // 2, -grad_vec_rot, grad_vec_rot)
    idx_vec_rot_hid = (idx_vec_origin_hid + HID // 2) % HID
    
    return grad_vec_origin, idx_vec_origin_hid, grad_vec_rot, idx_vec_rot_hid

@triton.jit
def _attention_scores_compute(
    # input tensors
    Q, stride_q_n, stride_q_tdst, stride_q_hid,
    K, stride_k_n, stride_k_tsrc, stride_k_hid,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    
    # output tensors
    INDICES, stride_indices_d, stride_indices_z,
    VALUES, stride_values_z,
    
    # input variables
    N, TDST, TSRC, HID,
    NUM_SINK,
    WINDOW_SIZE,
    
    # kernel constants
    BLOCK_HID: tl.constexpr,
):
    idx_n = tl.program_id(0).to(tl.int64)
    idx_tdst = tl.program_id(1).to(tl.int64)
    idx_k = tl.program_id(2).to(tl.int64)
    
    tdst = idx_tdst + TSRC - TDST
    
    if idx_k < NUM_SINK:
        idx_tsrc = idx_k
    else:
        window_offset = idx_k - NUM_SINK
        t_tsrc = tdst - WINDOW_SIZE + 1 + window_offset
        idx_tsrc = tl.maximum(idx_k, t_tsrc)
    
    # load key
    key, _, _, _, _ = load_rotary_embedded_vector(
        K, stride_k_n, stride_k_tsrc, stride_k_hid,
        COS, stride_cos_t, stride_cos_hid,
        SIN, stride_sin_t, stride_sin_hid,
        idx_n, idx_tsrc, idx_k,
        HID, BLOCK_HID
    )
    
    # load query
    query, _, _, _, _ = load_rotary_embedded_vector(
        Q, stride_q_n, stride_q_tdst, stride_q_hid,
        COS, stride_cos_t, stride_cos_hid,
        SIN, stride_sin_t, stride_sin_hid,
        idx_n, idx_tdst, tl.minimum(tdst, WINDOW_SIZE + NUM_SINK),
        HID, BLOCK_HID,
    )
    
    # calc dot product.
    score = tl.sum(query.to(tl.float32) * key.to(tl.float32))
    score = score * (1 / tl.sqrt(HID.to(tl.float32)))
    score = tl.where(idx_tsrc <= tdst, score, float('-inf'))
    
    # output
    idx_z = idx_n * TDST * (WINDOW_SIZE + NUM_SINK) + idx_tdst * (WINDOW_SIZE + NUM_SINK) + idx_k
    tl.store(
        VALUES +\
            idx_z * stride_values_z,
        value = score
    )
    tl.store(
        INDICES +\
            0 * stride_indices_d +\
            idx_z * stride_indices_z,
        value = idx_n
    )
    tl.store(
        INDICES +\
            1 * stride_indices_d +\
            idx_z * stride_indices_z,
        value = idx_tdst
    )
    tl.store(
        INDICES +\
            2 * stride_indices_d +\
            idx_z * stride_indices_z,
        value = idx_tsrc
    )

@triton.jit
def _attention_score_backward_compute(
    # input tensors
    GRAD_VALUES, stride_grad_values_z,
    Q, stride_q_n, stride_q_tdst, stride_q_hid,
    K, stride_k_n, stride_k_tsrc, stride_k_hid,
    INDICES, stride_indices_d, stride_indices_z,
    COS, stride_cos_t, stride_cos_hid,
    SIN, stride_sin_t, stride_sin_hid,
    
    # output tensors
    GRAD_Q, stride_grad_q_n, stride_grad_q_tdst, stride_grad_q_hid,
    GRAD_K, stride_grad_k_n, stride_grad_k_tsrc, stride_grad_k_hid,
    
    # input variables
    N, TDST, TSRC, HID, NNZ,
    NUM_SINK,
    WINDOW_SIZE,
    
    # block constant
    BLOCK_HID: tl.constexpr,
):
    idx_z = tl.program_id(0)
    
    idx_n = tl.load(
        INDICES +\
            0 * stride_indices_d +\
            idx_z * stride_indices_z
    ).to(tl.int64)
    idx_tdst = tl.load(
        INDICES +\
            1 * stride_indices_d +\
            idx_z * stride_indices_z
    ).to(tl.int64)
    idx_tsrc = tl.load(
        INDICES +\
            2 * stride_indices_d +\
            idx_z * stride_indices_z
    ).to(tl.int64)
    tdst = idx_tdst + TSRC - TDST
    
    idx_k = idx_z % (NUM_SINK + WINDOW_SIZE)
    
    # load key
    key, key_origin, key_rot, cos_k, sin_k = load_rotary_embedded_vector(
        K, stride_k_n, stride_k_tsrc, stride_k_hid,
        COS, stride_cos_t, stride_cos_hid,
        SIN, stride_sin_t, stride_sin_hid,
        idx_n, idx_tsrc, idx_k,
        HID, BLOCK_HID
    )
    
    # load query
    query, query_origin, query_rot, cos_q, sin_q = load_rotary_embedded_vector(
        Q, stride_q_n, stride_q_tdst, stride_q_hid,
        COS, stride_cos_t, stride_cos_hid,
        SIN, stride_sin_t, stride_sin_hid,
        idx_n, idx_tdst, tl.minimum(tdst, WINDOW_SIZE + NUM_SINK),
        HID, BLOCK_HID,
    )
    
    # load value grad
    grad_score = tl.load(
        GRAD_VALUES +\
            idx_z * stride_grad_values_z,
    )
    
    grad_score = tl.where(idx_tsrc <= tdst, grad_score, 0)
    grad_score = grad_score * (1 / tl.sqrt(HID.to(tl.float32)))
    
    grad_key = grad_score * query
    grad_query = grad_score * key
    
    grad_key_origin, idx_key_origin_hid, grad_key_rot, idx_key_rot_hid = grad_rotary_embedded_vector(
        grad_key, key_origin, key_rot, cos_k, sin_k,
        HID, BLOCK_HID
    )
    grad_query_origin, idx_query_origin_hid, grad_query_rot, idx_query_rot_hid = grad_rotary_embedded_vector(
        grad_query, query_origin, query_rot, cos_q, sin_q,
        HID, BLOCK_HID
    )
    
    mask_hid = tl.arange(0, BLOCK_HID) < HID
    
    tl.atomic_add(
        GRAD_K +\
            idx_n * stride_grad_k_n +\
            idx_tsrc * stride_grad_k_tsrc +\
            idx_key_origin_hid * stride_grad_k_hid,
        mask = mask_hid,
        val = grad_key_origin
    )
    tl.atomic_add(
        GRAD_K +\
            idx_n * stride_grad_k_n +\
            idx_tsrc * stride_grad_k_tsrc +\
            idx_key_rot_hid * stride_grad_k_hid,
        mask = mask_hid,
        val = grad_key_rot
    )
    
    tl.atomic_add(
        GRAD_Q +\
            idx_n * stride_grad_q_n +\
            idx_tdst * stride_grad_q_tdst +\
            idx_query_origin_hid * stride_grad_q_hid,
        mask = mask_hid,
        val = grad_query_origin
    )
    tl.atomic_add(
        GRAD_Q +\
            idx_n * stride_grad_q_n +\
            idx_tdst * stride_grad_q_tdst +\
            idx_query_rot_hid * stride_grad_q_hid,
        mask = mask_hid,
        val = grad_query_rot
    )

class AttentionScoreFunc(Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor, 
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
        num_sink: int,
        window_size: int,
    ):
        assert q.ndim == 3
        assert k.ndim == 3
        assert cos.ndim == 2, cos.shape
        assert sin.ndim == 2, sin.shape
        N, TDST, HID = q.shape
        _, TSRC, _ = k.shape
        assert k.shape == (N, TSRC, HID)
        assert cos.shape[-1] == HID
        assert sin.shape[-1] == HID
        
        device = q.device
        dtype = torch.float32 #q.dtype
        
        nnz = N * TDST * (num_sink + window_size)
        indices = torch.zeros((3, nnz), dtype=torch.int64, device=device)
        values = torch.zeros((nnz,), dtype=dtype, device=device)
        
        BLOCK_HID = triton.next_power_of_2(HID)
        
        grid = (N, TDST, num_sink + window_size)
        
        _device = torch.cuda.current_device()
        torch.cuda.set_device(q.device)
        _attention_scores_compute[grid](
            q, *q.stride(),
            k, *k.stride(),
            cos, *cos.stride(),
            sin, *sin.stride(),
            
            indices, *indices.stride(),
            values, *values.stride(),
            
            N, TDST, TSRC, HID,
            num_sink,
            window_size,
            
            BLOCK_HID,
            
            num_warps=1,
            num_stages=1,
        )
        torch.cuda.set_device(_device)
        
        ctx.save_for_backward(
            q, k, cos, sin, indices
        )
        ctx.num_sink = num_sink
        ctx.window_size = window_size
        
        return indices, values

    @staticmethod
    def backward(
        ctx, 
        grad_indices: Tensor, 
        grad_values: Tensor
    ):
        q, k, cos, sin, indices = ctx.saved_tensors
        num_sink = ctx.num_sink
        window_size = ctx.window_size
        
        N, TDST, HID = q.shape
        _, TSRC, _ = k.shape
        _, NNZ = indices.shape
        
        assert q.ndim == 3
        assert k.ndim == 3
        assert cos.ndim == 2
        assert sin.ndim == 2
        assert indices.ndim == 2
        assert grad_values.ndim == 1
        
        grad_q = torch.zeros_like(q, dtype=torch.float32)
        grad_k = torch.zeros_like(k, dtype=torch.float32)
        
        BLOCK_HID = triton.next_power_of_2(HID)
        
        grid = (NNZ,)
        _attention_score_backward_compute[grid](
            grad_values, *grad_values.stride(),
            q, *q.stride(),
            k, *k.stride(),
            indices, *indices.stride(),
            cos, *cos.stride(),
            sin, *sin.stride(),
            grad_q, *grad_q.stride(),
            grad_k, *grad_k.stride(),
            
            N, TDST, TSRC, HID, NNZ, 
            num_sink,
            window_size,
            
            BLOCK_HID,
            
            num_warps=1,
            num_stages=1,
        )
        
        return (
            grad_q, #q: Tensor
            grad_k, #k: Tensor
            None, #cos: Tensor
            None, #sin: Tensor
            None, #num_sink: int
            None, #window_size: int
        )

def attention_scores(
    q: Tensor, 
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    num_sink: int = 4,
    window_size: int = 512,
):
    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape
    
    window_size = min(window_size, TSRC - num_sink)
    
    indices, values = AttentionScoreFunc.apply(
        q, k, cos, sin, num_sink, window_size,
    )
    
    values = values\
        .view(-1, num_sink + window_size)\
        .softmax(-1)\
        .view(-1)\
        .contiguous()
    
    probs = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(N, TDST, TSRC),
        requires_grad=q.requires_grad,
        dtype=values.dtype,
        device=values.device,
        check_invariants=True,
    )
    
    return probs

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        N = q.shape[0]
        TDST = q.shape[1]
        position_ids = torch.arange(0, TDST, device=q.device)[None, :].expand(N, TDST)
    
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos  # [seq_len, dim]
    sin = sin  # [seq_len, dim]
    assert cos.ndim == 2
    cos = cos[position_ids]  # [bs, seq_len, dim]
    sin = sin[position_ids]  # [bs, seq_len, dim]
    assert position_ids.ndim == 2
    assert cos.ndim == 3
    
    q_embed = (q * cos) + (rotate_half(q) * sin) 
    
    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed    

def sink_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cos: Tensor,
    sin: Tensor,
    num_sink: int = 4,
    window_size: int = 512,
):
    _dtype = v.dtype
    
    # COO format
    probs = attention_scores(
        q, k, cos, sin,
        num_sink=num_sink,
        window_size=window_size,
    )
    
    if v.dtype in [torch.bfloat16, torch.float16]:
        v = v.to(torch.float32)
    context = torch.bmm(probs, v)
    context = context.to(_dtype)
    
    return context

if __name__ == '__main__':
    N = 1
    T = 8
    HID = 128
    NSINK = 2
    WIND = 4
    
    std = 0.5
    q = torch.nn.Parameter(torch.randn((N, T, HID), device=0) * std)
    k = torch.nn.Parameter(torch.randn((N, T, HID), device=0) * std)
    v = torch.nn.Parameter(torch.randn((N, T, HID), device=0) * std)
    cos = torch.randn((T, HID), device=q.device)
    sin = cos.clone()
    
    for istep in range(10000):
        x = sink_attention(
            q, k, v, cos, sin, 
            num_sink=NSINK,
            window_size=WIND
        )
        
        # look up diagonal
        loss = (x - v[:, :, :]).square().mean() * 1000
        loss.backward()
        
        lr = 1e-3
        
        # print(q.grad)
        
        q.data += -q.grad * lr
        q.grad = None
        k.data += -k.grad * lr
        k.grad = None
        v.data += -v.grad * lr * 0.1
        v.grad = None
        
        if (istep % 500) == 0:
            print('loss', loss.item())