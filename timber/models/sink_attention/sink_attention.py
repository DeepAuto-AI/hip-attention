import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function

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
    mask_tsrc = idx_tsrc <= tdst
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    
    idx_hid_rot = (idx_hid + HID // 2) % HID
    mask_hid_rot = mask_hid & (idx_hid_rot < HID)
    
    # load key
    key = tl.load(
        K +\
            idx_n * stride_k_n +\
            idx_tsrc * stride_k_tsrc +\
            idx_hid * stride_k_hid,
        mask = mask_tsrc & mask_hid,
        other = 0,
    )
    
    key_rot = tl.load(
        K +\
            idx_n * stride_k_n +\
            idx_tsrc * stride_k_tsrc +\
            idx_hid_rot * stride_k_hid,
        mask = mask_tsrc & mask_hid_rot,
        other = 0,
    )
    key_rot = tl.where(idx_hid < HID // 2, -key_rot, key_rot)
    
    cos_k = tl.load(
        COS +\
            idx_k * stride_cos_t +\
            idx_hid * stride_cos_hid,
        mask=mask_tsrc & mask_hid,
        other=0,
    )
    sin_k = tl.load(
        SIN +\
            idx_k * stride_sin_t +\
            idx_hid * stride_sin_hid,
        mask=mask_tsrc & mask_hid,
        other=0,
    )
    
    key = ((key.to(tl.float32) * cos_k) + (key_rot.to(tl.float32) * sin_k)).to(key.dtype)
    
    # load query
    query = tl.load(
        Q +\
            idx_n * stride_q_n +\
            tdst * stride_q_tdst +\
            idx_hid * stride_q_hid,
        mask = mask_hid,
        other = 0,
    )
    
    query_rot = tl.load(
        Q +\
            idx_n * stride_q_n +\
            tdst * stride_q_tdst +\
            idx_hid_rot * stride_q_hid,
        mask = mask_hid_rot,
        other = 0,
    )
    query_rot = tl.where(idx_hid < HID // 2, -query_rot, query_rot)
    
    cos_q = tl.load(
        COS +\
            # tdst * stride_cos_t +\
            tl.minimum(tdst, WINDOW_SIZE + NUM_SINK) * stride_cos_t +\
            idx_hid * stride_cos_hid,
        mask=mask_hid,
        other=0,
    )
    sin_q = tl.load(
        SIN +\
            # tdst * stride_sin_t +\
            tl.minimum(tdst, WINDOW_SIZE + NUM_SINK) * stride_sin_t +\
            idx_hid * stride_sin_hid,
        mask=mask_hid,
        other=0,
    )
    
    query = ((query.to(tl.float32) * cos_q) + (query_rot.to(tl.float32) * sin_q)).to(query.dtype)
    
    # calc dot product. NOTE: you need to scale query before pass into kernel
    score = tl.sum(query.to(tl.float32) * key.to(tl.float32))
    score = score / tl.sqrt(HID.to(tl.float32))
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
        
        ctx.save_for_backward(
            q, k, cos, sin
        )
        ctx.num_sink = num_sink
        ctx.window_size = window_size
        
        nnz = N * TDST * (num_sink + window_size)
        indices = torch.zeros((3, nnz), dtype=torch.int32, device=device)
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
            
            BLOCK_HID
        )
        torch.cuda.set_device(_device)
        
        return indices, values

    @staticmethod
    def backward(
        ctx, 
        grad_indices: Tensor, 
        grad_values: Tensor
    ):
        return (
            None, #q: Tensor
            None, #k: Tensor
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
    
    # print(probs.to_dense()[0])
    # q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, None)
    # print((q_embed @ k_embed.transpose(-1, -2))[0] / (128 ** 0.5))
    
    if v not in [torch.float32]:
        v = v.to(torch.float32)
    context = torch.bmm(probs, v)
    context = context.to(_dtype)
    
    return context

if __name__ == '__main__':
    N = 1
    T = 8
    HID = 128
    NSINK = 1
    WIND = 2
    
    q = torch.nn.Parameter(torch.randn((N, T, HID), device=0) * 0.02)
    k = torch.nn.Parameter(q.data.clone())
    v = torch.nn.Parameter(q.data.clone())
    cos = torch.randn((T, HID), device=q.device)
    sin = cos.clone()
    
    x = sink_attention(
        q, k, v, cos, sin, 
        num_sink=NSINK, window_size=WIND
    )
    x.mean().backward()
    
    print(x.mean())
    print(q.grad)
    print(k.grad)
    print(v.grad)