import torch
import warnings
from torch import nn

from src.models.tree_attention.attention1_gpu import (
    load_checkouts,
    sparse_attention,
    attention_matrix,
)

def test_sparse_attention():
    q, k, v, out = load_checkouts()
    
    w_start = 512
    n_patches = 128
    mask_k = 256
    scale_up = 2
    
    if q.dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        warnings.warn("tree attention does not support 32 bits right now.")
    
    with torch.autocast('cuda', torch.float32):
        indices, ks, probs = attention_matrix(
            q,
            k,
            
            w_start,
            n_patches,
            mask_k,
            scale_up,
        )
        
    v = nn.Parameter(v)
    probs = nn.Parameter(probs)
    
    # exam GD
    for i in range(1000):
        context = sparse_attention(
            v,
            indices,
            ks,
            probs,
        )
        
        loss = context.square().sum()
        loss.backward()
        
        v.data -= 0.1 * v.grad
        probs.data -= 0.001 * probs.grad
        
        v.grad = None
        probs.grad = None
        
        # print(loss.item())
    
    assert loss.item() < 0.01
    print('[pass] test_sparse_attention')

if __name__ == '__main__':
    test_sparse_attention()