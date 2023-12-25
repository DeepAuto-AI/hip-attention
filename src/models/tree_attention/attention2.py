"""
- Need to stop expansion when reach #patch
> multiple = 4, #patch = 16, k = 64, w = 8192
| w    | z    | z'   | k'   | keep?|
|------|------|------|------|------|
| 64   | 64   | 1    | 16   | True |
| 256  | 64   | 2    | 16   | True |
| 1024 | 64   | 8    | 16   | True |
| 4096 | 64   | 32   | 32   | done |
| 8192 | done | done | done | done |

- When approximator interation stops?
w / T * k >= p

if p and k is constant
w = (p/k)T
approximator is logN, but sparse attention is linear

if p=T/C
w = T^2/(kC) -- log w = 2log T - log kC
approximator is quadratic, but sparse attention is linear

if k=T/C
w = pC
approximator is linear, but sparse attention is quadratic

if p=T/C and k=T/C
w = T
approximator is log N, but sparse attention is quadratic
"""

from typing import Union
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from ...utils import get_bench

# get_bench().synchronize = True
# get_bench().disabled = False

timer = lambda x: get_bench().region(x)

def topk_prime(
    zs: Tensor, 
    ws: Tensor, 
    tsrcs: Tensor, 
    ks: Tensor, 
    ps: Tensor, 
    quary: Tensor, 
    key: Tensor
) -> Tensor:
    """
    Select topk' z from zs, and determine number of zs is larger than patches
    
    Args:
        zs (Tensor): fp[N, T_DST, Z]
        ws (Tensor): fp[N, T_DST, 1]
        tsrcs (Tensor): fp[N, T_DST, 1]
        ks (Tensor): top-k in final one
        ps (Tensor): patches to exit iteration
        quary (Tensor): fp[N, T_DST, HID]
        key (Tensor): fp[N, T_DST, HID]
    
    Return:
        bool[N, T_DST, 1]. bitmask of further topk-prime update requires
    """
    pass

def resize(
    zs: Tensor, 
    ws_from: Tensor, 
    ws_to: Tensor
) -> Tensor:
    """
    Resize zs from ws_from to ws_to

    Args:
        zs (Tensor): fp[N, T_DST, Z] List of non zero entry for each quary.
        ws_from (Tensor): fp[N, T_DST, 1] Current width
        ws_to (Tensor): fp[N, T_DST, 1] New width
    
    Return:
        new zs (Tensor)
    """
    pass

def approx_mask(
    query: Tensor,
    key: Tensor,
    ps: Union[int, Tensor],
    ks: Union[int, Tensor],
    init_w: int,
    scale_up: int,
) -> Tensor:
    """Generate Tree Attention Mask

    Args:
        query (Tensor): fp[N, T_DST, HID]
        key (Tensor): fp[N, T_SRC, HID]
        ps (Union[int, Tensor]): patches. accept int or fp[N, T_DST, 1]
        ks (Union[int, Tensor]): top-ks. accept int or fp[N, T_DST, 1]
    
    Return:
        fp[N, T_DST, T_SRC] <-- sparse_csr_tensor
    """
    pass

class TreeAttention(nn.Module):
    def __init__(
        self,
        causal: bool,
        k: int = 128,
        start_w: int = 4096,
        w: int = 64,
        scale_up: int = 4,
        p: int = 32,
    ):
        super().__init__()
        
        self.causal = causal
        self.k = k
        self.start_w = start_w
        self.w = w
        self.scale_up = scale_up
        self.p = p
        assert causal
    
    def forward_batch(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        N, H, T_SRC, HID = k.shape
        N, H, T_DST, HID = q.shape
        
        contexts = []
        t_dense = max(0, max(self.start_w, self.k) - T_SRC + T_DST)
        if t_dense > 0:
            q, q_dense = q[..., t_dense:, :], q[..., :t_dense, :]
            if mask is not None:
                mask, mask_dense = mask[..., t_dense:, :], mask[..., :t_dense, :]
                scores = torch.matmul(q_dense, k.transpose(-1, -2))
                scores = scores + mask_dense
                probs = torch.softmax(scores, -1)
                context = torch.matmul(probs, v)
            else:
                # need causal flash attention
                context = F.scaled_dot_product_attention(
                    q_dense,
                    k,
                    v,
                    is_causal=True,
                    scale=1,
                )
            contexts.append(context)
        
        t_sparse = T_DST - t_dense
        if t_sparse > 0:
            q = q.view(N*H, t_sparse, HID)
            k = k.view(N*H, T_SRC, HID)
            with timer("fmask"):
                mask_sparse = approx_mask(
                    query=q,
                    key=k,
                    ps=self.p,
                    ks=self.k,
                    init_w=self.w,
                    scale_up=self.scale_up,
                )
            
            with timer("attention"):
                scores = torch.sparse.sampled_addmm(
                    mask_sparse, q.to(torch.float32), k.transpose(-1, -2).to(torch.float32)
                )
                scores = torch.sparse_bsr_tensor(
                    crow_indices=scores.crow_indices(),
                    col_indices=scores.col_indices(),
                    values=scores.values().unsqueeze(-1).unsqueeze(-1),
                    size=scores.shape
                )
                import torch.sparse._triton_ops as triton_ops
                probs = triton_ops.bsr_softmax(scores) #.to_dense().to(q.dtype)
                bsz, tdst, tsrc = probs.shape
                cols = probs.col_indices() # N, A*Z
                values = probs.values().squeeze(-1).squeeze(-1) # N, A*Z
                
                nnz = values.shape[-1] // tdst
                indices = torch.concat([
                    torch.arange(bsz, device=cols.device).view(1, -1, 1, 1).expand(1, bsz, tdst, nnz),
                    torch.arange(tdst, device=cols.device).view(1, 1, -1, 1).expand(1, bsz, tdst, nnz),
                    cols.view(1, bsz, tdst, -1).contiguous()
                ], dim=0).view(3, -1)
                values = values.view(-1)
                probs = torch.sparse_coo_tensor(
                    indices=indices.long(),
                    values=values.to(torch.float32),
                    size=probs.shape
                )
                context = torch.bmm(probs, v.view(N*H, T_SRC, HID).to(torch.float32)).to(v.dtype)
                context = context.view(N, H, t_sparse, HID)
            contexts.append(context)
        
        contexts = torch.concat(contexts, dim=-2)
        
        return contexts