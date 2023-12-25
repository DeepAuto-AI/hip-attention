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

import math
import random
from typing import Optional, Tuple, Union
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor, nn
import torch
import torch.nn.functional as F
from ...utils import get_bench

# get_bench().synchronize = True
# get_bench().disabled = False

timer = lambda x: get_bench().region(x)

DBG_T = 6

def masked_bmm_masked_indices(a: Tensor, b: Tensor, xs: Tensor):
    """
    a: [N, A, B]
    b: [N, C, B]
    
    xs: [N, A, Z] < C
    xs_mask: [N, A, Z] \in {0, 1}
    value: float
    
    - return
    ys: [N, A, Z] \in { (R := a[i, :] \dot b[:, j]) if xs_mask == 1 else value }
    """
    
    device = a.device
    dtype = a.dtype
    
    N, A, B = a.shape
    _N, C, _B = b.shape
    assert B == _B
    assert N == _N
    _N, _A, Z = xs.shape
    assert N == _N
    assert A == _A
    
    xs = torch.clamp_max(xs, C - 1)
    
    # sparse implementation
    col_idx = xs.view(N, A * Z)
    crow_idx = (torch.arange(0, A+1, device=device) * Z)[None, :].expand(N, A+1)
    values = torch.zeros_like(col_idx, dtype=dtype)
    mask = torch.sparse_csr_tensor(
        crow_indices=crow_idx,
        col_indices=col_idx,
        values=values.to(torch.float32),
        size=(N, A, C)
    )
    with timer("addmm"):
        ys = torch.sparse.sampled_addmm(
            mask, a.to(torch.float32), b.transpose(-1, -2).to(torch.float32)
        )
    ys_values = ys.values()
    assert ys_values.shape == values.shape
    ys = ys_values.view(N, A, Z).to(dtype)
    return ys

def topk_prime(
    zs: Tensor, 
    zs_mask: Tensor,
    ws: Tensor, 
    tsrcs: Tensor, 
    ks: int, 
    ps: Tensor, 
    query: Tensor, 
    key: Tensor,
    last_loop: bool,
) -> Tuple[Tensor, Tensor]:
    """
    Select topk' z from zs, and determine number of zs is larger than patches. Returns new mask
    
    Args:
        zs (Tensor): fp[N, T_DST, Z]
        zs_mask (Tensor): long[N, T_DST, Z]
        ws (Tensor): fp[N, T_DST, 1]
        tsrcs (Tensor): fp[N, T_DST, 1]
        ks (int): top-k in final one
        ps (Tensor): fp[N, T_DST, 1] patches to exit iteration
        query (Tensor): fp[N, T_DST, HID]
        key (Tensor): fp[N, T_SRC, HID]
    
    Return: Tuple[
        bool[1, T_DST, 1]. bitmask of further topk-prime update requires,
        long[N, T_DST, Z]. new zs mask
    ]
    """
    N, T_DST, Z = zs.shape
    assert zs.shape == zs_mask.shape
    assert ws.shape == (N, T_DST, 1)
    assert tsrcs.shape == (N, T_DST, 1)
    assert isinstance(ks, int)
    assert ps.shape == (N, T_DST, 1)
    _, _, HID = query.shape
    assert query.shape[:2] == (N, T_DST)
    _, T_SRC, _ = key.shape
    assert key.shape == (N, T_SRC, HID)
    
    method = 'key'
    if method == 'cumsum':
        key_cumsum = key.cumsum(dim=-2)
        zs_start = torch.clamp(torch.round(zs * (tsrcs / ws)), 0, T_SRC-1).long().unsqueeze(-1)
        zs_end = torch.clamp(torch.round((zs+1) * (tsrcs / ws)), 0, T_SRC-1).long().unsqueeze(-1)
        ks_start = key_cumsum\
            .unsqueeze(1)\
            .expand(N, T_DST, T_SRC, HID)\
            .gather(dim=-2, index=zs_start.expand(N, T_DST, Z, HID))
        ks_end = key_cumsum\
            .unsqueeze(1)\
            .expand(N, T_DST, T_SRC, HID)\
            .gather(dim=-2, index=zs_end.expand(N, T_DST, Z, HID))
        ks_avgpool = (ks_end - ks_start) / (zs_end - zs_start)
        qs = query
        scores = torch.einsum("...td,...tzd->...tz", qs, ks_avgpool)
    elif method == 'key':
        # zs_start = torch.clamp(torch.round(zs * (tsrcs / ws)), 0, T_SRC-1).long().unsqueeze(-1)
        # ks_start = key\
        #     .unsqueeze(1)\
        #     .expand(N, T_DST, T_SRC, HID)\
        #     .gather(dim=-2, index=zs_start.expand(N, T_DST, Z, HID))
        # qs = query
        # scores = torch.einsum("...td,...tzd->...tz", qs, ks_start)
        zs_start = torch.clamp(torch.round(zs * (tsrcs / ws)), 0, T_SRC-1).long()
        scores = masked_bmm_masked_indices(query, key, zs_start)
    else:
        pass
    assert scores.shape == (N, T_DST, Z)
    scores[...,-1:] = 999
    scores.masked_fill_(zs_mask < 0.5, torch.finfo(scores.dtype).min)
    
    if not last_loop:
        topk = torch.clamp_max(torch.clamp_min(torch.round((ws / tsrcs) * ks), ps), zs_mask.sum(-1, keepdim=True))
        assert topk.shape == (N, T_DST, 1)
    else:
        topk = ks
    need_update = topk <= (ps * 2)
    if not last_loop:
        topk = topk * 1.5
    
    _, indices = torch.sort(scores, dim=-1, descending=True)
    assert indices.shape == (N, T_DST, Z)
    zs_mask = zs_mask.scatter(
        dim=-1,
        index=indices,
        src=(torch.arange(0, Z, device=zs.device)[None, None, :].expand(N, T_DST, Z) < topk).to(torch.long)
    )
    
    return need_update, zs_mask

def resize(
    zs: Tensor, 
    zs_mask: Tensor,
    ws_from: Tensor, 
    ws_to: Tensor,
    scale_up: float,
    Z_MAX: int,
) -> Tuple[Tensor, Tensor]:
    """
    Resize zs from ws_from to ws_to

    Args:
        zs (Tensor): fp[N, T_DST, Z] List of non zero entry for each query.
        ws_from (Tensor): fp[N, T_DST, 1] Current width
        ws_to (Tensor): fp[N, T_DST, 1] New width
    
    Return: 
    Tuple[
        zs (Tensor),
        zs_mask (Tensor),
    ]
    """
    max_scale_up = int(math.ceil(scale_up))
    with timer("resize.expand"):
        # TODO: resize from ws_new to ws
        N, A, Z = zs.shape
        scale = ws_to / ws_from
        zs_start = zs * scale
        zs_end = (zs + 1) * scale
        zs_start = torch.round(zs_start).long()
        zs_end = torch.round(zs_end).long()
        ps = zs_start[:, :, :, None].expand(N, A, Z, max_scale_up)
        ps_counts = (zs_end - zs_start)[:, :, :, None]
        reps = torch.arange(0, max_scale_up, device=zs.device)[None, None, None, :]
        reps = reps * zs_mask[:, :, :, None]
        ps = ps + torch.clamp_max(reps, torch.clamp_min(ps_counts - 1, 0)).long()
        ps = ps.view(N * A, Z * max_scale_up)
    
    with timer("resize.unique"):
        _, indices = torch.unique_consecutive(ps, return_inverse=True)
        indices -= indices.min(dim=1, keepdim=True)[0]
        result = torch.full_like(ps, -1)
        ps = result.scatter_(1, indices, ps)
        ps = ps.view(N, A, -1)
    
    with timer("resize.cleanup"):
        # ##print(ps[0, DBG_T])
        zs_mask = torch.logical_and(ps >= 0, ps < Z_MAX).to(torch.long)
        # ##print(zs_mask[0, DBG_T])
        zs = ps.masked_fill(torch.logical_or(ps < 0, ps >= Z_MAX), Z_MAX)
        max_z = int(zs_mask.sum(-1, dtype=torch.long).max().item())
        # if is_last_loop:
        #     max_z = self_k
        # else:
        #     max_z = int(round(self_k * 2))
        # ##print(is_last_loop, max_z, int(pixels_mask.sum(-1).max().item()))
        zs = zs[..., :max_z].contiguous()
        zs_mask = zs_mask[..., :max_z].contiguous()
        ##print(zs[0, DBG_T])
    return zs, zs_mask

def padd_zs(zs: Tensor, next_zs: Tensor, value: Union[float, int]):
    Z = zs.shape[-1]
    NEXT_Z = next_zs.shape[-1]
    if Z == NEXT_Z:
        return zs, next_zs
    elif Z < NEXT_Z:
        zs = F.pad(zs, (0, NEXT_Z-Z), value=value).to(zs.dtype)
        return zs, next_zs
    elif Z > NEXT_Z:
        next_zs = F.pad(next_zs, (0, Z-NEXT_Z), value=value).to(next_zs.dtype)
        return zs, next_zs
    else:
        raise Exception()

def imshow_pixels(ps, psmask, N, T_DST, T_SRC):
    pass

    return

    ps = ps * psmask.long()
    ps = torch.clamp(ps, 0, T_SRC-1)
    ret = torch.empty((N, T_DST, T_SRC), device=ps.device).fill_(0)
    ret.scatter_(dim=-1, index=ps, value=1)
    ret = ret.cpu().numpy()
    def max_pool(img, factor: int):
        """ Perform max pooling with a (factor x factor) kernel"""
        ds_img = np.full((img.shape[0] // factor, img.shape[1] // factor), -float('inf'), dtype=img.dtype)
        np.maximum.at(ds_img, (np.arange(img.shape[0])[:, None] // factor, np.arange(img.shape[1]) // factor), img)
        return ds_img
    ret = max_pool(ret[0], 1)
    plt.clf()
    plt.title(random.randint(0, 100000))
    plt.imshow(ret)
    plt.savefig("hello.png", dpi=320)
    input(">>>")

def approx_mask(
    query: Tensor,
    key: Tensor,
    ps: Union[int, Tensor],
    ks: Union[int, Tensor],
    init_w: int,
    scale_up: Union[float, int],
    mask_value: float,
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
    
    device = query.device
    dtype = query.dtype
    
    N, T_DST, HID = query.shape
    _N, T_SRC, _HID = key.shape
    assert _N == N
    assert _HID == HID
    
    if isinstance(ps, int):
        ps = torch.tensor(ps, device=device, dtype=torch.long).view(1, 1, 1).expand(N, T_DST, 1)
    elif isinstance(ps, float) or (isinstance(ps, Tensor) and ps.dtype != torch.long):
        raise Exception("ps should be int or int Tensor")
    else:
        assert isinstance(ps, Tensor)
    
    if isinstance(ks, Tensor):
        ks = ks.max().long().item()
    elif isinstance(ks, float):
        ks = int(round(ks))
    elif isinstance(ks, int):
        pass
    else:
        raise Exception()
    
    assert isinstance(init_w, int)
    assert isinstance(scale_up, (int, float))
    
    assert T_SRC > init_w
    
    zs = torch.arange(0, init_w, device=device).view(1, 1, init_w).expand(N, T_DST, init_w)
    Z_MAX = 9999999
    zs_mask = torch.ones_like(zs, dtype=torch.long)
    ws = torch.full((N, T_DST, 1), init_w, dtype=torch.long, device=device)
    tsrcs = torch.arange(T_SRC+1-T_DST, T_SRC+1, device=device).view(1, T_DST, 1).expand(N, T_DST, 1)
    need_update = torch.full((1, T_DST, 1), True, dtype=torch.bool, device=device)
    
    w = init_w
    life = 1
    last_loop = False
    while True:
        Z = zs.shape[-1]
        tzs = torch.masked_select(zs, need_update.expand(N, T_DST, Z)).view(N, -1, Z)
        tzs_mask = torch.masked_select(zs_mask, need_update.expand(N, T_DST, Z)).view(N, -1, Z)
        tws = torch.masked_select(ws, need_update.expand(N, T_DST, 1)).view(N, -1, 1)
        t_tsrcs = torch.masked_select(tsrcs, need_update.expand(N, T_DST, 1)).view(N, -1, 1)
        tks = ks # torch.masked_select(ks, need_update.expand(N, T_DST, 1)).view(N, -1, 1)
        tps = torch.masked_select(ps, need_update.expand(N, T_DST, 1)).view(N, -1, 1)
        tquery = torch.masked_select(query, need_update.expand(N, T_DST, HID)).view(N, -1, HID)
        
        ##print('bb', tzs_mask[0, DBG_T])
        
        next_need_update, next_zs_mask = topk_prime(
            zs=tzs,
            zs_mask=tzs_mask,
            ws=tws,
            tsrcs=t_tsrcs,
            ks=tks,
            ps=tps,
            query=tquery,
            key=key,
            last_loop=last_loop
        )
        assert next_zs_mask.dtype == torch.long
        
        ##print('bb', next_zs_mask[0, DBG_T])
        
        def pack_zs(zs: Tensor, zs_mask: Tensor) -> Tuple[Tensor, Tensor]:
            new_zs = zs.masked_fill(zs_mask < 0.5, Z_MAX)
            # ##print(zs_mask[0,0]<0.5)
            new_zs, idx = torch.sort(new_zs, dim=-1, descending=False)
            new_zs_mask = zs_mask.gather(dim=-1, index=idx)
            
            max_item = zs_mask.sum(dim=-1, dtype=torch.long).max().item()
            new_zs = new_zs[...,:max_item]
            new_zs_mask = new_zs_mask[...,:max_item]
            
            return new_zs, new_zs_mask
        
        ##print('b', tzs[0, DBG_T], next_zs_mask[0, DBG_T])
        
        tzs, next_zs_mask = pack_zs(tzs, next_zs_mask)
        
        ##print('c', tzs[0, DBG_T], next_zs_mask[0, DBG_T])
        
        next_tws = torch.min(tws * scale_up, t_tsrcs)
        tzs, next_zs_mask = resize(
            zs=tzs, 
            zs_mask=next_zs_mask,
            ws_from=tws,
            ws_to=next_tws,
            scale_up=scale_up,
            Z_MAX=Z_MAX
        )
        
        ##print('d', tzs[0, DBG_T], next_zs_mask[0, DBG_T])
        
        ws.masked_scatter_(need_update.expand(N, T_DST, 1), next_tws)
        zs, tzs = padd_zs(zs, tzs, value=Z_MAX)
        assert zs_mask.dtype == next_zs_mask.dtype
        zs_mask, next_zs_mask = padd_zs(zs_mask, next_zs_mask, value=0)
        assert zs.shape == zs_mask.shape, f"{zs.shape} == {zs_mask.shape}"
        assert zs_mask.dtype == next_zs_mask.dtype
        Z = zs.shape[-1]
        zs = zs.masked_scatter(need_update.expand(N, T_DST, Z), tzs)
        zs_mask = zs_mask.masked_scatter(need_update.expand(N, T_DST, Z), next_zs_mask)
        
        imshow_pixels(zs, zs_mask.float(), N, T_DST, T_SRC)
        
        # print('iter', zs.shape, zs_mask.shape, ws.max().item(), w, T_SRC)
        if next_need_update.any():
            # print('updated')
            need_update = need_update\
                .masked_scatter_(need_update, next_need_update)\
                .view(need_update.shape)
        else:
            # break
            if last_loop:
                # print('breaked')
                # if life == 0:
                #     break
                # life -= 1
                break
            last_loop = True
            need_update = torch.full((1, T_DST, 1), True, dtype=torch.bool, device=device)
            # break
        w = w * scale_up
    
    # ##print(ws)
    ##print(zs.shape, zs_mask.shape)
    
    zs, zs_mask = resize(
        zs=zs,
        zs_mask=zs_mask,
        ws_from=ws,
        ws_to=tsrcs,
        scale_up=scale_up,
        Z_MAX=Z_MAX
    )
    zs_mask = (zs < T_SRC).long() * zs_mask
    
    zs, zs_mask = pack_zs(zs, zs_mask)
    ##print(zs.shape, zs_mask.shape)
    zs = zs[...,:ks]
    zs_mask = zs_mask[...,:ks]
    
    imshow_pixels(zs, zs_mask.float(), N, T_DST, T_SRC)
    
    ##print(zs[0, 6], zs_mask[0, 6], zs_mask[0].sum(-1))
    ##print(zs.shape, zs_mask.shape, zs_mask.float().mean())
    
    Z = zs.shape[-1]
    
    col_idx = torch.clamp(zs.reshape(N, T_DST * Z), 0, T_SRC-1)
    crow_idx = (torch.arange(0, T_DST+1, device=device) * Z)[None, :].expand(N, T_DST+1)
    values = torch.full_like(col_idx, mask_value, dtype=dtype)
    mask = torch.sparse_csr_tensor(
        crow_indices=crow_idx,
        col_indices=col_idx,
        values=values.to(torch.float32),
        size=(N, T_DST, T_SRC)
    )
    return mask

class TreeAttention(nn.Module):
    def __init__(
        self,
        causal: bool,
        k: int = 256,
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
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Optional[Tensor] = None):
        N, H, T_SRC, HID = k.shape
        N, H, T_DST, HID = q.shape
        
        contexts = []
        t_dense = max(0, max(self.start_w, self.k) - T_SRC + T_DST)
        if t_dense > 0:
            q, q_dense = q[..., t_dense:, :], q[..., :t_dense, :]
            if attention_mask is not None:
                attention_mask, mask_dense = attention_mask[..., t_dense:, :], attention_mask[..., :t_dense, :]
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
            DENSE = False
            with timer("fmask"):
                mask_sparse = approx_mask(
                    query=q,
                    key=k,
                    ps=self.p,
                    ks=self.k,
                    init_w=self.w,
                    scale_up=self.scale_up,
                    mask_value=1 if DENSE else 0
                )
            
            if DENSE:
                scores = torch.matmul(q, k.transpose(-1, -2))
                scores = scores + (1 - torch.clamp(mask_sparse.to_dense(), 0, 1)) * -32000
                # scores = scores + attention_mask
                probs = torch.softmax(scores, -1)
                context = torch.matmul(probs, v)
            else:
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
                    with torch.autocast('cuda', torch.float32):
                        context = torch.bmm(probs, v.view(N*H, T_SRC, HID).to(torch.float32)).to(v.dtype)
                    context = context.view(N, H, t_sparse, HID)
            contexts.append(context)
        
        contexts = torch.concat(contexts, dim=-2)
        
        return contexts

if __name__ == '__main__':
    torch.random.manual_seed(42)
    
    device = 'cuda:0'
    batch_size = 1
    head_size = 16
    head_dim = 64
    seq_len = 128
    attention = TreeAttention(
        causal=True, 
        start_w=0,
        w=4,
        scale_up=4,
        p=4,
        k=16
    )
    
    query = torch.randn((batch_size, head_size, seq_len, head_dim), device=device)
    key = query.clone()
    value = query.clone()
    
    attention(query, key, value)