import math
import cv2
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor
from performer_pytorch import FastAttention
import numpy as np
from ...utils import get_bench

# get_bench().synchronize = True
# get_bench().disabled = False

timer = lambda x: get_bench().region(x)

def imshow_pixels(ps, psmask, N, T_DST, T_SRC):
    pass

    # ps = ps * psmask.long()
    # ret = torch.empty((N, T_DST, T_SRC), device=ps.device).fill_(0)
    # ret.scatter_(dim=-1, index=ps, value=1)
    # ret = ret.cpu().numpy()
    # def max_pool(img, factor: int):
    #     """ Perform max pooling with a (factor x factor) kernel"""
    #     ds_img = np.full((img.shape[0] // factor, img.shape[1] // factor), -float('inf'), dtype=img.dtype)
    #     np.maximum.at(ds_img, (np.arange(img.shape[0])[:, None] // factor, np.arange(img.shape[1]) // factor), img)
    #     return ds_img
    # ret = max_pool(ret[0], 1)
    # plt.clf()
    # plt.imshow(ret)
    # plt.savefig("hello.png", dpi=320)
    # input(">>>")

def masked_bmm_masked_indices(a: Tensor, b: Tensor, xs: Tensor, xs_mask: Tensor, value: float):
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
    
    # # [N, A, 1, B]
    # xs_a = a.view(N, A, 1, B)
    # # [N, A, Z, B]
    # xs_b = b.view(N, 1, C, B).expand(N, A, C, B)
    # xs_b_idx = xs.view(N, A, Z, 1).expand(N, A, Z, B)
    # xs_b = xs_b.gather(dim=-2, index=xs_b_idx)
    
    # ys = (xs_a * xs_b).sum(-1)
    # ys = torch.where(xs_mask > 0.5, ys, value)
    # return ys
    
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
    ys = torch.where(xs_mask > 0.5, ys, value)
    return ys
    
    # # dense implementation
    # with timer("mbmm_topk"):
    #     SPARQ = 32
    #     _, a_idx = torch.topk(a.abs(), k=SPARQ, dim=-1)
    #     a_mask = torch.zeros_like(a)
    #     a_mask.scatter_(dim=-1, index=a_idx, value=1)
    #     a = a_mask * a
    
    # with timer("mbmm_bmm"):
    #     ts = torch.bmm(a, b.transpose(-1, -2))
    #     ys = ts.gather(dim=-1, index=xs, sparse_grad=True)
    #     ys = torch.where(xs_mask > 0.5, ys, value)
    # return ys

def forward_mask(self_w: int, self_k: int, q: Tensor, k: Tensor, scale_up: float, oversample: float):
    device = q.device
    dtype = q.dtype
    N, T_SRC, HID = k.shape
    N, T_DST, HID = q.shape

    start_t_src = T_SRC - T_DST + 1
    end_t_src = start_t_src + T_DST
    tsrcs = torch.arange(start_t_src, end_t_src, device=device)[None, :, None].expand(N, T_DST, 1)
    
    w = self_w
    ws = torch.empty((T_DST,), device=device, dtype=torch.float)[None, :, None].fill_(self_w).expand(N, T_DST, 1)
    # w cannot be larger than t_src
    
    OVERSAMPLE = oversample
    SCALE_UP = scale_up
    max_pixels = max(int(round(self_k * OVERSAMPLE)), self_w)
    pixels = torch.arange(0, max_pixels, device=device)[None, None, :].expand(N, T_DST, max_pixels).contiguous()
    pixels_mask = torch.empty_like(pixels).fill_(0).float()
    pixels_mask[:, :, :self_w] = 1.0
    # pixel_counts = torch.empty((T_DST,), device=device, dtype=torch.long).fill_(self.w)[None, :, None].expand(N, T_DST, 1).contiguous()
    
    # need_expand = ws < tsrcs
    while True:
        with timer("loop"):
            # is_last_loop = w * SCALE_UP >= T_SRC
            is_last_loop = w == T_SRC
            
            imshow_pixels(pixels, pixels_mask, N, T_DST, T_SRC)
            
            # TODO: topk masked pixels
            t_tsrcs = tsrcs #torch.masked_select(tsrcs, need_expand).view(N, -1, 1)
            tws = ws #torch.masked_select(ws, need_expand).view(N, -1, 1)
            # print('a', t_tsrcs.shape, pixels.shape)
            tpixels = pixels #torch.masked_select(pixels, need_expand)
            # tpixels = tpixels.view(N, -1, pixels.shape[-1])
            tpixels_mask = pixels_mask #torch.masked_select(pixels_mask, need_expand).view(N, -1, pixels.shape[-1])
            tq = q #torch.masked_select(q, need_expand).view(N, -1, q.shape[-1])
            
            with timer("topk_mask"):
                txs = torch.round(tpixels * (t_tsrcs / tws)).long()
                with timer("masked_bmm"):
                    scores = masked_bmm_masked_indices(tq, k, txs, tpixels_mask, -32000.0)
                # k = clamp(round(tws / t_tsrcs * self.k)) * (1 if tww == t_tsrcs else 1.5), 1, tws)
                tks = torch.clamp(
                    # torch.round((tws / t_tsrcs * self_k) * torch.where(tws == t_tsrcs, 1, OVERSAMPLE)), 
                    torch.round((tws / t_tsrcs * self_k) * (1 if is_last_loop else OVERSAMPLE)), 
                    torch.tensor(1, device=device), 
                    torch.clamp_max(tws - 1, scores.shape[-1] - 1)
                ).long()
                # tks_max = tks.max().item()
                tks_max = int(math.ceil(min(w - 1, scores.shape[-1] -1, self_k * (1 if is_last_loop else OVERSAMPLE))))
                values, indices = torch.topk(scores, k=tks_max, dim=-1, sorted=True, largest=True)
                with timer("pixels_gather"):
                    new_pixels = pixels.gather(dim=-1, index=indices)
                new_pixels_mask = (torch.arange(0, indices.shape[-1], device=device)[None, None, :] < tks) * 1.0
            
            with timer("pad"):
                new_pixels = F.pad(new_pixels, (0, pixels.shape[-1] - new_pixels.shape[-1]), value=0.0)
                new_pixels_mask = F.pad(new_pixels_mask, (0, pixels.shape[-1] - new_pixels_mask.shape[-1]), value=0.0)
                pixels = new_pixels #torch.masked_scatter(pixels, need_expand, new_pixels)
                pixels_mask = new_pixels_mask #torch.masked_scatter(pixels_mask, need_expand, new_pixels_mask)
            
            with timer("update_ws"):
                # need_expand = ws < tsrcs
                ws_new = torch.min(tsrcs, ws * SCALE_UP)
            
            with timer("resize"):
                MAX_SCALE_UP = int(math.ceil(SCALE_UP))
                with timer("resize.expand"):
                    # TODO: resize from ws_new to ws
                    N, A, Z = pixels.shape
                    scale = ws_new / ws
                    ps_start = pixels * scale
                    ps_end = (pixels + 1) * scale
                    ps_start = torch.round(ps_start).long()
                    ps_end = torch.round(ps_end).long()
                    ps = ps_start[:, :, :, None].expand(N, A, Z, MAX_SCALE_UP)
                    ps_counts = (ps_end - ps_start)[:, :, :, None]
                    reps = torch.arange(0, MAX_SCALE_UP, device=device)[None, None, None, :]
                    reps = reps * pixels_mask[:, :, :, None]
                    ps = ps + torch.clamp_max(reps, torch.clamp_min(ps_counts - 1, 0)).long()
                    ps = ps.view(N * A, Z * MAX_SCALE_UP).contiguous()
                with timer("resize.sort"):
                    ps, _ = torch.sort(ps, dim=-1, descending=False)
                
                # print(ps.shape, self_k, self_w)
                
                with timer("resize.unique"):
                    _, indices = torch.unique_consecutive(ps, return_inverse=True)
                    indices -= indices.min(dim=1, keepdim=True)[0]
                    result = torch.full_like(ps, -1)
                    ps = result.scatter_(1, indices, ps)
                    ps = ps.view(N, A, -1)
                
                with timer("resize.cleanup"):
                    pixels_mask = torch.zeros_like(ps)
                    pixels_mask = pixels_mask.masked_fill_((ps >= 0), 1.0)
                    pixels = ps.masked_fill_((ps < 0), 0.0)
                    # max_z = int(pixels_mask.sum(-1).max().item())
                    if is_last_loop:
                        max_z = self_k
                    else:
                        max_z = int(round(self_k * 2))
                    # print(is_last_loop, max_z, int(pixels_mask.sum(-1).max().item()))
                    pixels = pixels[..., :max_z].contiguous()
                    pixels_mask = pixels_mask[..., :max_z].contiguous()
                
            ws = ws_new

            # manage break
            if w == T_SRC:
                break
            w = min(T_SRC, w * SCALE_UP)
    
    imshow_pixels(pixels, pixels_mask, N, T_DST, T_SRC)
    # pixels = pixels * pixels_mask.long()
    # ret = torch.empty((N, T_DST, T_SRC), device=device).fill_(-32000.0)
    # ret.scatter_(dim=-1, index=pixels, value=0)
    # return ret
    
    N, A, Z = pixels.shape
    _, C, _ = k.shape
    col_idx = pixels.view(N, A * Z)
    crow_idx = (torch.arange(0, A+1, device=device) * Z)[None, :].expand(N, A+1)
    values = torch.zeros_like(col_idx, dtype=dtype)
    mask = torch.sparse_csr_tensor(
        crow_indices=crow_idx,
        col_indices=col_idx,
        values=values.to(torch.float32),
        size=(N, A, C)
    )
    return mask

class TreeAttention(nn.Module):
    def __init__(
        self, 
        causal: bool, 
        k: int, 
        start_w: int, 
        w: int, 
        scale_up: float,
        oversample: float,
    ):
        super().__init__()
        
        self.causal = causal
        self.k = k
        self.start_w = start_w
        self.w = w
        self.scale_up = scale_up
        self.oversample = oversample
        assert causal
    
    def forward_single_quary(
        self, 
        q: Tensor, 
        kcum: Tensor,
        k: Tensor, 
        v: Tensor
    ):
        assert q.ndim == 3
        assert k.ndim == 3
        assert v.ndim == 3
        assert q.shape[1] == 1
        assert k.shape[1] == v.shape[1]
        
        if k.shape[1] < self.k:
            score = torch.bmm(q, k.transpose(-1, -2))
            probs = torch.softmax(score, dim=-1)
            context = torch.bmm(probs, v)
            return context, probs
        
        N, T_SRC, HID = k.shape
        N, T_DST, HID = q.shape
        
        w = 16
        assert self.w > w
        assert w <= T_SRC
        
        scores_mask = torch.ones((N, 1, w), dtype=torch.float32, device=q.device)
        
        while True:
            KEY_APPROX_METHOD = 'skip'
            if KEY_APPROX_METHOD == 'avg':
                # key avg
                idx = torch.round(torch.arange(0, w+1, dtype=torch.float32, device=q.device) * (T_SRC / (w+1))).long()
                start_idx = idx[:-1]
                end_idx = idx[1:]
                pixel_counts = end_idx - start_idx
                
                start_idx = start_idx[None, :, None].expand(N, w, HID)
                end_idx = end_idx[None, :, None].expand(N, w, HID)
                pixel_counts = pixel_counts[None, :, None].expand(N, w, HID)
                
                tk = kcum.gather(dim=1, index=end_idx) - kcum.gather(dim=1, index=start_idx)
                tk = tk / (pixel_counts + 1e-12)
            elif KEY_APPROX_METHOD == 'skip':
                # key skipping
                idx = torch.round(torch.arange(0, w, dtype=torch.float32, device=q.device) * (T_SRC / (w))).long()
                idx = idx[None, :, None].expand(N, w, HID)
                tk = k.gather(dim=1, index=idx)
            else:
                raise Exception()
            
            # sparq
            SPARQ = 32
            values, indices = torch.topk(q.abs(), dim=-1, k=SPARQ)
            tq = q.gather(dim=-1, index=indices)
            tk = tk.gather(dim=-1, index=indices.expand(N, tk.shape[1], -1))
            
            # perform attention
            scores = torch.softmax(
                torch.bmm(tq, tk.transpose(-1, -2))\
                    .masked_fill_(scores_mask < 0.5, torch.finfo(q.dtype).min),
                dim=-1
            )
            
            # sink attention
            SINKING = 2
            if SINKING > 0:
                scores[:, :, :SINKING] = 2.0
            values, indices = torch.topk(scores, k=int(round(min(w, max(1, round(w / T_SRC * self.k)) * (1 if w == T_SRC else 1.5)))), dim=-1)
            scores_mask.fill_(0)
            scores_mask.scatter_(dim=-1, index=indices, value=1)
            
            w = min(T_SRC, w * 2)
            scores_mask = F.interpolate(scores_mask, size=(w,), mode='nearest')
            if w == T_SRC:
                break
        
        scores = torch.bmm(q, k.transpose(-1, -2))
        scores = scores.masked_fill_(scores_mask < 0.5, torch.finfo(q.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        context = torch.bmm(probs, v)
        
        # # calc alpha with sparq
        # SPARQ = 16
        # values, indices = torch.topk(q.abs(), dim=-1, k=SPARQ)
        # tq = q.gather(dim=-1, index=indices)
        # tk = k.gather(dim=-1, index=indices.expand(N, k.shape[1], -1))
        # tprobs = torch.softmax(torch.bmm(tq, tk.transpose(-1, -2)), -1)
        # alpha = (scores_mask * tprobs).sum(-1, keepdim=True)
        # avg_context = v.sum(dim=-2, keepdim=True)
        
        # context = context * alpha + avg_context * (1 - alpha)
        
        return context, probs
    
    def forward_single(self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor):
        assert q.ndim == 4
        assert k.ndim == 4
        assert v.ndim == 4
        assert k.shape[-2] == v.shape[-2]
        
        N, H, T_SRC, HID = k.shape
        N, H, T_DST, HID = q.shape
        
        kcum = k.cumsum(dim=-2).view(N*H, T_SRC, HID)
        
        a_rows = []
        c_rows = []
        for t in range(T_DST):
            if (t % 1000) == 0:
                print(t)
            assert T_SRC-T_DST+t+1 > 0
            t_src = T_SRC-T_DST+t+1
            k_row = k[..., :t_src, :]
            v_row = v[..., :t_src, :]
            q_row = q[..., t:t+1, :]
            k_row = k_row.view(N*H, t_src, HID)
            v_row = v_row.view(N*H, t_src, HID)
            q_row = q_row.view(N*H, 1, HID)
            
            c_row, a_row = self.forward_single_quary(
                q=q_row,
                kcum=kcum, 
                k=k_row,
                v=v_row
            )
            c_rows.append(c_row)
            
            # a_row = F.pad(a_row, pad=(0, T_SRC - a_row.shape[-1]), value=0)
            # a_rows.append(a_row)
        
        # a_rows = torch.concat(a_rows, dim=-2)
        # score = torch.matmul(q, k.transpose(-1, -2)).view(N, H, T_DST, T_SRC) + attention_mask
        # probs = torch.softmax(score, dim=-1).view(N*H, T_DST, T_SRC)
        # torch.save({'a': a_rows, 's': probs}, 'dump.pth')
        
        context = torch.concat(c_rows, dim=-2)
        context = context.view(N, H, T_DST, HID)
        
        # performer
        with torch.autocast("cuda", torch.float32):
            pcontext = self.performer(
                q.to(torch.float32), 
                k.to(torch.float32), 
                v.to(torch.float32),
            )
        context = pcontext * 0.1 + context * 0.9
        
        return context
    
    def forward_batch(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        __q = q
        __k = k
        __v = v
        __mask = mask
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
        
        # t_mask = max(0, self.w - T_SRC + T_SRC)
        # if t_mask > t_dense:
        #     c_mask = t_mask - t_dense
        #     t_dense = t_mask
        #     q, q_dense = q[..., c_mask:, :], q[..., :c_mask, :]
        #     mask, mask_dense = mask[..., c_mask:, :], mask[..., :c_mask, :]
        #     scores = torch.matmul(q_dense, k.transpose(-1, -2))
        #     scores = scores + mask_dense
        #     _, idx = torch.topk(scores, k=self.k, dim=-1)
        #     new_scores = torch.full_like(scores, -32000)
        #     new_scores.scatter_(dim=-1, index=idx, src=scores)
        #     scores = new_scores
        #     probs = torch.softmax(scores, -1)
        #     context = torch.matmul(probs, v)
        #     contexts.append(context)
        
        t_sparse = T_DST - t_dense
        if t_sparse > 0:
            q = q.view(N*H, t_sparse, HID)
            k = k.view(N*H, T_SRC, HID)
            with timer("fmask"):
                mask_sparse = forward_mask(
                    self.w, 
                    self.k, 
                    q, 
                    k,
                    self.scale_up,
                    self.oversample,
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
                cols = probs.col_indices() # N*A, Z
                values = probs.values().squeeze(-1).squeeze(-1) # N*A, Z
                
                nnz = cols.shape[-1]
                indices = torch.concat([
                    torch.arange(bsz, device=cols.device).view(-1, 1, 1, 1),
                    torch.arange(tdst, device=cols.device).view(1, -1, 1, 1),
                    cols.view(bsz, tdst, -1, 1).contiguous()
                ], dim=-1).view(-1, 3)
                values = values.view(-1)
                probs = torch.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=probs.shape
                )
                
                # probs = torch.sparse_csr_tensor(
                #     crow_indices=probs.crow_indices(),
                #     col_indices=probs.col_indices(),
                #     values=probs.values().squeeze(-1).squeeze(-1),
                #     size=probs.shape
                # )
                context = torch.bmm(probs, v.view(N*H, T_SRC, HID))
                context = context.view(N, H, t_sparse, HID)
            
            # mask_sparse = mask_sparse.view(N, H, t_sparse, T_SRC)
            # scores = torch.matmul(q, k.transpose(-1, -2))
            # scores = scores + mask_sparse
            # probs = torch.softmax(scores, -1)
            # context = torch.matmul(probs, v)
            contexts.append(context)
        
        contexts = torch.concat(contexts, dim=-2)
        
        # print(mask_dense.shape, mask_sparse.shape)
        # mask = torch.concat([
        #     mask_dense.expand(mask_sparse.shape[0], mask_sparse.shape[1], -1, -1), 
        #     mask_sparse
        # ], dim=-2)
        # mask = (mask > -1) * 1.0
        
        # scores = torch.matmul(__q, __k.transpose(-1, -2))
        # scores = scores + mask
        # approx = torch.softmax(scores, dim=-1)
        
        # scores = torch.matmul(__q, __k.transpose(-1, -2))
        # scores = scores + __mask
        # probs = torch.softmax(scores, dim=-1)
        
        # torch.save({'a': approx, 's': probs}, 'dump.pth')
        
        return contexts
    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor):
        # return self.forward_single(q, k, v, attention_mask)
        return self.forward_batch(q, k, v, attention_mask)