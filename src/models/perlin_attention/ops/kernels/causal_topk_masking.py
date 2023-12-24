import torch, math

def causal_topk_masking(
    probs, 
    k,
    attention_mask, 
    dst_attention_mask, 
    causal_attention_mask, 
    not_padded=True, 
    k_flatten_dim='causal_batch',
    is_causal=True,
):
    # attention_mask is always for src
    assert k_flatten_dim == 'causal_batch'
    assert not_padded
    
    N, H, T_DST, T_M = probs.shape
    FP_MIN = torch.finfo(torch.float16).min * 0.5
    
    top_k_elems = None
    per_item_top_k = None
    # assert k_flatten_dim in ['head', 'batch', 'causal_batch']
        
    masked_estimated_attention_probs = (probs * (dst_attention_mask > -1))
    
    causal_token_length = (causal_attention_mask > -1).long().sum(-1).view(1, 1, T_DST, 1)
    
    t = masked_estimated_attention_probs.transpose(1, 2).reshape(N, T_DST, H*T_M)
    # NOTE consider causal token length
    if is_causal:
        per_item_top_k = torch.clamp((H * torch.floor(k * T_M / causal_token_length.squeeze(0))).view(1, T_DST, 1), 1, H*T_M)
    else:
        token_length = (attention_mask > -1).long().sum(-1).view(N, -1)
        per_item_top_k = (H * torch.round(k * T_M / token_length)).view(N, 1, 1)
    
    # NOTE to prevent 0 top-k when large T and small T_m
    per_item_top_k = torch.clamp_min(per_item_top_k, 1)
    
    top_k_elems = min(int(math.ceil(torch.max(per_item_top_k).item())), t.shape[-1])
        
    _, indices = torch.topk(
        input=t,
        k=top_k_elems, 
        dim=-1, 
        sorted=True #sorted true is important
    )
        
    partial_attention_mask = torch.empty(
        t.shape, 
        dtype=torch.long, 
        device=attention_mask.device,
    )
    partial_attention_mask.fill_(t.shape[-1])
    partial_attention_mask.scatter_(
        dim=-1,
        index=indices,
        src=torch.arange(
            top_k_elems, 
            dtype=torch.long,
            device=attention_mask.device, 
        )\
            .view((1, -1) if t.ndim == 2 else (1, 1, -1))\
            .expand(indices.shape)
    )
        
    t_alive_mask = partial_attention_mask < per_item_top_k
    partial_attention_mask = t_alive_mask.float()
    
    partial_attention_mask = partial_attention_mask.view(N, T_DST, H, T_M).transpose(1, 2)
    partial_attention_mask.masked_fill_(
        mask=dst_attention_mask < -1,
        value=FP_MIN
    )
    
    partial_attention_mask = partial_attention_mask.view(N, H, T_DST, T_M)
    
    return partial_attention_mask
