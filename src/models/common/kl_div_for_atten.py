import torch

def kl_div_attention(
    input: torch.Tensor, 
    target: torch.Tensor, 
    attention_mask: torch.Tensor, 
    log_target: bool = False
):
    assert torch.max(attention_mask).item() <= 0
    assert attention_mask.ndim == 4
    
    N, H, T, T = input.shape
    
    if not log_target: # default
        loss_pointwise = target * ((target + 1e-12).log() - input)
    else:
        loss_pointwise = target.exp() * (target - input)
    
    one_mask = mask = (attention_mask > -1) * 1.0
    mask = mask * mask.transpose(-1, -2)
    
    loss_pointwise = loss_pointwise * mask
    loss = loss_pointwise.sum() / (one_mask[:, :, 0, :].sum() + 1e-8)
    
    return loss