from torch import nn
import torch
import torch.nn.functional as F
import math

class LoraLinear(nn.Module):
    def __init__(self, inch, outch, dim_r):
        super().__init__()
        
        self.lora_a = nn.Parameter(torch.zeros((dim_r, inch)))
        self.lora_b = nn.Parameter(torch.zeros((outch, dim_r)))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor):
        x = F.linear(x, torch.mm(self.lora_b, self.lora_a))
        # x = F.linear(x, self.lora_b)
        # print(x[0,0,0])
        return x

# fused
def lora_forward(linear: nn.Linear, lora: LoraLinear, x: torch.Tensor, enabled: bool):
    if not enabled:
        return linear(x)
    assert linear.bias.ndim == 1
    assert x.ndim == 3
    
    x_fc = F.linear(x, linear.weight)
    x_lora = lora(x)
    x = x_fc + x_lora
    if linear.bias is not None:
        x = x + linear.bias.view(1, 1, linear.bias.shape[0])
    return x

# split for save memory
def lora_forward_linear(linear: nn.Linear, x: torch.Tensor):
    return F.linear(x, linear.weight, linear.bias)

def lora_forward_lora(linear: nn.Linear, linear_x: torch.Tensor, lora: LoraLinear, x: torch.Tensor, enabled: bool):
    # return linear_x
    if not enabled:
        return linear_x
        # assert linear.bias.ndim == 1
        # assert linear_x.ndim == 3
        # if linear.bias is not None:
        #     return linear_x + linear.bias.view(1, 1, linear.bias.shape[0])
        # return linear_x
    
    _x = x
    
    op_dtype = linear_x.dtype
    
    linear_x_flatten = False
    if linear_x.ndim == 4:
        linear_x_flatten = True
        N, H, T, D = linear_x.shape
        linear_x = linear_x.permute(0, 2, 1, 3).reshape(N, T, H*D)
    
    x_flatten = False
    if x.ndim == 4:
        x_flatten = True
        N, H, T, D = x.shape
        x = x.permute(0, 2, 1, 3).reshape(N, T, H*D)
    
    # print('wow', linear_x.shape, linear_x.dtype, x.shape, x.dtype, x_flatten, linear_x_flatten)
    
    if linear.bias is not None:
        linear_x = linear_x - linear.bias.view(1, 1, linear.bias.shape[0])
    if linear_x.dtype != op_dtype:
        linear_x = linear_x.to(op_dtype)
    
    lora_x = lora(x)
    
    # assert lora_x.requires_grad
    
    x = lora_x + linear_x
    
    # assert x.requires_grad
    
    if linear.bias is not None:
        x = x + linear.bias.view(1, 1, linear.bias.shape[0])
    if x.dtype != op_dtype:
        x = x.to(op_dtype)
    
    # assert x.requires_grad
    
    if x_flatten:
        x = x.view(N, T, H, D).permute(0, 2, 1, 3).contiguous()
    
    # assert x.requires_grad
    
    # print('w12', linear_x.dtype, _x.dtype, x.dtype, op_dtype)
    
    return x