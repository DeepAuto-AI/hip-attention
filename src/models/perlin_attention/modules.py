import math
import time
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim

BENCHMARKING = False

def interpolate(x: torch.Tensor, size, interp_mode: str = None):
    if x.shape[-2:] == size: return x
    
    interp_mode = ('bilinear' if size[-1] >= x.shape[-1] else 'area') if interp_mode is None else interp_mode
    
    if not BENCHMARKING:
        if torch.get_autocast_gpu_dtype() == torch.bfloat16: # F interpolate is not supported on bf16
            original_dtype = x.dtype
            with torch.autocast('cuda', torch.float32):
                if x.dtype != torch.float32:
                    x = x.to(torch.float32)
                x = F.interpolate(x, size, mode=interp_mode)
            if x.dtype != original_dtype:
                x = x.to(original_dtype)
        else:
            x = F.interpolate(x, size, mode=interp_mode)
    else:
        x = F.interpolate(x, size, mode=interp_mode)
    
    return x

class Residual(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.net = nn.Sequential(*args)
    
    def forward(self, x):
        y = self.net(x)
        return x + y

class KeepRes(nn.Module):
    def __init__(self, *args, output_width=None):
        super().__init__()
        self.net = nn.Sequential(*args)
        self.output_width = output_width
    
    def forward(self, x):
        x_shape = x.shape
        x = self.net(x)
        if self.output_width is None:
            x = interpolate(x, x_shape[-2:])
        else:
            x = interpolate(x, (x_shape[-2], self.output_width))
        return x

class ResBlock(nn.Module):
    def __init__(self, ch, padding=1, lnorm_size=None, padding_mode='zeros', causal=False, dilation=1):
        super().__init__()
        
        self.net = KeepRes(
            CausalConv2d(ch, ch, 3, padding=padding, padding_mode=padding_mode, causal=causal, dilation=dilation),
            # nn.BatchNorm2d(48),
            # nn.LayerNorm(lnorm_size),
            nn.ReLU(),
            CausalConv2d(ch, ch, 3, padding=padding, padding_mode=padding_mode, causal=causal, dilation=dilation),
            # nn.BatchNorm2d(48),
            # nn.LayerNorm(lnorm_size),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_out = self.net(x)
        x = self.relu(x_out + x)
        return x

class UpsampleFP32(nn.Module):
    def __init__(self, scale, dtype=torch.float32):
        super().__init__()
        self.scale = scale
        self.dtype = dtype
    
    def forward(self, x):
        x_type = x.dtype
        if not BENCHMARKING:
            if x.dtype != self.dtype and ((torch.get_autocast_gpu_dtype() == torch.bfloat16) or (x.dtype == torch.bfloat16)):
                x = x.to(self.dtype)
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if not BENCHMARKING:
            if x_type != x.dtype:
                x = x.to(x_type)
        return x
    
CAUSAL_CONV_FORCE_NON_CAUSAL = False
    
class CausalConv2d(nn.Module):
    def __init__(self, 
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        padding_mode: str = 'zeros',
        dilation: int = 1,
        causal: bool = False,
    ):
        global CAUSAL_CONV_FORCE_NON_CAUSAL
        
        super().__init__()
        
        self.causal = causal
        if CAUSAL_CONV_FORCE_NON_CAUSAL:
            self.causal = False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding)
        self.padding_mode = padding_mode
        self.dilation = dilation
        
        # to follow pytorch initializer
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size)
        w = conv2d.weight.data
        b = conv2d.bias.data
        
        self.bias = nn.Parameter(b)
        if not causal:
            self.weight = nn.Parameter(w)
        else:
            weight = torch.zeros((out_channels, in_channels, kernel_size * 2 - 1, kernel_size))
            weight[:,:,:kernel_size,:] = w
            self.weight = nn.Parameter(weight)
            
            weight_mask = torch.zeros((out_channels, in_channels, kernel_size * 2 - 1, kernel_size))
            weight_mask[:,:,:kernel_size,:] = 1.0
            self.register_buffer('weight_mask', None)
            self.weight_mask = weight_mask
            
            # assert padding == (kernel_size // 2), "always same padding allowed"
            d = dilation if isinstance(dilation, (int, float)) else dilation[0]
            self.padding = ((kernel_size-1) * d, padding)
    
    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(
                input=F.pad(
                    input, 
                    (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), 
                    mode=self.padding_mode
                ),
                weight=weight, 
                bias=bias, 
                stride=self.stride,
                padding=0, 
                dilation=self.dilation, 
                groups=1
            )
        
        # print(input.shape, weight.shape, weight.device, input.device, bias is None)
        
        return F.conv2d(
            input=input, 
            weight=weight, 
            bias=bias, 
            stride=self.stride,
            padding=self.padding, 
            dilation=self.dilation,
            groups=1,
        )
    
    def forward(self, x: torch.Tensor):
        w = self.weight.masked_fill(self.weight_mask == 0, 0) if self.causal else self.weight
        
        # if w.shape[0] == w.shape[1]:
        #     return x
        # if w.shape[0] < w.shape[1]:
        #     return x[:, :w.shape[0], ...]
        
        # torch.cuda.synchronize()
        # t = time.time()
        
        y = self._conv_forward(
            input=x,
            weight=w,
            bias=self.bias,
        )
        
        # torch.cuda.synchronize()
        # print(time.time()-t)
        
        return y
