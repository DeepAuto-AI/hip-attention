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
from torch import Tensor

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
) -> Tensor:
    """Generate Tree Attention Mask

    Args:
        query (Tensor): fp[N, T_DST, HID]
        key (Tensor): fp[N, T_SRC, HID]
        ps (Union[int, Tensor]): patches. accept int or fp[N, T_DST, 1]
        ks (Union[int, Tensor]): top-ks. accept int or fp[N, T_DST, 1]
    """
    pass