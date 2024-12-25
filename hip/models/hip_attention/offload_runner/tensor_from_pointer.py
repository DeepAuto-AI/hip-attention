import cupy
import numpy as np
import torch
from math import prod
import ctypes
from typing import Tuple
from torch.utils.cpp_extension import load
import os

module_path = os.path.dirname(__file__)
ops = load(
    "tensor_from_pointer",
    sources=[
        os.path.join(module_path, "tensor_from_pointer.cpp"),
    ],
)

def tensor_from_pointer(
    ptr: int, 
    shape: Tuple[int], 
    dtype: torch.dtype, 
    device_index: int
) -> torch.Tensor:
    if dtype == torch.float16:
        elem_size = 16
    elif dtype == torch.bfloat16:
        elem_size = 16
    elif dtype == torch.float32:
        elem_size = 32
    elif dtype in [torch.uint8, torch.float8_e5m2]:
        elem_size = 8
    else:
        raise NotImplementedError()
    tensor = ops.tensor_from_pointer(
        ptr, 
        prod(shape), 
        elem_size, 
        device_index
    ) # type: torch.Tensor
    tensor = tensor.view(shape).view(dtype)
    return tensor

if __name__ == '__main__':
    shape = (2, 2)
    dtype = torch.float16
    
    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = prod(shape)
    byte_size = elem_size * numel
    pointer = cupy.cuda.malloc_managed(byte_size)
    print(f'manged alloc {pointer.ptr}')
    
    ret = tensor_from_pointer(pointer.ptr, shape, dtype, 0)
    print(ret)
    
    from cupy.cuda.memory import MemoryPointer, UnownedMemory
    def device_ptr_2_cupy(pDevice: int, span: int, dtype: type, owner = None):
        sizeByte = span * 2
        mem = UnownedMemory(pDevice, sizeByte, owner)
        memptr = MemoryPointer(mem, 0)
        return cupy.ndarray(span, dtype=dtype, memptr=memptr)
    
    ret_cupy = device_ptr_2_cupy(pointer.ptr, numel, cupy.float16)
    print(ret_cupy)
    
    ret[0, 0] = 1
    ret[0, 1] = 2
    ret[1, 0] = 3
    ret[1, 1] = 4
    
    print(ret)
    print(ret_cupy)
    
    assert np.all(ret.view(-1).cpu().numpy() == cupy.asnumpy(ret_cupy))
    
    print('pass')