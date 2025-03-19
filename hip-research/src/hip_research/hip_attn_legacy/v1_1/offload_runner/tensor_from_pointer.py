import ctypes
import math
import os
from math import prod
from typing import Tuple, Union

import cuda
import cuda.cudart
import cupy
import numpy as np
import torch
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
load(
    "tensor_from_pointer",
    verbose=False,
    is_python_module=False,
    is_standalone=False,
    sources=[
        os.path.join(module_path, "tensor_from_pointer.cpp"),
    ],
    extra_cflags=["-O3"],
)


def sizeof(dtype: Union[torch.Tensor, torch.dtype]) -> int:
    if isinstance(dtype, torch.Tensor):
        return dtype.numel() * sizeof(dtype.dtype)

    if dtype in [
        torch.uint8,
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    ]:
        return 1
    elif dtype in [
        torch.uint16,
        torch.int16,
        torch.float16,
        torch.bfloat16,
    ]:
        return 2
    elif dtype in [
        torch.uint32,
        torch.int32,
        torch.float32,
    ]:
        return 4
    elif dtype in [
        torch.uint64,
        torch.int64,
        torch.float64,
    ]:
        return 8
    else:
        raise Exception()


def tensor_from_pointer(
    ptr: int, shape: Tuple[int], dtype: torch.dtype, device_index: int
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
    tensor = torch.ops.hip_attn.tensor_from_pointer(
        ptr, prod(shape), elem_size, device_index
    )  # type: torch.Tensor
    tensor = tensor.view(shape).view(dtype)
    return tensor


def alloc_managed_tensor(
    shape: tuple,
    dtype: torch.dtype,
    device: Union[str, torch.device],
):
    device = device
    if isinstance(device, str):
        device = torch.device(device)

    elem_size = sizeof(dtype)
    numel = math.prod(shape)
    align = 4096
    byte_size = elem_size * numel
    byte_size = byte_size + byte_size % align

    _result_code, pointer = cuda.cudart.cudaMallocManaged(
        byte_size, cuda.cudart.cudaMemAttachGlobal
    )

    t_gpu = tensor_from_pointer(pointer, shape, dtype, device.index)
    t_cpu = tensor_from_pointer(pointer, shape, dtype, -1)

    return t_gpu, t_cpu


if __name__ == "__main__":
    shape = (2, 2)
    dtype = torch.bfloat16

    elem_size = torch.tensor([], dtype=dtype).element_size()
    numel = prod(shape)
    byte_size = elem_size * numel
    pointer = cupy.cuda.malloc_managed(byte_size)
    print(f"manged alloc {pointer.ptr}")

    ret = tensor_from_pointer(pointer.ptr, shape, dtype, 0)
    print(ret)

    from cupy.cuda.memory import MemoryPointer, UnownedMemory

    def device_ptr_2_cupy(pDevice: int, span: int, dtype: type, owner=None):
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

    print("pass")
