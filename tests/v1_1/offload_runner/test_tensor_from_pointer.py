import unittest
from math import prod

import cupy
import numpy as np
import torch

from hip_attn.v1_1.offload_runner.tensor_from_pointer import tensor_from_pointer


class TestTensorFromPointer(unittest.TestCase):

    def test_tensor_from_pointer(self):
        shape = (2, 2)
        dtype = torch.float16

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
