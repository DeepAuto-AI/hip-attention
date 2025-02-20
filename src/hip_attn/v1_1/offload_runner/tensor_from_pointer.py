import os
from math import prod
from typing import Tuple

import torch
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
ops = load(
    "tensor_from_pointer",
    sources=[
        os.path.join(module_path, "tensor_from_pointer.cpp"),
    ],
)


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
    tensor = ops.tensor_from_pointer(
        ptr, prod(shape), elem_size, device_index
    )  # type: torch.Tensor
    tensor = tensor.view(shape).view(dtype)
    return tensor
