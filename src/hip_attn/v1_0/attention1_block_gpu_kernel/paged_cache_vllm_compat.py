from typing import Tuple

import torch
from torch import Tensor


class VllmCompat:
    pass


class PagedKeyCacheVllmCompat(VllmCompat):
    # interface
    dtype: torch.dtype
    device: torch.device
    shape: Tuple[int, int, int]
    ndim: int

    # vllm compat
    key_cache: Tensor
    block_table: Tensor
    context_length: Tensor
    max_context_length: int
    block_size: int

    def __init__(
        self,
        key_cache: Tensor,
        block_table: Tensor,
        context_length: Tensor,
        max_context_length: int,
    ):
        assert max_context_length > 0

        self.key_cache = key_cache
        self.block_table = block_table
        self.context_length = context_length
        self.max_context_length = max_context_length

        self.dtype = key_cache.dtype
        self.device = key_cache.device

        BATCH_SIZE, MAX_NUM_BLOCKS = block_table.shape
        assert context_length.shape == (BATCH_SIZE,)
        assert isinstance(max_context_length, int)

        NUM_BLOCKS, NUM_HEADS, HEAD_SIZE_DIV_X, BLOCK_SIZE, X = key_cache.shape
        HEAD_SIZE = HEAD_SIZE_DIV_X * X

        assert NUM_BLOCKS >= MAX_NUM_BLOCKS
        assert (BLOCK_SIZE * NUM_BLOCKS) >= max_context_length

        self.shape = (BATCH_SIZE * NUM_HEADS, max_context_length, HEAD_SIZE)
        self.block_size = BLOCK_SIZE
        self.ndim = 3

    def stride(self):
        return tuple(
            [
                1,
            ]
            * len(self.shape)
        )

    def data_ptr(self):
        return self.key_cache.data_ptr()


class PagedValueCacheVllmCompat(VllmCompat):
    # interface
    dtype: torch.dtype
    device: torch.device
    shape: Tuple[int, int, int]
    ndim: int

    # vllm compat
    value_cache: Tensor
    block_table: Tensor
    context_length: Tensor
    max_context_length: int
    block_size: int

    def __init__(
        self,
        key_cache: "PagedKeyCacheVllmCompat",
        value_cache: Tensor,
    ):
        self.block_size = key_cache.block_size
        block_table = key_cache.block_table
        context_length = key_cache.context_length
        max_context_length = key_cache.max_context_length

        self.value_cache = value_cache
        self.block_table = block_table
        self.context_length = context_length
        self.max_context_length = max_context_length

        self.dtype = value_cache.dtype
        self.device = value_cache.device

        BATCH_SIZE, MAX_NUM_BLOCKS = block_table.shape
        assert context_length.shape == (BATCH_SIZE,)
        assert isinstance(max_context_length, int)

        NUM_BLOCKS, NUM_HEADS, HEAD_SIZE, BLOCK_SIZE = value_cache.shape

        assert NUM_BLOCKS >= MAX_NUM_BLOCKS
        assert BLOCK_SIZE == self.block_size

        self.shape = (BATCH_SIZE * NUM_HEADS, max_context_length, HEAD_SIZE)
        self.ndim = 3

    def stride(self):
        return tuple(
            [
                1,
            ]
            * len(self.shape)
        )

    def data_ptr(self):
        return self.value_cache.data_ptr()
