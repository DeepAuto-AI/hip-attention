import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from hip_attn.v1_2.hip_config import HiPAttentionConfig
from hip_attn.v1_2.uvm_gpu_cache import HiPOffloadCache

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


class HiPModelOffloadCache:
    def __init__(
        self,
        max_token_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: torch.device,
        hip_config: HiPAttentionConfig,
        max_mask_cache_token_size: Optional[int] = None,
        max_sa_cache_token_size: Optional[int] = None,
        max_mask_cache_factor: Optional[float] = None,
        max_sa_cache_factor: Optional[float] = None,
    ):
        from hip_attn.v1_2.uvm_gpu_cache import HiPOffloadCache, format_size_bytes

        assert isinstance(device, torch.device)
        assert device.index is not None

        self.size = max_token_size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num

        # TODO: derive token sizes from size
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.online_update_cache = os.getenv("DEBUG_ONLINE", "0") == "1"
        self.layer_buffer = []
        for layer_id in range(layer_num):
            is_dense = layer_id in hip_config.dense_layers
            if len(hip_config.layers) == 2:
                layer_config = hip_config.layers[0 if is_dense else 1]
            else:
                layer_config = hip_config.layers[layer_id]

            if max_mask_cache_factor is None:
                cur_max_mask_cache_token_size = max_mask_cache_token_size * head_num
                if layer_id in hip_config.dense_layers:
                    cur_max_mask_cache_token_size *= 2
            else:
                base_mask_cache_tokens = (
                    layer_config.sink_token_size
                    + layer_config.sliding_window_size
                    + layer_config.second_stage_k
                )
                cur_max_mask_cache_token_size = math.ceil(
                    max_mask_cache_factor * base_mask_cache_tokens
                )

            if max_sa_cache_factor is None:
                cur_max_sa_cache_token_size = max_sa_cache_token_size * head_num
                if layer_id in hip_config.dense_layers:
                    cur_max_sa_cache_token_size *= 2
            else:
                base_sa_cache_tokens = (
                    max_token_size / layer_config.stages[0].stage_chunk_size
                )
                cur_max_sa_cache_token_size = math.ceil(
                    max_sa_cache_factor * base_sa_cache_tokens
                )

            self.layer_buffer.append(
                HiPOffloadCache(
                    layer_id=layer_id,
                    max_token_size=max_token_size + 1,
                    max_mask_cache_token_size=min(
                        max_token_size * head_num, cur_max_mask_cache_token_size
                    ),
                    max_sa_cache_token_size=min(
                        max_token_size * head_num, cur_max_sa_cache_token_size
                    ),
                    head_num=head_num,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    online_cache_update=self.online_update_cache,
                )
            )

            uvm_allocated_bytes, gpu_allocated_bytes = self._calc_allocated_bytes()
            logger.info(
                f"[{layer_id + 1}/{layer_num}] "
                f"CPU (UVM): {format_size_bytes(uvm_allocated_bytes)} and "
                f"GPU: {format_size_bytes(gpu_allocated_bytes)} are allocated. "
                f"({self.dtype} on {self.device})"
            )

        # (layer_id, batch_id) -> (K, V, seq_len)
        self.prefetch_threads: Dict[Tuple[int, int], threading.Thread] = {}
        self.prefetched_kv: Dict[Tuple[int, int], Tuple[Tensor, Tensor, int]] = {}

        self.async_set_threads: Set[threading.Thread] = set()

        self.enable_async = os.getenv("HIP_DISABLE_AYSNC", "0") == "0"

        # uvm_allocated_bytes, gpu_allocated_bytes = self.calc_allocated_bytes()
        # logger.info(
        #     f'Allocated total CPU (UVM) bytes: {format_size_bytes(uvm_allocated_bytes)}, '
        #     f'Allocated total GPU bytes: {format_size_bytes(gpu_allocated_bytes)}, '
        #     f'{self.dtype} on {self.device}'
        # )

        self.require_validation = os.getenv("HIP_OFFLOAD_CACHE_VALIDATION", "0") == "1"
        if self.require_validation:
            self.validation_cache = MHATokenToKVPool(
                max_token_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=layer_num,
                device=self.device,
            )
        else:
            self.validation_cache = None

    def is_online_cache_update_enabled(self):
        return self.online_update_cache

    def get_kv_buffer(
        self,
        layer_id: int,
    ) -> Tuple[HiPOffloadCache, Any]:
        # Use this function for decode, pass this to `k`
        if self.require_validation:
            return self.layer_buffer[layer_id], self.validation_cache.get_kv_buffer(
                layer_id
            )
        return self.layer_buffer[layer_id], None

    def get_fetched_prefix_kv_buffer(
        self,
        layer_id: int,
        batch_id: Optional[int] = None,
        # you need to pass KV for extend
        cache_k: Optional[Tensor] = None,
        cache_v: Optional[Tensor] = None,
        extend_seq_lens: Optional[Tensor] = None,
        extend_seq_lens_cpu: Optional[List[int]] = None,
    ) -> Tuple[
        Union[Tensor, List[Tensor]], Union[Tensor, List[Tensor]], Union[Any, List[Any]]
    ]:

        if batch_id is not None:
            return self._get_fetched_prefix_kv_buffer_single(
                layer_id=layer_id,
                batch_id=batch_id,
                cache_k=cache_k,
                cache_v=cache_v,
            )

        else:
            k_chunks = []
            v_chunks = []
            offloading_metadata_list = []

            start_len = 0
            for idx_batch, seq_len in enumerate(extend_seq_lens_cpu):
                if seq_len > 0:  # Skip empty sequences
                    k_chunk, v_chunk, offloading_metadata = (
                        self._get_fetched_prefix_kv_buffer_single(
                            layer_id,
                            idx_batch,
                            cache_k=cache_k[start_len : start_len + seq_len].unsqueeze(
                                0
                            ),
                            cache_v=cache_v[start_len : start_len + seq_len].unsqueeze(
                                0
                            ),
                        )
                    )
                    k_chunks.append(k_chunk)
                    v_chunks.append(v_chunk)
                    offloading_metadata_list.append(offloading_metadata)

                else:
                    k_chunks.append(None)
                    v_chunks.append(None)
                    offloading_metadata_list.append(None)

                start_len += seq_len

            return k_chunks, v_chunks, offloading_metadata_list

    def _get_fetched_prefix_kv_buffer_single(
        self,
        layer_id: int,
        batch_id: Optional[int] = None,
        # you need to pass KV for extend
        cache_k: Optional[Tensor] = None,
        cache_v: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Any]:
        # return cache_k, cache_v

        # Use this function for prefill
        handle_id = (layer_id, batch_id)
        prefetch_thread = self.prefetch_threads.get(handle_id, None)
        if prefetch_thread is not None:
            while handle_id not in self.prefetched_kv:
                time.sleep(0.0001)
            # print('start join', flush=True)
            # while True:
            #     try:
            #         prefetch_thread.join(timeout=1.0)
            #         print('joined')
            #         break
            #     except TimeoutError:
            #         print('timeout', layer_id, batch_id)
            #     except RuntimeError:
            #         print('runtime error wtf')
            #         raise RuntimeError('deadlock')

        assert handle_id in self.prefetched_kv, "did prefetch successed?"
        k, v, prefix_seq_len, table = self.prefetched_kv.pop(handle_id)

        assert isinstance(k, Tensor)
        assert isinstance(v, Tensor)
        assert isinstance(prefix_seq_len, int)
        assert k.shape == v.shape
        assert k.ndim == 4, f"{k.shape}"
        assert k.shape[0] == 1
        assert k.shape[1] >= prefix_seq_len
        assert k.shape[2] == self.head_num
        assert k.shape[3] == self.head_dim
        assert k.dtype == v.dtype
        assert k.dtype == self.dtype
        assert cache_k.ndim == 4
        assert cache_k.shape[0] == 1
        assert cache_k.shape[2] == self.head_num
        assert cache_k.shape[3] == self.head_dim
        assert k.shape[1] == prefix_seq_len + cache_k.shape[1]
        assert k.dtype in [
            torch.float8_e5m2,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ]

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        # if self.dtype not in [torch.float8_e5m2]:
        #     assert cache_k.dtype == self.dtype
        # else:
        #     if cache_k.dtype != self.dtype:
        #         cache_k = cache_k.to(self.dtype)
        #         cache_v = cache_v.to(self.dtype)

        k[:, prefix_seq_len:, :, :] = cache_k
        v[:, prefix_seq_len:, :, :] = cache_v

        if self.require_validation:
            k_valid, v_valid = self.validation_cache.get_kv_buffer(layer_id)

            assert k.dtype == k_valid.dtype

            k_valid_packed = k_valid[table].unsqueeze(0)
            v_valid_packed = v_valid[table].unsqueeze(0)

            k_err = ((k_valid_packed - k) ** 2).sum()
            v_err = ((v_valid_packed - v) ** 2).sum()

            assert k_err < 1e-5, k_err
            assert v_err < 1e-5, v_err

            return k, v, (k_valid, v_valid)
        else:
            return k, v, None

    def set_kv_buffer(
        self,
        layer_id: int,
        table: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        async_copy: bool = False,
        push_to_gpu_cache: bool = False,
    ):
        if self.require_validation:
            self.validation_cache.set_kv_buffer(
                layer_id,
                table,
                cache_k,
                cache_v,
            )

        if not self.enable_async:
            async_copy = False

        # pass async_copy=True when only prefill (eager mode)
        assert (not async_copy) or (
            async_copy and (not torch.cuda.is_current_stream_capturing())
        )

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if async_copy:
            start_event = torch.cuda.Event()
            start_event.record()

            def thread_main():
                try:
                    start_event.synchronize()
                    stream = torch.cuda.Stream(device=self.device)

                    with torch.cuda.stream(stream):
                        table_gpu = table.to(torch.int64)
                        table_cpu = table.to("cpu", non_blocking=False)
                        cache_k_cpu = cache_k.to("cpu", non_blocking=False)
                        cache_v_cpu = cache_v.to("cpu", non_blocking=False)
                        self.layer_buffer[layer_id].set_kv_buffer(
                            table=table_cpu,
                            table_gpu=table_gpu,
                            cache_k=cache_k_cpu,
                            cache_v=cache_v_cpu,
                        )
                    stream.synchronize()
                finally:
                    self.async_set_threads.remove(t)

            t = threading.Thread(target=thread_main, daemon=True)
            self.async_set_threads.add(t)
            t.start()
        else:
            self.layer_buffer[layer_id].set_kv_buffer(
                table=table,
                table_gpu=table,
                cache_k=cache_k,
                cache_v=cache_v,
            )

    def on_model_start(
        self,
        is_prefill,
        batch_size: int,
        req_to_token: Tensor,
        req_pool_indices: Tensor,
        extend_prefix_lens_cpu: np.array,
        extend_seq_lens_cpu: np.array,
    ):
        require_prefetch = is_prefill

        if require_prefetch:
            # FIXME: find better way to detect this.
            is_first_chunk = extend_prefix_lens_cpu[0] == 0
            # FIXME: find better way to detect this.
            is_inter_chunk = extend_seq_lens_cpu[0] in map(lambda x: 2**x, range(0, 20))
            # BUG(heejun): at the last chunk of prefill, prefetch layer sometimes failes... so disable async
            if not (batch_size == 1 and (is_first_chunk or is_inter_chunk)):
                self.onetime_disable = self.enable_async
                self.enable_async = False
            else:
                self.onetime_disable = False
            self._prefetch_layer(
                0,
                batch_size,
                req_to_token,
                req_pool_indices,
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
            )
            # self.wait_prefetch_layer(forward_batch, 0)

    def on_model_end(self, is_prefill: bool):
        require_prefetch = is_prefill

        if require_prefetch:
            self._synchronize()
            self.enable_async = self.enable_async or self.onetime_disable
            self.onetime_disable = False

    def on_layer_start(
        self,
        layer_id: int,
        is_prefill: bool,
        batch_size: int,
        req_to_token: Tensor,
        req_pool_indices: Tensor,
        extend_prefix_lens_cpu: np.array,
        extend_seq_lens_cpu: np.array,
    ):
        require_prefetch = is_prefill

        if require_prefetch and (layer_id < (self.layer_num - 1)):
            self._prefetch_layer(
                layer_id + 1,
                batch_size,
                req_to_token,
                req_pool_indices,
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
            )

    def on_layer_end(
        self,
        layer_id: int,
        is_prefill: bool,
    ):
        require_prefetch = is_prefill

        if require_prefetch and (layer_id < (self.layer_num - 1)):
            torch.cuda.current_stream(self.device).synchronize()

    def _prefetch_layer(
        self,
        layer_id: int,
        batch_size: int,
        req_to_token: Tensor,
        req_pool_indices: Tensor,
        extend_prefix_lens_cpu: np.array,
        extend_seq_lens_cpu: np.array,
    ):
        for ibatch in range(batch_size):
            curr_req_pool_indices = req_pool_indices[ibatch : ibatch + 1]
            block_table = req_to_token.index_select(dim=0, index=curr_req_pool_indices)[
                0,
                : extend_prefix_lens_cpu[ibatch] + extend_seq_lens_cpu[ibatch],
            ]
            # print(block_table, block_table.shape)
            self._prefetch_prefix_kv_buffer(
                layer_id=layer_id,
                batch_id=ibatch,
                table=block_table,
                prefix_seq_len=extend_prefix_lens_cpu[ibatch],
            )

    def _prefetch_prefix_kv_buffer(
        self, layer_id: int, batch_id: int, table: Tensor, prefix_seq_len: int
    ) -> threading.Thread:
        # you must call before get fetched prefix
        assert table.ndim == 1

        hip_offload_cache, _ = self.get_kv_buffer(layer_id)

        handle_id = (layer_id, batch_id)
        assert handle_id not in self.prefetch_threads, handle_id
        assert handle_id not in self.prefetched_kv, handle_id

        if self.enable_async:
            start_event = torch.cuda.Event()
            table = table.to(torch.int64).to("cpu")
            start_event.record()

            # torch.cuda.synchronize()
            def thread_main():
                try:
                    # BUG(heejun): i think this line is quite suspicious hmm
                    start_event.synchronize()
                    stream = torch.cuda.Stream(device=self.device, priority=0)

                    with torch.cuda.stream(stream):
                        k, v = hip_offload_cache.prefetch_prefix_kv_buffer(
                            table=table,
                            device=self.device,
                        )
                        assert k.device == self.device
                        assert v.device == self.device

                    stream.synchronize()
                    self.prefetched_kv[handle_id] = (k, v, prefix_seq_len, table)
                except Exception as ex:
                    print(f"{handle_id} thread dead")
                    raise Exception("thread dead") from ex
                finally:
                    self.prefetch_threads.pop(handle_id)

            t = threading.Thread(target=thread_main, daemon=True)
            self.prefetch_threads[handle_id] = t
            t.start()
        else:
            k, v = hip_offload_cache.prefetch_prefix_kv_buffer(
                table=table.to(torch.int64),
                device=self.device,
            )
            assert k.device == self.device
            assert v.device == self.device
            torch.cuda.synchronize()
            self.prefetched_kv[handle_id] = (k, v, prefix_seq_len, table)
        return

    def _synchronize(self):
        torch.cuda.synchronize(device=self.device)
        t = time.time()
        # you must call this function when finish prefill, before decode
        while (len(self.prefetch_threads) > 0) or (len(self.async_set_threads) > 0):
            time.sleep(0.001)
        assert len(self.prefetch_threads) == 0
        assert len(self.async_set_threads) == 0
        assert len(self.prefetched_kv) == 0
        elapsed = time.time() - t
        logger.debug(f"Final layer sync took {elapsed * 1024:.4f} ms")

    def _calc_allocated_bytes(self):
        uvm_allocated_bytes = 0
        gpu_allocated_bytes = 0
        for cache in self.layer_buffer:
            uvm_allocated_bytes += cache.k_uvm.allocated_cpu_bytes
            gpu_allocated_bytes += cache.k_uvm.allocated_gpu_bytes
            uvm_allocated_bytes += cache.v_uvm.allocated_cpu_bytes
            gpu_allocated_bytes += cache.v_uvm.allocated_gpu_bytes
            gpu_allocated_bytes += cache.mask_k_cache.allocated_gpu_bytes
            gpu_allocated_bytes += cache.sa_kv_cache.allocated_gpu_bytes
        return uvm_allocated_bytes, gpu_allocated_bytes


# For validation reference
class MHATokenToKVPool:
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.size = size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self._create_buffers()

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"Reference KV Cache is allocated. K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB."
        )

    def _create_buffers(self):
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            torch.empty(
                (self.size + 1, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.empty(
                (self.size + 1, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        flatten = torch.stack(
            [
                torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
                torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
            ]
        )
        return flatten

    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        for i in range(self.layer_num):
            self.k_buffer[i][indices] = k_data[i]
            self.v_buffer[i][indices] = v_data[i]

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
            self.v_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
        else:
            self.k_buffer[layer_id][loc] = cache_k
            self.v_buffer[layer_id][loc] = cache_v
