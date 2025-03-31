import math
import os
from typing import Optional, Tuple, Union

import cuda
import cuda.cudart
import torch
import tqdm
import triton
import triton.language as tl
from torch import Tensor

from hip_attn.v1_1.offload_runner.tensor_from_pointer import (
    alloc_managed_tensor,
    sizeof,
)
from hip_attn.v1_2.attention_metadata import (
    HiPAttentionCacheAccessStatistics,
    HiPAttentionOutputMetadata,
)

MAX_INT: tl.constexpr = tl.constexpr(90000000)
MAX_INT_ACQUIRED: tl.constexpr = tl.constexpr(90000001)


def format_size_bytes(tensor: Union[Tensor, Union[float, int]]) -> str:
    if isinstance(tensor, Tensor):
        byte_size = sizeof(tensor)
    elif isinstance(tensor, (int, float)):
        byte_size = tensor
    else:
        raise Exception()

    if byte_size < 1024:
        return f"{byte_size} B"
    elif byte_size < (1024**2):
        return f"{byte_size / 1024:.2f} KB"
    elif byte_size < (1024**3):
        return f"{byte_size / (1024 ** 2):.2f} MB"
    else:
        return f"{byte_size / (1024 ** 3):.2f} GB"


def debug_print(*args):
    print(f'[HiPOffloadKVPoolMHA] {" ".join(map(lambda x: str(x), args))}')


###############################################################################
#                               Data Structure
###############################################################################


def uvm_note_cpu(tensor: Tensor, prefetch: bool = False):
    cuda.cudart.cudaMemAdvise(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation,
        -1,
    )
    cuda.cudart.cudaMemAdvise(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy,
        tensor.device.index,
    )
    if prefetch:
        cuda.cudart.cudaMemPrefetchAsync(
            tensor.data_ptr(), tensor.numel() * tensor.element_size(), -1, 0
        )


class UVMCache:
    bank_cpu: Tensor
    bank_gpu: Tensor
    metadata: Tensor

    def __init__(
        self,
        layer_id: int,
        max_token_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.layer_id = layer_id
        self.max_token_size = max_token_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        if self.device.index is None:
            self.device = torch.get_default_device()

        self.bank_cpu, self.bank_gpu = self.alloc_uvm(
            [max_token_size, head_num, head_dim], dtype=self.dtype
        )

        # {
        #     Token Generation: uint32    # Increase one on every overwrite
        # }
        self.metadata = torch.full(
            [max_token_size, 1], dtype=torch.int32, device=device, fill_value=0
        )

        self.allocated_cpu_bytes = sizeof(self.bank_cpu)
        self.allocated_gpu_bytes = sizeof(self.metadata)

        # debug_print(f'UVMCache: bank={format_size_bytes(self.bank_cpu)}, metadata={format_size_bytes(self.metadata)}')

    def alloc_uvm(self, shape, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        t_gpu, t_cpu = alloc_managed_tensor(shape, dtype, self.device)

        uvm_note_cpu(t_gpu)
        t_cpu.fill_(0)

        return t_cpu, t_gpu

    def gather_cpu(self, table: Tensor, pin_memory=False) -> Tensor:
        assert table.ndim == 1
        assert table.device == self.bank_cpu.device

        # print('gather alloc', flush=True)
        t = torch.empty(
            (table.shape[0], self.bank_cpu.shape[1], self.bank_cpu.shape[2]),
            dtype=self.bank_cpu.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )

        view_dtype = torch.uint16
        view_dtype_np = np.uint16
        if self.bank_cpu.dtype in [torch.float32]:
            view_dtype = torch.uint32
            view_dtype_np = np.uint32
        elif self.bank_cpu.dtype in [torch.float16, torch.bfloat16]:
            view_dtype = torch.uint16
            view_dtype_np = np.uint16
        elif self.bank_cpu.dtype in [torch.uint8, torch.float8_e5m2]:
            view_dtype = torch.uint8
            view_dtype_np = np.uint8
        else:
            raise Exception()

        # t = np.empty(
        #     (table.shape[0], self.bank_cpu.shape[1], self.bank_cpu.shape[2]),
        #     dtype=view_dtype_np,
        # )

        # print('gather pin', flush=True)
        # if pin_memory:
        #     t = t.pin_memory()

        # print('gather index_copy', flush=True)
        index_copy(
            self.bank_cpu.view(dtype=view_dtype).numpy(),
            t.view(dtype=view_dtype).numpy(),
            table.numpy(),
            num_thread=os.cpu_count(),
        )
        # print('gather done', flush=True)

        # t = torch.from_numpy(t).view(self.bank_cpu.dtype)
        # print('convert done', flush=True)

        return t


import numba
import numpy as np


@numba.njit(parallel=True)
def index_copy(
    src: np.ndarray, out: np.ndarray, table: np.ndarray, num_thread: int = 32
):
    chunk_size = math.ceil(table.shape[0] / num_thread)
    for ithread in numba.prange(num_thread):
        for i in range(chunk_size):
            t = chunk_size * ithread + i
            if t < table.shape[0]:
                out[t] = src[table[t]]


def pad_to_cacheline(nelem: int, dtype: torch.dtype):
    byte_size = 4
    if dtype in [torch.int32, torch.uint32, torch.float32]:
        byte_size = 4
    elif dtype in [torch.int64, torch.uint64, torch.float64]:
        byte_size = 8
    elif dtype in [torch.int16, torch.uint16, torch.bfloat16, torch.float16]:
        byte_size = 2
    else:
        raise Exception()

    assert nelem > 0

    # in bytes
    cacheline_size = 32

    step = max(1, cacheline_size // byte_size)
    return nelem if (nelem % step) == 0 else (nelem + step - (nelem % step))


class GPUCache:
    global_metadata: Tensor
    bank: Tensor
    metadata: Tensor
    table: Tensor

    def __init__(
        self,
        k_uvm: UVMCache,
        v_uvm: Optional[UVMCache],
        max_cache_token_size: int,
        online_cache_update: bool,
    ):
        self.k_uvm = k_uvm
        self.v_uvm = v_uvm
        self.head_num = self.k_uvm.head_num
        self.head_dim = self.k_uvm.head_dim
        self.dtype = self.k_uvm.dtype
        self.device = self.k_uvm.device
        self.kv_packed = self.v_uvm is not None
        if self.kv_packed:
            assert self.head_num == self.v_uvm.head_num
            self.head_dim += self.v_uvm.head_dim
        self.max_cache_token_size = max_cache_token_size
        self.max_uvm_token_size = self.k_uvm.max_token_size
        if self.kv_packed:
            assert self.max_uvm_token_size == self.v_uvm.max_token_size

        """
        [
            CachelinePadded {
                current_tick: int32,
            }
            CachelinePadded {
                random_seed: int32,
            }
        ]
        """
        self.global_metadata = torch.zeros(
            (2, pad_to_cacheline(1, torch.int32)), dtype=torch.int32, device=self.device
        )

        self.bank = torch.zeros(
            (self.max_cache_token_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        """
        CachelinePadded {
            [0] Back reference to table: int64,         # initial handshake, store token index of UVM bank
            [1] Reference to UVM Cache: int64,          # MAX_TOKEN, for token generation check
            [2] Token Generation of UVM Cache: int64,   # To check the version of cached token
            [3] Last accessed tick: int64,
            [4] Not accessed duration: int64,           # Increse one every step
            [5] Did taken in this kernel step: int64    # Reset to zero on every step
            [6] Token hash: int64                       # for debug
        }
        """
        self.metadata = torch.zeros(
            (self.max_cache_token_size, pad_to_cacheline(7, torch.int64)),
            dtype=torch.int64,
            device=self.device,
        )
        # self.metadata[:, 0].fill_(0)
        # self.metadata[:, 1].fill_(0)
        # self.metadata[:, 2].fill_(0)
        # self.metadata[:, 3].fill_(0)
        self.metadata[:, 4].fill_(1)
        # self.metadata[:, 5].fill_(0)

        # NOTE: this table is way too large to pad... sorry
        self.table = torch.full(
            (self.head_num, self.max_uvm_token_size, 1),
            dtype=torch.int32,
            device=self.device,
            fill_value=MAX_INT.value,
        )

        self.allocated_gpu_bytes = (
            sizeof(self.global_metadata)
            + sizeof(self.bank)
            + sizeof(self.metadata)
            + sizeof(self.table)
        )

        self.step = 0

        self.online_update_cache = online_cache_update

    def handle_cache_miss(
        self,
        metadata: HiPAttentionOutputMetadata,
        stats: HiPAttentionCacheAccessStatistics,
    ):
        # self._verify_cache()

        # NOTE: increase not accessed timer
        self.metadata[:, 4].add_(1)
        self.metadata[:, 5].fill_(0)

        # if id(stats) == id(metadata.mask_cache_statistics): return
        if stats is None:
            return
        if self.online_update_cache:
            return

        # NOTE: this function should be capturable.
        # NOTE: this function will called only when mask is updated

        uvm_page_count = self.k_uvm.bank_cpu.shape[0]
        gpu_page_count = self.bank.shape[0]

        assert stats.cache_miss_counter.shape[1:] == (
            self.head_num,
            uvm_page_count,
        ), f"{stats.cache_miss_counter.shape[1:]} == [{self.head_num}, {uvm_page_count}]"

        # update LRU recency
        # increase LRU step
        self.global_metadata[0, 0].add_(1)

        if stats.access_counter.shape[0] != 1:
            # assert False
            # NOTE: if paged attention, stats should be single batch.
            accessed = stats.access_counter.sum(0)
        else:
            accessed = stats.access_counter.squeeze(0)

        assert accessed.ndim == 2
        assert accessed.shape == (self.head_num, uvm_page_count)
        assert self.k_uvm.metadata.shape == (uvm_page_count, 1)
        assert self.global_metadata.shape == (
            2,
            pad_to_cacheline(1, self.global_metadata.dtype),
        )
        assert self.metadata.shape == (
            self.bank.shape[0],
            pad_to_cacheline(5, self.metadata.dtype),
        )
        assert self.table.shape == (self.head_num, uvm_page_count, 1)

        BLOCK_SIZE = 128
        grid = (self.head_num * triton.cdiv(uvm_page_count, BLOCK_SIZE),)
        update_recency[grid](
            accessed,
            *accessed.stride(),
            self.k_uvm.metadata,
            *self.k_uvm.metadata.stride(),
            self.global_metadata,
            *self.global_metadata.stride(),
            self.metadata,
            *self.metadata.stride(),
            self.table,
            *self.table.stride(),
            uvm_page_count,
            self.k_uvm.bank_cpu.shape[1],
            BLOCK_SIZE,
            num_warps=4,
        )
        self.step += 1

        # perform LRU
        assert gpu_page_count <= (
            uvm_page_count * self.head_num
        ), f"{gpu_page_count} <= {(uvm_page_count * self.head_num)}"

        # cache_miss = ((stats.cache_miss_counter > 0) * stats.access_counter).sum(0).view(-1)
        if stats.cache_miss_counter.shape[0] == 1:
            cache_miss = stats.cache_miss_counter.view(-1)
        else:
            cache_miss = (
                ((stats.cache_miss_counter > 0) * stats.access_counter).sum(0).view(-1)
            )
        put_mask = cache_miss > 0
        # put_priority_list = cache_miss.argsort(-1, descending=True, stable=False)
        # put_priority_list = put_priority_list[:gpu_page_count]
        put_priority_list = cache_miss.topk(
            k=gpu_page_count, dim=-1, largest=True, sorted=False
        ).indices
        put_mask = put_mask[put_priority_list]

        slot_recency = self.metadata[:, 3]
        evict_priority_list = slot_recency.argsort(
            dim=-1, descending=False, stable=False
        )

        self.write_cache(
            put_list=put_priority_list,
            put_mask=put_mask,
            evict_list=evict_priority_list,
        )

        # NOTE: for debug
        self.verify_cache(put_mask)

    def verify_cache(
        self,
        put_mask: Optional[Tensor] = None,
        force: bool = False,
        max_verification: int = 10000,
    ):
        if (os.getenv("DEBUG_ONLINE_VERIFY", "0") == "0") and (not force):
            return
        if self.k_uvm.layer_id != 3 and (not force):
            return

        torch.cuda.synchronize()
        table = self.table.cpu()
        metadata = self.metadata.cpu()
        bank = self.bank.cpu()
        uvm_metadata = self.k_uvm.metadata.cpu()
        uvm_k_bank = self.k_uvm.bank_cpu
        uvm_v_bank = self.v_uvm.bank_cpu if self.kv_packed else None

        total_table_hit = 0
        total_back_ref_hit = 0
        total_uvm_ref_hit = 0
        total_token_gen_hit = 0
        total_hash_hit = 0
        total_cache_hit = 0
        for idx_head in range(table.shape[0]):
            for idx_page in tqdm.tqdm(
                range(min(table.shape[1], max_verification)),
                dynamic_ncols=True,
                leave=False,
            ):
                target_slot = table[idx_head, idx_page].item()
                if target_slot < MAX_INT:
                    total_table_hit += 1
                    (
                        back_ref,
                        ref_to_uvm,
                        token_gen,
                        last_tick,
                        sleep_tick,
                        is_touched,
                        token_hash,
                    ) = metadata[target_slot, :7]
                    if (back_ref // table.shape[0]) == idx_page:
                        total_back_ref_hit += 1
                        if ref_to_uvm < MAX_INT:
                            total_uvm_ref_hit += 1
                            if uvm_metadata[ref_to_uvm, 0] == token_gen:
                                total_token_gen_hit += 1
                                gpu_value = bank[target_slot]
                                if not self.kv_packed:
                                    cpu_value = uvm_k_bank[idx_page, idx_head]
                                else:
                                    cpu_value = torch.cat(
                                        [
                                            uvm_k_bank[idx_page, idx_head],
                                            uvm_v_bank[idx_page, idx_head],
                                        ],
                                        dim=0,
                                    )
                                gpu_hash = (
                                    gpu_value.view(torch.uint16)
                                    .to(torch.uint32)
                                    .sum()
                                    .item()
                                )
                                cpu_hash = (
                                    cpu_value.view(torch.uint16)
                                    .to(torch.uint32)
                                    .sum()
                                    .item()
                                )
                                mse = ((gpu_value - cpu_value) ** 2).mean().item()
                                check_pass = (mse < 1e-4) or True
                                if not check_pass:
                                    error_location = (
                                        uvm_k_bank.view(torch.uint16)
                                        .to(torch.uint32)
                                        .sum(dim=-1)
                                        == token_hash
                                    ).nonzero()
                                    error_cpu_location = (
                                        uvm_k_bank.view(torch.uint16)
                                        .to(torch.uint32)
                                        .sum(dim=-1)
                                        == gpu_hash
                                    ).nonzero()
                                    error_msg = f"""
cache_hit={total_cache_hit}, token_gen={total_token_gen_hit}, ref_to_uvm={total_uvm_ref_hit}, back_ref={total_back_ref_hit}
GPU = {gpu_value} ({gpu_value.shape})
-----
UVM = {cpu_value} ({cpu_value.shape})
token_hash={token_hash}, gpu_hash={gpu_hash}, cpu_hash={cpu_hash}, token_found={error_location}, uvm_found={error_cpu_location}
head={idx_head}, page={idx_page}, slot={target_slot}, backref={back_ref}, uvmref={ref_to_uvm}, gen={token_gen}, is_touched={is_touched}, mse={mse}"""
                                    assert False, error_msg
                                total_cache_hit += 1
                                if gpu_hash == cpu_hash:
                                    total_hash_hit += 1

        if put_mask is not None:
            print("lastly put", put_mask.sum().item())
        tqdm.tqdm.write(
            f"verified kv_packed={self.kv_packed}, cache_hit={total_cache_hit}, token_gen={total_token_gen_hit}, ref_to_uvm={total_uvm_ref_hit}, back_ref={total_back_ref_hit}, hash_hit={total_hash_hit}"
        )

    def write_cache(
        self,
        put_list: Tensor,
        put_mask: Tensor,
        evict_list: Tensor,
    ):
        assert put_list.shape == put_mask.shape
        assert evict_list.shape == put_list.shape

        BLOCK_SIZE = 128

        qsize = put_list.shape[0]

        grid = (triton.cdiv(qsize, BLOCK_SIZE),)
        write_cache[grid](
            put_list,
            *put_list.stride(),
            put_mask,
            *put_mask.stride(),
            evict_list,
            *evict_list.stride(),
            self.bank,
            *self.bank.stride(),
            self.metadata,
            *self.metadata.stride(),
            self.table,
            *self.table.stride(),
            self.k_uvm.metadata,
            *self.k_uvm.metadata.stride(),
            self.k_uvm.bank_gpu,
            *self.k_uvm.bank_gpu.stride(),
            self.v_uvm.bank_gpu if self.kv_packed else None,
            *(self.v_uvm.bank_gpu.stride() if self.kv_packed else (0, 0, 0)),
            self.global_metadata,
            *self.global_metadata.stride(),
            qsize,
            self.k_uvm.bank_gpu.shape[0],
            self.k_uvm.bank_gpu.shape[1],
            self.kv_packed,
            BLOCK_SIZE,
            self.k_uvm.bank_gpu.shape[-1],
        )

    def on_set_kv_buffer(
        self,
        table: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # NOTE: cache_k, and cache_v might not on GPU (during prefill)
        # if table is allocated to valid slots, copy tensors to bank
        # if slot is not valid, unlink the table
        # NOTE: but for temporary, just unlink table always

        assert table.device == self.device

        self.table[:, :, 0].index_fill_(
            dim=1, index=table.to(torch.int64), value=MAX_INT.value
        )

    def flush(self):
        self.table[:, :, 0].fill_(MAX_INT.value)


class HiPOffloadCache:
    def __init__(
        self,
        layer_id: int,
        max_token_size: int,
        max_mask_cache_token_size: int,
        max_sa_cache_token_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        online_cache_update: bool,
    ):
        self.k_uvm = UVMCache(
            layer_id=layer_id,
            max_token_size=max_token_size,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        self.v_uvm = UVMCache(
            layer_id=layer_id,
            max_token_size=max_token_size,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        self.mask_k_cache = GPUCache(
            k_uvm=self.k_uvm,
            v_uvm=None,
            max_cache_token_size=max_mask_cache_token_size,
            online_cache_update=online_cache_update,
        )

        self.sa_kv_cache = GPUCache(
            k_uvm=self.k_uvm,
            v_uvm=self.v_uvm,
            max_cache_token_size=max_sa_cache_token_size,
            online_cache_update=online_cache_update,
        )

    def get_page_count(self):
        assert self.k_uvm.bank_cpu.shape == self.v_uvm.bank_cpu.shape
        return self.k_uvm.bank_cpu.shape[0]

    def prefetch_prefix_kv_buffer(
        self, table: Tensor, device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        if table.device != torch.device("cpu"):
            table = table.to("cpu", non_blocking=False)
        k = self.k_uvm.gather_cpu(table, pin_memory=True)
        v = self.v_uvm.gather_cpu(table, pin_memory=True)
        k = k.to(device, non_blocking=True).unsqueeze(0)
        v = v.to(device, non_blocking=True).unsqueeze(0)
        return k, v

    def set_kv_buffer(
        self,
        table: torch.Tensor,
        table_gpu: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        cache_device = cache_k.device
        assert table.device == cache_device
        assert cache_v.device == cache_device

        if cache_device == torch.device("cpu"):
            # self.k_uvm.bank_cpu[table] = cache_k
            # self.v_uvm.bank_cpu[table] = cache_v
            if cache_k.dtype in [torch.float16, torch.bfloat16]:
                view_dtype = torch.uint16
            elif cache_k.dtype in [torch.float32]:
                view_dtype = torch.uint32
            elif cache_k.dtype in [torch.uint8, torch.float8_e5m2]:
                view_dtype = torch.uint8
            else:
                raise Exception()

            set_kv_buffer_(
                self.k_uvm.bank_cpu.view(view_dtype).numpy(),
                cache_k.view(view_dtype).numpy(),
                table.numpy(),
                os.cpu_count(),
            )

            set_kv_buffer_(
                self.v_uvm.bank_cpu.view(view_dtype).numpy(),
                cache_v.view(view_dtype).numpy(),
                table.numpy(),
                os.cpu_count(),
            )
        else:
            assert cache_device == self.k_uvm.device
            self.k_uvm.bank_gpu[table] = cache_k
            self.v_uvm.bank_gpu[table] = cache_v

        self.mask_k_cache.on_set_kv_buffer(
            table=table_gpu,
            cache_k=cache_k,
            cache_v=cache_v,
        )
        self.sa_kv_cache.on_set_kv_buffer(
            table=table_gpu,
            cache_k=cache_k,
            cache_v=cache_v,
        )

        self.k_uvm.metadata.index_put_(
            indices=(table,),
            values=torch.index_select(self.k_uvm.metadata, index=table_gpu, dim=0) + 1,
        )
        self.v_uvm.metadata.index_put_(
            indices=(table,),
            values=torch.index_select(self.v_uvm.metadata, index=table_gpu, dim=0) + 1,
        )

    def handle_cache_miss(self, metadata: HiPAttentionOutputMetadata):
        if metadata.mask_cache_statistics is not None:
            self.mask_k_cache.handle_cache_miss(
                metadata=metadata, stats=metadata.mask_cache_statistics
            )
            self.sa_kv_cache.handle_cache_miss(
                metadata=metadata, stats=metadata.sa_cache_statistics
            )
        else:
            self.mask_k_cache.handle_cache_miss(metadata=metadata, stats=None)
            self.sa_kv_cache.handle_cache_miss(metadata=metadata, stats=None)


###############################################################################
#                               Kernel Function
###############################################################################


@numba.njit(parallel=True, fastmath=True)
def set_kv_buffer_(
    bank: np.ndarray, cache: np.ndarray, table: np.ndarray, num_threads: int
):
    assert table.shape[0] == cache.shape[0]

    chunk_size = math.ceil(table.shape[0] / num_threads)
    for ithread in numba.prange(num_threads):
        for i in range(chunk_size):
            t = ithread * chunk_size + i
            if t < table.shape[0]:
                bank[table[t]] = cache[t]


@triton.jit
def validate_bank_metadata_slots(
    UVM_METADATA,
    stride_uvm_metadata_token,
    stride_uvm_metadata_k,
    METADATA,
    stride_metadata_slot,
    stride_metadata_k,
    idx_slot,
    idx_page,  # this is optional. if given, check backref
    cache_hit,
    HEAD_KV,
):
    cache_hit = (idx_slot < MAX_INT) & cache_hit

    back_ref = tl.load(
        METADATA + idx_slot.to(tl.int64) * stride_metadata_slot + 0 * stride_metadata_k,
        mask=cache_hit,
    )

    if idx_page is not None:
        cache_hit = (
            ((back_ref // HEAD_KV) == idx_page) & (back_ref < MAX_INT) & cache_hit
        )
    else:
        cache_hit = (back_ref < MAX_INT) & cache_hit

    ref_to_uvm = tl.load(
        METADATA + idx_slot.to(tl.int64) * stride_metadata_slot + 1 * stride_metadata_k,
        mask=cache_hit,
    ).to(tl.int64)
    cache_hit = (ref_to_uvm < MAX_INT) & cache_hit

    uvm_token_gen = tl.load(
        UVM_METADATA
        + ref_to_uvm.to(tl.int64) * stride_uvm_metadata_token
        + 0 * stride_uvm_metadata_k,
        mask=cache_hit,
    )
    cache_token_gen = tl.load(
        METADATA + idx_slot.to(tl.int64) * stride_metadata_slot + 2 * stride_metadata_k,
        mask=cache_hit,
    )
    cache_hit = (
        (uvm_token_gen < MAX_INT) & (uvm_token_gen == cache_token_gen) & cache_hit
    )

    tl.static_assert(cache_hit.dtype == tl.int1)

    return cache_hit


@triton.jit
def load_tokens(
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_KV_PACKED: tl.constexpr,
    GPU_BANK_COUNT: int,
    OFFLOAD_CACHE_LOAD_VALUE: tl.constexpr,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
    stride_offload_cache_gpu_global_metadata_k,
    stride_offload_cache_gpu_global_metadata_pad,
    OFFLOAD_CACHE_GPU_BANK,
    stride_offload_cache_gpu_bank_token,
    stride_offload_cache_gpu_bank_hid,
    OFFLOAD_CACHE_GPU_METADATA,
    stride_offload_cache_gpu_metadata_token,
    stride_offload_cache_gpu_metadata_k,
    OFFLOAD_CACHE_GPU_TABLE,
    stride_offload_cache_gpu_table_head_kv,
    stride_offload_cache_gpu_table_token,
    strdie_offload_cache_gpu_table_k,
    ACCESS_COUNTER,
    stride_access_counter_bsz,
    stride_access_counter_head_kv,
    stride_access_counter_tsrc,
    CACHE_MISS_COUNTER,
    stride_cache_miss_counter_bsz,
    stride_cache_miss_counter_head_kv,
    stride_cache_miss_counter_tsrc,
    idx_bsz,
    idx_tsrc,
    idx_kv_head,
    idx_hid,
    mask_keys,
    HEAD_KV: int,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    HID_DIM,
    IS_BSA: tl.constexpr = False,
    UPDATE_CACHE: tl.constexpr = False,
    V_CACHE=None,
    stride_v_cache_page=None,
    stride_v_cache_offset=None,
    stride_v_cache_kv_head=None,
    stride_v_cache_hid=None,
):
    # DEBUG: to load nothing
    # mask_keys = mask_keys & False

    # tl.static_print(OFFLOAD_CACHE_METHOD)

    if not USING_PAGES:
        tl.static_assert(not USING_OFFLOAD_CACHE)

        if ACCESS_COUNTER is not None:
            # tl.atomic_add(
            #     ACCESS_COUNTER +\
            #         idx_bsz.to(tl.int64) * stride_access_counter_bsz +\
            #         idx_kv_head * stride_access_counter_head_kv +\
            #         idx_tsrc * stride_access_counter_tsrc,
            #     mask=mask_keys,
            #     val=1
            # )

            tl.store(
                ACCESS_COUNTER
                + idx_bsz.to(tl.int64) * stride_access_counter_bsz
                + idx_kv_head * stride_access_counter_head_kv
                + idx_tsrc * stride_access_counter_tsrc,
                mask=mask_keys,
                value=1,
            )

        keys = tl.load(
            K
            + idx_bsz.to(tl.int64) * stride_k_bsz
            + idx_tsrc.to(tl.int64) * stride_k_tsrc
            + idx_kv_head.to(tl.int64) * stride_k_head
            + idx_hid.to(tl.int64) * stride_k_hid,
            mask=mask_keys & (idx_hid < HID_DIM),
            other=0.0,
            # cache_modifier='.cs', # TODO: uncomment this
        )
    else:
        seq_len = tl.load(
            CACHE_SEQ_LENS + idx_bsz.to(tl.int64) * stride_cache_seq_lens_b,
        )
        mask_tsrc = (idx_tsrc >= 0) & (idx_tsrc < seq_len)
        ptrs = (
            BLOCK_TABLE
            + idx_bsz.to(tl.int64) * stride_block_table_bsz
            + (idx_tsrc // PAGE_SIZE).to(tl.int64) * stride_block_table_page
        )
        idx_page = tl.load(
            ptrs,
            mask=mask_tsrc,
            other=0,
        ).to(tl.int64)
        offset_page = idx_tsrc % PAGE_SIZE

        if ACCESS_COUNTER is not None:
            # tl.atomic_add(
            #     ACCESS_COUNTER +\
            #         idx_bsz.to(tl.int64) * stride_access_counter_bsz +\
            #         idx_kv_head * stride_access_counter_head_kv +\
            #         idx_page * stride_access_counter_tsrc,
            #     mask=mask_keys,
            #     val=1
            # )

            tl.store(
                ACCESS_COUNTER
                + idx_bsz.to(tl.int64) * stride_access_counter_bsz
                + idx_kv_head * stride_access_counter_head_kv
                + idx_page * stride_access_counter_tsrc,
                mask=mask_keys,
                value=1,
            )

        original_mask_keys = mask_keys

        if USING_OFFLOAD_CACHE:
            tl.static_assert(PAGE_SIZE == 1)

            idx_slots = tl.load(
                OFFLOAD_CACHE_GPU_TABLE
                + idx_page.to(tl.int64) * stride_offload_cache_gpu_table_token
                + idx_kv_head * stride_offload_cache_gpu_table_head_kv
                + 0 * strdie_offload_cache_gpu_table_k,
                mask=mask_keys,
            )
            idx_slot_has_reference_to_bank = (idx_slots < MAX_INT) & mask_keys
            idx_slots = idx_slots * idx_slot_has_reference_to_bank

            ALWAYS_VALIDATE_LINK: tl.constexpr = False  # not UPDATE_CACHE

            if ALWAYS_VALIDATE_LINK:
                validated_slots = validate_bank_metadata_slots(
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    idx_slots,
                    idx_page,
                    idx_slot_has_reference_to_bank,
                    HEAD_KV,
                )

                mask_slot_cache_hit = validated_slots & idx_slot_has_reference_to_bank
            else:
                mask_slot_cache_hit = idx_slot_has_reference_to_bank

            # if OFFLOAD_CACHE_LOAD_VALUE:
            #     mask_slot_cache_hit = mask_slot_cache_hit & False

            idx_hid_cached = idx_hid
            if OFFLOAD_CACHE_LOAD_VALUE:
                idx_hid_cached += BLOCK_HID
            keys_cached = tl.load(
                OFFLOAD_CACHE_GPU_BANK
                + idx_slots.to(tl.int64) * stride_offload_cache_gpu_bank_token
                + idx_hid_cached * stride_offload_cache_gpu_bank_hid,
                mask=mask_slot_cache_hit,
                other=0.0,
            )
            if keys_cached.dtype == tl.uint8:
                keys_cached = keys_cached.to(tl.float8e5, bitcast=True).to(tl.bfloat16)
            if keys_cached.dtype == tl.float8e5:
                keys_cached = keys_cached.to(tl.bfloat16)

            if UPDATE_CACHE:
                idx_slots_verify = tl.load(
                    OFFLOAD_CACHE_GPU_TABLE
                    + idx_kv_head.to(tl.int64) * stride_offload_cache_gpu_table_head_kv
                    + idx_page * stride_offload_cache_gpu_table_token
                    + 0 * strdie_offload_cache_gpu_table_k,
                    mask=mask_slot_cache_hit,
                )
                mask_slot_cache_hit = (
                    (idx_slots_verify < MAX_INT)
                    & (idx_slots == idx_slots_verify)
                    & mask_slot_cache_hit
                )

            if mask_slot_cache_hit.shape[0] == 1:
                keys_cached_hash = tl.sum(
                    keys_cached.to(tl.uint16, bitcast=True), axis=0, keep_dims=True
                ).to(tl.uint64)
            elif mask_slot_cache_hit.shape[1] == 1:
                keys_cached_hash = tl.sum(
                    keys_cached.to(tl.uint16, bitcast=True), axis=1, keep_dims=True
                ).to(tl.uint64)
            else:
                raise Exception()
            tl.debug_barrier()

            tl.inline_asm_elementwise(
                "MEMBAR.SC.GPU;", "=r", [], dtype=tl.int32, is_pure=True, pack=1
            )

            truth_hash = tl.load(
                OFFLOAD_CACHE_GPU_METADATA
                + idx_slots.to(tl.int64) * stride_offload_cache_gpu_metadata_token
                + 6 * stride_offload_cache_gpu_metadata_k,
                mask=mask_slot_cache_hit,
            ).to(tl.uint64)
            hash_mask = tl.full((1,), value=1, dtype=tl.uint64)
            hash_mask = (hash_mask << 32) - 1
            if OFFLOAD_CACHE_LOAD_VALUE:
                truth_hash = (truth_hash >> 32) & hash_mask
            else:
                truth_hash = truth_hash & hash_mask
            tl.debug_barrier()
            if UPDATE_CACHE:
                mask_slot_cache_hit = (
                    truth_hash == (keys_cached_hash & hash_mask)
                ) & mask_slot_cache_hit

            tl.static_assert(mask_slot_cache_hit.dtype == tl.int1)

            mask_keys_cache_miss = mask_keys & (~mask_slot_cache_hit)
            mask_slot_cache_hit = mask_keys & (~mask_keys_cache_miss)
            mask_keys = mask_keys_cache_miss

            tl.store(
                OFFLOAD_CACHE_GPU_METADATA
                + idx_slots.to(tl.int64) * stride_offload_cache_gpu_metadata_token
                + 4 * stride_offload_cache_gpu_metadata_k,
                value=0,
                mask=mask_slot_cache_hit,
            )
            # tl.store(
            #     CACHE_MISS_COUNTER +\
            #         idx_bsz.to(tl.int64) * stride_cache_miss_counter_bsz +\
            #         idx_kv_head * stride_cache_miss_counter_head_kv +\
            #         idx_page * stride_cache_miss_counter_tsrc,
            #     mask=mask_slot_cache_hit,
            #     value=0,
            # )

        keys = tl.load(
            K_CACHE
            + idx_page.to(tl.int64) * stride_k_cache_page
            + offset_page.to(tl.int64) * stride_k_cache_offset
            + idx_kv_head.to(tl.int64) * stride_k_cache_kv_head
            + idx_hid.to(tl.int64) * stride_k_cache_hid,
            mask=mask_keys & (idx_hid < HID_DIM),
            other=0.0,
        )
        if keys.dtype == tl.uint8:
            keys = keys.to(tl.float8e5, bitcast=True).to(tl.bfloat16)
        if keys.dtype == tl.float8e5:
            keys = keys.to(tl.bfloat16)

        if USING_OFFLOAD_CACHE:
            keys = tl.where(
                mask_keys,
                keys,
                keys_cached,
            )

            if CACHE_MISS_COUNTER is not None:
                if UPDATE_CACHE:
                    tl.debug_barrier()
                    tl.store(
                        CACHE_MISS_COUNTER
                        + idx_bsz.to(tl.int64) * stride_cache_miss_counter_bsz
                        + idx_kv_head * stride_cache_miss_counter_head_kv
                        + idx_page * stride_cache_miss_counter_tsrc,
                        mask=mask_keys_cache_miss,
                        value=1,
                    )
                    # mask_victim_slots = (cache_miss_counter != 1) & mask_keys_cache_miss
                    mask_victim_slots = mask_keys_cache_miss  # NOTE: init value if cache miss counter is ignored
                    # table is protected by cache miss counter
                    previous_table = tl.atomic_xchg(
                        OFFLOAD_CACHE_GPU_TABLE
                        + idx_page.to(tl.int64) * stride_offload_cache_gpu_table_token
                        + idx_kv_head * stride_offload_cache_gpu_table_head_kv
                        + 0 * strdie_offload_cache_gpu_table_k,
                        val=MAX_INT_ACQUIRED,
                        mask=mask_victim_slots,
                    )
                    mask_victim_slots = (
                        previous_table != MAX_INT_ACQUIRED
                    ) & mask_victim_slots
                    mask_table_acquired = mask_victim_slots

                    tl.debug_barrier()

                    idx_victim_slots = tl.zeros_like(idx_slots).to(tl.int64) + MAX_INT
                    max_not_accessed_time = tl.zeros_like(idx_slots).to(tl.int64) - 1
                    victim_slot_not_acquired = mask_victim_slots
                    seed = tl.atomic_add(
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA
                        + 1 * stride_offload_cache_gpu_global_metadata_k
                        + 0 * stride_offload_cache_gpu_global_metadata_pad,
                        val=1,
                    )
                    for i in range(10):
                        pid = (
                            tl.program_id(0).to(tl.int64)
                            * tl.num_programs(1)
                            * tl.num_programs(2)
                            + tl.program_id(1) * tl.num_programs(2)
                            + tl.program_id(2)
                        )
                        if original_mask_keys.shape[0] == 1:
                            idx_randint = tl.randint(
                                seed,
                                tl.arange(0, BLOCK_SIZE_K)[None, :] + i * BLOCK_SIZE_K,
                                10,
                            ).to(tl.int64) & ((1 << 30) - 1)
                        else:
                            idx_randint = tl.randint(
                                seed,
                                tl.arange(0, BLOCK_SIZE_K)[:, None] + i * BLOCK_SIZE_K,
                                10,
                            ).to(tl.int64) & ((1 << 30) - 1)
                        # if IS_BSA:
                        #     idx_victim_slots_try = idx_randint % (8192 * HEAD_KV)
                        # else:
                        #     idx_victim_slots_try = idx_randint % (8192 * HEAD_KV)
                        # idx_victim_slots_try = idx_victim_slots_try * 128 + tl.extra.cuda.smid()
                        idx_victim_slots_try = idx_randint
                        # idx_victim_slots_try = idx_randint * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        idx_victim_slots_try = idx_victim_slots_try % GPU_BANK_COUNT
                        # if IS_BSA:
                        #     idx_victim_slots_try = idx_victim_slots_try % (10000 * HEAD_KV)
                        # else:
                        #     idx_victim_slots_try = idx_victim_slots_try % (32000 * HEAD_KV)

                        acquired = victim_slot_not_acquired

                        not_accessed_time = tl.load(
                            OFFLOAD_CACHE_GPU_METADATA
                            + idx_victim_slots_try.to(tl.int64)
                            * stride_offload_cache_gpu_metadata_token
                            + 4 * stride_offload_cache_gpu_metadata_k,
                            mask=mask_victim_slots,
                        )
                        new_old_slot = (
                            not_accessed_time > max_not_accessed_time
                        ) & mask_victim_slots
                        # if already acquired, release it
                        tl.atomic_xchg(
                            OFFLOAD_CACHE_GPU_METADATA
                            + idx_victim_slots.to(tl.int64)
                            * stride_offload_cache_gpu_metadata_token
                            + 5 * stride_offload_cache_gpu_metadata_k,
                            val=0,
                            mask=new_old_slot & (~victim_slot_not_acquired),
                        )
                        tl.atomic_xchg(
                            OFFLOAD_CACHE_GPU_METADATA
                            + idx_victim_slots.to(tl.int64)
                            * stride_offload_cache_gpu_metadata_token
                            + 0 * stride_offload_cache_gpu_metadata_k,
                            val=MAX_INT,
                            mask=new_old_slot & (~victim_slot_not_acquired),
                        )
                        max_not_accessed_time = tl.maximum(
                            max_not_accessed_time, not_accessed_time
                        )
                        victim_slot_not_acquired = (
                            victim_slot_not_acquired | new_old_slot
                        )
                        acquired = victim_slot_not_acquired & (not_accessed_time > 0)

                        # check already written or not
                        previous_state = tl.atomic_xchg(
                            OFFLOAD_CACHE_GPU_METADATA
                            + idx_victim_slots_try.to(tl.int64)
                            * stride_offload_cache_gpu_metadata_token
                            + 5 * stride_offload_cache_gpu_metadata_k,
                            val=1,  # NOTE: this should be MAX_INT_1, but just for temporary.
                            mask=acquired,
                        )
                        acquired = (previous_state != 1) & acquired

                        # check acquired or not
                        previous_state = tl.atomic_xchg(
                            OFFLOAD_CACHE_GPU_METADATA
                            + idx_victim_slots_try.to(tl.int64)
                            * stride_offload_cache_gpu_metadata_token
                            + 0 * stride_offload_cache_gpu_metadata_k,
                            val=MAX_INT_ACQUIRED,  # NOTE: this should be MAX_INT_1, but just for temporary.
                            mask=acquired,
                        )
                        acquired = (previous_state != MAX_INT_ACQUIRED) & acquired

                        previously_acquired = (previous_state < MAX_INT) & acquired
                        previous_idx_page = previous_state // HEAD_KV
                        previous_idx_head_kv = previous_state % HEAD_KV
                        tl.atomic_xchg(
                            OFFLOAD_CACHE_GPU_TABLE
                            + previous_idx_page.to(tl.int64)
                            * stride_offload_cache_gpu_table_token
                            + previous_idx_head_kv
                            * stride_offload_cache_gpu_table_head_kv
                            + 0 * strdie_offload_cache_gpu_table_k,
                            val=MAX_INT,
                            mask=previously_acquired,
                        )

                        idx_victim_slots = tl.where(
                            acquired,
                            idx_victim_slots_try,
                            idx_victim_slots,
                        )

                        victim_slot_not_acquired = (
                            ~acquired
                        ) & victim_slot_not_acquired
                        tl.debug_barrier()
                    tl.debug_barrier()
                    # mask_victim_slots = mask_victim_slots & (idx_victim_slots < MAX_INT)
                    mask_victim_slots = (
                        mask_victim_slots
                        & (~victim_slot_not_acquired)
                        & mask_keys_cache_miss
                        & (idx_victim_slots != MAX_INT)
                        & original_mask_keys
                    )
                    # if not IS_BSA:
                    #     mask_victim_slots = mask_victim_slots & (~victim_slot_not_acquired) & (idx_victim_slots < (32000 * HEAD_KV)) & (idx_victim_slots > -1)
                    # else:
                    #     mask_victim_slots = mask_victim_slots & (~victim_slot_not_acquired) & (idx_victim_slots < (10000 * HEAD_KV)) & (idx_victim_slots > -1)
                    idx_victim_slots = idx_victim_slots * mask_victim_slots
                    tl.debug_barrier()

                    tl.store(
                        OFFLOAD_CACHE_GPU_BANK
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_bank_token
                        + idx_hid_cached * stride_offload_cache_gpu_bank_hid,
                        value=keys,
                        mask=mask_victim_slots,
                    )

                    # take token hash for debug
                    if mask_victim_slots.shape[0] == 1:
                        keys_hash = tl.sum(
                            keys.to(tl.uint16, bitcast=True), axis=0, keep_dims=True
                        ).to(tl.uint64)
                    elif mask_victim_slots.shape[1] == 1:
                        keys_hash = tl.sum(
                            keys.to(tl.uint16, bitcast=True), axis=1, keep_dims=True
                        ).to(tl.uint64)
                    else:
                        raise Exception()

                    if IS_BSA:
                        values = tl.load(
                            V_CACHE
                            + idx_page.to(tl.int64) * stride_v_cache_page
                            + offset_page.to(tl.int64) * stride_v_cache_offset
                            + idx_kv_head.to(tl.int64) * stride_v_cache_kv_head
                            + idx_hid.to(tl.int64) * stride_v_cache_hid,
                            mask=mask_victim_slots,
                            other=0.0,
                        )
                        if values.dtype == tl.uint8:
                            values = values.to(tl.float8e5, bitcast=True).to(
                                tl.bfloat16
                            )
                        if values.dtype == tl.float8e5:
                            values = values.to(tl.bfloat16)
                        tl.store(
                            OFFLOAD_CACHE_GPU_BANK
                            + idx_victim_slots.to(tl.int64)
                            * stride_offload_cache_gpu_bank_token  # idx_hid * stride_offload_cache_gpu_bank_hid,
                            + ((idx_hid_cached + BLOCK_HID) % (BLOCK_HID * 2))
                            * stride_offload_cache_gpu_bank_hid,
                            value=values,
                            mask=mask_victim_slots,
                        )

                        if mask_victim_slots.shape[0] == 1:
                            values_hash = tl.sum(
                                values.to(tl.uint16, bitcast=True),
                                axis=0,
                                keep_dims=True,
                            ).to(tl.uint64)
                        elif mask_victim_slots.shape[1] == 1:
                            values_hash = tl.sum(
                                values.to(tl.uint16, bitcast=True),
                                axis=1,
                                keep_dims=True,
                            ).to(tl.uint64)
                        else:
                            raise Exception()
                        if not OFFLOAD_CACHE_LOAD_VALUE:
                            keys_hash = keys_hash | (values_hash << 32)
                        else:
                            keys_hash = (keys_hash << 32) | values_hash

                    tl.store(
                        OFFLOAD_CACHE_GPU_METADATA
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_metadata_token
                        + 6 * stride_offload_cache_gpu_metadata_k,
                        value=keys_hash,
                        mask=mask_victim_slots,
                    )

                    uvm_token_gen = tl.load(
                        OFFLOAD_CACHE_UVM_METADATA
                        + idx_page.to(tl.int64)
                        * stride_offload_cache_uvm_metadata_token
                        + 0 * stride_offload_cache_uvm_metadata_k,
                        mask=mask_victim_slots,
                    )

                    tl.store(
                        OFFLOAD_CACHE_GPU_METADATA
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_metadata_token
                        + 1 * stride_offload_cache_gpu_metadata_k,
                        value=idx_page,
                        mask=mask_victim_slots,
                    )
                    tl.store(
                        OFFLOAD_CACHE_GPU_METADATA
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_metadata_token
                        + 2 * stride_offload_cache_gpu_metadata_k,
                        value=uvm_token_gen,
                        mask=mask_victim_slots,
                    )
                    tl.store(
                        OFFLOAD_CACHE_GPU_METADATA
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_metadata_token
                        + 4 * stride_offload_cache_gpu_metadata_k,
                        value=0,
                        mask=mask_victim_slots,
                    )

                    tl.debug_barrier()

                    # core.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=core.int32, is_pure=True, pack=1,
                    #                    _builder=_builder)
                    tl.inline_asm_elementwise(
                        "MEMBAR.SC.GPU;", "=r", [], dtype=tl.int32, is_pure=True, pack=1
                    )

                    # release slot
                    tl.atomic_xchg(
                        OFFLOAD_CACHE_GPU_METADATA
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_metadata_token
                        + 0 * stride_offload_cache_gpu_metadata_k,
                        val=idx_page * HEAD_KV + idx_kv_head,
                        mask=mask_victim_slots,
                    )
                    # release table
                    table_slots = tl.where(
                        mask_table_acquired & (~mask_victim_slots),
                        MAX_INT,
                        idx_victim_slots,
                    )
                    tl.atomic_xchg(
                        OFFLOAD_CACHE_GPU_TABLE
                        + idx_page.to(tl.int64) * stride_offload_cache_gpu_table_token
                        + idx_kv_head * stride_offload_cache_gpu_table_head_kv
                        + 0 * strdie_offload_cache_gpu_table_k,
                        val=table_slots,
                        mask=mask_table_acquired,
                    )
                else:
                    tl.store(
                        CACHE_MISS_COUNTER
                        + idx_bsz.to(tl.int64) * stride_cache_miss_counter_bsz
                        + idx_kv_head * stride_cache_miss_counter_head_kv
                        + idx_page * stride_cache_miss_counter_tsrc,
                        mask=mask_keys_cache_miss,
                        value=1,
                    )
    if keys.dtype == tl.uint8:
        keys = keys.to(tl.float8e5, bitcast=True).to(tl.float16)
    if keys.dtype == tl.float8e5:
        keys = keys.to(tl.float16)

    return keys


def update_recency_pytorch(
    accessed_ptr: Tensor,
    uvm_metadata: Tensor,
    global_metadata: Tensor,
    metadata: Tensor,
    table: Tensor,
    head_num: int,
    uvm_page_count: int,
):
    for idx_head_kv in range(head_num):
        for idx_token in tqdm.tqdm(
            range(uvm_page_count), dynamic_ncols=True, leave=False
        ):
            current_tick = global_metadata[0, 0]

            accessed = accessed_ptr[idx_head_kv, idx_token] > 0
            cache_hit = True & accessed
            if not cache_hit:
                continue

            idx_table = table[idx_head_kv, idx_token]
            cache_hit = (idx_table != MAX_INT) & cache_hit
            if not cache_hit:
                continue

            back_ref = metadata[idx_table, 0]
            cache_hit = (back_ref == idx_token) & cache_hit
            if not cache_hit:
                continue

            ref_to_uvm = metadata[idx_table, 1]
            cache_hit = (ref_to_uvm != MAX_INT) & cache_hit
            if not cache_hit:
                continue

            uvm_token_gen = uvm_metadata[ref_to_uvm, 0]
            cache_token_gen = metadata[idx_table, 2]
            cache_hit = (
                (uvm_token_gen != MAX_INT)
                & (uvm_token_gen == cache_token_gen)
                & cache_hit
            )
            if not cache_hit:
                continue

            metadata[idx_table, 3] = current_tick.to(metadata.dtype)


@triton.jit
def update_recency(
    ACCESSED,
    stride_accessed_head_kv,
    stride_accessed_token,
    UVM_METADATA,
    stride_uvm_metadata_token,
    stride_uvm_metadata_k,
    GLOBAL_METADTA,
    stride_global_metadata_k,
    stride_global_metadata_pad,
    METADATA,
    stride_metadata_slot,
    stride_metadata_k,
    TABLE,
    stride_table_head_kv,
    stride_table_token,
    stride_table_k,
    page_count: int,
    HEAD_KV: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    idx_block = pid % tl.cdiv(page_count, BLOCK_SIZE)
    idx_head_kv = pid // tl.cdiv(page_count, BLOCK_SIZE)

    idx_token = idx_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_token = idx_token < page_count

    current_tick = tl.load(
        GLOBAL_METADTA + 0 * stride_global_metadata_k + 0 * stride_global_metadata_pad
    )

    # TODO: merge with load tokens, verify cache
    accessed = (
        tl.load(
            ACCESSED
            + idx_head_kv.to(tl.int64) * stride_accessed_head_kv
            + idx_token * stride_accessed_token,
            mask=mask_token,
            other=0,
        )
        > 0
    )
    cache_hit = mask_token & accessed

    table = tl.load(
        TABLE
        + idx_head_kv.to(tl.int64) * stride_table_head_kv
        + idx_token * stride_table_token
        + 0 * stride_table_k,
        mask=cache_hit,
        other=MAX_INT,
    ).to(tl.int64)
    cache_hit = (table < MAX_INT) & cache_hit

    ALWAYS_VALIDATE_LINK: tl.constexpr = False  # True

    if ALWAYS_VALIDATE_LINK:
        validated_cache_hit = validate_bank_metadata_slots(
            UVM_METADATA,
            stride_uvm_metadata_token,
            stride_uvm_metadata_k,
            METADATA,
            stride_metadata_slot,
            stride_metadata_k,
            table,
            idx_token,
            cache_hit,
            HEAD_KV,
        )
        cache_hit = cache_hit & validated_cache_hit

    tl.store(
        METADATA + table.to(tl.int64) * stride_metadata_slot + 3 * stride_metadata_k,
        mask=cache_hit,
        value=current_tick,
    )


@triton.jit
def write_cache(
    PUT,
    stride_put_t,
    MASK,
    stride_mask_t,
    EVICT,
    stride_evict_t,
    BANK,
    stride_bank_t,
    stride_bank_hid,
    METADATA,
    stride_metadata_t,
    stride_metadata_k,
    TABLE,
    stride_table_head_kv,
    stride_table_t,
    stride_table_k,
    UVM_METADATA,
    stride_uvm_metadata_t,
    stride_uvm_metadata_k,
    UVM_K_BANK,
    stride_uvm_k_bank_t,
    stride_uvm_k_bank_head_kv,
    stride_uvm_k_bank_hid,
    UVM_V_BANK,
    stride_uvm_v_bank_t,
    stride_uvm_v_bank_head_kv,
    stride_uvm_v_bank_hid,
    GLOBAL_METADATA,
    stride_global_metadata_t,
    stride_global_metadata_k,
    qsize: int,
    page_count: int,
    HEAD_KV: int,
    KV_PACKED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_queue = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_queue = idx_queue < qsize

    put_list = tl.load(
        PUT + idx_queue.to(tl.int64) * stride_put_t,
        mask=mask_queue,
    )
    idx_page = put_list % page_count
    idx_head_kv = put_list // page_count

    mask_put = (
        tl.load(MASK + idx_queue.to(tl.int64) * stride_mask_t, mask=mask_queue, other=0)
        != 0
    )
    idx_evict = tl.load(EVICT + idx_queue.to(tl.int64) * stride_evict_t, mask=mask_put)

    # check still it is cache miss
    idx_slot = tl.load(
        TABLE
        + idx_head_kv.to(tl.int64) * stride_table_head_kv
        + idx_page * stride_table_t
        + 0 * stride_table_k,
        mask=mask_put,
        other=MAX_INT,
    )
    is_valid_slot = validate_bank_metadata_slots(
        UVM_METADATA,
        stride_uvm_metadata_k,
        stride_uvm_metadata_k,
        METADATA,
        stride_metadata_t,
        stride_metadata_k,
        idx_slot,
        idx_page,
        mask_put,
        HEAD_KV,
    )
    mask_put = mask_put & (~is_valid_slot)

    # unlink bank <-> table
    victim_table_entry = tl.load(
        METADATA + idx_evict.to(tl.int64) * stride_metadata_t + 0 * stride_metadata_k,
        mask=mask_put,
    )
    tl.store(
        TABLE
        + (victim_table_entry.to(tl.int64) % HEAD_KV) * stride_table_head_kv
        + (victim_table_entry // HEAD_KV) * stride_table_t
        + 0 * stride_table_k,
        value=MAX_INT,
        mask=mask_put & (victim_table_entry < MAX_INT),
    )

    # setup metadata
    tl.store(
        METADATA + idx_evict.to(tl.int64) * stride_metadata_t + 0 * stride_metadata_k,
        mask=mask_put,
        value=idx_page * HEAD_KV + idx_head_kv,
    )
    tl.store(
        METADATA + idx_evict.to(tl.int64) * stride_metadata_t + 1 * stride_metadata_k,
        mask=mask_put,
        value=idx_page,
    )
    token_gen = tl.load(
        UVM_METADATA
        + idx_page.to(tl.int64) * stride_uvm_metadata_t
        + 0 * stride_uvm_metadata_k,
        mask=mask_put,
    )
    tl.store(
        METADATA + idx_evict.to(tl.int64) * stride_metadata_t + 2 * stride_metadata_k,
        mask=mask_put,
        value=token_gen,
    )
    current_tick = tl.load(
        GLOBAL_METADATA + 0 * stride_global_metadata_t + 0 * stride_global_metadata_k,
    )
    tl.store(
        METADATA + idx_evict.to(tl.int64) * stride_metadata_t + 3 * stride_metadata_k,
        mask=mask_put,
        value=current_tick,
    )

    # setup table
    tl.store(
        TABLE
        + idx_page.to(tl.int64) * stride_table_t
        + idx_head_kv * stride_table_head_kv
        + 0 * stride_table_k,
        mask=mask_put,
        value=idx_evict,
    )

    # copy values
    idx_hid = tl.arange(0, BLOCK_HID)

    keys = tl.load(
        UVM_K_BANK
        + idx_page[:, None].to(tl.int64) * stride_uvm_k_bank_t
        + idx_head_kv[:, None] * stride_uvm_k_bank_head_kv
        + idx_hid[None, :] * stride_uvm_k_bank_hid,
        mask=mask_put[:, None],
    )
    tl.store(
        BANK
        + idx_evict[:, None].to(tl.int64) * stride_bank_t
        + idx_hid[None, :] * stride_bank_hid,
        mask=mask_put[:, None],
        value=keys,
    )

    if KV_PACKED:
        values = tl.load(
            UVM_V_BANK
            + idx_page[:, None].to(tl.int64) * stride_uvm_v_bank_t
            + idx_head_kv[:, None] * stride_uvm_v_bank_head_kv
            + idx_hid[None, :] * stride_uvm_v_bank_hid,
            mask=mask_put[:, None],
        )
        tl.store(
            BANK
            + idx_evict[:, None].to(tl.int64) * stride_bank_t
            + (idx_hid + BLOCK_HID)[None, :] * stride_bank_hid,
            mask=mask_put[:, None],
            value=values,
        )
