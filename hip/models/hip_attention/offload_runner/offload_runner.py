import argparse
import gc
import random
import cuda.cudart
import torch.distributed
import tqdm
from transformers import AutoTokenizer
from hip.models.hip_attention.offload_runner.llama_model import LlamaForCausalLM, LlamaDecoderLayer, LlamaAttention
import torch, time, os
from typing import List, Literal, Optional, Dict, Union, Any, Tuple
from transformers.cache_utils import Cache, PretrainedConfig, is_torchdynamo_compiling
from transformers import BitsAndBytesConfig
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
import cupy
import ctypes
import cuda
from hip.models.hip_attention.offload_runner.tensor_from_pointer import tensor_from_pointer
from math import prod
import pynvml
import threading
from hip import HiPAttentionArgs, HiPAttentionOutputMetadata
import triton
import triton.language as tl

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_memory_free_MiB(torch_gpu_index):
    pynvml.nvmlInit()
    gpu_index = int(torch_gpu_index)
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_index = int(os.environ['CUDA_VISIBLE_DEVICES'].split(',')[gpu_index])
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free = max(mem_info.free, mem_info.total - torch.cuda.memory_reserved(torch_gpu_index))
    return free // 1024 ** 2

@triton.jit
def linear_scan_and_calc_frequency_cuda(
    LOG,
    stride_log_n, stride_log_k,
    FREQ,
    stride_freq_n, stride_freq_k,
    
    LOG_K, MAX_PTR,
):
    idx_n = tl.program_id(0)
    
    counter = 0
    for idx_k in range(LOG_K):
        current_ptr = tl.load(
            LOG +\
                idx_n * stride_log_n +\
                idx_k * stride_log_k,
        )
        next_ptr = tl.load(
            LOG +\
                idx_n * stride_log_n +\
                (idx_k + 1) * stride_log_k,
            mask=(idx_k + 1) < LOG_K,
            other=MAX_PTR,
        )
        counter += 1
        if current_ptr != next_ptr:
            tl.store(
                FREQ +\
                    idx_n * stride_freq_n +\
                    idx_k * stride_freq_k,
                value=counter
            )
            counter = 0

def linear_scan_and_calc_frequency(
    access_log: torch.Tensor,
    MAX_PTR,
):
    access_freq = torch.zeros_like(access_log)
    pre_device = torch.get_default_device()
    
    grid = (access_freq.shape[0], )
    
    torch.set_default_device(access_log.device)
    linear_scan_and_calc_frequency_cuda[grid](
        access_log, *access_log.stride(),
        access_freq, *access_freq.stride(),
        access_log.shape[-1], MAX_PTR
    )
    torch.set_default_device(pre_device)
    
    return access_freq

@triton.jit
def copy_to_banks_from_state_cuda(
    STATE,
    stride_state_n, stride_state_t, stride_state_head_kv, stride_state_hid,
    ORIGINAL_TABLE,
    stride_original_table_cache, stride_original_table_t,
    ORIGINAL_BANK,
    stride_original_bank_cache, stride_original_bank_page, stride_original_bank_offset, stride_original_bank_hid,
    
    TABLE,
    stride_table_cache, stride_table_t,
    BANK,
    stride_bank_cache, stride_bank_page, stride_bank_offset, stride_bank_hid,
    BANK_STATS,
    stride_bank_stats_cache, stride_bank_stats_page, stride_bank_stats_k,
    TABLE_DELTA,
    stride_table_delta_cache, stride_table_delta_update,
    BANK_LOC,
    stride_bank_loc_cache, stride_bank_loc_update,
    
    offset,
    TSRC,
    HEAD_KV,
    BLOCK_SIZE_K,
    BUDGET,
    TABLE_LEN,
    
    HID: tl.constexpr,
):
    # tsrc_log: int[NCACHE, UPDATE]
    # tsrc_log_loc: int[NCACHE, UPDATE]
    # tsrc_log -> tsrc_log_loc
    
    idx_cache = tl.program_id(1).to(tl.int64)
    idx_head_kv = (idx_cache % HEAD_KV).to(tl.int64)
    idx_bsz = (idx_cache // HEAD_KV).to(tl.int64)
    
    idx_update = tl.program_id(0).to(tl.int64)
    
    table_delta = tl.load(
        TABLE_DELTA +\
            idx_cache * stride_table_delta_cache +\
            idx_update * stride_table_delta_update,
    ).to(tl.int64)
    table_delta = tl.minimum(table_delta + offset, TSRC - 1)
    
    if (table_delta // BLOCK_SIZE_K) >= (TABLE_LEN - 1):
        return
    
    if ORIGINAL_TABLE is not None:
        orignal_bank_loc = tl.load(
            ORIGINAL_TABLE +\
                idx_cache * stride_original_table_cache +\
                (table_delta // BLOCK_SIZE_K) * stride_original_table_t
        )
    else:
        orignal_bank_loc = 65535
    idx_hid = tl.arange(0, HID)
    if ((orignal_bank_loc >= 0) & (orignal_bank_loc < BUDGET) & (orignal_bank_loc != 65535)):
        state = tl.load(
            ORIGINAL_BANK +\
                idx_cache * stride_original_bank_cache +\
                orignal_bank_loc * stride_original_bank_page +\
                offset * stride_original_bank_offset +\
                idx_hid * stride_original_bank_hid,
        ).to(BANK.dtype.element_ty)
    else:
        state = tl.load(
            STATE +\
                idx_bsz * stride_state_n +\
                table_delta * stride_state_t +\
                idx_head_kv * stride_state_head_kv +\
                idx_hid * stride_state_hid,
        ).to(BANK.dtype.element_ty)
    
    bank_loc = tl.load(
        BANK_LOC+\
            idx_cache * stride_bank_loc_cache +\
            idx_update * stride_bank_loc_update
    ).to(tl.int64)
    
    if bank_loc >= BUDGET:
        return
    
    new_bank_backref = (table_delta // BLOCK_SIZE_K).to(tl.int32)
    
    if offset == 0:
        current_bank_backref = tl.load(
            BANK_STATS +\
                idx_cache * stride_bank_stats_cache +\
                bank_loc * stride_bank_stats_page +\
                1 * stride_bank_stats_k,
        )
        
        if  (current_bank_backref != new_bank_backref) &\
            (current_bank_backref < TABLE_LEN) &\
            (current_bank_backref >= 0):
            tl.atomic_cas(
                TABLE +\
                    idx_cache * stride_table_cache +\
                    current_bank_backref * stride_table_t,
                cmp=bank_loc.to(tl.uint16),
                val=tl.full([], value=65535, dtype=tl.uint16)
            )
            # update backref
        
        tl.store(
            BANK_STATS +\
                idx_cache * stride_bank_stats_cache +\
                bank_loc * stride_bank_stats_page +\
                1 * stride_bank_stats_k,
            value=new_bank_backref,
        )
    
    tl.store(
        BANK +\
            idx_cache * stride_bank_cache +\
            bank_loc * stride_bank_page +\
            offset * stride_bank_offset +\
            tl.arange(0, HID) * stride_bank_hid,
        value=state
    )

def copy_to_banks_from_state(
    state: torch.Tensor,
    original_table: torch.Tensor,
    original_bank: torch.Tensor,
    table: torch.Tensor,
    bank: torch.Tensor,
    bank_stats: torch.Tensor,
    bank_loc: torch.Tensor,
    table_delta: torch.Tensor,
    offset: int,
):
    N_CACHE, BUDGET, BLOCK_SIZE_K, HID = bank.shape
    BSZ, TSRC, HEAD_KV, __ = state.shape
    assert N_CACHE == (BSZ * HEAD_KV), f'{N_CACHE} == ({BSZ} * {HEAD_KV})'
    if original_bank is not None:
        assert original_bank.shape == bank.shape
        assert original_table.shape[0] == N_CACHE
    N_CACHE, TABLE_LEN = table.shape
    
    N_CACHE, N_UPDATE = table_delta.shape
    assert table_delta.shape == bank_loc.shape, f'{table_delta.shape} == {bank_loc.shape}'
    
    assert bank_stats.ndim == 3, bank_stats.shape
    assert table.ndim == 2
    
    grid = (N_UPDATE, N_CACHE)
    pre_device = torch.get_default_device()
    torch.set_default_device(table_delta.device)
    copy_to_banks_from_state_cuda[grid](
        state, *state.stride(),
        original_table, *(original_table.stride() if original_bank is not None else (0, 0,)),
        original_bank if original_bank is not None else state, *(original_bank.stride() if original_bank is not None else (0, 0, 0, 0)),
        
        table, *table.stride(),
        bank, *bank.stride(),
        bank_stats, *bank_stats.stride(),
        table_delta, *table_delta.stride(),
        bank_loc, *bank_loc.stride(),
        
        offset,
        TSRC,
        HEAD_KV,
        BLOCK_SIZE_K,
        BUDGET,
        TABLE_LEN,
        
        HID,
        
        num_warps=1,
    )
    torch.set_default_device(pre_device)
    
    # bank.view(N_CACHE, BUDGET * BLOCK_SIZE_K, HID).scatter_(
    #     dim=1,
    #     index=(bank_loc * BLOCK_SIZE_K + offset)[:, :, None]\
    #         .expand(N_CACHE, -1, HID),
    #     src=state.permute(0, 2, 1, 3).reshape(N_CACHE, TSRC, HID)\
    #         .gather(
    #             dim=1,
    #             index=torch.clamp_max(table_delta.long() + offset, TSRC - 1)[:, :, None]\
    #                 .expand(-1, -1, HID)
    #         )
    # )

from dataclasses import dataclass

@dataclass
class DecodeStats:
    num_accessed: int
    num_cache_hit: int
    cache_hit_ratio: float
    cache_access_utilization: float

class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> past_kv_length = outputs.past_key_values # access cache filled with key/values from generation
        ```
    """

    def __init__(
        self, 
        config: PretrainedConfig, 
        max_batch_size: int, 
        max_cache_len: int, 
        device: torch.device,
        dtype: torch.dtype, 
        cache_backend: Literal['cuda', 'uvm'] = 'cuda',
        share = 1,
        use_offload_cache: bool = False,
        uvm_offload_key = True,
        uvm_offload_value = True,
        cache_size = 8192,
        kv_group_size = 4,
        mask_k = 512,
        block_size_k = 2,
        sliding_window_size = 256,
    ) -> None:
        super().__init__()
        
        if isinstance(device, (int, str)):
            device = torch.device(device)
        self.device = device
        
        torch.cuda.synchronize()
        free_memory_mb = get_memory_free_MiB(device.index)
        
        print(f'allocatable {free_memory_mb:,} MB')
        
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )
        self.cache_backend = cache_backend

        sliding_window_size += 64 # BUG: WTF?
        self.sliding_window_size = sliding_window_size
        self.share = share
        self.uvm_offload_key = uvm_offload_key
        self.uvm_offload_value = uvm_offload_value
        self.cache_budget = cache_size // block_size_k
        self.sparse_attention_budget = (mask_k * kv_group_size + sliding_window_size) // block_size_k
        self.max_seq_len = max_cache_len
        self.block_size_k = block_size_k
        self.offload_cache_dtype = torch.float8_e5m2
        
        self.using_offload_cache = use_offload_cache
        if self.using_offload_cache:
            self.num_layers = config.num_hidden_layers
            self.num_caches = self.num_layers * self.max_batch_size * self.num_key_value_heads
            self.num_caches_per_layer = self.max_batch_size * self.num_key_value_heads
            self.offload_cache_null_pointer = 65535
            assert self.cache_budget < self.offload_cache_null_pointer # smaller than uint16
            
            # this cache is updateable
            # in table, last slot is trash
            def new_tables(len):
                return torch.full(
                    (self.num_caches, len), 
                    dtype=torch.uint16, 
                    device=device, 
                    fill_value=self.offload_cache_null_pointer
                )
            def new_banks(len):
                return torch.zeros(
                    (self.num_caches, len, self.block_size_k, self.head_dim), 
                    dtype=self.offload_cache_dtype, 
                    device=device
                ).fill_(42)
            def new_bank_stats(len):
                stats = torch.zeros(
                    (self.num_caches, len, 2),
                    dtype=torch.int32,
                    device=device
                ) # Tuple[num_accessed, backref_table_index]
                stats[:, :, 1].fill_(torch.iinfo(stats.dtype).max)
                return stats
            
            self.masking_key_tables = new_tables(self.max_seq_len // self.block_size_k + 1)
            self.masking_key_banks = new_banks(self.cache_budget)
            self.masking_key_bank_stats = new_bank_stats(self.cache_budget)
            
            # this caches are protected
            self.sa_key_tables = new_tables(self.max_seq_len // self.block_size_k + 1)
            self.sa_key_banks = new_banks(self.sparse_attention_budget)
            self.sa_key_bank_stats = new_bank_stats(self.sparse_attention_budget)
            
            self.sa_value_tables = new_tables(self.max_seq_len // self.block_size_k + 1)
            self.sa_value_banks = new_banks(self.sparse_attention_budget)
            self.sa_value_bank_stats = new_bank_stats(self.sparse_attention_budget)
            
            self.counters = torch.zeros(
                (self.num_caches, 2), 
                dtype=torch.int64, 
                device=device
            ) # [accessed, hit]
            print(
                f'allocated for cache | '
                f'masking keys: (bank = {self.masking_key_banks.numel() * self.masking_key_banks.element_size()/1024/1024} MB, table = {self.masking_key_tables.numel() * self.masking_key_tables.element_size() / 1024 / 1024} MB), '
                f'SA keys: (bank = {self.sa_key_banks.numel() * self.sa_key_banks.element_size()/1024/1024} MB, table = {self.sa_key_tables.numel() * self.sa_key_tables.element_size() / 1024 / 1024} MB), '
                f'SA values: (bank = {self.sa_value_banks.numel() * self.sa_value_banks.element_size()/1024/1024} MB, table = {self.sa_value_tables.numel() * self.sa_value_tables.element_size() / 1024 / 1024} MB)'
            )
            
            # dummy initialize
            dummy_init_offload_cache = False
            if dummy_init_offload_cache:
                def dummy_init(tables, banks, hit_ratio):
                    N, NBANK, PAGE_SIZE, _ = banks.shape
                    t_tables = (torch.rand_like(tables, dtype=torch.float32) * (NBANK - 1)).to(tables.dtype)
                    mask = torch.rand_like(t_tables, dtype=torch.float32) <= hit_ratio
                    t_tables = torch.where(mask, t_tables.int(), 65535).to(torch.uint16)
                    tables.copy_(t_tables)
                dummy_init(self.masking_key_tables, self.masking_key_banks, 0.8)
                dummy_init(self.sa_key_tables, self.sa_key_banks, 0.98)
                dummy_init(self.sa_value_tables, self.sa_value_banks, 0.98)
        
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (max_batch_size, self.max_cache_len, self.num_key_value_heads, self.head_dim)
        total_bytes = 0
        for idx_group in range(config.num_hidden_layers // share):
            new_layer_key_cache = self.allocate_tensor(
                cache_shape, 
                self.dtype, 
                device, 
                self.cache_backend if (idx_group > -10) and self.uvm_offload_key else 'cuda'
            )
            new_layer_value_cache = self.allocate_tensor(
                cache_shape, 
                self.dtype, 
                device, 
                self.cache_backend if (idx_group > -10) and self.uvm_offload_value else 'cuda'
            )
            if cache_backend == 'cuda':
                byte_size = 2 * new_layer_key_cache.numel() * new_layer_key_cache.element_size()
            elif cache_backend == 'uvm':
                byte_size = 2 * new_layer_key_cache[0].numel() * new_layer_key_cache[0].element_size()
            else:
                raise Exception()
            total_bytes += byte_size
            for _ in range(share):
                self.key_cache.append(new_layer_key_cache)
                self.value_cache.append(new_layer_value_cache)
        total_mb = total_bytes / 1024 / 1024
        print(f'allocated {total_mb:,} MB {cache_shape}. compression ratio: {total_mb / free_memory_mb:.3f}')
        
        self.prompt_copy_stream = torch.cuda.Stream(self.device)
        self.prompt_copy_threads: List[threading.Thread] = []
        
        # torch.cuda.synchronize()
        # gc.collect()
        # torch.cuda.empty_cache()
    
    def has_offload_cache(self, layer_idx: int):
        return self.using_offload_cache
    
    def get_offload_cache(self, layer_idx: int):
        def get_layer(t: torch.Tensor):
            return t[self.num_caches_per_layer * layer_idx: self.num_caches_per_layer * (layer_idx + 1)]
        
        masking_key_tables = get_layer(self.masking_key_tables)
        masking_key_banks = get_layer(self.masking_key_banks)
        masking_key_bank_stats = get_layer(self.masking_key_bank_stats)
        
        sa_key_tables = get_layer(self.sa_key_tables)
        sa_key_banks = get_layer(self.sa_key_banks)
        sa_key_bank_stats = get_layer(self.sa_key_bank_stats)
        
        sa_value_tables = get_layer(self.sa_value_tables)
        sa_value_banks = get_layer(self.sa_value_banks)
        sa_value_bank_stats = get_layer(self.sa_value_bank_stats)
        
        counters = get_layer(self.counters)
        
        return (
            masking_key_tables, masking_key_banks, masking_key_bank_stats,
            sa_key_tables, sa_key_banks, sa_key_bank_stats,
            sa_value_tables, sa_value_banks, sa_value_bank_stats,
            counters
        )
    
    def allocate_tensor(
        self, 
        shape: Tuple[int], 
        dtype: torch.dtype, 
        device: torch.device,
        cache_backend: str,
    ):
        if cache_backend == 'cuda':
            return torch.zeros(shape, dtype=dtype, device=device)
        elif cache_backend == 'uvm':
            elem_size = torch.tensor([], dtype=dtype).element_size()
            numel = prod(shape)
            align = 2048
            byte_size = elem_size * numel
            byte_size = byte_size + byte_size % align
            r, pointer = cuda.cudart.cudaMallocManaged(byte_size, cuda.cudart.cudaMemAttachGlobal)
            if isinstance(device, str): device = torch.device(device)
            t_gpu = tensor_from_pointer(pointer, shape, dtype, device.index)
            t_cpu = tensor_from_pointer(pointer, shape, dtype, -1)
            # print(f'managed alloc result={r}, ptr=0x{pointer:02X}, bytes={byte_size:3,}, {t_gpu.device} {t_cpu.device}')
            self.note_cpu(t_gpu, prefetch=True)
            return t_gpu, t_cpu
        raise NotImplementedError()

    def note_device(self, tensor: Tuple[torch.Tensor, torch.Tensor], advise = True):
        if advise:
            cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation, tensor.device.index)
            cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy, tensor.device.index)
        cuda.cudart.cudaMemPrefetchAsync(tensor.data_ptr(), tensor.numel() * tensor.element_size(), tensor.device.index, 0)
    
    def note_cpu(self, tensors: Tuple[torch.Tensor, torch.Tensor], prefetch = True):
        if isinstance(tensors, tuple):
            tensor, _ = tensors
        else:
            tensor = tensors
        cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation, -1)
        cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy, tensor.device.index)
        if prefetch:
            cuda.cudart.cudaMemPrefetchAsync(tensor.data_ptr(), tensor.numel() * tensor.element_size(), -1, 0)
        
    def note_decode(self, tensors: Tuple[torch.Tensor, torch.Tensor]):
        if isinstance(tensors, tuple):
            tensor, _ = tensors
        else:
            tensor = tensors
        cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviceSetReadMostly, tensor.device.index)
    
    def unnote_decode(self, tensors: Tuple[torch.Tensor, torch.Tensor]):
        if isinstance(tensors, tuple):
            tensor, _ = tensors
        else:
            tensor = tensors
        cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviceUnsetReadMostly, tensor.device.index)
    
    def note_prompt(self, tensors: Tuple[torch.Tensor, torch.Tensor]):
        if isinstance(tensors, tuple):
            tensor, _ = tensors
        else:
            tensor = tensors
        cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy, -1)
    
    def unnote_prompt(self, tensors: Tuple[torch.Tensor, torch.Tensor]):
        if isinstance(tensors, tuple):
            tensor, _ = tensors
        else:
            tensor = tensors
        cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy, tensor.device.index)
    
    def decode_start(self):
        torch.cuda.synchronize()
        if self.uvm_offload_key: map(self.note_cpu, self.key_cache)
        if self.uvm_offload_value: map(self.note_cpu, self.value_cache)
        if self.uvm_offload_key: map(self.note_decode, self.key_cache)
        if self.uvm_offload_value: map(self.note_decode, self.value_cache)
    
    def decode_end(self):
        if self.uvm_offload_key: map(self.note_cpu, self.key_cache)
        if self.uvm_offload_value: map(self.note_cpu, self.value_cache)
        if self.uvm_offload_key: map(self.unnote_decode, self.key_cache)
        if self.uvm_offload_value: map(self.unnote_decode, self.value_cache)
    
    def prompt_start(self):
        if self.uvm_offload_key: map(self.note_cpu, self.key_cache)
        if self.uvm_offload_value: map(self.note_cpu, self.value_cache)
        if self.uvm_offload_key: map(self.note_prompt, self.key_cache)
        if self.uvm_offload_value: map(self.note_prompt, self.value_cache)
        
        assert len(self.prompt_copy_threads) == 0, "every copy threads should be reaped"
    
    def prompt_end(self):
        self.prompt_copy_stream.synchronize()
        for thread in self.prompt_copy_threads:
            thread.join()
        self.prompt_copy_threads.clear()
        
        if self.uvm_offload_key: map(self.note_cpu, self.key_cache)
        if self.uvm_offload_value: map(self.note_cpu, self.value_cache)
        if self.uvm_offload_key: map(self.unnote_prompt, self.key_cache)
        if self.uvm_offload_value: map(self.unnote_prompt, self.value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            raise Exception()

        # evict previous, prefetch next
        is_prompt = cache_kwargs.get('is_prompt', False)
        
        # prompt = CPU, decode = UVM
        if self.cache_backend == 'uvm':
            if self.uvm_offload_key: 
                k_out = self.key_cache[layer_idx][1 if is_prompt else 0]
            else:
                k_out = self.key_cache[layer_idx]
            if self.uvm_offload_value:
                v_out = self.value_cache[layer_idx][1 if is_prompt else 0]
            else:
                v_out = self.value_cache[layer_idx]
        elif self.cache_backend == 'cuda':
            k_out = self.key_cache[layer_idx]
            v_out = self.value_cache[layer_idx]
        else:
            raise Exception()
        
        if ('batch_index' in cache_kwargs):
            ibatch = cache_kwargs['batch_index']
            k_out = k_out[ibatch:ibatch+1]
            v_out = v_out[ibatch:ibatch+1]
        
        if (layer_idx % self.share) == 0:
            if is_prompt:
                if self.cache_backend == 'uvm' and self.uvm_offload_key:
                    assert k_out.device == torch.device('cpu')
                self.prompt_copy_stream.wait_stream(torch.cuda.default_stream(cache_position.device))
                with torch.cuda.stream(self.prompt_copy_stream):
                    cache_position_key = cache_position.to(k_out.device, non_blocking=True)
                    cache_position_value = cache_position.to(v_out.device, non_blocking=True)
                    key_states_cpu = key_states.to(k_out.dtype).to(k_out.device, non_blocking=True)
                    value_states_cpu = value_states.to(k_out.dtype).to(v_out.device, non_blocking=True)
                @torch.inference_mode(True)
                def job():
                    self.prompt_copy_stream.synchronize()
                    k_out.index_copy_(1, cache_position_key, key_states_cpu)
                    v_out.index_copy_(1, cache_position_value, value_states_cpu)
                t = threading.Thread(target=job, daemon=True)
                self.prompt_copy_threads.append(t)
                t.start()
            else:
                k_out.index_copy_(1, cache_position, key_states.to(k_out.dtype))
                v_out.index_copy_(1, cache_position, value_states.to(k_out.dtype))
        
        if is_prompt:
            return key_states, value_states

        return k_out, v_out

    def copy_to_banks(
        self,
        tables,
        banks,
        bank_stats,
        states,
        table_delta, 
        bank_loc = None,
        check_masking_bank_validity = False,
        copy_from_original = False,
    ):
        NCACHE, BUDGET, BLOCK_SIZE_K, _ = banks.shape
        NCACHE, TABLE_SIZE = tables.shape
        _, N_UPDATE = table_delta.shape
        
        if bank_loc is None:
            bank_loc = torch\
                .arange(0, N_UPDATE, device=table_delta.device)[None, :]\
                .expand(NCACHE, -1)
        elif bank_loc.ndim == 1:
            bank_loc = bank_loc[None, :]\
                .expand(NCACHE, -1)
        else:
            assert isinstance(bank_loc, torch.Tensor)
        
        original_table = tables.clone() if copy_from_original else None
        
        # if not copy_from_original:
        #     bank_stats[:, :, 1].fill_(torch.iinfo(bank_stats.dtype).max)
        #     tables.fill_(65535)
        
        tables.copy_(
            tables.long().scatter(
                dim=-1, 
                index=table_delta.long() // BLOCK_SIZE_K, 
                src=bank_loc,
            ).to(tables.dtype)
        )
        
        for ioffset in range(BLOCK_SIZE_K):
            copy_to_banks_from_state(
                states,
                original_table,
                banks.clone() if copy_from_original else None,
                tables,
                banks,
                bank_stats,
                bank_loc,
                table_delta,
                ioffset,
            )
        
        if check_masking_bank_validity:
            for idx in tqdm.tqdm(range(tables.shape[-1] - 1)):
                idx_bank = tables[0, idx].item()
                if idx_bank < 0 or idx_bank >= BUDGET:
                    continue
                for idx_k in range(BLOCK_SIZE_K):
                    token = banks[0, idx_bank, idx_k].float()
                    truth = states[0, idx * BLOCK_SIZE_K + idx_k, 0].float()
                    if (token - truth).abs().mean().item() >= 0.15:
                        print(
                            f'{(token - truth).abs().mean().item()} {token} {truth}\n'
                            f'{idx_bank} = tables[0, {idx}]\n'
                            f'banks[0, {idx_bank}, {idx_k}] != states[0, {idx} * {BLOCK_SIZE_K} + {idx_k}, 0]\n'
                            f'(table_delta[0] == idx).int().sum() => {((table_delta[0] // BLOCK_SIZE_K) == idx).int().sum()} (is updated now?)\n'
                        )
                        
                        raise Exception()

    def decode_reset_stats(self):
        # print(self.masking_key_tables[0].cpu().tolist())
        if self.using_offload_cache:
            num_accessed, num_hit = self.counters.sum(0).cpu().tolist()
            cache_access_utilization = (self.masking_key_bank_stats[:, :, 0] > 0)\
                .float().mean().cpu()
            
            self.counters.fill_(0)
            self.masking_key_bank_stats[:, :, 0].fill_(0)
            
            return DecodeStats(
                num_accessed=num_accessed,
                num_cache_hit=num_hit,
                cache_hit_ratio=num_hit / (num_accessed + 1e-20),
                cache_access_utilization=cache_access_utilization,
            )
        else:
            return DecodeStats(
                num_accessed=0,
                num_cache_hit=0,
                cache_hit_ratio=0,
                cache_access_utilization=0,
            )
    
    def copy_state_to_bank_using_mask(
        self, 
        tables: torch.Tensor, 
        banks: torch.Tensor, 
        bank_stats: torch.Tensor,
        states: torch.Tensor,
        metadata: HiPAttentionOutputMetadata,
        position_ids: torch.Tensor,
        REPEAT_BATCH: int = 1,
    ):
        NCACHE, TABLE_LEN = tables.shape
        NCACHE, BUDGET, BLOCK_SIZE_K, HID = banks.shape
        _, MAX_TSRC, HEAD_KV, _ = states.shape
        
        BSZ = NCACHE // HEAD_KV
        if position_ids.shape[0] == 1:
            position_ids = position_ids.repeat(BSZ, 1)
        if position_ids.shape[1] > 1:
            position_ids = position_ids.amax(dim=-1, keepdim=True)
        
        LARGE_INT = (TABLE_LEN - 1) * BLOCK_SIZE_K
        
        # print(position_ids.shape) # torch.Size([1, 1])
        sw_mask = (((position_ids - self.sliding_window_size) // BLOCK_SIZE_K) * BLOCK_SIZE_K) +\
            torch.arange(
                0, 
                self.sliding_window_size, 
                BLOCK_SIZE_K, 
                device=tables.device
            )[None, :]
        sw_mask = torch.where(sw_mask >= 0, sw_mask, LARGE_INT)
        sw_mask = sw_mask.repeat_interleave(HEAD_KV, 0)
        # self.copy_to_banks(
        #     tables=tables,
        #     banks=banks,
        #     bank_stats=bank_stats,
        #     table_delta=sw_mask,
        #     bank_loc=torch\
        #         .arange(BUDGET - self.sliding_window_size // BLOCK_SIZE_K, BUDGET, device=tables.device),
        #     states=states,
        #     copy_from_original=True,
        # )
        
        mask = metadata.indices[:, -1, :].repeat(REPEAT_BATCH, 1).view(NCACHE, -1).clone()
        mask = torch.where(mask >= 0, torch.clamp_max(mask, LARGE_INT), LARGE_INT).sort(dim=-1).values
        mask = torch.where(mask != torch.roll(mask, dims=-1, shifts=1), mask, LARGE_INT)\
            .sort(dim=-1).values[:, :BUDGET - self.sliding_window_size // BLOCK_SIZE_K]
        # print(mask.shape, sw_mask.shape, banks.shape, self.sliding_window_size, banks.shape, HEAD_KV)
        mask = torch.cat([mask, sw_mask], dim=1)
        
        self.copy_to_banks(
            tables=tables,
            banks=banks,
            bank_stats=bank_stats,
            table_delta=mask,
            states=states,
            copy_from_original=True,
        )

    def copy_stats_to_bank_using_access_stats(
        self,
        tables: torch.Tensor,
        banks: torch.Tensor,
        bank_stats: torch.Tensor,
        block_access_log: torch.Tensor,
        key_cache: torch.Tensor,
        method = 'lru_approx',
    ):
        N_CACHE, BUDGET, BLOCK_SIZE_K, HID = banks.shape
        N_CACHE, TABLE_LEN = tables.shape
        N_CACHE, BUDGET, STAT_K = bank_stats.shape
        N, BDST, LOG_K = block_access_log.shape
        MAX_BSZ, MAX_TSRC, HEAD_KV, HID = key_cache.shape
        BSZ = N_CACHE // HEAD_KV
        
        MAX_PTR = (TABLE_LEN - 1) * BLOCK_SIZE_K
        
        assert N_CACHE == (BSZ * HEAD_KV)
        
        block_access_log = block_access_log[:, -1, :].reshape(N_CACHE, -1)[:, :, None]# + torch.arange(0, 2, device=block_access_log.device)
        block_access_log =  block_access_log.reshape(N_CACHE, -1) * BLOCK_SIZE_K
        
        if method == 'overwrite':
            # print('asfd', block_access_log[0, :20], BUDGET, block_access_log.shape, LOG_K, block_access_log[0, :].unique().shape)
            # print('asdfadsas', (block_access_log[0, :] == 2).any())
            
            block_access_log = torch.where(
                torch.logical_and(block_access_log >= 0, block_access_log < MAX_TSRC), 
                block_access_log, 
                MAX_PTR
            )
            # print(block_access_log[0].sort().values.tolist())
            
            block_access_log = block_access_log.sort(dim=-1, stable=False).values
            block_access_log_frequency = linear_scan_and_calc_frequency(block_access_log, MAX_PTR)
            block_access_log_frequency = torch.where(block_access_log < MAX_PTR, block_access_log_frequency, 0)
            sort_indices = torch.argsort(block_access_log_frequency, dim=-1, descending=True, stable=False)
            block_access_log = block_access_log.gather(dim=-1, index=sort_indices)[:, :BUDGET]
            
            # block_access_log = (torch.arange(0, BUDGET, device=block_access_log.device, dtype=block_access_log.dtype) * BLOCK_SIZE_K)[None, :].expand(block_access_log.shape)
            # block_access_log = torch.clamp(block_access_log, 0, MAX_PTR)
            
            self.copy_to_banks(
                tables=tables,
                banks=banks,
                bank_stats=bank_stats,
                table_delta=block_access_log,
                states=key_cache,
                check_masking_bank_validity=False,
                copy_from_original=True,
            )
            
            # print('after', tables[0, :20], tables.shape)
        elif method == 'lru_approx':
            # block_access_log = block_access_log[:, -1, :].reshape(N_CACHE, -1) * BLOCK_SIZE_K
            block_access_log = torch.where(
                torch.logical_and(block_access_log >= 0, block_access_log < MAX_TSRC), 
                block_access_log, 
                MAX_PTR
            )
            
            # print('a')
            
            # calc cache miss mask
            table_lookup = tables.long().gather(dim=1, index=(block_access_log // BLOCK_SIZE_K).long())
            cache_miss_mask = torch.logical_or(table_lookup >= BUDGET, table_lookup == 65535)
            
            # calc frequency of cache miss mask, and cut it as budget. create temporary bank
            cache_miss_log = torch.where(cache_miss_mask, block_access_log, MAX_PTR)
            cache_miss_log, sorted_indices = torch.sort(cache_miss_log, dim=-1, stable=False, descending=False)
            cache_miss_mask = cache_miss_mask.gather(dim=-1, index=sorted_indices)
            # print(cache_miss_log[0])
            cache_miss_log_frequency = linear_scan_and_calc_frequency(cache_miss_log, MAX_PTR)
            # print(cache_miss_log_frequency[0])
            cache_miss_log_frequency = torch.where(cache_miss_log < MAX_PTR, cache_miss_log_frequency, 0)
            sort_indices = torch.argsort(cache_miss_log_frequency, dim=-1, descending=True, stable=False)
            cache_miss_log = cache_miss_log.gather(dim=-1, index=sort_indices)[:, :BUDGET]
            cache_miss_mask = cache_miss_mask.gather(dim=-1, index=sort_indices)[:, :BUDGET]
            # print(cache_miss_log[0])
            # print(cache_miss_log_frequency.gather(dim=-1, index=sort_indices)[0])
            
            # copy temporary bank to banks using copy_to_banks
            victim_mask = (bank_stats[:, :, 0] <= 1).int()
            victim_mask, victim_bank_indices = victim_mask.sort(dim=-1, descending=True)
            victim_bank_indices = torch.where(victim_mask.bool(), victim_bank_indices, BUDGET)
            
            # REPLACE_BUDGET = int(BUDGET * 0.3)
            REPLACE_BUDGET = BUDGET
            
            # print('hi', victim_mask[0], cache_miss_log[:, :REPLACE_BUDGET], victim_bank_indices[:, :REPLACE_BUDGET])
            
            final_mask = torch.logical_and(victim_mask.bool(), cache_miss_mask)
            cache_miss_log = torch.where(final_mask, cache_miss_log, MAX_PTR)
            victim_bank_indices = torch.where(final_mask, victim_bank_indices, BUDGET)
            
            table_delta = cache_miss_log[:, :REPLACE_BUDGET]
            bank_loc = victim_bank_indices[:, :REPLACE_BUDGET]
            
            self.copy_to_banks(
                tables=tables,
                banks=banks,
                bank_stats=bank_stats,
                states=key_cache,
                table_delta=table_delta,
                bank_loc=bank_loc,
                check_masking_bank_validity=False,
                copy_from_original=True,
            )
        else:
            raise Exception()

    def process_block_access_log(
        self, 
        layer_index: int,
        position_ids: torch.Tensor,
        query_states: torch.Tensor,
        new_key_state: torch.Tensor,
        new_value_state: torch.Tensor,
        metadata: HiPAttentionOutputMetadata,
        mask_updated: bool,
    ):
        # NOTE: this is temporary
        query_states = query_states.expand(self.max_batch_size, -1, -1, -1)
        new_key_state = new_key_state.expand(self.max_batch_size, -1, -1, -1)
        new_value_state = new_value_state.expand(self.max_batch_size, -1, -1, -1)
        
        BSZ, TDST, HEAD, HID = query_states.shape
        _, NEW_TSRC, HEAD_KV, _ = new_key_state.shape
        
        on_demand_sa_cache_update = True
        
        (
            masking_key_tables, masking_key_banks, masking_key_bank_stats,
            sa_key_tables, sa_key_banks, sa_key_bank_stats,
            sa_value_tables, sa_value_banks, sa_value_bank_stats,
            _,
        ) = self.get_offload_cache(layer_index)
        
        if (new_key_state.shape[1] == 1):
            # on decoding
            if mask_updated and on_demand_sa_cache_update:
                # print('update SA kv cache')
                if self.uvm_offload_key:
                    key_cache = self.key_cache[layer_index][0]
                else:
                    key_cache = self.key_cache[layer_index]
                
                if self.uvm_offload_value:
                    value_cache = self.value_cache[layer_index][0]
                else:
                    value_cache = self.value_cache[layer_index]
                
                _, MAX_TSRC, HEAD_KV, _ = key_cache.shape
                
                self.copy_state_to_bank_using_mask(
                    sa_key_tables, 
                    sa_key_banks, 
                    sa_key_bank_stats,
                    key_cache,
                    metadata, 
                    position_ids,
                )
                self.copy_state_to_bank_using_mask(
                    sa_value_tables, 
                    sa_value_banks, 
                    sa_value_bank_stats,
                    value_cache,
                    metadata, 
                    position_ids,
                )
                
                self.copy_stats_to_bank_using_access_stats(
                    masking_key_tables,
                    masking_key_banks,
                    masking_key_bank_stats,
                    metadata.block_access_log,
                    key_cache,
                    method='lru_approx',
                )
        else:
            self.copy_state_to_bank_using_mask(
                sa_key_tables, 
                sa_key_banks, 
                sa_key_bank_stats,
                new_key_state,
                metadata, 
                position_ids[:, -1:].repeat(BSZ, 1),
                REPEAT_BATCH=BSZ,
            )
            self.copy_state_to_bank_using_mask(
                sa_value_tables, 
                sa_value_banks, 
                sa_value_bank_stats,
                new_value_state,
                metadata, 
                position_ids[:, -1:].repeat(BSZ, 1),
                REPEAT_BATCH=BSZ,
            )
            
            self.copy_stats_to_bank_using_access_stats(
                masking_key_tables,
                masking_key_banks,
                masking_key_bank_stats,
                metadata.block_access_log[:, -1:, :].repeat(BSZ, 1, 1),
                new_key_state,
                method='overwrite',
            )
            
            # (
            #     masking_key_tables, masking_key_banks, masking_key_bank_stats,
            #     _, _, _,
            #     _, _, _,
            #     _
            # ) = self.get_offload_cache(layer_index)
            
            # # self.copy_stats_to_bank_using_access_stats(
            # #     masking_key_tables,
            # #     masking_key_banks,
            # #     masking_key_bank_stats,
            # #     metadata.block_access_log[:, -1:, :].repeat(BSZ, 1, 1),
            # #     new_key_state,
            # # )
            
            # NCACHE, TABLE_LEN = masking_key_tables.shape
            # NCACHE, BUDGET, BLOCK_SIZE_K, HID = masking_key_banks.shape
            
            # LARGE_INT = (TABLE_LEN - 1) * BLOCK_SIZE_K #torch.iinfo(metadata.block_access_log.dtype).max
            
            # tsrc_log = metadata.block_access_log[:, -1, :].repeat(BSZ, 1) * self.block_size_k
            
            # tsrc_log = tsrc_log.view(BSZ, HEAD_KV, HEAD // HEAD_KV * tsrc_log.shape[-1])
            # tsrc_log = torch.where(tsrc_log >= 0, tsrc_log, LARGE_INT)
            # tsrc_log = tsrc_log.sort(dim=-1).values
            # tsrc_log = torch.where(tsrc_log != torch.roll(tsrc_log, shifts=1, dims=-1), tsrc_log, LARGE_INT)
            # tsrc_log = tsrc_log.sort(dim=-1).values
            
            # tsrc_log = tsrc_log.reshape(NCACHE, -1).sort(dim=-1).values
            # tsrc_log = torch.where(tsrc_log != torch.roll(tsrc_log, shifts=1, dims=-1), tsrc_log, LARGE_INT)
            
            # tsrc_log = tsrc_log.sort(dim=-1).values[:, :BUDGET]
            
            # self.copy_to_banks(
            #     tables=masking_key_tables,
            #     banks=masking_key_banks,
            #     bank_stats=masking_key_bank_stats,
            #     states=new_key_state, 
            #     table_delta=tsrc_log,
            #     copy_from_original=True,
            # )
            pass

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            if isinstance(self.key_cache[layer_idx], torch.Tensor):
                self.key_cache[layer_idx].zero_()
            else:
                self.key_cache[layer_idx][1].zero_()
            if isinstance(self.value_cache[layer_idx], torch.Tensor):
                self.value_cache[layer_idx].zero_()
            else:
                self.value_cache[layer_idx][1].zero_()
        
        def reset_cache(tables, banks, bank_stats):
            tables.fill_(65535)
            banks.fill_(0)
            bank_stats[:, :, 0].fill_(0)
            bank_stats[:, :, 1].fill_(65535)
        reset_cache(self.masking_key_tables, self.masking_key_banks, self.masking_key_bank_stats)
        reset_cache(self.sa_key_tables, self.sa_key_banks, self.sa_key_bank_stats)
        reset_cache(self.sa_value_tables, self.sa_value_banks, self.sa_value_bank_stats)

def convert_llama_to_vllm(model: LlamaForCausalLM):
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models.llama import LlamaMLP
    for ilayer, layer in enumerate(model.model.layers):
        layer = layer # type: LlamaDecoderLayer
        input_layernorm = RMSNorm(layer.input_layernorm.weight.shape[0]).to(model.device).half()
        input_layernorm.load_state_dict(layer.input_layernorm.state_dict())
        output_layernorm = RMSNorm(layer.post_attention_layernorm.weight.shape[0]).to(model.device).half()
        output_layernorm.load_state_dict(layer.post_attention_layernorm.state_dict())
        layer.input_layernorm = input_layernorm
        layer.post_attention_layernorm = output_layernorm
        
        mlp = LlamaMLP(
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            hidden_act=model.config.hidden_act,
            bias=getattr(model.config, "mlp_bias", False),
            prefix=f"layer{ilayer}.mlp",
        ).to(model.device).half()
        mlp.down_proj.load_state_dict(layer.mlp.down_proj.state_dict())
        mlp.gate_up_proj.load_state_dict({
            'weight': torch.cat([
                layer.mlp.gate_proj.weight,
                layer.mlp.up_proj.weight,
            ], dim=0)
        })
        layer.mlp = mlp
        
        self_attn = layer.self_attn # type: LlamaAttention
        qkv_proj = QKVParallelLinear(
            model.config.hidden_size, 
            self_attn.head_dim, 
            self_attn.num_heads, 
            self_attn.num_key_value_heads, 
            self_attn.q_proj.bias is not None, 
            params_dtype=torch.float16
        ).to(model.device).half()
        qkv_proj.load_state_dict({
            'weight': torch.cat([
                self_attn.q_proj.weight,
                self_attn.k_proj.weight,
                self_attn.v_proj.weight,
            ], dim=0)
        })
        self_attn.q_proj = None
        self_attn.k_proj = None
        self_attn.v_proj = None
        self_attn.qkv_proj = qkv_proj
    
    norm = RMSNorm(model.model.norm.weight.shape[0]).to(model.device).half()
    norm.load_state_dict(model.model.norm.state_dict())
    model.model.norm = norm
    
    return model

class CUDACapture:
    def __init__(self, model):
        self.model = model
        self.graph = None
        
    def try_capture_and_forward(self, *args, **kwargs):
        if self.need_capture():
            self.capture(*args, **kwargs)
        return self.forward(*args, **kwargs)
    
    def capture(self, *args, **kwargs):
        assert self.graph is None
        
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(5):
                self.model.forward(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.graph_output = self.model.forward(*args, **kwargs)
        self.graph = g
        self.graph_args = args
        self.graph_kwargs = kwargs
    
    def forward(self, *args, **kwargs):
        assert self.graph is not None
        
        for src, dst in zip(args, self.graph_args):
            if isinstance(src, torch.Tensor):
                dst.copy_(src, non_blocking=True)
        
        for key in kwargs:
            if isinstance(kwargs[key], torch.Tensor):
                src = kwargs[key]
                dst = self.graph_kwargs[key]
                assert src.shape == dst.shape, key
                dst.copy_(src, non_blocking=True)
        
        self.graph.replay()
        
        return self.graph_output

    def need_capture(self):
        return self.graph is None

class Runner:
    def __init__(
        self, 
        model_id: str, 
        method: str, 
        cache_backend: Literal['cuda', 'uvm'],
        kv_share: int,
        using_offload_cache: bool,
        cache_size: int,
        refresh_interval: int,
        prefix_query: bool,
        hip_args: HiPAttentionArgs,
    ):
        import vllm.distributed
        import torch.distributed
        
        assert refresh_interval > 0
        
        if not torch.distributed.is_initialized():
            os.environ['MASTER_PORT'] = str(random.randint(32000, 33000))
            os.environ['MASTER_ADDR'] = 'localhost'
            torch.distributed.init_process_group(world_size=1, rank=0)
            vllm.distributed.init_distributed_environment(1, 0, local_rank=0)
            vllm.distributed.initialize_model_parallel()
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = LlamaForCausalLM.from_pretrained(
            model_id, 
            device_map={'': 'cuda:0'}, 
            torch_dtype=torch.float16,
            attn_implementation='flash_attention_2',
        )
        for module in model.modules():
            if isinstance(module, LlamaAttention):
                module.attention_method = method
                module.hip_args = hip_args
                module.hip_prefix_query_length = refresh_interval - 1
        
        self.tokenizer = tokenizer
        self.model = convert_llama_to_vllm(model.half()).eval()
        self.method = method
        self.decode_step = 0
        self.cache_backend = cache_backend
        self.prefix_query = prefix_query
        self.hip_args = hip_args
        
        self.capture = CUDACapture(self.model)
        
        self.capture_hip_refresh = CUDACapture(self.model)
        self.capture_hip_cache = CUDACapture(self.model)
        
        self.hip_refresh_interval = refresh_interval
        
        self.using_offload_cache = using_offload_cache
        self.kv_share = kv_share
        self.kv_offload_cache = using_offload_cache
        self.kv_offload_cache_size = cache_size
        
        self.refresh_step_prefix_query = dict()
        self.cache_step_last_query = dict()
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
    
    @torch.inference_mode(True)
    def decode_forward(self, *args, **kwargs):
        if self.method == 'hip':
            if (not self.using_offload_cache) or True:
                if self.capture_hip_refresh.need_capture():
                    for m in self.model.modules():
                        if isinstance(m, LlamaAttention):
                            assert m.hip_prefix_query is not None
                            
                            self.refresh_step_prefix_query[m] = m.hip_prefix_query
                            
                            m.hip_cache = None
                            m.hip_last_cache = None
                            m.hip_use_cache = False
                            m.hip_checkout_cache = True
                            m.using_prefix_query = True
                    self.capture_hip_refresh.capture(*args, **kwargs)
                    
                    for m in self.model.modules():
                        if isinstance(m, LlamaAttention):
                            assert m.hip_last_cache is not None
                            m.hip_cache = m.hip_last_cache
                            m.hip_use_cache = True
                            m.hip_checkout_cache = False
                            m.using_prefix_query = False
                    self.capture_hip_cache.capture(*args, **kwargs)
                    
                    for m in self.model.modules():
                        if isinstance(m, LlamaAttention):
                            assert m.hip_cache is not None
                            
                            self.cache_step_last_query[m] = m.hip_last_query
                            
                            m.hip_cache = None
                            m.hip_last_cache = None
                            m.hip_use_cache = False
                            m.hip_checkout_cache = False
                            m.using_prefix_query = False
                
                if (self.decode_step % self.hip_refresh_interval) == 0:
                    return self.capture_hip_refresh.forward(*args, **kwargs)
                else:
                    output = self.capture_hip_cache.forward(*args, **kwargs)
                    
                    for m in self.model.modules():
                        if isinstance(m, LlamaAttention):
                            prefix = self.refresh_step_prefix_query[m]
                            query = self.cache_step_last_query[m]
                            offset = (self.decode_step % self.hip_refresh_interval) - 1
                            prefix[:, offset:offset+1].copy_(query, non_blocking=True)
                            # print('a')
                    
                    return output
            else:
                if self.capture_hip_refresh.need_capture():
                    for m in self.model.modules():
                        if isinstance(m, LlamaAttention):
                            m.hip_cache = None
                            m.hip_last_cache = None
                            m.hip_use_cache = False
                            m.hip_checkout_cache = True
                    self.capture_hip_refresh.capture(*args, **kwargs)
                    
                    for m in self.model.modules():
                        if isinstance(m, LlamaAttention):
                            assert m.hip_last_cache is not None
                            m.hip_cache = m.hip_last_cache
                            m.hip_use_cache = True
                            m.hip_checkout_cache = False
                    self.capture_hip_cache.capture(*args, **kwargs)
                    
                    for m in self.model.modules():
                        if isinstance(m, LlamaAttention):
                            assert m.hip_cache is not None
                            m.hip_cache = None
                            m.hip_last_cache = None
                            m.hip_use_cache = False
                            m.hip_checkout_cache = False
                
                if (self.decode_step % self.hip_refresh_interval) == 0:
                    return self.capture_hip_refresh.forward(*args, **kwargs)
                else:
                    return self.capture_hip_cache.forward(*args, **kwargs)
        else:
            return self.capture.try_capture_and_forward(*args, **kwargs)
    
    @torch.inference_mode(True)
    def sample(self, logits: torch.Tensor):
        logits = logits[:, -1:, :] # type: torch.Tensor
        next_token_id = logits.argmax(dim=-1)
        return next_token_id
    
    @torch.inference_mode(True)
    def reset_prefix(self):
        for m in self.model.modules():
            if isinstance(m, LlamaAttention):
                if m.hip_prefix_query is not None:
                    m.hip_prefix_query.fill_(0)
    
    @torch.inference_mode(True)
    def push_last_query_to_prefix(self, max_batch_size):
        for m in self.model.modules():
            if isinstance(m, LlamaAttention):
                last_query = m.hip_last_query
                if m.hip_prefix_query is None:
                    m.hip_prefix_query = torch.zeros((
                        max_batch_size, 
                        m.hip_prefix_query_length, 
                        last_query.shape[2], 
                        last_query.shape[3]
                    ), dtype=last_query.dtype, device=last_query.device)
                m.hip_prefix_query\
                    .copy_(last_query\
                        .repeat_interleave(
                            max_batch_size // m.hip_last_query.shape[0], 0
                        )
                    )
    
    @torch.inference_mode(True)
    def generate(self, text, max_tokens=256, item_repeat=24, n_prefill_warmup=1):
        chunked_prefill_size = 8192
        
        input_ids = self.tokenizer([text, ], return_tensors="pt", padding=True).input_ids.to(self.model.device)
        input_ids = input_ids.repeat(item_repeat, 1)
        bsz, context_len = input_ids.shape
        
        slack_memory = torch.empty((1 * 1024 * 1024 * 1024), dtype=torch.uint8, device=self.model.device) # NOTE: resever 1GB for cuda-graph
        
        cache = StaticCache(
            self.model.config,
            max_batch_size=bsz,
            max_cache_len=context_len + max_tokens, 
            device=self.model.device,
            dtype=torch.float16,
            share=self.kv_share,
            cache_backend=self.cache_backend,
            use_offload_cache=self.kv_offload_cache,
            block_size_k=self.hip_args.block_size_k,
            sliding_window_size=self.hip_args.sliding_window_size,
            cache_size=self.kv_offload_cache_size,
        )
        
        prompt_cache_pos = torch.arange(0, context_len, dtype=torch.long, device=self.model.device)
        self.reset_prefix()
        cache.prompt_start()
        for _ in range(n_prefill_warmup):
            with torch.autocast('cuda', torch.float16):
                self.model(
                    input_ids=input_ids[0:0+1], 
                    position_ids=prompt_cache_pos.unsqueeze(0).expand(1, -1), 
                    cache_position=prompt_cache_pos, 
                    past_key_values=cache,
                    num_logits_to_keep=1,
                )
        cache.prompt_end()
        self.push_last_query_to_prefix(item_repeat)
        
        # compile decode step
        decode_input_ids = torch.zeros((bsz, 1), dtype=torch.long, device=self.model.device)
        decode_cache_pos = torch.zeros((1, ), dtype=torch.long, device=self.model.device)
        decode_cache_pos.fill_(context_len)
        cache.decode_start()
        with torch.autocast('cuda', torch.float16):
            self.decode_forward(
                input_ids=decode_input_ids, 
                position_ids=decode_cache_pos.unsqueeze(0).expand(bsz, 1), 
                cache_position=decode_cache_pos, 
                past_key_values=cache
            )
        cache.decode_end()
        cache.decode_reset_stats()
        cache.reset()
        self.decode_step = 0
        print('decode compiled')
        
        decoded_tokens = []
        
        del slack_memory
        torch.cuda.synchronize()
        
        event_prefill_start = torch.cuda.Event(True)
        event_prefill_end = torch.cuda.Event(True)
        event_decode_start = torch.cuda.Event(True)
        event_decode_end = torch.cuda.Event(True)
        
        logits = []
        torch.cuda.synchronize()
        event_prefill_start.record()
        ibatch = 0
        for module in self.model.modules():
            if isinstance(module, LlamaAttention):
                module.prompt_batch_index = ibatch
        self.reset_prefix()
        cache.prompt_start()
        with torch.autocast('cuda', torch.float16):
            prompt_output = self.model(
                input_ids=input_ids[ibatch:ibatch+1], 
                position_ids=prompt_cache_pos.unsqueeze(0).expand(1, -1), 
                cache_position=prompt_cache_pos, 
                past_key_values=cache,
                num_logits_to_keep=1,
            )
        cache.prompt_end()
        self.push_last_query_to_prefix(item_repeat)
        
        for _ in range(bsz):
            logits.append(prompt_output.logits)
        
        for ilayer in range(len(cache.key_cache)):
            for ibatch in range(bsz):
                if self.cache_backend == 'uvm':
                    cache.key_cache[ilayer][1][ibatch].copy_(cache.key_cache[ilayer][1][0], non_blocking=True)
                    cache.value_cache[ilayer][1][ibatch].copy_(cache.value_cache[ilayer][1][0], non_blocking=True)
                else:
                    cache.key_cache[ilayer][ibatch].copy_(cache.key_cache[ilayer][0], non_blocking=True)
                    cache.value_cache[ilayer][ibatch].copy_(cache.value_cache[ilayer][0], non_blocking=True)
        
        logits = torch.cat(logits, dim=0)
        next_token = self.sample(logits)
        decoded_tokens.append(next_token)
        decode_input_ids.copy_(next_token, non_blocking=True)
        del prompt_output
        event_prefill_end.record()
        
        event_prefill_end.synchronize()
        elapsed_prefill = event_prefill_start.elapsed_time(event_prefill_end)
        print(f'prefill took {elapsed_prefill:.3f} ms')
        
        if self.kv_offload_cache:
            cache.decode_reset_stats()
        elapsed_decode = 0
        hit_ratio_sa = []
        hit_ratio_mask = []
        cache_active_ratio_sa = []
        cache_active_ratio_mask = []
        cache.decode_start()
        for istep in tqdm.tqdm(range(max_tokens), dynamic_ncols=True, leave=False, desc='decode'):
            with torch.autocast('cuda', torch.float16):
                event_decode_start.record()
                decode_output = self.decode_forward(
                    input_ids=decode_input_ids, 
                    position_ids=decode_cache_pos.unsqueeze(0).expand(bsz, 1), 
                    cache_position=decode_cache_pos, 
                    past_key_values=cache
                )
            next_token = self.sample(decode_output.logits)
            decoded_tokens.append(next_token)
            decode_input_ids.copy_(next_token, non_blocking=True)
            decode_cache_pos.add_(1)
            event_decode_end.record()
            event_decode_end.synchronize()
            
            elapsed_decode += event_decode_start.elapsed_time(event_decode_end) / 1000
            
            if self.kv_offload_cache:
                stats = cache.decode_reset_stats()
                if (istep % self.hip_refresh_interval) == 0:
                    hit_ratio_mask.append(stats.cache_hit_ratio)
                    cache_active_ratio_mask.append(stats.cache_access_utilization)
                else:
                    hit_ratio_sa.append(stats.cache_hit_ratio)
                    cache_active_ratio_sa.append(stats.cache_access_utilization)
                if stats.num_accessed > (100 * 1024 * 1024):
                    unit = 'G'
                    unit_scale = 1024 ** 3
                else:
                    unit = 'M'
                    unit_scale = 1024 ** 2
                tqdm.tqdm.write(
                    f'[{istep}] \t '
                    f'hit ratio {stats.cache_hit_ratio * 100:.2f} % \t '
                    f'(accessed = {stats.num_accessed / unit_scale:.2f} {unit}, '
                    f'hit = {stats.num_cache_hit / unit_scale:.2f} {unit}) \t '
                    f'cache access util: {stats.cache_access_utilization * 100:.2f} %'
                )

            self.decode_step += 1
        cache.decode_end()
        
        torch.cuda.synchronize()
        
        elapsed_prefill = event_prefill_start.elapsed_time(event_prefill_end) / 1000
        
        gen_out = torch.cat(decoded_tokens, dim=-1)
        text_outs = self.tokenizer.batch_decode(gen_out, skip_special_tokens=False)
        
        print(
            f"Time taken for {tuple(input_ids.shape)}:  "
            f"{input_ids.shape[-1] / elapsed_prefill:.2f} tok/s {elapsed_prefill:.2f} s  |  "
            f"{gen_out.numel() / elapsed_decode:.2f} tok/s {elapsed_decode:.2f} s"
        )
        print(
            f'Cache statistics: '
            f'Mask(hit ratio = {sum(hit_ratio_mask)/len(hit_ratio_mask)*100:.2f} %, '
            f'active = {sum(cache_active_ratio_mask)/len(cache_active_ratio_mask)*100:.2f} %), '
            f'SA(hit ratio = {sum(hit_ratio_sa)/(len(hit_ratio_sa) + 1e-20)*100:.2f} %, '
            f'active = {sum(cache_active_ratio_sa)/(len(cache_active_ratio_sa) + 1e-20)*100:.2f} %), '
        )
        
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        return text_outs
    
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--method', default='hip', type=str)
        parser.add_argument('--cache_backend', default='cuda', type=str)
        parser.add_argument('--input', default='./samples/32k.md', type=str)
        parser.add_argument('--model', default='llama3.1_8b', type=str)
        parser.add_argument('--batch_size', default=16, type=int)
        parser.add_argument('--kv_share', default=1, type=int)
        parser.add_argument('--max_tokens', default=64, type=int)
        parser.add_argument('--k', default=512, type=int)
        parser.add_argument('--sw', default=256, type=int)
        parser.add_argument('--cache_size', default=8192, type=int)
        parser.add_argument('--offload-cache', action=argparse.BooleanOptionalAction)
        parser.add_argument('--block_size_k', default=2, type=int)
        parser.add_argument('--refresh_interval', default=8, type=int)
        
        args = parser.parse_args()
        
        print(args)
        
        with open(args.input, 'r') as f:
            document = f.read()
        
        sample_input = f'''<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi, can you describe about following document? Here is document, 

```
{document}
```

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
        results = Runner(
            {
                'llama3.1_8b': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
                'llama2_7b': 'meta-llama/Llama-2-7b-chat-hf',
                'llama2_13b': 'meta-llama/Llama-2-13b-chat-hf',
            }[args.model],
            method=args.method,
            cache_backend=args.cache_backend,
            kv_share=args.kv_share,
            using_offload_cache=args.offload_cache,
            cache_size=args.cache_size,
            refresh_interval=args.refresh_interval,
            prefix_query=True,
            hip_args=HiPAttentionArgs(
                mask_k=args.k,
                block_size_k=args.block_size_k,
                sliding_window_size=args.sw,
                sample_method='last',
            ),
        )\
            .generate(
                sample_input,
                item_repeat=args.batch_size,
                max_tokens=args.max_tokens,
            )
        print('-' * 20, 'example', '-' * 20)
        print(results[-1])
        print('-' * 50)
        for result in results:
            result = result.replace("\n", "\\n")
            print(f'{result[:100]} [...] {len(result)}')
    finally:
        import torch.distributed
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()