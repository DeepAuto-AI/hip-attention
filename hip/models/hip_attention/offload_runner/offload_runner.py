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

def get_memory_free_MiB(gpu_index):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free = max(mem_info.free, mem_info.total - torch.cuda.memory_reserved(gpu_index))
    return free // 1024 ** 2

@triton.jit
def copy_to_banks_from_state_cuda(
    STATE,
    stride_state_n, stride_state_t, stride_state_head_kv, stride_state_hid,
    BANK,
    stride_bank_cache, stride_bank_page, stride_bank_offset, stride_bank_hid,
    TABLE_DELTA,
    stride_table_delta_cache, stride_table_delta_update,
    BANK_LOC,
    stride_bank_loc_cache, stride_bank_loc_update,
    
    offset,
    TSRC,
    HEAD_KV,
    
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
    state = tl.load(
        STATE +\
            idx_bsz * stride_state_n +\
            table_delta * stride_state_t +\
            idx_head_kv * stride_state_head_kv +\
            tl.arange(0, HID) * stride_state_hid,
    )
    bank_loc = tl.load(
        BANK_LOC+\
            idx_cache * stride_bank_loc_cache +\
            idx_update * stride_bank_loc_update
    ).to(tl.int64)
    tl.store(
        BANK +\
            idx_cache * stride_bank_cache +\
            bank_loc * stride_bank_page +\
            offset * stride_bank_offset +\
            tl.arange(0, HID) * stride_bank_hid,
        value=state.to(BANK.dtype.element_ty)
    )

def copy_to_banks_from_state(
    state: torch.Tensor,
    bank: torch.Tensor,
    bank_loc: torch.Tensor,
    table_delta: torch.Tensor,
    offset: int,
):
    N_CACHE, BUDGET, BLOCK_SIZE_K, HID = bank.shape
    BSZ, TSRC, HEAD_KV, __ = state.shape
    assert N_CACHE == (BSZ * HEAD_KV), f'{N_CACHE} == ({BSZ} * {HEAD_KV})'
    
    N_CACHE, N_UPDATE = table_delta.shape
    assert table_delta.shape == bank_loc.shape
    
    grid = (N_UPDATE, N_CACHE)
    pre_device = torch.get_default_device()
    torch.set_default_device(table_delta.device)
    copy_to_banks_from_state_cuda[grid](
        state, *state.stride(),
        bank, *bank.stride(),
        table_delta, *table_delta.stride(),
        bank_loc, *bank_loc.stride(),
        
        offset,
        TSRC,
        HEAD_KV,
        HID,
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
        gc.collect()
        torch.cuda.empty_cache()
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

        self.sliding_window_size = sliding_window_size
        self.share = share
        self.uvm_offload_key = uvm_offload_key
        self.uvm_offload_value = uvm_offload_value
        self.cache_budget = cache_size // block_size_k
        self.sparse_attention_budget = (mask_k * kv_group_size + sliding_window_size) // block_size_k
        self.max_seq_len = max_cache_len + self.sliding_window_size
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
            self.masking_key_tables = torch.full((self.num_caches, self.max_seq_len // self.block_size_k + 1), dtype=torch.uint16, device=device, fill_value=self.offload_cache_null_pointer)
            self.masking_key_banks = torch.zeros((self.num_caches, self.cache_budget, self.block_size_k, self.head_dim), dtype=self.offload_cache_dtype, device=device)
            
            # this caches are protected
            self.sa_key_tables = torch.full((self.num_caches, self.max_seq_len // self.block_size_k + 1), dtype=torch.uint16, device=device, fill_value=self.offload_cache_null_pointer)
            self.sa_key_banks = torch.zeros((self.num_caches, self.sparse_attention_budget, self.block_size_k, self.head_dim), dtype=self.offload_cache_dtype, device=device)
            self.sa_value_tables = torch.full((self.num_caches, self.max_seq_len // self.block_size_k + 1), dtype=torch.uint16, device=device, fill_value=self.offload_cache_null_pointer)
            self.sa_value_banks = torch.zeros((self.num_caches, self.sparse_attention_budget, self.block_size_k, self.head_dim), dtype=self.offload_cache_dtype, device=device)
            
            self.counters = torch.zeros((self.num_caches, 2), dtype=torch.int64, device=device) # [accessed, hit]
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
            byte_size = 2 * new_layer_key_cache[0].numel() * new_layer_key_cache[0].element_size()
            total_bytes += byte_size
            for _ in range(share):
                self.key_cache.append(new_layer_key_cache)
                self.value_cache.append(new_layer_value_cache)
        print(f'allocated {total_bytes / 1024 / 1024:,} MB')
        
        self.prompt_copy_stream = torch.cuda.Stream(self.device)
        self.prompt_copy_threads: List[threading.Thread] = []
    
    def has_offload_cache(self, layer_idx: int):
        return self.using_offload_cache
    
    def get_offload_cache(self, layer_idx: int):
        def get_layer(t: torch.Tensor):
            return t[self.num_caches_per_layer * layer_idx: self.num_caches_per_layer * (layer_idx + 1)]
        
        masking_key_tables = get_layer(self.masking_key_tables)
        masking_key_banks = get_layer(self.masking_key_banks)
        
        sa_key_tables = get_layer(self.sa_key_tables)
        sa_key_banks = get_layer(self.sa_key_banks)
        sa_value_tables = get_layer(self.sa_value_tables)
        sa_value_banks = get_layer(self.sa_value_banks)
        
        counters = get_layer(self.counters)
        
        return (
            masking_key_tables, masking_key_banks,
            sa_key_tables, sa_key_banks,
            sa_value_tables, sa_value_banks,
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
        masking_key_tables,
        masking_key_banks,
        new_key_state,
        tsrc_log, 
        BLOCK_SIZE_K, 
        NCACHE, 
        BUDGET, 
        HID, 
        NEW_TSRC, 
        tsrc_log_bank_loc = None,
        check_masking_bank_validity = False,
    ):  
        if tsrc_log_bank_loc is None:
            tsrc_log_bank_loc = torch\
                .arange(0, tsrc_log.shape[-1], device=tsrc_log.device)[None, :]\
                .expand(masking_key_tables.shape[0], -1)
        elif tsrc_log_bank_loc.ndim == 1:
            tsrc_log_bank_loc = tsrc_log_bank_loc[None, :]\
                .expand(masking_key_tables.shape[0], -1)
        else:
            assert isinstance(tsrc_log_bank_loc, torch.Tensor)
        
        masking_key_tables.copy_(
            masking_key_tables.long().scatter(
                dim=-1, 
                index=tsrc_log.long() // BLOCK_SIZE_K, 
                src=tsrc_log_bank_loc,
            ).to(masking_key_tables.dtype)
        )
        
        for ioffset in range(BLOCK_SIZE_K):
            copy_to_banks_from_state(
                new_key_state,
                masking_key_banks,
                tsrc_log_bank_loc,
                tsrc_log,
                ioffset,
            )
        
        if check_masking_bank_validity:
            for idx in tqdm.tqdm(range(masking_key_tables.shape[-1] - 1)):
                idx_bank = masking_key_tables[0, idx].item()
                if idx_bank < 0 or idx_bank > BUDGET:
                    continue
                for idx_k in range(BLOCK_SIZE_K):
                    token = masking_key_banks[0, idx_bank, idx_k].float()
                    truth = new_key_state[0, idx * BLOCK_SIZE_K + idx_k, 0].float()
                    assert (token - truth).abs().mean().item() < 1e-4, f'{(token - truth).abs().sum().item()}'

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
        
        if (new_key_state.shape[1] == 1):
            # on decoding
            if mask_updated and on_demand_sa_cache_update:
                # print('update SA kv cache')
                (
                    _, _,
                    sa_key_tables, sa_key_banks,
                    sa_value_tables, sa_value_banks,
                    _
                ) = self.get_offload_cache(layer_index)
                
                if self.uvm_offload_key:
                    key_cache = self.key_cache[layer_index][0]
                else:
                    key_cache = self.key_cache[layer_index]
                
                if self.uvm_offload_value:
                    value_cache = self.value_cache[layer_index][0]
                else:
                    value_cache = self.value_cache[layer_index]
                
                _, MAX_TSRC, HEAD_KV, _ = key_cache.shape
                
                def copy_state_using_mask(
                    tables: torch.Tensor, banks: torch.Tensor, states: torch.Tensor
                ):
                    NCACHE, TABLE_LEN = tables.shape
                    NCACHE, BUDGET, BLOCK_SIZE_K, HID = banks.shape
                    
                    LARGE_INT = (TABLE_LEN - 1) * BLOCK_SIZE_K
                    
                    # print(position_ids.shape) # torch.Size([1, 1])
                    sw_mask = ((position_ids - self.sliding_window_size) // BLOCK_SIZE_K * BLOCK_SIZE_K)\
                        .repeat(HEAD_KV, 1)\
                        .amax(dim=-1)[:, None] +\
                            torch.arange(
                                0, 
                                self.sliding_window_size, 
                                BLOCK_SIZE_K, 
                                device=tables.device
                            )[None, :]
                    sw_mask = torch.where(sw_mask >= 0, sw_mask, LARGE_INT)
                    self.copy_to_banks(
                        masking_key_tables=tables,
                        masking_key_banks=banks,
                        tsrc_log=sw_mask,
                        tsrc_log_bank_loc=torch\
                            .arange(BUDGET - self.sliding_window_size // BLOCK_SIZE_K, BUDGET, device=tables.device),
                        new_key_state=states,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        BUDGET=BUDGET,
                        HID=HID,
                        NCACHE=NCACHE,
                        NEW_TSRC=MAX_TSRC,
                    )
                    
                    mask = metadata.indices[:, -1, :].view(NCACHE, -1)
                    mask = torch.where(mask >= 0, torch.clamp_max(mask, LARGE_INT), LARGE_INT).sort(dim=-1).values
                    mask = torch.where(mask != torch.roll(mask, dims=-1, shifts=1), mask, LARGE_INT)\
                        .sort(dim=-1).values[:, :BUDGET - self.sliding_window_size // BLOCK_SIZE_K]
                    
                    self.copy_to_banks(
                        masking_key_tables=tables,
                        masking_key_banks=banks,
                        tsrc_log=mask,
                        new_key_state=states,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        BUDGET=BUDGET,
                        HID=HID,
                        NCACHE=NCACHE,
                        NEW_TSRC=MAX_TSRC,
                    )
                
                copy_state_using_mask(sa_key_tables, sa_key_banks, key_cache)
                copy_state_using_mask(sa_value_tables, sa_value_banks, value_cache)
        else:
            (
                masking_key_tables, masking_key_banks,
                _, _,
                _, _,
                _
            ) = self.get_offload_cache(layer_index)
            
            NCACHE, TABLE_LEN = masking_key_tables.shape
            NCACHE, BUDGET, BLOCK_SIZE_K, HID = masking_key_banks.shape
            
            LARGE_INT = (TABLE_LEN - 1) * BLOCK_SIZE_K #torch.iinfo(metadata.block_access_log.dtype).max
            
            tsrc_log = metadata.block_access_log[:, -1, :].repeat(BSZ, 1) * self.block_size_k
            
            tsrc_log = tsrc_log.view(BSZ, HEAD_KV, HEAD // HEAD_KV * tsrc_log.shape[-1])
            tsrc_log = torch.where(tsrc_log >= 0, tsrc_log, LARGE_INT)
            tsrc_log = tsrc_log.sort(dim=-1).values
            tsrc_log = torch.where(tsrc_log != torch.roll(tsrc_log, shifts=1, dims=-1), tsrc_log, LARGE_INT)
            tsrc_log = tsrc_log.sort(dim=-1).values
            
            tsrc_log = tsrc_log.reshape(NCACHE, -1).sort(dim=-1).values
            tsrc_log = torch.where(tsrc_log != torch.roll(tsrc_log, shifts=1, dims=-1), tsrc_log, LARGE_INT)
            
            tsrc_log = tsrc_log.sort(dim=-1).values[:, :BUDGET]
            
            self.copy_to_banks(
                masking_key_banks=masking_key_banks,
                masking_key_tables=masking_key_tables,
                new_key_state=new_key_state, 
                tsrc_log=tsrc_log, 
                BLOCK_SIZE_K=BLOCK_SIZE_K, 
                NCACHE=NCACHE, 
                BUDGET=BUDGET,
                HID=HID,
                NEW_TSRC=NEW_TSRC,
            )

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
            for i in range(3):
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
        hip_args: HiPAttentionArgs,
    ):
        import vllm.distributed
        import torch.distributed
        
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
        
        self.tokenizer = tokenizer
        self.model = convert_llama_to_vllm(model.half()).eval()
        self.method = method
        self.decode_step = 0
        self.cache_backend = cache_backend
        self.hip_args = hip_args
        
        self.capture = CUDACapture(self.model)
        
        self.capture_hip_refresh = CUDACapture(self.model)
        self.capture_hip_cache = CUDACapture(self.model)
        
        self.hip_refresh_interval = 8
        
        self.using_offload_cache = using_offload_cache
        self.kv_share = kv_share
        self.kv_offload_cache = using_offload_cache
        self.kv_offload_cache_size = cache_size
    
    @torch.inference_mode(True)
    def decode_forward(self, *args, **kwargs):
        if self.method == 'hip':
            if not self.using_offload_cache:
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
    def generate(self, text, max_tokens=256, item_repeat=24, n_prefill_warmup=1):
        input_ids = self.tokenizer([text, ] * item_repeat, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        bsz, context_len = input_ids.shape
        
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
        
        # compile decode step
        decode_input_ids = torch.zeros((bsz, 1), dtype=torch.long, device=self.model.device)
        decode_cache_pos = torch.zeros((1, ), dtype=torch.long, device=self.model.device)
        cache.decode_start()
        with torch.autocast('cuda', torch.float16):
            self.decode_forward(
                input_ids=decode_input_ids, 
                position_ids=decode_cache_pos.unsqueeze(0).expand(bsz, 1), 
                cache_position=decode_cache_pos, 
                past_key_values=cache
            )
        cache.decode_end()
        cache.reset()
        self.decode_step = 0
        print('decode compiled')
        
        prompt_cache_pos = torch.arange(0, context_len, dtype=torch.long, device=self.model.device)
        decode_cache_pos.fill_(context_len)
        decoded_tokens = []
        
        torch.cuda.synchronize()
        
        event_prefill_start = torch.cuda.Event(True)
        event_prefill_end = torch.cuda.Event(True)
        event_decode_start = torch.cuda.Event(True)
        event_decode_end = torch.cuda.Event(True)
        
        logits = []
        for idx_warmup in range(n_prefill_warmup+1):
            if idx_warmup == n_prefill_warmup:
                print('prefill warmup done')
                torch.cuda.synchronize()
                event_prefill_start.record()
            ibatch = 0
            for module in self.model.modules():
                if isinstance(module, LlamaAttention):
                    module.prompt_batch_index = ibatch
            cache.prompt_start()
            prompt_output = self.model(
                input_ids=input_ids[ibatch:ibatch+1], 
                position_ids=prompt_cache_pos.unsqueeze(0).expand(1, -1), 
                cache_position=prompt_cache_pos, 
                past_key_values=cache,
                num_logits_to_keep=1,
            )
            cache.prompt_end()
        for _ in range(bsz):
            logits.append(prompt_output.logits)
        
        for ilayer in range(len(cache.key_cache)):
            for ibatch in range(bsz):
                cache.key_cache[ilayer][1][ibatch].copy_(cache.key_cache[ilayer][1][0], non_blocking=True)
                cache.value_cache[ilayer][1][ibatch].copy_(cache.value_cache[ilayer][1][0], non_blocking=True)
        
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
            cache.counters.fill_(0)
        elapsed_decode = 0
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
                counters = cache.counters.sum(0)
                accessed, hit = counters.cpu().tolist()
                accessed += 1e-20
                cache.counters.fill_(0)
                tqdm.tqdm.write(f'[{istep}] \t hit ratio {hit / accessed * 100:.2f} % \t (accessed = {accessed / 1024 / 1024:.2f} M, hit = {hit / 1024 / 1024:.2f} M)')
            
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
        for result in results[:8]:
            result = result.replace("\n", "\\n")
            print(f'{result[:80]} [...] {len(result)}')
    finally:
        import torch.distributed
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()