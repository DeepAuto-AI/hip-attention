import math
import cuda
import torch
from torch import Tensor
from typing import Optional, Tuple, Union
import triton
import triton.language as tl

from hip.models.hip_attention.offload_runner.tensor_from_pointer import (
    tensor_from_pointer
)
from hip.models.hip_attention.gen3.attention_metadata import (
    HiPAttentionOutputMetadata,
    HiPAttentionCacheAccessStatistics,
)

MAX_INT: tl.constexpr = 2_147_483_647

def sizeof(dtype: Union[Tensor, torch.dtype]) -> int:
    if isinstance(dtype, Tensor):
        return dtype.numel() * sizeof(dtype.dtype)
    
    if dtype in [
        torch.uint8, 
        torch.int8, 
        torch.float8_e4m3fn, 
        torch.float8_e4m3fnuz, 
        torch.float8_e5m2, 
        torch.float8_e5m2fnuz
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

def format_size_bytes(tensor: Union[Tensor, Union[float, int]]) -> str:
    if isinstance(tensor, Tensor):
        byte_size = sizeof(tensor)
    elif isinstance(tensor, (int, float)):
        byte_size = tensor
    else:
        raise Exception()

    if byte_size < 1024:
        return f'{byte_size} B'
    elif byte_size < (1024 ** 2):
        return f'{byte_size / 1024:.2f} KB'
    elif byte_size < (1024 ** 3):
        return f'{byte_size / (1024 ** 2):.2f} MB'
    else:
        return f'{byte_size / (1024 ** 3):.2f} GB'

def debug_print(self, *args):
    print('[HiPOffloadKVPoolMHA]', *args)

###############################################################################
#                               Data Structure
###############################################################################

def uvm_note_cpu(tensor: Tensor, prefetch: bool = False):
    cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation, -1)
    cuda.cudart.cudaMemAdvise(tensor.data_ptr(), tensor.numel() * tensor.element_size(), cuda.cudart.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy, tensor.device.index)
    if prefetch:
        cuda.cudart.cudaMemPrefetchAsync(tensor.data_ptr(), tensor.numel() * tensor.element_size(), -1, 0)

class UVMCache:
    bank_cpu: Tensor
    bank_gpu: Tensor
    
    def __init__(
        self, 
        max_token_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_token_size = max_token_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        self.bank_cpu, self.bank_gpu = self.alloc_uvm([max_token_size, head_num, head_dim])
        
        # {
        #     Token Generation: uint32    # Increase one on every overwrite
        # }
        self.metadata = torch.full(
            [max_token_size, 1], 
            dtype=torch.uint32, 
            device=device,
            fill_value=MAX_INT
        )
        
        self.allocated_cpu_bytes = sizeof(self.bank_cpu)
        self.allocated_gpu_bytes = sizeof(self.metadata)
        
        debug_print(f'UVMCache: bank={format_size_bytes(self.bank_cpu)}, metadata={format_size_bytes(self.metadata)}')
    
    def alloc_uvm(self, shape, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        device = self.device
        if isinstance(device, str):
            device = torch.device(device)
            
        elem_size = sizeof(dtype)
        numel = math.prod(shape)
        align = 4096
        byte_size = elem_size * numel
        byte_size = byte_size + byte_size % align
        
        _result_code, pointer = cuda.cudart.cudaMallocManaged(
            byte_size, 
            cuda.cudart.cudaMemAttachGlobal
        )
        
        t_gpu = tensor_from_pointer(pointer, shape, dtype, device.index)
        t_cpu = tensor_from_pointer(pointer, shape, dtype, -1)
        
        uvm_note_cpu(t_gpu)
        t_cpu.fill_(0)
        
        return t_cpu, t_gpu
    
    def gather_cpu(self, table: Tensor, pin_memory = False) -> Tensor:
        assert table.ndim == 1
        assert table.device == self.bank_cpu.device
        t = self.bank_cpu[table]
        if pin_memory:
            t = t.pin_memory()
        return t
    
class GPUCache:
    def __init__(
        self, 
        k_uvm: UVMCache, 
        v_uvm: Optional[UVMCache],
        max_cache_token_size: int,
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
        
        self.bank = torch.zeros(
            (self.max_cache_token_size, self.head_dim), 
            dtype=self.dtype, 
            device=self.device
        )
        
        """
        {
            Back reference to table: uint32,        # initial handshake
            Reference to UVM Cache: uint32,         # MAX_TOKEN
            Token Generation of UVM Cache: uint32,  # To check the version of cached token
        }
        """
        self.metadata = torch.full(
            (self.max_cache_token_size, 3),
            dtype=torch.uint32,
            device=self.device,
            fill_value=MAX_INT,
        )
        
        self.table = torch.full(
            (self.head_num, self.max_uvm_token_size, 1),
            dtype=torch.uint32,
            device=self.device,
            fill_value=MAX_INT,
        )
        
        self.allocated_gpu_bytes = (
            sizeof(self.bank) + sizeof(self.metadata) + sizeof(self.table)
        )
        debug_print(
            f'[GPUCache] bank={format_size_bytes(self.bank)}, '
            f'metadata={format_size_bytes(self.metadata)}, '
            f'table={format_size_bytes(self.table)}, '
            f'total={format_size_bytes(self.allocated_gpu_bytes)}'
        )
    
    def handle_cache_access(
        self,
        stats: HiPAttentionCacheAccessStatistics
    ):
        # LRU access recency update, if needed
        return
    
    def handle_cache_miss(
        self,
        stats: HiPAttentionCacheAccessStatistics
    ):
        # forcely overwrite whole cache
        return
    
    ########################
    # cache update methods
    ########################
    
    def force_push_indices(self, slot_indices: Tensor):
        assert slot_indices.ndim == 2
        assert slot_indices.shape[1] == 2
        # slot_indices: int32[N_UPDATES, {idx_head, idx_slot}]

class HiPOffloadCache:
    def __init__(
        self,
        max_token_size: int,
        max_mask_cache_token_size: int,
        max_sa_cache_token_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.k_uvm = UVMCache(
            max_token_size=max_token_size,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        
        self.v_uvm = UVMCache(
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
        )
        
        self.sa_kv_cache = GPUCache(
            k_uvm=self.k_uvm,
            v_uvm=self.v_uvm,
            max_cache_token_size=max_sa_cache_token_size,
        )
    
    def get_page_count(self):
        assert self.k_uvm.bank_cpu.shape == self.v_uvm.bank_cpu.shape
        return self.k_uvm.bank_cpu.shape[0]
    
    def prefetch_prefix_kv_buffer(
        self,
        table: Tensor,
        device: torch.device
    ) -> Tuple[Tensor, Tensor]:
        table = table.cpu()
        k = self.k_uvm.gather_cpu(table, pin_memory=True)
        v = self.v_uvm.gather_cpu(table, pin_memory=True)
        k = k.to(device, non_blocking=True)
        v = v.to(device, non_blocking=True)
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
        
        if cache_device == torch.device('cpu'):
            self.k_uvm.bank_cpu[table] = cache_k
            self.v_uvm.bank_cpu[table] = cache_v
        else:
            assert cache_device == self.k_uvm.device
            self.k_uvm.bank_gpu[table] = cache_k
            self.v_uvm.bank_gpu[table] = cache_v
        
        self.k_uvm.metadata.index_put_(
            indices=table, 
            values=torch.index_select(self.k_uvm.metadata, index=table_gpu, dim=0) + 1
        )
        self.v_uvm.metadata.index_put_(
            indices=table, 
            values=torch.index_select(self.v_uvm.metadata, index=table_gpu, dim=0) + 1
        )
    
    # call this after decode masking step
    def handle_cache_miss(self, metadata: HiPAttentionOutputMetadata):
        self.mask_k_cache.handle_cache_access(
            stats=metadata.sa_cache_statistics
        )
        self.mask_k_cache.handle_cache_miss(
            stats=metadata.sa_cache_statistics
        )
        
        self.sa_kv_cache.handle_cache_access(
            stats=metadata.sa_cache_statistics
        )
        self.sa_kv_cache.handle_cache_miss(
            stats=metadata.sa_cache_statistics
        )

###############################################################################
#                               Kernel Function
###############################################################################

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
    OFFLOAD_CACHE_LOAD_VALUE: tl.constexpr,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
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
    
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    # DEBUG: to load nothing
    # mask_keys = mask_keys & False
    
    # tl.static_print(OFFLOAD_CACHE_METHOD)
    
    if not USING_PAGES:
        tl.static_assert(not USING_OFFLOAD_CACHE)
        
        tl.atomic_add(
            ACCESS_COUNTER +\
                idx_bsz.to(tl.int64) * stride_access_counter_bsz +\
                idx_kv_head * stride_access_counter_head_kv +\
                idx_tsrc * stride_access_counter_tsrc,
            mask=mask_keys,
            val=1
        )
        
        keys = tl.load(
            K +\
                idx_bsz.to(tl.int64) * stride_k_bsz +\
                idx_tsrc.to(tl.int64) * stride_k_tsrc +\
                idx_kv_head.to(tl.int64) * stride_k_head +\
                idx_hid.to(tl.int64) * stride_k_hid,
            mask = mask_keys,
            other = 0.0,
            # cache_modifier='.cs', # TODO: uncomment this
        )
    else:
        seq_len = tl.load(
            CACHE_SEQ_LENS +\
                idx_bsz.to(tl.int64) * stride_cache_seq_lens_b,
        )
        mask_tsrc = idx_tsrc < seq_len
        ptrs = BLOCK_TABLE +\
            idx_bsz.to(tl.int64) * stride_block_table_bsz + \
            (idx_tsrc // PAGE_SIZE).to(tl.int64) * stride_block_table_page
        idx_page = tl.load(
            ptrs,
            mask=mask_tsrc,
            other=0,
        ).to(tl.int64)
        offset_page = idx_tsrc % PAGE_SIZE
        
        tl.atomic_add(
            ACCESS_COUNTER +\
                idx_bsz.to(tl.int64) * stride_access_counter_bsz +\
                idx_kv_head * stride_access_counter_head_kv +\
                idx_page * stride_access_counter_tsrc,
            mask=mask_keys,
            val=1
        )
        
        if USING_OFFLOAD_CACHE:
            tl.static_assert(PAGE_SIZE == 1)
            original_mask_keys = mask_keys
            
            idx_slots = tl.load(
                OFFLOAD_CACHE_GPU_TABLE +\
                    idx_page * stride_offload_cache_gpu_table_token +\
                    idx_kv_head * stride_offload_cache_gpu_table_head_kv +\
                    0 * strdie_offload_cache_gpu_table_k,
                mask=mask_keys,
                other=MAX_INT,
            )
            idx_slot_has_reference_to_bank = idx_slots != MAX_INT
            
            slot_metadata_backref_to_table = tl.load(
                OFFLOAD_CACHE_GPU_METADATA +\
                    idx_slots * stride_offload_cache_gpu_metadata_token +\
                    0 * stride_offload_cache_gpu_metadata_k,
                mask=idx_slot_has_reference_to_bank,
                other=MAX_INT,
            )
            idx_slot_is_valid_link = (
                slot_metadata_backref_to_table == idx_page
            ) & idx_slot_has_reference_to_bank
            
            slot_metadata_ref_to_uvm = tl.load(
                OFFLOAD_CACHE_GPU_METADATA +\
                    idx_slots * stride_offload_cache_gpu_metadata_token +\
                    1 * stride_offload_cache_gpu_metadata_k,
                mask=idx_slot_is_valid_link,
                other=MAX_INT,
            )
            slot_metadata_token_gen = tl.load(
                OFFLOAD_CACHE_GPU_METADATA +\
                    idx_slots * stride_offload_cache_gpu_metadata_token +\
                    2 * stride_offload_cache_gpu_metadata_k,
                mask=idx_slot_is_valid_link,
                other=MAX_INT,
            )
            idx_slot_is_valid_link = (
                slot_metadata_ref_to_uvm != MAX_INT
            ) & idx_slot_is_valid_link
            
            uvm_metadata_token_gen = tl.load(
                OFFLOAD_CACHE_UVM_METADATA +\
                    slot_metadata_ref_to_uvm * stride_offload_cache_uvm_metadata_token +\
                    0 * stride_offload_cache_uvm_metadata_k,
                mask=idx_slot_is_valid_link,
                other=MAX_INT
            )
            
            mask_slot_cache_hit = (
                uvm_metadata_token_gen != MAX_INT
            ) & (
                uvm_metadata_token_gen == slot_metadata_token_gen
            ) & idx_slot_is_valid_link
            
            idx_hid_cached = idx_hid
            if OFFLOAD_CACHE_LOAD_VALUE:
                idx_hid_cached += BLOCK_HID
            keys_cached = tl.load(
                OFFLOAD_CACHE_GPU_BANK +\
                    idx_slots * stride_offload_cache_gpu_bank_token +\
                    idx_hid * stride_offload_cache_gpu_bank_hid,
                mask=mask_slot_cache_hit,
                other=0,
            )
            
            mask_keys = mask_keys & (~mask_slot_cache_hit)
        
        keys = tl.load(
            K_CACHE +\
                idx_page.to(tl.int64) * stride_k_cache_page +\
                offset_page.to(tl.int64) * stride_k_cache_offset +\
                idx_kv_head.to(tl.int64) * stride_k_cache_kv_head +\
                idx_hid.to(tl.int64) * stride_k_cache_hid,
            mask=mask_keys,
            other=0.0,
        )
        
        if USING_OFFLOAD_CACHE:
            keys = tl.where(
                mask_slot_cache_hit,
                keys_cached,
                keys,
            )
            
            tl.atomic_add(
                CACHE_MISS_COUNTER +\
                    idx_bsz.to(tl.int64) * stride_cache_miss_counter_bsz +\
                    idx_kv_head * stride_cache_miss_counter_head_kv +\
                    idx_page * stride_cache_miss_counter_tsrc,
                mask=mask_keys,
                val=1,
            )
    
    if keys.dtype == tl.uint8:
        keys = keys.to(tl.float8e5, bitcast=True).to(tl.float16)
    if keys.dtype == tl.float8e5:
        keys = keys.to(tl.float16)
    
    return keys