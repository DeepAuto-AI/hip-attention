import triton
import triton.language as tl
import torch
from torch import Tensor
import math
from typing import Optional, Union, List, Tuple
from timber.models.timber_attention.attention1_block_gpu_kernel.paged_cache_vllm_compat import (
    PagedKeyCacheVllmCompat, PagedValueCacheVllmCompat
)

def next_multiple_of(x: int, multiple_by: int = 16):
    return triton.next_power_of_2(max(x, multiple_by))

@triton.jit
def _calc_prob_return_context_acc_compute(
    K, stride_k_n, stride_k_tsrc, stride_k_hid,
    V, stride_v_n, stride_v_tsrc, stride_v_hid, 
    CONTEXT_LENGTH, 
    
    queries,
    idx_n,
    idx_tsrc,
    mask_tsrc,
    idx_hid,
    mask_hid,
    idx_tdst,
    mask_tdst,
    context_length,
    acc,
    l_i,
    m_i,
    
    KV_REPEAT_INTERLEAVE,
    IS_CAUSAL,
    TDST,
    TSRC,
    
    CACHE_METHOD,
    
    VLLM_NUM_KV_HEADS,
    VLLM_BLOCK_SIZE,
    VLLM_X,
    
    stride_k_vllm_num_blocks,
    stride_k_vllm_num_kv_heads,
    stride_k_vllm_head_size_x,
    stride_k_vllm_block_size,
    stride_k_vllm_x,
    
    stride_v_vllm_num_blocks,
    stride_v_vllm_num_kv_heads,
    stride_v_vllm_head_size,
    stride_v_vllm_block_size,
    
    BLOCK_TABLES,
    stride_block_tables_num_seqs,
    stride_block_tables_max_num_blocks_per_seq,
):
    # keys := [BLOCK_HID: hid, BLOCK_BK * BLOCK_SIZE_K: tsrc]
    # queries := [BLOCK_SIZE_Q: tdst, BLOCK_HID: hid]
    # scores := [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]

    if CACHE_METHOD == 'cont':
        keys = tl.load(
            K +\
                (idx_n // KV_REPEAT_INTERLEAVE) * stride_k_n +\
                idx_tsrc[None, :] * stride_k_tsrc +\
                idx_hid[:, None] * stride_k_hid,
            mask = mask_tsrc[None, :] & mask_hid[:, None],
            other = 0,
        )
    elif CACHE_METHOD == 'vllm':
        """
        idx_block = block_tables[idx_batch, idx_tsrc // block_size]
        offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
        key = key_cache[idx_block, idx_head, :, offset_block, :].reshape(-1)
        """
        idx_batch = ((idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS).to(tl.int64)
        idx_head = ((idx_n // KV_REPEAT_INTERLEAVE) % VLLM_NUM_KV_HEADS).to(tl.int64)
        idx_block = tl.load(
            BLOCK_TABLES +\
                idx_batch * stride_block_tables_num_seqs +\
                (idx_tsrc // VLLM_BLOCK_SIZE) * stride_block_tables_max_num_blocks_per_seq,
            mask = mask_tsrc,
        ).to(tl.int64)
        offset_block = (idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)).to(tl.int64)
        
        # [BLOCK_HID: hid, BLOCK_BK: bk, BLOCK_SIZE_K_PADDED: tsrc]
        keys = tl.load(
            K +\
                idx_block[None, :] * stride_k_vllm_num_blocks +\
                idx_head * stride_k_vllm_num_kv_heads +\
                (idx_hid[:, None] // VLLM_X) * stride_k_vllm_head_size_x +\
                offset_block[None, :] * stride_k_vllm_block_size +\
                (idx_hid[:, None] % VLLM_X) * stride_k_vllm_x,
            mask = mask_tsrc[None, :] & mask_hid[:, None],
            other = 0,
        )
    else:
        raise Exception()
    
    if keys.dtype == tl.uint8:
        keys = keys.to(tl.float8e5, bitcast=True).to(queries.dtype)
    
    qk = tl.dot(queries, keys).to(tl.float32) * 1.44269504
    
    if IS_CAUSAL:
        qk += (
            (idx_tdst[:, None] + TSRC - TDST) < idx_tsrc[None, :] |
            (~(mask_tdst[:, None] & mask_tsrc[None, :]))
        ) * (-1.0e+6)
    else:
        qk += (
            ~(mask_tdst[:, None] & mask_tsrc[None, :])
        ) * (-1.0e+6)
    if CONTEXT_LENGTH is not None:
        qk += (
            (idx_tsrc[None, :] >= context_length)
        ) * (-1.0e+6)
    
    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1)[:, None])
    qk = qk - m_ij
    # [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]
    p = tl.math.exp2(qk)
    
    if IS_CAUSAL:
        p *= (
            ((idx_tdst[:, None] + TSRC - TDST) >= idx_tsrc[None, :]) &
            (mask_tdst[:, None] & mask_tsrc[None, :])
        )
    else:
        p *= (
            (mask_tdst[:, None] & mask_tsrc[None, :])
        )
    
    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    l_ij = tl.sum(p, axis=1)
    
    # -- update m_i and l_i
    alpha = tl.math.exp2(m_i - m_ij)
    # tl.device_print('ff', l_ij)
    l_i = l_i * alpha + l_ij[:, None]
    
    # -- update output accumulator --
    acc = acc * alpha
    
    if CACHE_METHOD == 'cont':
        values = tl.load(
            V +\
                (idx_n // KV_REPEAT_INTERLEAVE) * stride_v_n +\
                idx_tsrc[:, None] * stride_v_tsrc +\
                idx_hid[None, :] * stride_v_hid,
            mask = mask_tsrc[:, None] & mask_hid[None, :],
            other = 0
        )
    elif CACHE_METHOD == 'vllm':
        """
        idx_block = block_tables[idx_batch, idx_tsrc // block_size]
        offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
        value = value_cache[idx_block, idx_head, :, offset_block].reshape(-1)
        """
        idx_batch = (idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS
        idx_head = (idx_n // KV_REPEAT_INTERLEAVE) % VLLM_NUM_KV_HEADS
        
        idx_block = tl.load(
            BLOCK_TABLES +\
                idx_batch * stride_block_tables_num_seqs +\
                (idx_tsrc // VLLM_BLOCK_SIZE) * stride_block_tables_max_num_blocks_per_seq,
            mask = mask_tsrc,
            other = 0
        ).to(tl.int64)
        mask_block = (idx_tsrc // VLLM_BLOCK_SIZE) < tl.cdiv(TSRC, VLLM_BLOCK_SIZE)
        offset_block = idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)
        
        # value: [BLOCK_SIZE_PADDED: tsrc, BLOCK_HID: hid]
        values = tl.load(
            V +\
                idx_block[:, None] * stride_v_vllm_num_blocks+\
                idx_head * stride_v_vllm_num_kv_heads+\
                idx_hid[None, :].to(tl.int64) * stride_v_vllm_head_size +\
                offset_block[:, None] * stride_v_vllm_block_size,
            mask = mask_tsrc[:, None] & mask_hid[None, :] & mask_block[:, None],
            other = 0
        )
    else:
        raise Exception()
    
    if values.dtype == tl.uint8:
        values = values.to(tl.float8e5, bitcast=True).to(tl.float16)
    
    # update acc
    acc += tl.dot(p.to(values.dtype), values).to(tl.float32)
    
    # update m_i and l_i
    m_i = m_ij
    
    return acc, l_i, m_i

@triton.autotune(
    configs=[
        triton.Config(kwargs={}, num_warps=16),
        triton.Config(kwargs={}, num_warps=8),
        triton.Config(kwargs={}, num_warps=4),
        triton.Config(kwargs={}, num_warps=2),
    ],
    key=['BLOCK_HID', 'BLOCK_BK'],
    warmup=3,
    rep=50,
)
@triton.jit
def _calc_prob_return_context_compute(
    # input matrices
    Q, stride_q_n, stride_q_tdst, stride_q_hid,
    K, stride_k_n, stride_k_tsrc, stride_k_hid,
    V, stride_v_n, stride_v_tsrc, stride_v_hid,
    ATTEN_MASK, stride_atten_mask_n, stride_atten_mask_tsrc,
    
    # indices metrices
    INDICES, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    KS, stride_ks_n, stride_ks_bdst,
    
    # output matrices,
    CONTEXT, stride_context_n, stride_context_tdst, stride_context_hid,
    
    # input variables
    KV_REPEAT_INTERLEAVE, N, TDST, TSRC, HID: tl.constexpr, BDST, BSRC, BK,
    
    # vllm compat
    stride_k_vllm_num_blocks, 
    stride_k_vllm_num_kv_heads, 
    stride_k_vllm_head_size_x, 
    stride_k_vllm_block_size, 
    stride_k_vllm_x,
    
    stride_v_vllm_num_blocks,
    stride_v_vllm_num_kv_heads,
    stride_v_vllm_head_size,
    stride_v_vllm_block_size,
    
    BLOCK_TABLES,
    stride_block_tables_num_seqs,
    stride_block_tables_max_num_blocks_per_seq,
    
    CONTEXT_LENGTH,
    stride_context_length_num_seqs,
    
    VLLM_NUM_BLOCKS,
    VLLM_NUM_KV_HEADS,
    VLLM_HEAD_SIZE_X,
    VLLM_BLOCK_SIZE: tl.constexpr,
    VLLM_X: tl.constexpr,
    VLLM_HEAD_SIZE,
    
    # sliding window support
    USING_SLIDING_WINDOW: tl.constexpr,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    
    SLIDING_WINDOW_MASK,
    stride_sliding_window_mask_n,
    stride_sliding_window_mask_bdst,
    stride_sliding_window_mask_tsrc,
    
    # block constant
    CACHE_METHOD: tl.constexpr,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    pid_n = pid // BDST
    pid_bdst = pid % BDST
    
    # pid_n = tl.program_id(0).to(tl.int64)
    # pid_bdst = tl.program_id(1).to(tl.int64)
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED).to(tl.int64)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    
    idx_n = pid_n
    
    idx_bdst = pid_bdst
    idx_tdst = (idx_block_q + idx_bdst * BLOCK_SIZE_Q).to(tl.int64)
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    idx_hid = tl.arange(0, BLOCK_HID)
    if BLOCK_HID != HID:
        mask_hid = idx_hid < HID
    else:
        mask_hid = True
    
    if CONTEXT_LENGTH is not None:
        context_length = tl.load(
            CONTEXT_LENGTH +\
                ((idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS) * stride_context_length_num_seqs,
        )
    else:
        context_length = None
    
    # TODO replace to read from global memory
    tsrc = TSRC - (idx_bdst * BLOCK_SIZE_Q + BLOCK_SIZE_Q - 1)
    idx_sliding_tsrc_start = tl.maximum(0, tsrc - SLIDING_WINDOW_SIZE)
    idx_sliding_tsrc_end = tl.minimum(tsrc, idx_sliding_tsrc_start + SLIDING_WINDOW_SIZE)
    
    ks = tl.load(
        KS + \
            idx_n * stride_ks_n +
            idx_bdst * stride_ks_bdst,
    ).to(tl.int64)
    
    acc = tl.zeros((BLOCK_SIZE_Q_PADDED, BLOCK_HID), dtype=tl.float32)
    # scores_rowmax_state: [BLOCK_SIZE_Q: tdst, 1: tsrc]
    m_i = tl.full((BLOCK_SIZE_Q_PADDED, 1), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_SIZE_Q_PADDED, 1), 1.0, dtype=tl.float32)
    
    queries = tl.load(
        Q +\
            idx_n * stride_q_n +\
            idx_tdst[:, None] * stride_q_tdst +\
            idx_hid[None, :] * stride_q_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        other = 0
    )
    
    # perform main flash attention
    for idx_bbk in range(tl.cdiv(ks, BLOCK_BK)):
        idx_bk = (tl.arange(0, BLOCK_BK) + idx_bbk * BLOCK_BK).to(tl.int64)
        mask_bk = (idx_bk < ks) & (idx_bk < BK)
        
        # [BLOCK_BK,]
        idx_tsrc_block_start = tl.load(
            INDICES +\
                idx_n * stride_indices_n +\
                idx_bdst * stride_indices_bdst +\
                idx_bk * stride_indices_bk,
            mask = mask_bk,
            other = TSRC,
        ).to(tl.int64)
        
        # [BLOCK_BK, BLOCK_SIZE_K]
        idx_tsrc = tl.arange(0, BLOCK_SIZE_K)[None, :].to(tl.int64) + idx_tsrc_block_start[:, None]
        mask_tsrc = (idx_tsrc < TSRC) & mask_bk[:, None]
        if CONTEXT_LENGTH is not None:
            mask_tsrc = mask_tsrc & (idx_tsrc < context_length)
        
        # [BLOCK_BK * BLOCK_SIZE_K; multiple of 16]
        idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K,))
        mask_tsrc = tl.reshape(mask_tsrc, (BLOCK_BK * BLOCK_SIZE_K,))
        
        if USING_SLIDING_WINDOW:
            # submit mask
            offset_to_submit = idx_tsrc - idx_sliding_tsrc_start
            mask_to_submit = (idx_tsrc >= idx_sliding_tsrc_start) & (idx_tsrc < idx_sliding_tsrc_end)
            tl.store(
                SLIDING_WINDOW_MASK +\
                    idx_n * stride_sliding_window_mask_n +\
                    idx_bdst * stride_sliding_window_mask_bdst +\
                    offset_to_submit * stride_sliding_window_mask_tsrc,
                mask = mask_to_submit,
                value = 1,
            )
            tl.debug_barrier()
        
        acc, l_i, m_i = _calc_prob_return_context_acc_compute(
            K, stride_k_n, stride_k_tsrc, stride_k_hid,
            V, stride_v_n, stride_v_tsrc, stride_v_hid, 
            CONTEXT_LENGTH, 
            
            queries,
            idx_n,
            idx_tsrc,
            mask_tsrc,
            idx_hid,
            mask_hid,
            idx_tdst,
            mask_tdst,
            context_length,
            acc,
            l_i,
            m_i,
            
            KV_REPEAT_INTERLEAVE,
            IS_CAUSAL,
            TDST,
            TSRC,
            
            CACHE_METHOD,
            
            VLLM_NUM_KV_HEADS,
            VLLM_BLOCK_SIZE,
            VLLM_X,
            
            stride_k_vllm_num_blocks,
            stride_k_vllm_num_kv_heads,
            stride_k_vllm_head_size_x,
            stride_k_vllm_block_size,
            stride_k_vllm_x,
            
            stride_v_vllm_num_blocks,
            stride_v_vllm_num_kv_heads,
            stride_v_vllm_head_size,
            stride_v_vllm_block_size,
            
            BLOCK_TABLES,
            stride_block_tables_num_seqs,
            stride_block_tables_max_num_blocks_per_seq,
        )
    
    # perform longformer flash attention
    if USING_SLIDING_WINDOW:
        for idx_slide_block in range(tl.cdiv(SLIDING_WINDOW_SIZE, BLOCK_BK * BLOCK_SIZE_K)):
            idx_sliding = tl.arange(0, BLOCK_BK * BLOCK_SIZE_K) + idx_slide_block * (BLOCK_BK * BLOCK_SIZE_K)
            mask_sliding = idx_sliding < SLIDING_WINDOW_SIZE

            idx_tsrc = idx_sliding + idx_sliding_tsrc_start
            mask_tsrc = (idx_tsrc < TSRC) & (~tl.load(
                SLIDING_WINDOW_MASK +\
                    idx_n * stride_sliding_window_mask_n +\
                    idx_bdst * stride_sliding_window_mask_bdst +\
                    idx_sliding * stride_sliding_window_mask_tsrc,
                mask = mask_sliding,
                other = 1,
            ).to(tl.int1))
            if CONTEXT_LENGTH is not None:
                mask_tsrc = mask_tsrc & (idx_tsrc < context_length)
            
            acc, l_i, m_i = _calc_prob_return_context_acc_compute(
                K, stride_k_n, stride_k_tsrc, stride_k_hid,
                V, stride_v_n, stride_v_tsrc, stride_v_hid, 
                CONTEXT_LENGTH, 
                
                queries,
                idx_n,
                idx_tsrc,
                mask_tsrc,
                idx_hid,
                mask_hid,
                idx_tdst,
                mask_tdst,
                context_length,
                acc,
                l_i,
                m_i,
                
                KV_REPEAT_INTERLEAVE,
                IS_CAUSAL,
                TDST,
                TSRC,
                
                CACHE_METHOD,
                
                VLLM_NUM_KV_HEADS,
                VLLM_BLOCK_SIZE,
                VLLM_X,
                
                stride_k_vllm_num_blocks,
                stride_k_vllm_num_kv_heads,
                stride_k_vllm_head_size_x,
                stride_k_vllm_block_size,
                stride_k_vllm_x,
                
                stride_v_vllm_num_blocks,
                stride_v_vllm_num_kv_heads,
                stride_v_vllm_head_size,
                stride_v_vllm_block_size,
                
                BLOCK_TABLES,
                stride_block_tables_num_seqs,
                stride_block_tables_max_num_blocks_per_seq,
            )
    
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i
    tl.store(
        CONTEXT +\
            idx_n * stride_context_n +\
            idx_tdst[:, None] * stride_context_tdst +\
            idx_hid[None, :] * stride_context_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        value = acc.to(CONTEXT.type.element_ty)
    )
    

def calc_prob_return_context(
    # input matrices
    queries: Tensor, 
    keys: Union[Tensor, "PagedKeyCacheVllmCompat"], 
    values: Union[Tensor, "PagedValueCacheVllmCompat"], 
    attention_mask: Optional[Tensor],
    # indices metrices
    indices: Tensor, ks: Tensor,
    # block constant
    KV_REPEAT_INTERLEAVE: int,
    BLOCK_SIZE_Q: int,
    BLOCK_SIZE_K: int,
    IS_CAUSAL: bool,
    USING_SLIDING_WINDOW: bool,
    SLIDING_WINDOW_SIZE: int,
):
    """
    implement flash attention 1, not 2.
    """
    
    N, TDST, HID = queries.shape
    _N, TSRC, HID = keys.shape
    assert keys.shape == values.shape
    assert attention_mask is None or attention_mask.shape == (N, TDST)
    
    BSRC = triton.cdiv(TSRC, BLOCK_SIZE_K)
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    _, _, BK = indices.shape
    assert ks.shape == (N, BDST)
    
    # BLOCK_BK = max(1, 256 // BLOCK_SIZE_K)
    # BLOCK_BK = max(1, triton.next_power_of_2(BK) // 2)
    BLOCK_BK = triton.cdiv(64 if queries.dtype == torch.float32 else 128, BLOCK_SIZE_K)
    # print(256 // BLOCK_SIZE_K, BK)
    BLOCK_HID = triton.next_power_of_2(HID)
    BLOCK_SIZE_Q_PADDED = next_multiple_of(BLOCK_SIZE_Q, 16)
    
    # print(BK, BLOCK_BK)
    
    assert values.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.uint8]
    context = torch.zeros(
        (N, TDST, HID),
        dtype=queries.dtype,
        device=queries.device,
    )
    
    if isinstance(keys, Tensor) and isinstance(values, Tensor):
        CACHE_METHOD = 'cont'
        
        VLLM_NUM_BLOCKS =\
        VLLM_NUM_KV_HEADS =\
        VLLM_HEAD_SIZE_X =\
        VLLM_BLOCK_SIZE =\
        VLLM_X =\
        VLLM_HEAD_SIZE = 0
        
        vllm_keys_strides = (0, 0, 0, 0, 0)
        vllm_values_strides = (0, 0, 0, 0)
        
        block_tables = keys
        block_tables_strides = (0, 0)
        
        context_length = None
        context_length_strides = (0, )
    elif isinstance(keys, PagedKeyCacheVllmCompat) and isinstance(values, PagedValueCacheVllmCompat):
        """
        vLLM compatible paged attention
        
        q: [num_seqs, num_heads, head_size]
        k: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        v: [num_blocks, num_kv_heads, head_size, block_size]
        block_tables: [num_seqs, max_num_blocks_per_seq]
        context_lens: [num_seqs]
        """
        
        CACHE_METHOD = 'vllm'
        
        (
            VLLM_NUM_BLOCKS,
            VLLM_NUM_KV_HEADS, 
            VLLM_HEAD_SIZE_X,
            VLLM_BLOCK_SIZE,
            VLLM_X,
        ) = keys.key_cache.shape
        VLLM_HEAD_SIZE = VLLM_HEAD_SIZE_X * VLLM_X
        
        block_tables = keys.block_table
        block_tables_strides = block_tables.stride()
        assert len(block_tables_strides) == 2
        
        context_length = keys.context_length
        context_length_strides = context_length.stride()
        assert len(context_length_strides) == 1
        
        vllm_keys_strides = keys.key_cache.stride()
        assert len(vllm_keys_strides) == 5
        
        vllm_values_strides = values.value_cache.stride()
        assert len(vllm_values_strides) == 4
    else:
        raise Exception("not supported")
    
    if USING_SLIDING_WINDOW:
        sliding_window_mask = torch.zeros(
            (N, BDST, SLIDING_WINDOW_SIZE), 
            dtype=torch.int16, 
            device=queries.device
        )
        sliding_window_mask_strides = sliding_window_mask.stride()
    else:
        sliding_window_mask = None
        sliding_window_mask_strides = (0, 0, 0)
    assert len(sliding_window_mask_strides) == 3
    
    # grid = (N, BDST, )
    grid = (N * BDST, )
    
    assert attention_mask is None, "attention mask is not supported yet"
    assert queries.ndim == 3
    assert keys.ndim == 3
    assert values.ndim == 3
    assert attention_mask is None or attention_mask.ndim == 3
    assert indices.ndim == 3
    assert ks.ndim == 2
    assert context.ndim == 3
    
    # print(f'{queries.data_ptr():X}')
    # print(f'{keys.data_ptr():X} {vllm_keys_strides}')
    # print(f'{values.data_ptr():X} {vllm_values_strides}')
    # print(f'{context.data_ptr():X}')
    # print(f'{block_tables.data_ptr():X} {block_tables_strides}')
    # print(f'{context_length.data_ptr():X} {sliding_window_mask_strides}')
    # print(f'{sliding_window_mask.data_ptr():X} {sliding_window_mask_strides}')
    
    _calc_prob_return_context_compute[grid](
        queries, *queries.stride(),
        keys, *keys.stride(),
        values, *values.stride(),
        attention_mask, *((0, 0) if attention_mask is None else attention_mask.stride()),
        
        indices, *indices.stride(),
        ks, *ks.stride(),
        
        context, *context.stride(),
        
        KV_REPEAT_INTERLEAVE, 
        N, 
        TDST, 
        TSRC, 
        HID, 
        BDST, 
        BSRC, 
        BK,
        
        # vllm key value cache compat
        *vllm_keys_strides,
        *vllm_values_strides,
        
        block_tables,
        *block_tables_strides,
        
        context_length,
        *context_length_strides,
        
        VLLM_NUM_BLOCKS,
        VLLM_NUM_KV_HEADS,
        VLLM_HEAD_SIZE_X,
        VLLM_BLOCK_SIZE,
        VLLM_X,
        VLLM_HEAD_SIZE,
        
        # sliding window support
        USING_SLIDING_WINDOW,
        SLIDING_WINDOW_SIZE,
        sliding_window_mask,
        *sliding_window_mask_strides,
        
        CACHE_METHOD,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_Q_PADDED, 
        BLOCK_SIZE_K,
        BLOCK_HID,
        BLOCK_BK,
        IS_CAUSAL,
        
        # num_warps=8,
        # num_stages=2,
    )
    
    return context