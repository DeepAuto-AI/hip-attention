"""
block version of attention1
score = reduce_fn(score[block_ptrs])

k = 256 (16 block)
scale_up = 2

# infer by heuristics
n_patches = 128 (8 block)
w_start = 512 (32 block)

> example of block scale
1024: 64 block
2048: 128 block
4096: 256 block
"""

import json
import random
import gc
import warnings
from matplotlib import pyplot as plt
import numpy as np
import skimage.measure
import skimage
import torch
from torch import Tensor
import tqdm
import triton
import triton.language as tl
from typing import Literal, Optional, Tuple, List, Union
import os
import math
from torch.autograd import Function

assert (triton.__version__ in ['2.2.0', '2.1.0']) or ('nightly' in triton.__version__), triton.__version__
assert hasattr(tl, 'sort'), f'check triton version {triton.__version__}'

from timber.utils import get_bench, seed
from timber.models.timber_attention.common import load_checkouts
from timber.models.timber_attention.attention1_block_gpu_kernel.paged_cache_vllm_compat import (
    PagedKeyCacheVllmCompat, PagedValueCacheVllmCompat
)
from timber.models.timber_attention.attention1_block_gpu_kernel.masking_iteration import masking_iteration
from timber.models.timber_attention.attention1_block_gpu_kernel.safe_indices import safe_indices
from timber.models.timber_attention.attention1_block_gpu_kernel.calc_prob_return_context import calc_prob_return_context

timer = lambda x: get_bench().region(x)

DEBUG = os.environ.get('TIMBER_DEBUG', '0') == '1'

def next_multiple_of(x: int, multiple_by: int = 16):
    return triton.next_power_of_2(max(x, multiple_by))

@triton.jit
def _calc_score_compute(
    # input matrix
    QUERIES, stride_queries_n, stride_queries_tdst, stride_queries_hid,
    KEYS, stride_keys_n, stride_keys_tsrc, stride_keys_hid,
    ATTEN_MASK, stride_atten_mask_n, stride_atten_mask_tsrc,
    
    # block indices
    INDICES, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    KS, stride_ks_n, stride_ks_bdst,
    
    # out matrix
    SCORES, stride_scores_n, stride_scores_tdst, stride_scores_k,
    
    # input variables
    KV_REPEAT_INTERLEAVE, N, TDST, TSRC, HID, BK, K, BDST, BSRC, IS_CAUSAL,
    
    # vllm key cache compat
    stride_keys_vllm_num_bocks,
    stride_keys_vllm_num_kv_heads,
    stride_keys_vllm_head_size_x,
    stride_keys_vllm_block_size,
    stride_keys_vllm_x,
    
    VLLM_NUM_BLOCKS,
    VLLM_NUM_KV_HEADS,
    VLLM_HEAD_SIZE_X,
    VLLM_BLOCK_SIZE,
    VLLM_X,
    VLLM_HEAD_SIZE,
    
    BLOCK_TABLES,
    stride_block_tables_num_seqs,
    stride_block_tables_max_num_blocks_per_seq,
    
    # kernel constatnts
    KEY_CACHE_METHOD: tl.constexpr,
    BLOCK_BK: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    idx_n = tl.program_id(0).to(tl.int64)
    idx_bdst = tl.program_id(1).to(tl.int64)
    pid_bk = tl.program_id(2).to(tl.int64)
    
    ks = tl.load(
        KS +\
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst,
    )
    
    # if (pid_bk + 1) * BLOCK_BK > ks:
    #     return
    
    idx_bk = tl.arange(0, BLOCK_BK) + pid_bk * BLOCK_BK
    mask_bk = idx_bk < ks
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_tsrc = tl.load(
        INDICES +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
        mask = mask_bk,
    )
    # [BLOCK_BK: bk, BLOCK_SIZE_K_PADDED]
    idx_tsrc = idx_tsrc[:, None] + idx_block_k[None, :]
    mask_tsrc = (idx_tsrc < TSRC) & mask_block_k[None, :] & mask_bk[:, None]
    
    # [BLOCK_BK: bk, BLOCK_SIZE_K_PADDED]
    if ATTEN_MASK is not None:
        key_mask = tl.load(
            ATTEN_MASK +\
                idx_n * stride_atten_mask_n +\
                idx_tsrc * stride_atten_mask_tsrc,
            mask = mask_tsrc,
            other = False,
        ).to(tl.int1)
        mask_tsrc = mask_tsrc & key_mask
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    if ATTEN_MASK is not None:
        query_mask = tl.load(
            ATTEN_MASK +\
                idx_n * stride_atten_mask_n +\
                (idx_tdst + TSRC - TDST) * stride_atten_mask_tsrc,
            mask = mask_tdst,
            other = False,
        ).to(tl.int1)
        mask_tdst = mask_tdst & query_mask
    
    # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_BK: bk, BLOCK_SIZE_K_PADDED: tsrc]
    scores = tl.zeros((BLOCK_SIZE_Q_PADDED, BLOCK_BK, BLOCK_SIZE_K_PADDED), dtype=tl.float32)
    for pid_hid in range(tl.cdiv(HID, BLOCK_HID)):
        idx_hid = (tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID).to(tl.int64)
        mask_hid = idx_hid < HID
        
        # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_HID: hid]
        queries = tl.load(
            QUERIES +\
                idx_n * stride_queries_n +\
                idx_tdst[:, None] * stride_queries_tdst +\
                idx_hid[None, :] * stride_queries_hid,
            mask = mask_tdst[:, None] & mask_hid[None, :],
            other = 0
        )
        
        if KEY_CACHE_METHOD == 'cont':
            # [BLOCK_HID: hid, BLOCK_BK: bk, BLOCK_SIZE_K_PADDED: tsrc]
            keys = tl.load(
                KEYS +\
                    (idx_n // KV_REPEAT_INTERLEAVE) * stride_keys_n +\
                    idx_tsrc[None, :, :] * stride_keys_tsrc +\
                    idx_hid[:, None, None] * stride_keys_hid,
                mask = mask_tsrc[None, :, :] & mask_hid[:, None, None],
                other = 0
            )
        elif KEY_CACHE_METHOD == 'vllm':
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
                KEYS +\
                    idx_block[None, :, :] * stride_keys_vllm_num_bocks +\
                    idx_head * stride_keys_vllm_num_kv_heads +\
                    (idx_hid[:, None, None] // VLLM_X) * stride_keys_vllm_head_size_x +\
                    offset_block[None, :, :] * stride_keys_vllm_block_size +\
                    (idx_hid[:, None, None] % VLLM_X) * stride_keys_vllm_x,
                mask = mask_tsrc[None, :, :] & mask_hid[:, None, None],
                other = 0,
            )
        else:
            raise Exception()
        keys = tl.reshape(keys, (BLOCK_HID, BLOCK_BK * BLOCK_SIZE_K_PADDED))
        
        # TOOD: WIP
        
        if keys.dtype == tl.uint8:
            keys = keys.to(tl.float8e5, bitcast=True).to(queries.dtype)
        scores_mini = tl.dot(queries, keys)
        scores_mini = tl.reshape(scores_mini, (BLOCK_SIZE_Q_PADDED, BLOCK_BK, BLOCK_SIZE_K_PADDED))
        
        scores += scores_mini.to(scores.dtype)
    
    idx_scorek = (idx_bk[:, None] * BLOCK_SIZE_K + idx_block_k[None, :])
    mask_scorek = (idx_scorek < K) & mask_block_k[None, :] & mask_bk[:, None]
    
    scores_mask = (
        (mask_tdst[:, None, None] & mask_tsrc[None, :, :]) &
        mask_scorek[None, :] &
        True
    )
    
    if IS_CAUSAL:
        scores_mask = scores_mask & ((idx_tdst[:, None, None] + (TSRC - TDST)) >= idx_tsrc[None, :, :])
    
    tl.store(
        SCORES +\
            idx_n * stride_scores_n +\
            idx_tdst[:, None, None] * stride_scores_tdst +\
            idx_scorek[None, :, :] * stride_scores_k,
        mask = scores_mask,
        value = scores,
    )

@triton.jit
def _calc_score_compute_bwd_queries(
    # input matrices
    KS, stride_ks_n, stride_ks_bdst,
    INDICES, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    KEYS, stride_keys_n, stride_keys_tsrc, stride_keys_hid,
    
    # grad output (read)
    GRAD_SCORES, stride_grad_scores_n, stride_grad_scores_tdst, stride_grad_scores_k,
    
    # grad input (write)
    GRAD_QUERIES, stride_grad_queries_n, stride_grad_queries_tdst, stride_grad_queries_hid,
    
    # input variables
    N, TDST, TSRC, HID, BK, K,
    
    # block constant
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    """
    ks: int[N, TDST]
    indices: int[N, TDST, K]
    keys: fp[N, TSRC, HID]
    grad_scores: fp[N, TDST, K]
    grad_queries: fp[N, TDST, HID]
    -----
    foreach n in [..N]
    foreach tdst in [..TDST]
    
    scalar_ks = ks[n, tdst]
    
    acc = zeros(HID)
    for k in [..K]:
        idx_tsrc = indices[n, tdst, k]
        mask_tsrc = idx_tsrc < T_SRC & k < scalar_ks
        acc += grad_scores[n, tdst, k] * keys[n, idx_tsrc, :]
    grad_queries[n, tdst, :] = acc
    """
    
    idx_n = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    
    scalar_ks = tl.load(
        KS +\
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst
    )
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_tdst = (idx_bdst * BLOCK_SIZE_Q + idx_block_q)
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    
    accumulator = tl.zeros((BLOCK_SIZE_Q_PADDED, BLOCK_HID,), dtype=tl.float32)
    for idx_bk in range(BK):
        idx_tsrc = tl.load(
            INDICES + \
                idx_n * stride_indices_n + \
                idx_bdst * stride_indices_bdst + \
                idx_bk * stride_indices_bk,
        )
        
        idx_tsrc = idx_tsrc + idx_block_k
        mask_tsrc = (idx_tsrc < TSRC) & mask_block_k & (idx_tsrc < scalar_ks)
        
        idx_k = idx_bk * BLOCK_SIZE_K + idx_block_k
        mask_k = (idx_k < K) & mask_block_k
        
        # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_SIZE_K_PADDED: score]
        grad_score = tl.load(
            GRAD_SCORES +\
                idx_n * stride_grad_scores_n +\
                idx_tdst[:, None] * stride_grad_scores_tdst + \
                idx_k[None, :] * stride_grad_scores_k,
            mask = mask_tdst[:, None] & (mask_tsrc & mask_k)[None, :],
            other = 0,
        )
        
        # [BLOCK_SIZE_K_PADDED: score, BLOCK_HID: hid]
        key = tl.load(
            KEYS +\
                idx_n * stride_keys_n +\
                idx_tsrc[:, None] * stride_keys_tsrc +\
                idx_hid[None, :] * stride_keys_hid,
            mask = mask_hid[None, :] & (mask_tsrc & mask_k)[:, None],
            other = 0
        )
        
        # tl.device_print("", idx_tsrc)
        accumulator += tl.dot(grad_score, key).to(accumulator.dtype)
    
    tl.store(
        GRAD_QUERIES +\
            idx_n * stride_grad_queries_n +\
            idx_tdst[:, None] * stride_grad_queries_tdst +\
            idx_hid[None, :] * stride_grad_queries_hid,
        mask = mask_hid[None, :] & mask_tdst[:, None],
        value = accumulator
    )

@triton.jit
def _calc_score_compute_bwd_keys(
    # input matrices
    ks, stride_ks_n, stride_ks_bdst,
    indices, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    queries, stride_queries_n, stride_queries_tdst, stride_queries_hid,
    # grad output (read)
    grad_scores, stride_grad_scores_n, stride_grad_scores_tdst, stride_grad_scores_k,
    # grad input (write)
    grad_keys, stride_grad_keys_n, stride_grad_keys_tsrc, stride_grad_keys_hid,
    # input variables
    N, TDST, TSRC, HID, BK, K,
    # block constant
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    """
    indices: int[N, TDST, K]
    ks: int[N, TDST, K]
    queries: int[N, TDST, HID]
    grad_scores: fp[N, TDST, K]
    grad_keys: fp[N, TSRC, HID]
    -----
    foreach n in [..N]
    foreach tdst in [..TDST]
    foreach k in [..K]
    
    scalar_ks = ks[n, tdst]
    if k >= scalar_ks: return
    
    grad_keys[n, indices[n, tdst, k], hid] +=(atomic)
        grad_scores[n, tdst, k] * queries[n, tdst, :]
    """
    idx_n = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_bk = tl.program_id(2)
    
    scalar_ks = tl.load(
        ks +\
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst,
    )
    # mask_job = idx_bk < scalar_ks
    if idx_bk >= scalar_ks: return
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = (idx_hid < HID)
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    idx_k = idx_bk * BLOCK_SIZE_K + idx_block_k
    mask_k = (idx_k < K) & mask_block_k
    
    # [BLOCK_SIZE_K_PADDED: tsrc, BLOCK_SIZE_Q_PADDED: tdst]
    grad_score = tl.load(
        grad_scores +\
            idx_n * stride_grad_scores_n +\
            idx_tdst[None, :] * stride_grad_scores_tdst +\
            idx_k[:, None] * stride_grad_scores_k,
        mask = mask_tdst[None, :] & mask_k[:, None],
        other = 0
    )
    # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_HID: hid]
    query = tl.load(
        queries +\
            idx_n * stride_queries_n +\
            idx_tdst[:, None] * stride_queries_tdst +\
            idx_hid[None, :] * stride_queries_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        other = 0,
    )
    # [BLOCK_SIZE_K_PADDED: tsrc, BLOCK_HID: hid]
    scores = tl.dot(grad_score, query)
    
    idx_tsrc = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
    )
    idx_tsrc = idx_tsrc + idx_block_k
    mask_tsrc = (idx_tsrc < TSRC) & mask_block_k
    tl.atomic_add(
        grad_keys +\
            idx_n * stride_grad_keys_n +\
            idx_tsrc[:, None] * stride_grad_keys_tsrc +\
            idx_hid[None, :] * stride_grad_keys_hid,
        val = scores,
        mask = mask_tsrc[:, None] & mask_hid[None, :]
    )

# NOTE: you have to perform softmax after this
class CalcScoreAutoGradFn(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx, 
        # matrices
        queries: Tensor, keys: Union[Tensor, "PagedKeyCacheVllmCompat"], attention_mask: Tensor,
        # indices matrices
        indices: Tensor, ks: Tensor,
        # block constant
        KV_REPEAT_INTERLEAVE: int,
        BLOCK_SIZE_Q: int,
        BLOCK_SIZE_K: int,
        IS_CAUSAL: bool
    ):
        ctx.save_for_backward(queries, keys, indices, ks)
        ctx.BLOCK_SIZE_Q = BLOCK_SIZE_Q
        ctx.BLOCK_SIZE_K = BLOCK_SIZE_K
        
        N, TDST, HID = queries.shape
        _N, TSRC, _ = keys.shape
        _, _, BK = indices.shape
        
        BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
        BSRC = triton.cdiv(TSRC, BLOCK_SIZE_K)
        
        assert keys.shape == (_N, TSRC, HID)
        assert indices.shape == (N, BDST, BK)
        assert ks.shape == (N, BDST)
        
        K = BK * BLOCK_SIZE_K
        scores = torch.full(
            (N, TDST, K), 
            torch.finfo(queries.dtype).min,
            device=queries.device, 
            dtype=queries.dtype
        )
        
        BLOCK_SIZE_Q_PADDED = next_multiple_of(BLOCK_SIZE_Q, 16)
        BLOCK_SIZE_K_PADDED = next_multiple_of(BLOCK_SIZE_K, 1)
        BLOCK_BK = next_multiple_of(128 // BLOCK_SIZE_K_PADDED, 1)
        # BLOCK_BK = 1
        BLOCK_HID = triton.next_power_of_2(HID)
        # BLOCK_HID = max(BLOCK_SIZE_Q_PADDED, BLOCK_SIZE_K_PADDED)
        BLOCK_HID = 32
        
        if isinstance(keys, Tensor):
            KEY_CACHE_METHOD = 'cont'
            
            VLLM_NUM_BLOCKS =\
            VLLM_NUM_KV_HEADS =\
            VLLM_HEAD_SIZE_X =\
            VLLM_BLOCK_SIZE =\
            VLLM_X =\
            VLLM_HEAD_SIZE = 0
            
            vllm_keys_strides = (0, 0, 0, 0, 0)
            
            block_tables = keys
            block_tables_strides = (0, 0)
        elif isinstance(keys, PagedKeyCacheVllmCompat):
            """
            vLLM compatible paged attention
            
            q: [num_seqs, num_heads, head_size]
            k: [num_blocks, num_kv_heads, head_size/x, block_size, x]
            v: [num_blocks, num_kv_heads, head_size, block_size]
            block_tables: [num_seqs, max_num_blocks_per_seq]
            context_lens: [num_seqs]
            """
            
            KEY_CACHE_METHOD = 'vllm'
            
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
            
            vllm_keys_strides = keys.key_cache.stride()
            assert len(vllm_keys_strides) == 5            
        else:
            raise Exception()
        
        grid = (N, BDST, triton.cdiv(BK, BLOCK_BK))
        
        # print(grid)
        
        assert queries.ndim == 3
        assert keys.ndim == 3
        if attention_mask is not None:
            assert attention_mask.ndim == 2
            assert attention_mask.dtype == torch.bool
        assert indices.ndim == 3
        assert ks.ndim == 2
        assert scores.ndim == 3
        with timer("_calc_score_compute"):
            _calc_score_compute[grid](
                # input matrix
                queries, *queries.stride(),
                keys, *keys.stride(),
                attention_mask, *(attention_mask.stride() if attention_mask is not None else (0, 0)),
                
                # block indices
                indices, *indices.stride(),
                ks, *ks.stride(),
                
                # out matrix
                scores, *scores.stride(),
                
                # input variables
                KV_REPEAT_INTERLEAVE, 
                N, 
                TDST, 
                TSRC, 
                HID, 
                BK, 
                K, 
                BDST, 
                BSRC, 
                IS_CAUSAL,
                
                # vllm key cache compat
                *vllm_keys_strides,
                VLLM_NUM_BLOCKS,
                VLLM_NUM_KV_HEADS,
                VLLM_HEAD_SIZE_X,
                VLLM_BLOCK_SIZE,
                VLLM_X,
                VLLM_HEAD_SIZE,
                
                block_tables, *block_tables_strides,
                
                # kernel constatnts
                KEY_CACHE_METHOD,
                BLOCK_BK,
                BLOCK_SIZE_Q,
                BLOCK_SIZE_Q_PADDED,
                BLOCK_SIZE_K,
                BLOCK_SIZE_K_PADDED,
                BLOCK_HID,
                
                num_warps=4,
                num_stages=2,
                enable_warp_specialization=False,
            )
            
        # print(scores[0, 300, :])
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        ENABLED = True
        
        queries, keys, indices, ks = ctx.saved_tensors
        BLOCK_SIZE_Q = ctx.BLOCK_SIZE_Q
        BLOCK_SIZE_K = ctx.BLOCK_SIZE_K
        grad_queries = grad_keys = None
        
        N, T_DST, HID = queries.shape
        _, T_SRC, _HID = keys.shape
        assert HID == _HID
        _, _, BK = indices.shape
        _, _, K = grad_scores.shape

        # for queries
        if ctx.needs_input_grad[0]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE_Q))
            BLOCK_HID = triton.next_power_of_2(HID)
            
            grad_queries = torch.zeros_like(queries)
            
            if ENABLED:
                assert ks.ndim == 2
                assert indices.ndim == 3
                assert keys.ndim == 3
                assert grad_scores.ndim == 3
                assert  grad_queries.ndim == 3
                
                _calc_score_compute_bwd_queries[grid](
                    ks, ks.stride(0), ks.stride(1),
                    indices, indices.stride(0), indices.stride(1), indices.stride(2), 
                    keys, keys.stride(0), keys.stride(1), keys.stride(2),
                    
                    grad_scores, grad_scores.stride(0), grad_scores.stride(1), grad_scores.stride(2),
                    
                    grad_queries, grad_queries.stride(0), grad_queries.stride(1), grad_queries.stride(2),
                    
                    N, T_DST, T_SRC, HID, BK, K,
                    
                    BLOCK_SIZE_Q,
                    next_multiple_of(BLOCK_SIZE_Q, 16),
                    BLOCK_SIZE_K,
                    next_multiple_of(BLOCK_SIZE_K, 16),
                    BLOCK_HID,
                )
        
        # for keys
        if ctx.needs_input_grad[1]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE_Q), BK)
            BLOCK_HID = triton.next_power_of_2(HID)
            
            grad_keys = torch.zeros_like(keys)
            
            if ENABLED:
                _calc_score_compute_bwd_keys[grid](
                    ks, ks.stride(0), ks.stride(1),
                    indices, indices.stride(0), indices.stride(1), indices.stride(2), 
                    queries, queries.stride(0), queries.stride(1), queries.stride(2),
                    
                    grad_scores, grad_scores.stride(0), grad_scores.stride(1), grad_scores.stride(2),
                    
                    grad_keys, grad_keys.stride(0), grad_keys.stride(1), grad_keys.stride(2),
                    
                    N, T_DST, T_SRC, HID, BK, K,
                    
                    BLOCK_SIZE_Q,
                    next_multiple_of(BLOCK_SIZE_Q, 16),
                    BLOCK_SIZE_K,
                    next_multiple_of(BLOCK_SIZE_K, 16),
                    BLOCK_HID,
                )
        
        return (
            grad_queries, 
            grad_keys, 
            None,
            None, 
            None, 
            None,
            None,
            None,
            None,
        )

def calc_score_return_prob(
    queries: Tensor, keys: Tensor, attention_mask: Tensor,
    indices: Tensor, ks: Tensor,
    
    KV_REPEAT_INTERLEAVE: int,
    BLOCK_SIZE_Q: int,
    BLOCK_SIZE_K: int,
    IS_CAUSAL: bool,
):
    scores = CalcScoreAutoGradFn.apply(
        queries, keys, attention_mask,
        indices, ks,
        
        KV_REPEAT_INTERLEAVE, BLOCK_SIZE_Q, BLOCK_SIZE_K, IS_CAUSAL
    ) # type: Tensor
    
    with timer("calc_score_return_prob.softmax"):
        probs = scores.softmax(-1).to(scores.dtype)
    
    assert probs.dtype == queries.dtype
    
    N, TDST, K = scores.shape
    if attention_mask is not None:
        _, TSRC = attention_mask.shape
        if probs.requires_grad:
            probs = probs * attention_mask[:, TSRC-TDST:, None]
        else:
            probs.masked_fill_(~attention_mask[:, TSRC-TDST:, None], 0)
    
    assert scores.dtype == queries.dtype
    assert probs.dtype == queries.dtype
    
    return scores, probs

def debug_print(
    w_curr,
    mask, ws, ks, N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K
):
    plt.clf()
    indices = safe_indices(mask, ws, BLOCK_SIZE_K, allow_collision=True)
    # indices = torch.clamp(indices, 0, triton.cdiv(T_SRC, BLOCK_SIZE) - 1)
    x = to_dense(
        indices.cpu().numpy(),
        ks.cpu().unsqueeze(-1).numpy(), 
        None,
        N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K,
    )[0]
    x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
    # x = np.repeat(x, BLOCK_SIZE_Q, 0)
    # x = np.repeat(x, 1, 1)
    if x.shape[0] == 1:
        x = x.repeat(32, 0)
    plt.title(f'sum:{x.sum()} (collision allowed)')
    plt.imshow(x)
    plt.colorbar()
    path = f'saves/models/timber_attention/block_{w_curr}.png'
    # path = f'saves/models/timber_attention/block.png'
    print('saved', path, N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K, x.shape)
    plt.savefig(path, dpi=96, bbox_inches='tight')

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin) 
    
    if k is not None:
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed

def attention_matrix(
    queries: Tensor, 
    keys: Tensor,
    values: Tensor,
    attention_mask: Tensor,
    kv_repeat_interleave: int,
    
    w_start: int,
    n_patches: int,
    mask_k: int,
    scale_up: int,
    is_causal: bool,
    
    BLOCK_SIZE_Q: int = 16,
    BLOCK_SIZE_K: int = 1,
    REDUCE_METHOD: Literal['first', 'max', 'sum'] = 'max',
    REDUCE_STRIDE: int = 1,
    
    SPARQ: bool = True,
    SPARQ_START_TSRC: int = 2048,
    SPARQ_START_BK: int = 128,
    SPARQ_HID: int = 32,
    SPARQ_REDUCE_METHOD: Literal['sum', 'max'] = 'sum',
    
    IS_FLASH: bool = False,
    
    # NOTE: this improve latency quite well, but hurt accuracy
    ESTIMATOR_LOWER_RESOLUTION: int = 2,
    ESTIMATOR_LOWER_RESOLUTION_STOP_N_BLOCKS: int = 512,
    
    SAMPLING_METHOD: str = 'first',
    
    USING_SLIDING_WINDOW=True,
    SLIDING_WINDOW_SIZE=256,
    
    ROPE_METHOD='none',
    ROPE_COS=None,
    ROPE_SIN=None,
    POSITION_IDS=None,
    
    SELF_EXTEND_SCALE=None,
    SELF_EXTEND_WINDOW=None,
    
    GRID_SRC_STRIDE=1,
    GRID_K_STRIDE=1,
) -> Tuple[Tensor, Tensor, Tensor]:
    global DEBUG
    
    if DEBUG:
        print('attention_matrix', queries.shape, keys.shape, w_start, n_patches, mask_k, scale_up, BLOCK_SIZE_Q, BLOCK_SIZE_K)
        os.makedirs('saves/models/timber_attention/', exist_ok=True)
    
    N, T_DST, HID = queries.shape
    _, T_SRC, _ = keys.shape
    assert T_DST <= T_SRC, f"{queries.shape}, {keys.shape}"
    
    if triton.cdiv(mask_k, BLOCK_SIZE_K) <= ESTIMATOR_LOWER_RESOLUTION_STOP_N_BLOCKS:
        ESTIMATOR_LOWER_RESOLUTION = 1
    
    if ESTIMATOR_LOWER_RESOLUTION > 1:
        mask_k = mask_k // ESTIMATOR_LOWER_RESOLUTION
        w_start = w_start // ESTIMATOR_LOWER_RESOLUTION
        n_patches = n_patches // ESTIMATOR_LOWER_RESOLUTION
    
    if SPARQ and ((mask_k // BLOCK_SIZE_K) < SPARQ_START_BK):
        SPARQ = False
    if SPARQ and (T_SRC < SPARQ_START_TSRC):
        SPARQ = False
    if ROPE_METHOD in ['self_extend']:
        # assert (mask_k // BLOCK_SIZE_K) <= 128, "oh this is bug,,, i need help"
        # SPARQ_HID = 16
        SPARQ = False
    
    if SPARQ:
        warnings.warn('sparq is enabled')
    
    dtype = queries.dtype
    device = queries.device
    # assert queries.device == keys.device
    
    assert isinstance(BLOCK_SIZE_Q, int)
    assert isinstance(BLOCK_SIZE_K, int)
    BLOCK_SIZE_Q = int(BLOCK_SIZE_Q)
    BLOCK_SIZE_K = int(BLOCK_SIZE_K)
    
    if attention_mask is not None:
        assert attention_mask.shape == (N, T_SRC)
        assert attention_mask.dtype == torch.bool
    
    # NOTE: width of last query
    w_curr = round(w_start / scale_up)
    assert w_curr <= mask_k, f'{w_curr} <= {mask_k}'
    
    with timer('matrix.setup'):
        # vectors
        tsrcs_offset = max(BLOCK_SIZE_Q, BLOCK_SIZE_K) - 1
        tsrcs = torch.arange(
            tsrcs_offset+T_SRC-T_DST+1, tsrcs_offset+T_SRC+1, BLOCK_SIZE_Q, 
            dtype=torch.int64,
            device=device,
        )\
            .view(1, -1)\
            .expand(N, -1)\
            .contiguous()
        tsrcs.clamp_max_(T_SRC)
        if not is_causal:
            tsrcs.fill_(T_SRC)
        # NOTE: store non blocked width
        ws = torch.clamp(tsrcs, 0, w_curr)
        # NOTE: store num blocks
        ks = torch.ceil(ws / (BLOCK_SIZE_K)).to(torch.int64)
        # assert tsrcs.dtype == torch.int64
        # assert ws.dtype == torch.int64
        # assert ks.dtype == torch.int64
        
        # matrices
        # NOTE: float16 -> int64 seems not possible
        """
        mask_k_block = triton.cdiv(mask_k, BLOCK_SIZE_K)
        mask = torch.arange(mask_k_block, device=device, dtype=torch.float32).view(1, 1, mask_k_block) / ks.unsqueeze(-1)
        tmask = torch.zeros(
            (mask.shape[0], mask.shape[1], mask_k_block * math.ceil(scale_up)), 
            dtype=torch.float32, 
            device=device
        )
        """
        
        B_SRC = triton.cdiv(T_SRC, BLOCK_SIZE_K)
        B_DST = triton.cdiv(T_DST, BLOCK_SIZE_Q)
        
        sparq_indices = None
        sparq_indices_strides = (1, 1, 1)
        if SPARQ:
            with timer('matrix.setup.sparq'):
                q_scale = 1 / math.sqrt(HID)
                queries_scores = queries
                if ROPE_METHOD in ['self_extend']:
                    queries_scores, _ = apply_rotary_pos_emb(
                        queries / q_scale, 
                        None, 
                        ROPE_COS, 
                        ROPE_SIN, 
                        POSITION_IDS
                    )
                    queries_scores *= q_scale
                queries_scores = queries_scores.abs()
                if T_DST > 1 and (B_DST * BLOCK_SIZE_Q) != T_DST:
                    queries_scores = F.pad(
                        queries_scores.unsqueeze(0), 
                        (0, 0, 0, B_DST * BLOCK_SIZE_Q - T_DST), 
                        value=0
                    ).squeeze(0)
                # print(queries_scores.shape, B_DST, BLOCK_SIZE_Q, T_DST, T_DST > 1 and (B_DST * BLOCK_SIZE_Q) != T_DST)
                # TODO: padding
                queries_scores = queries_scores.view(N, B_DST, -1, HID)
                if SPARQ_REDUCE_METHOD == 'sum':
                    queries_scores = queries_scores.sum(-2)
                elif SPARQ_REDUCE_METHOD == 'max':
                    queries_scores = queries_scores.max(-2)[0]
                else:
                    raise Exception()
                _, sparq_indices = torch.topk(
                    queries_scores, 
                    k=SPARQ_HID, 
                    dim=-1, 
                    sorted=True
                )
                sparq_indices = sparq_indices.to(torch.int16)
                # sparq_indices = torch.arange(0, SPARQ_HID, device=queries.device)[None, None, :].repeat(N, B_DST, 1)
                sparq_indices_strides = sparq_indices.stride()
    
    # NOTE: mask is not available from here.
    # if DEBUG:
    #     debug_print(w_curr, mask, ws, ks, N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K)
    
    # NOTE: Calc. num iteration. this should be replaced with log_base. but i am lazy haha
    n_iteration = 0
    _w_curr = w_curr
    while w_curr < T_SRC:
        w_curr = round(w_curr * scale_up)
        n_iteration += 1
    # w_curr = _w_curr
    
    n_completed = _w_curr
    with timer("iterations"):
        i_iteration = 0
        mask, ks = masking_iteration(
            # input matrices
            queries, keys, attention_mask,
            
            # input metrices (blocked) 
            # mask, tmask,
            sparq_indices, sparq_indices_strides,
            
            # temp vectors (blocked)
            ws, ks, tsrcs, 
            
            # operator variables
            scale_up,
            triton.cdiv(n_patches, BLOCK_SIZE_K), 
            triton.cdiv(mask_k, BLOCK_SIZE_K), 
            is_causal,
            
            # iteration controls
            i_iteration, n_iteration,
            
            # rope config
            ROPE_METHOD,
            ROPE_COS,
            ROPE_SIN,
            POSITION_IDS,
            SELF_EXTEND_SCALE,
            SELF_EXTEND_WINDOW,
            
            # input constant
            kv_repeat_interleave,
            N,
            T_DST,
            T_SRC,
            B_DST,
            B_SRC,
            HID,
            SPARQ, 
            SPARQ_HID, 
            max(0, triton.cdiv(n_completed, BLOCK_SIZE_Q) - (triton.cdiv(T_SRC, BLOCK_SIZE_Q) - triton.cdiv(T_DST, BLOCK_SIZE_Q))),
            
            # kernel constant
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
            REDUCE_METHOD,
            REDUCE_STRIDE,
            
            SAMPLING_METHOD,
            
            GRID_SRC_STRIDE,
            GRID_K_STRIDE,
            
            DEBUG,
        )
        if DEBUG:
            debug_print(w_curr, mask, ws, ks, N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K)
    
    with timer('matrix.cleanup'):
        if ESTIMATOR_LOWER_RESOLUTION > 1:
            mask = torch.repeat_interleave(mask, ESTIMATOR_LOWER_RESOLUTION, dim=-1)
            ks = ks * ESTIMATOR_LOWER_RESOLUTION
            mask_k = mask_k * ESTIMATOR_LOWER_RESOLUTION
        indices = safe_indices(mask, ws, BLOCK_SIZE_K)
    
    # # NOTE: are you sure this function is the only thing can differentiate?
    with timer("score" if not IS_FLASH else "flash_atten"):
        if not IS_FLASH:
            assert ROPE_METHOD in ['none']
            scores, probs = calc_score_return_prob(
                queries=queries, keys=keys, attention_mask=attention_mask,
                indices=indices, ks=ks,
                KV_REPEAT_INTERLEAVE=kv_repeat_interleave,
                BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                IS_CAUSAL=is_causal,
            )
            assert probs.dtype == queries.dtype, f"{probs.dtype} == {queries.dtype}"
        else:
            assert ROPE_METHOD in ['self_extend', 'none']
            context = calc_prob_return_context(
                queries=queries, keys=keys, values=values, 
                attention_mask=attention_mask,
                indices=indices, ks=ks,
                KV_REPEAT_INTERLEAVE=kv_repeat_interleave,
                BLOCK_SIZE_Q=BLOCK_SIZE_Q, 
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                IS_CAUSAL=is_causal,
                USING_SLIDING_WINDOW=USING_SLIDING_WINDOW,
                SLIDING_WINDOW_SIZE=SLIDING_WINDOW_SIZE,
                ROPE_METHOD=ROPE_METHOD,
                ROPE_COS=ROPE_COS,
                ROPE_SIN=ROPE_SIN,
                POSITION_IDS=POSITION_IDS,
                SELF_EXTEND_SCALE=SELF_EXTEND_SCALE,
                SELF_EXTEND_WINDOW=SELF_EXTEND_WINDOW,
            )

            return indices, ks, context, None
    
    if DEBUG:
        x = to_dense(
            indices.cpu().numpy(),
            ks.cpu().numpy(),
            probs.detach().cpu().to(torch.float32).numpy(),
            N, 
            T_DST, 
            T_SRC, 
            BLOCK_SIZE_Q, 
            BLOCK_SIZE_K,
        )[0]
        x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
        if x.shape[0] == 1:
            x = x.repeat(32, 0)
        plt.imshow(x)
        path = 'saves/models/timber_attention/block_est.png'
        print('saved', path)
        plt.savefig(path, dpi=200, bbox_inches='tight')
        
        # x = np.matmul(
        #     queries[0].cpu().numpy(), 
        #     keys[0].cpu().numpy().transpose((-1, -2))
        # )
        if isinstance(keys, Tensor):
            x = (queries[0] @ keys[0].transpose(-1, -2)).detach().to(torch.float32).cpu().numpy()
            if is_causal:
                x = x + (1 - np.tri(*x.shape, T_SRC-T_DST)) * (-10000)
            x = np.exp(x - x.max(-1, keepdims=True))
            x = x / x.sum(-1, keepdims=True)
            x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
            plt.imshow(x)
            path = 'saves/models/timber_attention/block_truth.png'
            print('saved', path)
            plt.savefig(path, dpi=200, bbox_inches='tight')
            # print(ks)
            input('>>>')
    
    return indices, ks, probs, scores


@triton.jit
def _sdbmm_compute(
    # inputs
    INDICES, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    KS, stride_ks_n, stride_ks_bdst, 
    PROBS, stride_probs_n, stride_probs_tdst, stride_probs_k,
    VALUES, stride_values_n, stride_values_tsrc, stride_values_hid,
    
    # output
    CONTEXT, stride_context_n, stride_context_tdst, stride_context_hid,
    
    # variables
    KV_REPEAT_INTERLEAVE, N, TSRC, TDST, HID, K, BK, BSRC, BDST,
    
    # vllm value cache compat,
    stride_values_vllm_num_blocks,
    stride_values_vllm_num_kv_heads,
    stride_values_vllm_head_size,
    stride_values_vllm_block_size,
    
    VLLM_NUM_BLOCKS,
    VLLM_NUM_KV_HEADS,
    VLLM_HEAD_SIZE,
    VLLM_BLOCK_SIZE,
    
    BLOCK_TABLES,
    stride_block_tables_num_seqs,
    stride_block_tables_max_num_blocks_per_seq,
    
    # kernel blocks
    VALUE_CACHE_METHOD: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    idx_n = tl.program_id(0)
    # if idx_n >= N: return
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_bdst = tl.program_id(1)
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    pid_hid = tl.program_id(2)
    idx_hid = tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID
    mask_hid = idx_hid < HID
    
    n_bk = tl.load(
        KS +\
            idx_n * stride_ks_n+\
            idx_bdst * stride_ks_bdst,
    )
    
    scores = tl.zeros((BLOCK_SIZE_Q_PADDED, BLOCK_HID), dtype=tl.float32)
    for idx_bk in range(BK):
        mask_bk = idx_bk < n_bk
        _idx_tsrc = tl.load(
            INDICES +\
                idx_n * stride_indices_n +\
                idx_bdst * stride_indices_bdst +\
                idx_bk * stride_indices_bk,
            mask = mask_bk,
            # other = TSRC,
        ).to(tl.int64)
        # atten_indices: [BLOCK_SIZE_PADDED]
        idx_tsrc = _idx_tsrc + idx_block_k
        mask_tsrc = (idx_tsrc < TSRC) & mask_block_k & mask_bk
        
        # atten_probs: [BLOCK_SIZE_PADDED: tdst, BLOCK_SIZE_PADDED: tsrc]
        idx_prob_k = (idx_bk * BLOCK_SIZE_K + idx_block_k)
        mask_prob_k = (idx_prob_k < K) & mask_block_k & mask_bk
        atten_probs = tl.load(
            PROBS +\
                idx_n * stride_probs_n +\
                idx_tdst[:, None] * stride_probs_tdst +\
                idx_prob_k[None, :] * stride_probs_k,
            mask = \
                mask_tdst[:, None] &\
                mask_prob_k[None, :] &\
                ((idx_tdst[:, None] + TSRC - TDST) >= idx_tsrc[None, :]) & \
                mask_bk,
            other = 0,
        )
        # DEBUG: tl.device_assert(tl.max(idx_tsrc * mask_tsrc) < TSRC, "TSRC")
        
        if VALUE_CACHE_METHOD == 'cont':
            # value: [BLOCK_SIZE_PADDED: tsrc, BLOCK_HID: hid]
            value = tl.load(
                VALUES +\
                    (idx_n // KV_REPEAT_INTERLEAVE).to(tl.int64) * stride_values_n +\
                    idx_tsrc[:, None].to(tl.int64) * stride_values_tsrc +\
                    idx_hid[None, :].to(tl.int64) * stride_values_hid,
                mask = mask_tsrc[:, None] & mask_hid[None, :] & mask_bk,
                other = 0,
            )
        elif VALUE_CACHE_METHOD == 'vllm':
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
                mask = mask_tsrc & mask_bk,
                other = 0
            ).to(tl.int64)
            mask_block = (idx_tsrc // VLLM_BLOCK_SIZE) < tl.cdiv(TSRC, VLLM_BLOCK_SIZE)
            offset_block = idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)
            
            # value: [BLOCK_SIZE_PADDED: tsrc, BLOCK_HID: hid]
            value = tl.load(
                VALUES +\
                    idx_block[:, None] * stride_values_vllm_num_blocks+\
                    idx_head * stride_values_vllm_num_kv_heads+\
                    idx_hid[None, :].to(tl.int64) * stride_values_vllm_head_size +\
                    offset_block[:, None] * stride_values_vllm_block_size,
                mask = mask_tsrc[:, None] & mask_hid[None, :] & mask_bk & mask_block[:, None],
                other = 0
            )
        else:
            raise Exception()
        
        # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
        if value.dtype == tl.uint8:
            value = value.to(tl.float8e5, bitcast=True).to(atten_probs.dtype)
        scores_mini = tl.dot(atten_probs, value)
        scores += scores_mini.to(scores.dtype)
        
        # scores += tl.sum(value)
        
    tl.store(
        CONTEXT +\
            idx_n * stride_context_n +\
            idx_tdst[:, None] * stride_context_tdst +\
            idx_hid[None, :] * stride_context_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        value = scores
    )

@triton.jit
def _sdbmm_compute_bwd_values(
    # input matrices
    probs, stride_probs_n, stride_probs_tdst, stride_probs_k,
    indices, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    # grad output (read)
    grad_context, stride_grad_context_n, stride_grad_context_tdst, stride_grad_context_hid,
    # grad input (write)
    grad_values, stride_grad_values_n, stride_grad_values_tsrc, stride_grad_values_hid,
    # input variables
    N, TDST, TSRC, HID, BK, K,
    # block constant
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    """
    probs: fp[N, TDST, K]
    indices: int[N, TDST, K]
    
    grad_context: fp[N, TDST, HID]
    grad_values: fp[N, TSRC, HID]
    ----
    foreach n in range(N)
    foreach tdst in range(TDST)
    foreach k in range(K)
    
    grad_values[n, indices[n, tdst, k], :] +=(atmoic) probs[n, tdst, k] * grad_context[n, tdst, :]
    """
    
    idx_n = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_bk = tl.program_id(2)
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    idx_k = idx_bk * BLOCK_SIZE_K + idx_block_k
    mask_k = (idx_k < K) & mask_block_k
    
    idx_tsrc = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk
    )
    idx_tsrc = idx_tsrc + idx_block_k
    mask_tsrc = (idx_tsrc < TSRC) & mask_block_k
    
    # [BLOCK_SIZE_PADDED: tsrc, BLOCK_SIZE_PADDED: tdst]
    prob = tl.load(
        probs +\
            idx_n * stride_probs_n +\
            idx_tdst[None, :] * stride_probs_tdst +\
            idx_k[:, None] * stride_probs_k,
        mask = mask_tdst[None, :] & mask_k[:, None],
        other = 0
    )
    # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
    grad = tl.load(
        grad_context +\
            idx_n * stride_grad_context_n +\
            idx_tdst[:, None] * stride_grad_context_tdst +\
            idx_hid[None, :] * stride_grad_context_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        other = 0
    )
    # [BLOCK_SIZE_PADED: tsrc, BLOCK_HID: hid]
    output = tl.dot(prob, grad)
    
    tl.atomic_add(
        grad_values +\
            idx_n * stride_grad_values_n +\
            idx_tsrc[:, None] * stride_grad_values_tsrc +\
            idx_hid[None, :] * stride_grad_values_hid,
        val = output,
        mask = mask_tsrc[:, None] & mask_hid[None, :],
    )

@triton.jit
def _sdbmm_compute_bwd_probs(
    # input indices
    indices, stride_indices_n, stride_indices_bdst, stride_indices_bk,
    values, stride_values_n, stride_values_trsc, stride_values_hid,
    # grad output (read)
    grad_context, stride_grad_context_n, stride_grad_context_tdst, stride_grad_context_hid,
    # grad input (write)
    grad_probs, stride_grad_probs_n, stride_grad_probs_tdst, stride_grad_probs_k,
    # input variables
    N, TDST, TSRC, HID, BK, K,
    # blcok constant
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    """
    indices: fp[N, TDST, K]
    values: fp[N, TSRC, HID]
    grad_context: fp[N, TDST, HID]
    grad_probs: fp[N, TDST, K]
    -----
    foreach n in [..N]
    foreach tdst in [..TDST]
    foreach k in [..K]
    
    grad_probs[n, tdst, k] = sum(
        values[n, indices[n, tdst, k], :] * grad_context[n, tdst, :]
    )
    """
    
    idx_n = tl.program_id(0)
    idx_bdst = tl.program_id(1)
    idx_bk = tl.program_id(2)
    
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < HID
    
    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    mask_block_q = idx_block_q < BLOCK_SIZE_Q
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    mask_block_k = idx_block_k < BLOCK_SIZE_K
    
    idx_tsrc = tl.load(
        indices +\
            idx_n * stride_indices_n +\
            idx_bdst * stride_indices_bdst +\
            idx_bk * stride_indices_bk,
    )
    idx_tsrc = idx_tsrc + idx_block_k
    mask_tsrc = (idx_tsrc < TSRC) & mask_block_k
    
    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    
    # [BLOCK_HID: hid, BLOCK_SIZE_PADDED: tsrc]
    value = tl.load(
        values +\
            idx_n * stride_values_n +\
            idx_tsrc[None, :] * stride_values_trsc +\
            idx_hid[:, None] * stride_values_hid,
        mask = mask_tsrc[None, :] & mask_hid[:, None],
        other = 0,
    )
    # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
    vec_grad_context = tl.load(
        grad_context +\
            idx_n * stride_grad_context_n +\
            idx_tdst[:, None] * stride_grad_context_tdst +\
            idx_hid[None, :] * stride_grad_context_hid,
        mask = mask_tdst[:, None] & mask_hid[None, :],
        other = 0
    )
    # [BLOCK_SIZE_PADDED: tdst, BLOCK_SIZE_PADDED: tsrc]
    score = tl.dot(vec_grad_context, value)
    
    idx_k = idx_bk * BLOCK_SIZE_K + idx_block_k
    mask_k = (idx_k < K) & mask_block_k
    
    tl.store(
        grad_probs +\
            idx_n * stride_grad_probs_n +\
            idx_tdst[:, None] * stride_grad_probs_tdst +\
            idx_k[None, :] * stride_grad_probs_k,
        value = score,
        mask = mask_tdst[:, None] & mask_k[None, :]
    )

class SparseAttentionAutoGradFn(Function):
    @staticmethod
    def forward(
        ctx, 
        # attention values
        values: Union[Tensor, "PagedValueCacheVllmCompat"],
        
        # attention matrix
        indices: Tensor,
        ks: Tensor,
        probs: Tensor,
        
        KV_REPEAT_INTERLEAVE: int,
        BLOCK_SIZE_Q: int,
        BLOCK_SIZE_K: int,
    ):
        global DEBUG
        
        ctx.save_for_backward(values, indices, ks, probs)
        ctx.BLOCK_SIZE_Q = BLOCK_SIZE_Q
        ctx.BLOCK_SIZE_K = BLOCK_SIZE_K
    
        N, BDST, BK = indices.shape
        _N, TDST, K = probs.shape
        __N, TSRC, HID = values.shape
        assert N == _N
        assert N == (__N * KV_REPEAT_INTERLEAVE)
        # assert N == __N
        assert ks.shape == (N, BDST)
        
        BSRC = triton.cdiv(TSRC, BLOCK_SIZE_K)
        
        context_dtype = values.dtype
        if context_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
            context_dtype = probs.dtype
        assert context_dtype in [torch.float16, torch.bfloat16, torch.float32]
        context = torch.zeros((N, TDST, HID), dtype=context_dtype, device=values.device)
        
        BLOCK_SIZE_Q_PADDED = next_multiple_of(BLOCK_SIZE_Q, 16)
        BLOCK_SIZE_K_PADDED = next_multiple_of(BLOCK_SIZE_K, 16)
        BLOCK_HID = triton.next_power_of_2(HID)
        
        if isinstance(values, Tensor):
            VALUE_CACHE_METHOD = 'cont'
            
            block_tables = values
            block_tables_strides = (0, 0)
            
            VLLM_NUM_BLOCKS =\
            VLLM_NUM_KV_HEADS =\
            VLLM_HEAD_SIZE =\
            VLLM_BLOCK_SIZE = 0
            
            vllm_values_strides = (0, 0, 0, 0)
        elif isinstance(values, PagedValueCacheVllmCompat):
            """
            vLLM compatible paged attention
            
            q: [num_seqs, num_heads, head_size]
            k: [num_blocks, num_kv_heads, head_size/x, block_size, x]
            v: [num_blocks, num_kv_heads, head_size, block_size]
            block_tables: [num_seqs, max_num_blocks_per_seq]
            context_lens: [num_seqs]
            """
            
            VALUE_CACHE_METHOD = 'vllm'
            
            block_tables = values.block_table
            block_tables_strides = block_tables.stride()
            assert len(block_tables_strides) == 2
            
            (
                VLLM_NUM_BLOCKS,
                VLLM_NUM_KV_HEADS,
                VLLM_HEAD_SIZE,
                VLLM_BLOCK_SIZE
            ) = values.value_cache.shape
            vllm_values_strides = values.value_cache.stride()
            assert len(vllm_values_strides) == 4 
        else:
            raise Exception()
        
        grid = (N, BDST, triton.cdiv(HID, BLOCK_HID))
        # grid = (1, 1, 1)
        
        # NOTE: I have no idea what this sprase matrix format LOL, but for temporary
        if DEBUG:
            # print('sdbmm', grid, BLOCK_K, BLOCK_HID)
            # assert indices.max() < TSRC
            assert indices.min() >= 0
            assert indices.is_contiguous()
            assert ks.is_contiguous()
            assert probs.is_contiguous()
            # assert values.is_contiguous()
            assert context.is_contiguous()
            torch.cuda.synchronize()
        
        # print(values.shape[0] * values.stride(0))
        
        assert indices.shape[0] == N
        assert ks.shape[0] == N
        assert probs.shape[0] == N, f'{probs.shape} == {N}'
        # assert values.shape[0] == N
        assert context.shape[0] == N
        assert ks.ndim == 2
        assert probs.ndim == 3
        assert values.ndim == 3
        assert context.ndim == 3
        # assert values.dtype == probs.dtype, f"{values.dtype} == {probs.dtype}"
        # assert values.dtype == context.dtype
        _sdbmm_compute[grid](
            # inputs
            indices, *indices.stride(),
            ks, *ks.stride(),
            probs, *probs.stride(),
            values, *values.stride(),
            
            # output
            context, *context.stride(),
            
            # input variables
            KV_REPEAT_INTERLEAVE, N, TSRC, TDST, HID, K, BK, BSRC, BDST,
            
            # vllm value cache compat
            *vllm_values_strides,
            VLLM_NUM_BLOCKS,
            VLLM_NUM_KV_HEADS,
            VLLM_HEAD_SIZE,
            VLLM_BLOCK_SIZE,
            
            block_tables,
            *block_tables_strides,
            
            # blocks
            VALUE_CACHE_METHOD,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_Q_PADDED,
            BLOCK_SIZE_K,
            BLOCK_SIZE_K_PADDED,
            BLOCK_HID,
            
            num_warps=BLOCK_HID//32,
        )
        
        return context
    
    @staticmethod
    def backward(ctx, grad_context):
        ENABLED_VALUES = True
        ENABLED_PROBS = True
        
        values, indices, ks, probs = ctx.saved_tensors
        BLOCK_SIZE_Q = ctx.BLOCK_SIZE_Q
        BLOCK_SIZE_K = ctx.BLOCK_SIZE_K
        grad_values = grad_probs = None
        
        N, T_SRC, HID = values.shape
        _, B_DST, BK = indices.shape
        _, T_DST, K = probs.shape
        assert ks.shape == (N, B_DST)
        assert probs.shape == (N, T_DST, K)
        assert indices.shape[0] == N

        # for values
        if ctx.needs_input_grad[0]:
            grid = (N, B_DST, BK)
            BLOCK_HID = triton.next_power_of_2(HID)

            grad_values = torch.zeros(
                (N, T_SRC, HID), 
                device=values.device, 
                dtype=values.dtype,
            )
            
            if ENABLED_VALUES:
                _sdbmm_compute_bwd_values[grid](
                    probs, probs.stride(0), probs.stride(1), probs.stride(2),
                    indices, indices.stride(0), indices.stride(1), indices.stride(2),
                    
                    grad_context, grad_context.stride(0), grad_context.stride(1), grad_context.stride(2),
                    
                    grad_values, grad_values.stride(0), grad_values.stride(1), grad_values.stride(2),
                    
                    N, T_DST, T_SRC, HID, BK, K,
                    
                    BLOCK_SIZE_Q,
                    next_multiple_of(BLOCK_SIZE_Q, 16),
                    BLOCK_SIZE_K,
                    next_multiple_of(BLOCK_SIZE_K, 16),
                    BLOCK_HID,
                )
            
            # print(grad_values.abs().sum())
        
        # for probs
        if ctx.needs_input_grad[3]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE_Q), BK)
            BLOCK_HID = triton.next_power_of_2(HID)
            
            grad_probs = torch.zeros(
                (N, T_DST, K),
                device=probs.device,
                dtype=probs.dtype,
            )
            
            if ENABLED_PROBS:
                _sdbmm_compute_bwd_probs[grid](
                    indices, indices.stride(0), indices.stride(1), indices.stride(2),
                    values, values.stride(0), values.stride(1), values.stride(2), 
                    
                    grad_context, grad_context.stride(0), grad_context.stride(1), grad_context.stride(2),
                    
                    grad_probs, grad_probs.stride(0), grad_probs.stride(1), grad_probs.stride(2),
                    
                    N, T_DST, T_SRC, HID, BK, K,
                    
                    BLOCK_SIZE_Q,
                    next_multiple_of(BLOCK_SIZE_Q, 16),
                    BLOCK_SIZE_K,
                    next_multiple_of(BLOCK_SIZE_K, 16),
                    BLOCK_HID,
                )

        return (
            grad_values, 
            None, 
            None, 
            grad_probs, 
            None,
            None,
            None,
        )

def sparse_attention(
    # attention values
    values: Tensor,
    
    # attention matrix
    indices: Tensor,
    ks: Tensor,
    probs: Tensor,
    
    KV_REPEAT_INTERLEAVE: int,
    BLOCK_SIZE_Q: int,
    BLOCK_SIZE_K: int,
):
    context = SparseAttentionAutoGradFn.apply(
        values, indices, ks, probs, 
        KV_REPEAT_INTERLEAVE, BLOCK_SIZE_Q, BLOCK_SIZE_K,
    )
    
    return context

import numba
@numba.njit
def to_dense(
    indices: np.ndarray, 
    ks: np.ndarray, 
    value: np.ndarray,
    N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K
):
    # print(indices.shape, ks.shape, value.shape, T_DST, T_SRC)
    out = np.zeros((N, T_DST, T_SRC), dtype=np.float32)
    for idx_n in numba.prange(N):
        for idx_bdst in range(indices.shape[1]):
            for idx_k in range(indices.shape[2]):
                if idx_k < ks[idx_n, idx_bdst]:
                    idx_tsrc = indices[idx_n, idx_bdst, idx_k]
                    if value is not None:
                        dst = out[
                            idx_n, 
                            idx_bdst * BLOCK_SIZE_Q: (idx_bdst + 1) * BLOCK_SIZE_Q, 
                            idx_tsrc: idx_tsrc + BLOCK_SIZE_K
                        ]
                        src = value[
                            idx_n,
                            idx_bdst * BLOCK_SIZE_Q: (idx_bdst + 1) * BLOCK_SIZE_Q, 
                            idx_k * BLOCK_SIZE_K: (idx_k + 1) * BLOCK_SIZE_K
                        ]
                        if src.shape == dst.shape:
                            dst[:, :] = src[:, :]
                    else:
                        out[
                            idx_n, 
                            idx_bdst * BLOCK_SIZE_Q: (idx_bdst + 1) * BLOCK_SIZE_Q, 
                            idx_tsrc: idx_tsrc + BLOCK_SIZE_K
                        ] = 1
    return out

def paged_timber_attention(
    q: Tensor, 
    q_scale: float,
    k: Tensor, 
    v: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    max_context_len: int,
    # optional mask
    attention_mask: Tensor = None,
    
    # heuristics: w_start == mask_k * scale_up
    w_start: int = None,
    # heuristics: n_patches == mask_k // scale_up
    n_patches: int = None,
    mask_k: int = 512,
    scale_up: float = 2,
    
    block_size_q: int = 8,
    block_size_k: int = 1,
    reduce_method: str = 'max',
    reduce_stride: int = 2,
    
    rope_method: str = 'none',
    rope_cos: Tensor = None,
    rope_sin: Tensor = None,
    position_ids: Tensor = None,
    
    self_extend_scale: int = 8,
    self_extend_window: int = 1024,
):
    """
    vLLM compatible paged attention
    
    q: [num_seqs, num_heads, head_size]
    k: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    v: [num_blocks, num_kv_heads, head_size, block_size]
    block_tables: [num_seqs, max_num_blocks_per_seq]
    context_lens: [num_seqs]
    """
    
    with timer('scaling'):
        q = q * q_scale
        q = q.view(q.shape[0] * q.shape[1], 1, q.shape[2])
    
    with timer('compat'):
        if max_context_len < 0:
            max_context_len = block_tables.shape[1] * k.shape[3]
        paged_k = PagedKeyCacheVllmCompat(
            key_cache=k,
            block_table=block_tables,
            context_length=context_lens,
            max_context_length=max_context_len,
        )
        
        paged_v = PagedValueCacheVllmCompat(
            key_cache=paged_k,
            value_cache=v,
        )
    
    # print('paged qkv cache shape', q.shape, paged_k.shape, paged_v.shape)
    
    # if (not torch.cuda.is_current_stream_capturing()) and hasattr(k, 'readonly_start'):
    #     k.readonly_start()
    # if (not torch.cuda.is_current_stream_capturing()) and hasattr(v, 'readonly_start'):
    #     v.readonly_start()
    
    out = timber_attention(
        q=q,
        k=paged_k,
        v=paged_v,
        attention_mask=attention_mask,
        
        w_start=w_start,
        n_patches=n_patches,
        mask_k=mask_k,
        scale_up=scale_up,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        
        reduce_method=reduce_method,
        reduce_stride=reduce_stride,
        
        dense_queries_exp=0,
        
        rope_method=rope_method,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        position_ids=position_ids,
        
        self_extend_scale=self_extend_scale,
        self_extend_window=self_extend_window,
    )
    
    # if (not torch.cuda.is_current_stream_capturing()) and hasattr(k, 'readonly_end'):
    #     k.readonly_end()
    # if (not torch.cuda.is_current_stream_capturing()) and hasattr(v, 'readonly_end'):
    #     v.readonly_end()
    
    return out

def timber_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor,
    # optional mask
    attention_mask: Tensor = None,
    
    # heuristics: w_start == mask_k * scale_up
    w_start: int = None,
    # heuristics: n_patches == mask_k // scale_up
    n_patches: int = None,
    mask_k: int = 512,
    scale_up: float = 2,
    is_causal: bool = True,
    
    block_size_q: int = 32,
    block_size_k: int = 2,
    reduce_method: str = 'max',
    reduce_stride: int = 2,
    
    chunking: bool = False,
    chunk_size: int = 2048,
    
    is_flash: bool = True,
    enable_sparq: bool = True,
    
    sampling_method: str = 'random',
    
    using_sliding_window: bool = True,
    sliding_window_size: int = 128,
    
    dense_queries_exp: Optional[int] = None,
    
    rope_method: Literal['none', 'self_extend'] = 'none',
    rope_cos: Optional[Tensor] = None,
    rope_sin: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    
    self_extend_scale: int = 8,
    self_extend_window: int = 1024,
):
    assert sampling_method in ['random', 'first']
    
    if q.requires_grad:
        is_flash = False
    
    assert rope_method in ['none', 'self_extend']
    if rope_method == 'self_extend':
        assert dense_queries_exp == 0
        assert rope_sin is not None
        assert rope_cos is not None
        # assert position_ids is not None
        assert is_flash
    
    is_prompt = isinstance(k, Tensor) and isinstance(v, Tensor) and (q.shape[1] > 32)
    if is_prompt:
        if dense_queries_exp is None:
            dense_queries_exp = int(((math.log2(k.shape[1] / mask_k / 2)) * mask_k + mask_k) * 3)
        dense_queries = int(max(0, dense_queries_exp - k.shape[1] + q.shape[1]))
        # print('dense queries', dense_queries_exp, dense_queries, q.shape[1], k.shape[1], block_size_q, block_size_k)
        if is_causal and (dense_queries > 0) and (dense_queries_exp > 0):
            contexts = []
            
            dense_q = q[:, :dense_queries, :]
            dense_k = k[:, :dense_queries + k.shape[1] - q.shape[1], :]
            dense_v = v[:, :dense_queries + k.shape[1] - q.shape[1], :]
            
            dense_q = dense_q.unsqueeze(-2)
            dense_k = dense_k.unsqueeze(-2)
            dense_v = dense_v.unsqueeze(-2)
            
            if dense_q.shape[0] != dense_k.shape[0]:
                kv_repeat = dense_q.shape[0] // dense_k.shape[0]
                dense_k = torch.repeat_interleave(dense_k, kv_repeat, 0)
                dense_v = torch.repeat_interleave(dense_v, kv_repeat, 0)

            dense_context, _ = flash_attention(
                dense_q,
                dense_k,
                dense_v,
                is_causal=True
            )
            dense_context = dense_context.squeeze(-2)
            contexts.append(dense_context)
            
            if dense_queries < q.shape[1]:
                sparse_q = q[:, dense_queries:, :]
                sparse_k = k[:, :, :]
                sparse_v = v[:, :, :]
                sparse_context, _ = timber_attention(
                    sparse_q,
                    sparse_k,
                    sparse_v,
                    
                    attention_mask=attention_mask,
                    
                    w_start=w_start,
                    n_patches=n_patches,
                    mask_k=mask_k,
                    scale_up=scale_up,
                    
                    is_causal=is_causal,
                    
                    block_size_q=block_size_q,
                    block_size_k=block_size_k,
                    
                    reduce_method=reduce_method,
                    reduce_stride=reduce_stride,
                    
                    chunking=chunking,
                    chunk_size=chunk_size,
                    
                    is_flash=is_flash,
                    
                    enable_sparq=enable_sparq,
                    sampling_method=sampling_method,
                    
                    using_sliding_window=using_sliding_window,
                    sliding_window_size=sliding_window_size,
                    
                    dense_queries_exp=dense_queries_exp,
                    
                    rope_method=rope_method,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    position_ids=position_ids,
                    
                    self_extend_scale=self_extend_scale,
                    self_extend_window=self_extend_window,
                )
                contexts.append(sparse_context)
            
            if len(contexts) > 1:
                return torch.cat(contexts, dim=1), None
            else:
                return contexts[0], None
        
    CHUNKING = chunking
    CHUNK_SIZE = chunk_size
    if q.shape[1] > CHUNK_SIZE and CHUNKING:
        N, T_DST, HID = q.shape
        N, T_SRC, HID = k.shape
        
        contexts = []
        
        for ichunk in range(triton.cdiv(T_DST, CHUNK_SIZE)):
            q_chunk = q[:, ichunk*CHUNK_SIZE:(ichunk+1)*CHUNK_SIZE, :]
            cache_chunk_end = T_SRC-T_DST+(ichunk+1)*CHUNK_SIZE
            k_chunk = k[:, :cache_chunk_end, :]
            v_chunk = v[:, :cache_chunk_end, :]
            if attention_mask is not None:
                attention_mask_chunk = attention_mask[:, :cache_chunk_end]
            else:
                attention_mask_chunk = None
            
            context, _ = timber_attention(
                q_chunk, 
                k_chunk, 
                v_chunk, 
                attention_mask=attention_mask_chunk,
                
                w_start=w_start,
                n_patches=n_patches,
                mask_k=mask_k,
                scale_up=scale_up,
                
                is_causal=is_causal,
                
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                
                reduce_method=reduce_method,
                reduce_stride=reduce_stride,
                
                is_flash=is_flash,
                enable_sparq=enable_sparq,
                
                sampling_method=sampling_method,
                
                using_sliding_window=using_sliding_window,
                sliding_window_size=sliding_window_size,
                
                dense_queries_exp=dense_queries_exp,
                
                rope_method=rope_method,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                position_ids=position_ids,
                
                self_extend_scale=self_extend_scale,
                self_extend_window=self_extend_window,
            )
            contexts.append(context)
            
        contexts = torch.cat(contexts, dim=1)    
        
        return contexts, None
    
    global DEBUG
    DENSE_SPARSE_ATTENTION = False
    
    if w_start is None:
        w_start = math.ceil(mask_k * scale_up)
        # w_start = math.ceil(mask_k * scale_up * scale_up)
        # w_start = math.ceil(mask_k / scale_up)
        # w_start = mask_k
    if n_patches is None:
        n_patches = math.ceil(mask_k / scale_up)
        # n_patches = mask_k / scale_up
    
    assert q.ndim == 3
    assert k.ndim == 3
    assert v.ndim == 3
    N, T_DST, HID = q.shape
    _N, T_SRC, _HID = k.shape
    assert k.shape[:-1] == v.shape[:-1]
    assert (N % _N) == 0
    assert HID == _HID
    KV_REPEAT_INTERLEAVE = N // _N
    
    # assert q.dtype == k.dtype, f'{q.dtype} == {k.dtype}'
    # assert q.dtype == v.dtype
    
    # if attention_mask is None:
    #     attention_mask = torch.full((N, T_SRC), True, dtype=torch.bool, device=q.device)
    # if attention_mask.dtype != torch.bool:
    #     # mask should mark alive token as True
    #     attention_mask = attention_mask > 0.5
    # assert attention_mask.dtype == torch.bool
    
    assert isinstance(block_size_q, int)
    assert isinstance(block_size_k, int)
    
    block_size_q = min(block_size_q, triton.next_power_of_2(T_DST))
    block_size_k = min(block_size_k, triton.next_power_of_2(T_SRC))
    
    if DEBUG:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    with timer('timber_attention'):
        with timer('attention_matrix'):
            indices, ks, probs_or_context, scores = attention_matrix(
                queries=q,
                keys=k,
                values=v,
                attention_mask=attention_mask,
                kv_repeat_interleave=KV_REPEAT_INTERLEAVE,
                
                w_start=w_start,
                n_patches=n_patches,
                mask_k=mask_k,
                scale_up=scale_up,
                is_causal=is_causal,
                
                BLOCK_SIZE_Q=block_size_q,
                BLOCK_SIZE_K=block_size_k,
                REDUCE_METHOD=reduce_method,
                REDUCE_STRIDE=reduce_stride,
                
                IS_FLASH=is_flash,
                SPARQ=enable_sparq,
                SAMPLING_METHOD=sampling_method,
                
                USING_SLIDING_WINDOW=using_sliding_window,
                SLIDING_WINDOW_SIZE=sliding_window_size,
                
                ROPE_METHOD=rope_method,
                ROPE_COS=rope_cos,
                ROPE_SIN=rope_sin,
                POSITION_IDS=position_ids,
                
                SELF_EXTEND_SCALE=self_extend_scale,
                SELF_EXTEND_WINDOW=self_extend_window,
            )
            
            if is_flash:
                return probs_or_context, (indices, ks, None)
            else:
                probs = probs_or_context
                assert rope_method in ['none'] # self_extend is not supported
            
            # assert probs.dtype == v.dtype, f"{probs.dtype} == {v.dtype}"
        
        with timer('sparse_attention'):
            if DENSE_SPARSE_ATTENTION:
                probs_dense = torch.tensor(to_dense(
                    indices.to(torch.float32).cpu().numpy(), 
                    ks.to(torch.float32).cpu().numpy(), 
                    probs.to(torch.float32).cpu().numpy(), 
                    N, T_DST, T_SRC, block_size_q, block_size_k,
                )).to(v.dtype).to(indices.device)
                
                # scores_dense = to_dense(
                #     indices.cpu(), ks.cpu(), scores.cpu(),
                #     N, T_DST, T_SRC, block_size_q, block_size_k,
                # ).to(indices.device)
                
                mask_dense = probs_dense <= 1e-7
                mask_dense = mask_dense.to(probs.dtype) * torch.finfo(probs.dtype).min
                
                scores_truth = torch.bmm(q, k.transpose(-1, -2))
                probs_truth = (scores_truth + mask_dense).softmax(dim=-1)
                
                context = torch.bmm(probs_truth, v)
            else:
                context = sparse_attention(
                    v,
                    indices,
                    ks,
                    probs,
                    KV_REPEAT_INTERLEAVE=KV_REPEAT_INTERLEAVE,
                    BLOCK_SIZE_Q=block_size_q,
                    BLOCK_SIZE_K=block_size_k,
                )
                
                # v_cumsum = v.cumsum(dim=1)
                # v_avg = v_cumsum / torch.arange(1, v.shape[1]+1, device=v.device)[None, :, None]
                
                # exp_norm = torch.exp(scores - torch.max(scores, dim=-1, keepdim=True)[0])
                # min_exp_norm = torch.where(scores > -100, exp_norm, 1000.0).min(dim=-1, keepdim=True)[0]
                # sum_exp_norm = exp_norm.sum(dim=-1, keepdim=True)
                # ctx_exp_norm = min_exp_norm * torch.clamp_min(torch.arange(1, v.shape[1]+1, device=v.device)[None, :, None] - mask_k, 0)
                # sum_exp_norm = sum_exp_norm + ctx_exp_norm
                # ctx_ratio = (ctx_exp_norm / sum_exp_norm) * 0.1
                
                # context = context * (1 - ctx_ratio) + v_avg * ctx_ratio
    
    return context, (indices, ks, probs)

import torch.nn.functional as F


def torch_attention(q: Tensor, k: Tensor, v: Tensor):
    scores = torch.bmm(q, k.transpose(-1, -2))
    probs = torch.softmax(scores, dim=-1)
    context = torch.bmm(probs, v)
    return context, probs

def flash_attention(q: Tensor, k: Tensor, v: Tensor, is_causal=True):
    # context = F.scaled_dot_product_attention(
    #     q, k, v, is_causal=False, scale=None,
    # )
    # return context, None
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_with_kvcache
    
    assert q.shape[0] == k.shape[0], f"{q.shape}, {k.shape}"
    assert k.shape[0] == v.shape[0]
    
    return flash_attn_with_kvcache(q, k, v, causal=is_causal, softmax_scale=1.0), None

def landmark_attention(q: Tensor, k: Tensor, v: Tensor):
    """
    https://arxiv.org/pdf/2305.16300.pdf
    this paper claimed, they are faster than original attetnion... but seems not?
    """
    from timber.models.landmark_attention import fused_landmark_attention
    
    seqlen_k = k.shape[1]
    block_size = 64
    is_mem = torch.arange(0, seqlen_k, device=q.device) % block_size == (block_size - 1)
    return fused_landmark_attention(q, k, v, is_mem, block_size=block_size)

def main_latency_benchmark():
    global DEBUG
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dups', type=int, default=2)
    parser.add_argument('--query_size', type=int, default=1)
    parser.add_argument('--method', type=str, default='timber')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--block_size_q', type=int, default=16)
    parser.add_argument('--block_size_k', type=int, default=1)
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--scale_up', type=int, default=2)
    parser.add_argument('--not_causal', action='store_true')
    args = parser.parse_args()
    
    DEBUG = args.debug
    TRACE = args.trace
    BSIZE = args.batch_size
    DUPS = args.dups
    QUERY_SIZE = args.query_size
    METHOD = args.method
    n_samples = args.samples
    is_causal = not args.not_causal
    
    if DEBUG:
        seed()
    
    get_bench().disabled = not TRACE
    get_bench().synchronize = True

    CHUNK_LEN = 1024
    q, k, v, out = load_checkouts(idx=0, window=40, seq_len=CHUNK_LEN)
    HID = q.shape[-1]
    
    q = q.cpu()
    k = k.cpu()
    v = v.cpu()
    
    q = q.repeat(BSIZE, max(1, triton.cdiv(QUERY_SIZE, 1024)), 1)[:, :QUERY_SIZE, :].contiguous()
    k = k.repeat(BSIZE, DUPS, 1)
    v = v.repeat(BSIZE, DUPS, 1)
    started = False
    
    if METHOD in 'flash':
        q = q.view(BSIZE, -1, QUERY_SIZE, HID).permute(0, 2, 1, 3).contiguous()
        k = k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).permute(0, 2, 1, 3).contiguous()
        v = v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).permute(0, 2, 1, 3).contiguous()
    elif METHOD in 'landmark':
        q = q.view(BSIZE, -1, QUERY_SIZE, HID).contiguous()
        k = k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).contiguous()
        v = v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).contiguous()
    
    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    
    timber_attention_mask = torch.full((q.shape[0], k.shape[1]), True, dtype=torch.bool, device=q.device)
    
    def sample():
        with torch.no_grad():
            if METHOD in ['torch', 'none', 'default']:
                torch_attention(q, k, v)
            elif METHOD == 'flash':
                flash_attention(q, k, v, is_causal=is_causal)
            elif METHOD == 'landmark':
                landmark_attention(q, k, v)
            elif METHOD == 'timber':
                timber_attention(
                    q,
                    k,
                    v,
                    # attention_mask=timber_attention_mask,
                    mask_k=args.k,
                    block_size_q=args.block_size_q,
                    block_size_k=args.block_size_k,
                    scale_up=args.scale_up,
                    is_causal=is_causal,
                )
            else:
                raise Exception()
    
    s = torch.cuda.Stream()
    graph = None
    samples = []
    for i in tqdm.tqdm(range(n_samples)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        if i < 3:
            s.wait_stream(torch.cuda.current_stream())
            sample()
            torch.cuda.current_stream().wait_stream(s)
        elif args.trace:
            sample()
        elif graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                sample()
        else:
            graph.replay()
        
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
        
        if i > n_samples * 0.1:
            if not started:
                get_bench().reset_measures()
                get_bench().reset_trace()
                started = True
            samples.append(elapsed)
    
    if TRACE:
        print(get_bench().format_tracetree())
    
    samples = np.array(samples)
    print(f'[{METHOD}] {np.mean(samples):.4f}ms +- {np.std(samples):.4f}ms (q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)})')
    
    os.makedirs('./cache/attention1_block_gpu/', exist_ok=True)
    with open('./cache/attention1_block_gpu/result.json', 'w') as f:
        json.dump({
            'method': METHOD,
            'mean': np.mean(samples),
            'std': np.std(samples),
            'query_length': q.shape[-2],
            'keyvalue_length': k.shape[-2],
        }, f, indent=2)

def main_debug():
    global DEBUG
    DEBUG = True
    
    block = 1024
    block = 256
    q, k, v, out = load_checkouts(
        dtype=torch.float16, 
        seq_len=block * 4, 
        idx=6, 
        window=1
    )
    
    q = q[:, block * 2:, :]
    out = out[:, block * 2:, :]
    
    print('q', q.shape)
    print('k', k.shape)
    print('v', v.shape)
    print('out', out.shape)
    
    context, _ = timber_attention(
        q,
        k,
        v,
        mask_k=256,
        block_size_q=16,
        block_size_k=2,
        is_flash=False,
        dense_queries_exp=0,
    )
    
    stderr = (out - context).abs().mean().item()
    stdcontext = torch.std_mean(out)[0].item()
    
    print(f'err = {stderr:.6f} ({stderr/stdcontext:.4f} sigma), out_std = {stdcontext:.6f}')

def main_debug_mask():
    global DEBUG
    DEBUG = True
    
    seed()
    q, k, v, out = load_checkouts(dtype=torch.float16, seq_len=1024 * 2, idx=24, window=1)
    
    q = q[:, 512:, :]
    out = out[:, 512:, :]
    
    N, TSRC, HID = k.shape
    mask = torch.full((N, TSRC), 1, dtype=torch.float32, device=k.device)
    for i in range(N):
        mask[i, :1024] = 0
    
    context, (atten_indices, atten_ks, atten_probs) = timber_attention(
        q,
        k,
        v,
        attention_mask=mask,
    )
    
    stderr = (out - context).abs().mean().item()
    stdcontext = torch.std_mean(out)[0].item()
    
    print(f'err = {stderr:.6f} ({stderr/stdcontext:.4f} sigma), out_std = {stdcontext:.6f}')

if __name__ == '__main__':
    import sys
    if sys.argv[-1] == 'debug':
        main_debug()
    elif sys.argv[-1] == 'debug_mask':
        main_debug_mask()
    else:
        main_latency_benchmark()