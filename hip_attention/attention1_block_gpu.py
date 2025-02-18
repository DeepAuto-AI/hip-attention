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

import gc
import json
import math
import os
import warnings
from typing import List, Literal, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numba
import numpy as np
import torch
import tqdm
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from transformers.utils import logging

# assert (triton.__version__ in ['2.3.0', '2.2.0', '2.1.0']) or ('nightly' in triton.__version__), triton.__version__
# assert hasattr(tl, 'sort'), f'check triton version {triton.__version__}'

if not hasattr(tl, 'sort'):
    warnings.warn(
        "Triton Language does not contain `sort` function. "
        "This will cause the compilation problem. Please upgrade `triton >= 2.2.0`"
    )

from hip_attention.utils.benchmarking import get_bench
from hip_attention.tests.utils.seed import seed
from hip_attention.common import load_checkouts
from hip_attention.attention1_block_gpu_kernel.paged_cache_vllm_compat import (
    PagedKeyCacheVllmCompat, PagedValueCacheVllmCompat
)
from hip_attention.attention1_block_gpu_kernel.masking_iteration import masking_iteration
from hip_attention.attention1_block_gpu_kernel.safe_indices import safe_indices
from hip_attention.attention1_block_gpu_kernel.calc_prob_return_context import calc_prob_return_context
from hip_attention.attention1_block_gpu_kernel.calc_score_return_prob import calc_score_return_prob

logger = logging.get_logger(__name__)
timer = lambda x: get_bench().region(x)

DEBUG = os.environ.get('hip_DEBUG', '0') == '1'

def next_multiple_of(x: int, multiple_by: int = 16):
    return triton.next_power_of_2(max(x, multiple_by))

def debug_print(
    w_curr,
    mask, ws, ks, N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K
):
    from matplotlib import pyplot as plt
    import skimage
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
    plt.title(f'sum:{x.sum()}')
    plt.imshow(x)
    plt.colorbar()
    path = f'saves/models/hip_attention/block_{w_curr}.png'
    # path = f'saves/models/hip_attention/block.png'
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

def hip_attention_mask(
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
    
    block_size_q: int = 16,
    block_size_k: int = 1,
    reduce_method: Literal['first', 'max', 'sum'] = 'max',
    reduce_stride: int = 1,
    
    enable_sparq: bool = True,
    sparq_start_tsrc: int = 2048,
    sparq_start_bk: int = 128,
    sparq_hid: int = 32,
    sparq_reduce_method: Literal['sum', 'max'] = 'sum',
    
    is_flash: bool = True,
    
    # NOTE: this improve latency quite well, but hurt accuracy
    estimator_lower_resolution: int = 2,
    estimator_lower_resolution_stop_n_blocks: int = 64,
    
    sampling_method: str = 'first',

    using_sliding_window=True,
    sliding_window_size=128,
    
    rope_method='none',
    rope_cos=None,
    rope_sin=None,
    position_ids=None,
    
    self_extend_scale=None,
    self_extend_window=None,
    
    grid_src_stride=1,
    grid_k_stride=1,
    
    maximum_ks=None,
    maximum_ks_config=None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    global DEBUG
    
    if DEBUG:
        print('hip_attention_mask', queries.shape, keys.shape, w_start, n_patches, mask_k, scale_up, block_size_q, block_size_k)
        os.makedirs('saves/models/hip_attention/', exist_ok=True)
    
    N, T_DST, HID = queries.shape
    _, T_SRC, _ = keys.shape
    assert T_DST <= T_SRC, f"{queries.shape}, {keys.shape}"
    
    if triton.cdiv(mask_k, block_size_k) <= estimator_lower_resolution_stop_n_blocks:
        estimator_lower_resolution = 1
    
    if estimator_lower_resolution > 1:
        mask_k = mask_k // estimator_lower_resolution
        w_start = w_start // estimator_lower_resolution
        n_patches = n_patches // estimator_lower_resolution
        if maximum_ks is not None:
            maximum_ks = maximum_ks // estimator_lower_resolution
            assert maximum_ks_config is not None
            maximum_ks_config = list([x // estimator_lower_resolution for x in maximum_ks_config])
    
    if enable_sparq and ((mask_k // block_size_k) < sparq_start_bk):
        enable_sparq = False
    if enable_sparq and (T_SRC < sparq_start_tsrc):
        enable_sparq = False
    if rope_method in ['self_extend']:
        # assert (mask_k // BLOCK_SIZE_K) <= 128, "oh this is bug,,, i need help"
        # SPARQ_HID = 16
        enable_sparq = False
    
    # if SPARQ:
    #     warnings.warn('sparq is enabled')
    
    dtype = queries.dtype
    device = queries.device
    # assert queries.device == keys.device
    
    assert isinstance(block_size_q, int)
    assert isinstance(block_size_k, int)
    block_size_q = int(block_size_q)
    block_size_k = int(block_size_k)
    
    if attention_mask is not None:
        assert attention_mask.shape == (N, T_SRC)
        assert attention_mask.dtype == torch.bool
    
    # NOTE: width of last query
    w_curr = round(w_start / scale_up)
    assert w_curr <= mask_k, f'{w_curr} <= {mask_k}'
    
    with timer('matrix.setup'):
        # vectors
        tsrcs_offset = max(block_size_q, block_size_k) - 1
        tsrcs = torch.arange(
            tsrcs_offset+T_SRC-T_DST+1, tsrcs_offset+T_SRC+1, block_size_q, 
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
        # assert tsrcs.dtype == torch.int64
        # assert ws.dtype == torch.int64
        # assert ks.dtype == torch.int64
        
        # matrices
        # NOTE: float16 -> int64 seems not possible
        bws = torch.ceil(ws / block_size_k)
        ks = torch.ceil(bws / grid_src_stride).to(torch.int64)
        mask_k_block = triton.cdiv(triton.cdiv(mask_k, block_size_k), grid_k_stride)
        mask = torch.arange(
            mask_k_block, device=device, dtype=torch.float32
        )\
            .view(1, 1, 1, mask_k_block)\
            .expand(1, 1, grid_src_stride, mask_k_block) \
                / ks.unsqueeze(-1).unsqueeze(-1)
        mask = mask + (
            torch.arange(grid_src_stride, device=device, dtype=torch.float32).view(1, 1, grid_src_stride, 1)
        ) * (1 / bws.unsqueeze(-1).unsqueeze(-1))
        tmask = torch.zeros(
            (
                mask.shape[0], 
                mask.shape[1], 
                grid_src_stride,
                mask_k_block * math.ceil(scale_up)
            ), 
            dtype=torch.float32, 
            device=device
        )
        
        B_SRC = triton.cdiv(T_SRC, block_size_k)
        B_DST = triton.cdiv(T_DST, block_size_q)
        
        sparq_indices = None
        sparq_indices_strides = (1, 1, 1)
        if enable_sparq:
            with timer('matrix.setup.sparq'):
                q_scale = 1 / math.sqrt(HID)
                queries_scores = queries
                if rope_method in ['self_extend']:
                    queries_scores, _ = apply_rotary_pos_emb(
                        queries / q_scale, 
                        None, 
                        rope_cos, 
                        rope_sin, 
                        position_ids
                    )
                    queries_scores *= q_scale
                queries_scores = queries_scores.abs()
                if T_DST > 1 and (B_DST * block_size_q) != T_DST:
                    queries_scores = F.pad(
                        queries_scores.unsqueeze(0), 
                        (0, 0, 0, B_DST * block_size_q - T_DST), 
                        value=0
                    ).squeeze(0)
                # print(queries_scores.shape, B_DST, BLOCK_SIZE_Q, T_DST, T_DST > 1 and (B_DST * BLOCK_SIZE_Q) != T_DST)
                # TODO: padding
                queries_scores = queries_scores.view(N, B_DST, -1, HID)
                if sparq_reduce_method == 'sum':
                    queries_scores = queries_scores.sum(-2)
                elif sparq_reduce_method == 'max':
                    queries_scores = queries_scores.max(-2)[0]
                else:
                    raise Exception()
                _, sparq_indices = torch.topk(
                    queries_scores, 
                    k=sparq_hid, 
                    dim=-1, 
                    sorted=True
                )
                sparq_indices = sparq_indices.to(torch.int16)
                # sparq_indices = torch.arange(0, SPARQ_HID, device=queries.device)[None, None, :].repeat(N, B_DST, 1)
                sparq_indices_strides = sparq_indices.stride()
    # if DEBUG:
    #     debug_print(w_curr, mask, ws, ks, N, T_DST, T_SRC, BLOCK_SIZE_Q, BLOCK_SIZE_K)
        
    # NOTE: Calc. Mask
    n_iteration = 0
    _w_curr = w_curr
    while w_curr < T_SRC:
        w_curr = round(w_curr * scale_up)
        n_iteration += 1
    # w_curr = _w_curr
    
    n_completed = 0 #_w_curr
    with timer("iterations"):
        i_iteration = 0
        mask, ws, ks = masking_iteration(
            # input matrices
            queries, keys, attention_mask,
            
            # input metrices (blocked) 
            mask, tmask, 
            sparq_indices, sparq_indices_strides,
            
            # temp vectors (blocked)
            ws, ks, tsrcs, 
            
            # operator variables
            scale_up, 
            triton.cdiv(n_patches, block_size_k), 
            triton.cdiv(mask_k, block_size_k), 
            is_causal,
            
            # iteration controls
            i_iteration, n_iteration,
            
            # rope config
            rope_method,
            rope_cos,
            rope_sin,
            position_ids,
            self_extend_scale,
            self_extend_window,
            
            # dynamic k per query
            maximum_ks,
            maximum_ks_config,
            
            # input constant
            kv_repeat_interleave,
            N,
            T_DST,
            T_SRC,
            B_DST,
            B_SRC,
            HID,
            enable_sparq, 
            sparq_hid, 
            max(0, triton.cdiv(n_completed, block_size_q) - (triton.cdiv(T_SRC, block_size_q) - triton.cdiv(T_DST, block_size_q))),
            
            # kernel constant
            block_size_q,
            block_size_k,
            reduce_method,
            reduce_stride,
            
            sampling_method,
            
            grid_src_stride,
            grid_k_stride,
            
            using_sliding_window,
            sliding_window_size,
            
            DEBUG,
        )
        if DEBUG:
            debug_print(w_curr, mask, ws, ks, N, T_DST, T_SRC, block_size_q, block_size_k)
    
    with timer('matrix.cleanup'):
        if estimator_lower_resolution > 1:
            mask = torch.repeat_interleave(mask, estimator_lower_resolution, dim=-1)
            ks = ks * estimator_lower_resolution
            mask_k = mask_k * estimator_lower_resolution
        indices = safe_indices(mask, ws, block_size_k)
    
    # # NOTE: are you sure this function is the only thing can differentiate?
    with timer("score" if not is_flash else "flash_atten"):
        if not using_sliding_window:
            warnings.warn('you are not using sliding window, WARN: this may degrade performance')
        if not is_flash:
            assert rope_method in ['none']
            assert not using_sliding_window
            assert not enable_sparq
            warnings.warn('you are not using flash attention, WARN: this may degrade performance & latency')
        else:
            assert rope_method in ['self_extend', 'none']
    
    return indices, ks


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

        orig_device = torch.cuda.current_device()
        torch.cuda.set_device(indices.device)
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
        torch.cuda.set_device(orig_device)
        
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
                dtype=torch.float32,
            )
            
            if ENABLED_VALUES:
                orig_device = torch.cuda.current_device()
                torch.cuda.set_device(indices.device)
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
                torch.cuda.set_device(orig_device)

            grad_values = grad_values.to(values.dtype)
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

@numba.njit(parallel=True)
def to_dense(
    indices: np.ndarray, 
    ks: np.ndarray, 
    value: Optional[np.ndarray],
    N: int, 
    T_DST: int, 
    T_SRC: int, 
    BLOCK_SIZE_Q: int, 
    BLOCK_SIZE_K: int, 
):
    # print(indices.shape, ks.shape, value.shape, T_DST, T_SRC)
    out = np.zeros((N, T_DST, T_SRC), dtype=np.float32)
    for idx_n in numba.prange(N):
        for idx_bdst in numba.prange(indices.shape[1]):
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

def paged_hip_attention(
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
    
    using_precomputed_mask: bool = False,
    precomputed_indices: Tensor = None,
    precomputed_ks: Tensor = None,
    
    query_format: Literal['N_H_D', 'NH_TDST_D'] = 'N_H_D',
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
        
        if query_format == 'N_H_D':
            q = q.view(q.shape[0] * q.shape[1], 1, q.shape[2])
        elif query_format == 'NH_TDST_D':
            pass
        else:
            raise Exception(f'unknown format {query_format}')
    
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
    
    out = hip_attention(
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
        
        using_precomputed_mask=using_precomputed_mask,
        precomputed_ks=precomputed_ks,
        precomputed_indices=precomputed_indices,
    )
    
    # if (not torch.cuda.is_current_stream_capturing()) and hasattr(k, 'readonly_end'):
    #     k.readonly_end()
    # if (not torch.cuda.is_current_stream_capturing()) and hasattr(v, 'readonly_end'):
    #     v.readonly_end()
    
    return out

def hip_attention(
    q: Tensor, 
    k: Tensor, 
    v: Tensor,
    # optional attention mask
    attention_mask: Optional[Tensor] = None,
    
    # NOTE: do not touch w_start, n_patches, scale_up unless you really understand what are they.
    # NOTE: heuristics: w_start == mask_k * scale_up
    w_start: int = None,
    # NOTE: heuristics: n_patches == mask_k // scale_up
    n_patches: int = None,
    mask_k: int = 512,
    scale_up: float = 2,
    is_causal: bool = True,
    
    # block approximation hyperparameter
    block_size_q: int = 32,
    block_size_k: int = 2,
    reduce_method: str = 'max',
    reduce_stride: int = 2,
    
    # chunk the hip_attention computation, to prevent allocate large temp buffers
    chunking: bool = False,
    chunk_size: int = 2048,
    
    # using flash attention for sparse attention
    is_flash: bool = True,
    force_return_scores: bool = False,
    
    # control SparQ attention for score approximation in masking
    enable_sparq: bool = True,
    
    # representative token sampling method
    sampling_method: str = 'random',
    
    # sliding window
    using_sliding_window: bool = True,
    sliding_window_size: int = 128,
    
    # control query length for dense attention
    dense_queries_exp: Optional[int] = None,
    
    # rope (experimental)
    rope_method: Literal['none', 'self_extend'] = 'none',
    rope_cos: Optional[Tensor] = None,
    rope_sin: Optional[Tensor] = None,
    position_ids: Optional[Tensor] = None,
    
    # self-extend (experimental)
    self_extend_scale: int = 8,
    self_extend_window: int = 1024,
    
    # cached masking
    using_precomputed_mask: bool = False,
    precomputed_indices: Optional[Tensor] = None,
    precomputed_ks: Optional[Tensor] = None,
    
    # dynamic k per query support
    maximum_ks: Optional[Tensor] = None,
    maximum_ks_config: Optional[List[int]] = None,
    
    # number of sink tokens, default to size of tensor core
    num_sink: Optional[int] = None,
):
    assert sampling_method in ['random', 'first']
    
    if q.requires_grad:
        logger.warning_once('q requires grad, turning off flash')
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
                sparse_context, _ = hip_attention(
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
                    force_return_scores=force_return_scores,
                    
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
                    
                    using_precomputed_mask=using_precomputed_mask,
                    precomputed_ks=precomputed_ks,
                    precomputed_indices=precomputed_indices,
                    
                    maximum_ks=maximum_ks,
                    maximum_ks_config=maximum_ks_config,
                    
                    num_sink=num_sink,
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
            
            context, _ = hip_attention(
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
                force_return_scores=force_return_scores,
                
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
                
                using_precomputed_mask=using_precomputed_mask,
                precomputed_ks=precomputed_ks,
                precomputed_indices=precomputed_indices,
                
                maximum_ks=maximum_ks,
                maximum_ks_config=maximum_ks_config,
                
                num_sink=num_sink,
            )
            contexts.append(context)
            
        contexts = torch.cat(contexts, dim=1)    
        
        return contexts, None
    
    global DEBUG
    
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
    
    assert isinstance(block_size_q, int)
    assert isinstance(block_size_k, int)
    
    block_size_q = min(block_size_q, triton.next_power_of_2(T_DST))
    block_size_k = min(block_size_k, triton.next_power_of_2(T_SRC))
    
    if DEBUG:
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    with timer('hip_attention'):
        if not using_precomputed_mask:
            with timer('attention_matrix'):
                # if prompt (exceed single tensor-core block), 
                # do not use topk strding. this will cause more resource
                estimated_ksrc_stride = min(32, max(1, round(mask_k / (block_size_k * 16))))
                if q.shape[1] > 32:
                    estimated_ksrc_stride = 1
                
                indices, ks = hip_attention_mask(
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
                    
                    block_size_q=block_size_q,
                    block_size_k=block_size_k,
                    reduce_method=reduce_method,
                    reduce_stride=reduce_stride,
                    
                    is_flash=is_flash,
                    enable_sparq=enable_sparq,
                    sampling_method=sampling_method,
                    
                    using_sliding_window=using_sliding_window,
                    sliding_window_size=sliding_window_size,
                    
                    rope_method=rope_method,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    position_ids=position_ids,
                    
                    self_extend_scale=self_extend_scale,
                    self_extend_window=self_extend_window,
                    
                    grid_src_stride=estimated_ksrc_stride,
                    grid_k_stride=estimated_ksrc_stride,
                    
                    maximum_ks=maximum_ks,
                    maximum_ks_config=maximum_ks_config,
                )
        else:
            assert precomputed_ks is not None
            assert precomputed_indices is not None
            
            indices = precomputed_indices
            ks = precomputed_ks
            
            assert indices.shape[:-1] == (N, T_DST), f'{indices.shape}, {(N, T_DST)}'
            assert ks.shape == (N, T_DST), f'{ks.shape}, {(N, T_DST)}'
        
        with timer('sparse_attention'):
            if not is_flash:
                assert rope_method in ['none'] # self_extend is not supported
                
                scores, probs = calc_score_return_prob(
                    queries=q, keys=k, attention_mask=attention_mask,
                    indices=indices, ks=ks,
                    KV_REPEAT_INTERLEAVE=KV_REPEAT_INTERLEAVE,
                    BLOCK_SIZE_Q=block_size_q,
                    BLOCK_SIZE_K=block_size_k,
                    IS_CAUSAL=is_causal,
                )
                assert probs.dtype == q.dtype, f"{probs.dtype} == {q.dtype}"
                
                # if DEBUG:
                #     x = to_dense(
                #         indices.cpu().numpy(),
                #         ks.cpu().numpy(),
                #         probs.detach().cpu().to(torch.float32).numpy(),
                #         N, 
                #         T_DST, 
                #         T_SRC, 
                #         block_size_q, 
                #         block_size_k,
                #     )[0]
                #     x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
                #     if x.shape[0] == 1:
                #         x = x.repeat(32, 0)
                #     plt.imshow(x)
                #     path = 'saves/models/hip_attention/block_est.png'
                #     print('saved', path)
                #     plt.savefig(path, dpi=200, bbox_inches='tight')
                #
                #     if isinstance(k, Tensor):
                #         x = (q[0] @ k[0].transpose(-1, -2)).detach().to(torch.float32).cpu().numpy()
                #         if is_causal:
                #             x = x + (1 - np.tri(*x.shape, T_SRC-T_DST)) * (-10000)
                #         x = np.exp(x - x.max(-1, keepdims=True))
                #         x = x / x.sum(-1, keepdims=True)
                #         x = skimage.measure.block_reduce(x, (1, 1), np.max) ** 0.1
                #         plt.imshow(x)
                #         path = 'saves/models/hip_attention/block_truth.png'
                #         print('saved', path)
                #         plt.savefig(path, dpi=200, bbox_inches='tight')
                #         # print(ks)
                #         # input('>>>')
                
                context = sparse_attention(
                    v,
                    indices,
                    ks,
                    probs,
                    KV_REPEAT_INTERLEAVE=KV_REPEAT_INTERLEAVE,
                    BLOCK_SIZE_Q=block_size_q,
                    BLOCK_SIZE_K=block_size_k,
                )
            else:
                probs = None
                t = calc_prob_return_context(
                    queries=q, 
                    keys=k, 
                    values=v, 
                    attention_mask=attention_mask,
                    indices=indices, 
                    ks=ks,
                    KV_REPEAT_INTERLEAVE=KV_REPEAT_INTERLEAVE,
                    BLOCK_SIZE_Q=block_size_q, 
                    BLOCK_SIZE_K=block_size_k,
                    IS_CAUSAL=is_causal,
                    USING_SLIDING_WINDOW=using_sliding_window,
                    SLIDING_WINDOW_SIZE=sliding_window_size,
                    ROPE_METHOD=rope_method,
                    ROPE_COS=rope_cos,
                    ROPE_SIN=rope_sin,
                    POSITION_IDS=position_ids,
                    SELF_EXTEND_SCALE=self_extend_scale,
                    SELF_EXTEND_WINDOW=self_extend_window,
                    RETURN_SCORES=force_return_scores,
                    NUM_SINK=num_sink,
                )
                if force_return_scores:
                    context, probs = t
                else:
                    context = t
    
    return context, (indices, ks, probs)

import torch.nn.functional as F

def torch_attention(q: Tensor, k: Tensor, v: Tensor):
    scores = torch.bmm(q, k.transpose(-1, -2))
    probs = torch.softmax(scores, dim=-1)
    context = torch.bmm(probs, v)
    return context, probs

def flash_attention(q: Tensor, k: Tensor, v: Tensor, is_causal=True, backend='flash_attn'):
    if backend == 'sdpa':
        context = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=None,
        )
        return context, None
    elif backend == 'flash_attn':
        from flash_attn import flash_attn_with_kvcache
        
        assert q.shape[0] == k.shape[0], f"{q.shape}, {k.shape}"
        assert k.shape[0] == v.shape[0]
        
        return flash_attn_with_kvcache(q, k, v, causal=is_causal, softmax_scale=1.0), None
    else:
        raise Exception()

def landmark_attention(q: Tensor, k: Tensor, v: Tensor):
    """
    https://arxiv.org/pdf/2305.16300.pdf
    this paper claimed, they are faster than original attetnion... but seems not?
    """
    from hip.models.landmark_attention import fused_landmark_attention
    
    seqlen_k = k.shape[1]
    block_size = 64
    is_mem = torch.arange(0, seqlen_k, device=q.device) % block_size == (block_size - 1)
    return fused_landmark_attention(q, k, v, is_mem, block_size=block_size)

@torch.inference_mode(True)
def streaming_attention(q: Tensor, k: Tensor, v: Tensor, cos: Tensor, sin: Tensor, window_size: int):
    from hip.models.sink_attention.sink_attention import sink_attention
    
    return sink_attention(q, k, v, cos, sin, window_size=window_size)

from hip_attention.attention2_draft_prefetch import hip_attention as hip_attention_11

def main_latency_benchmark():
    global DEBUG
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dups', type=int, default=2)
    parser.add_argument('--query_size', type=int, default=1)
    parser.add_argument('--method', type=str, default='hip1.1')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--block_size_q', type=int, default=32)
    parser.add_argument('--block_stride_q', type=int, default=None)
    parser.add_argument('--block_size_k', type=int, default=1)
    parser.add_argument('--block_stride_k', type=int, default=None)
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--scale_up', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=-1)
    parser.add_argument('--head_size', type=int, default=-1)
    parser.add_argument('--refresh_interval', type=int, default=8)
    parser.add_argument('--not_causal', action='store_true')
    parser.add_argument('--head_groups', type=int, default=1)
    args = parser.parse_args()
    
    if args.query_size > 1:
        args.refresh_interval = -1
    
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
    q, k, v, out, cos, sin = load_checkouts(idx=0, window=40, seq_len=CHUNK_LEN)
    HID = q.shape[-1]
    
    q = q.cpu()
    k = k.cpu()
    v = v.cpu()
    
    if args.head_size > 0 and args.head_size != q.shape[0]:
        head_reps = int(math.ceil(args.head_size / q.shape[0]))
        q = q.repeat(head_reps, 1, 1)[:args.head_size, :, :].contiguous()
        k = k.repeat(head_reps, 1, 1)[:args.head_size, :, :].contiguous()
        v = v.repeat(head_reps, 1, 1)[:args.head_size, :, :].contiguous()
    
    q = q.repeat(BSIZE, max(1, triton.cdiv(QUERY_SIZE, 1024)), 1)[:, :QUERY_SIZE, :].contiguous()
    k = k.repeat(BSIZE, DUPS, 1)
    v = v.repeat(BSIZE, DUPS, 1)
    started = False
    
    if args.hidden_size > 0 and args.hidden_size != HID:
        hid_reps = int(math.ceil(args.hidden_size / HID))
        q = q.repeat(1, 1, hid_reps)[:, :, :args.hidden_size].contiguous()
        k = k.repeat(1, 1, hid_reps)[:, :, :args.hidden_size].contiguous()
        v = v.repeat(1, 1, hid_reps)[:, :, :args.hidden_size].contiguous()
        HID = args.hidden_size
    
    head_size = q.shape[0]
    cos = sin = torch.randn((k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device)
    
    if METHOD in ['flash', 'fa2', 'hip1.1']:
        q = q.view(BSIZE, -1, QUERY_SIZE, HID).permute(0, 2, 1, 3).contiguous()
        k = k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID)[:, ::args.head_groups, :, :].permute(0, 2, 1, 3).contiguous()
        v = v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID)[:, ::args.head_groups, :, :].permute(0, 2, 1, 3).contiguous()
    elif METHOD in ['landmark']:
        q = q.view(BSIZE, -1, QUERY_SIZE, HID).contiguous()
        k = k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).contiguous()
        v = v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).contiguous()
    elif METHOD in ['streaming', 'hyper']:
        k = k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).repeat(q.shape[0] // k.shape[0], 1, 1, 1).view(q.shape[0], -1, q.shape[2])
        v = v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).repeat(q.shape[0] // v.shape[0], 1, 1, 1).view(q.shape[0], -1, q.shape[2])
    
    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    cos = cos.cuda()
    sin = sin.cuda()
    if METHOD in ['hip1.1']:
        # q_quant = q.to(torch.float8_e5m2).view(torch.uint8)
        # k_quant = k.to(torch.float8_e5m2).view(torch.uint8)
        q_quant = q
        k_quant = k
    
    hip_attention_mask = torch.full((q.shape[0], k.shape[1]), True, dtype=torch.bool, device=q.device)
    
    from hip.models.hyper_attention.hyper_attn import HyperAttention
    hyper_attention = HyperAttention(
        input_dim=q.shape[-1],
        lsh_num_projs=7, # not very meaningful after 7
        block_size=64, # smaller better
        sample_size=1024, # larger better
        min_seq_len=32, # this factor is kind of random. usually smaller better
        cuda=True, 
    ).to('cuda').to(q.dtype)
    
    def sample(state = None):
        with torch.no_grad():
            if METHOD in ['torch', 'none', 'default']:
                torch_attention(q, k, v)
            elif METHOD in ['flash', 'fa2']:
                flash_attention(q, k, v, is_causal=is_causal)
            elif METHOD == 'landmark':
                landmark_attention(q, k, v)
            elif METHOD == 'streaming':
                streaming_attention(q, k, v, cos, sin, window_size=args.k)
            elif METHOD == 'hyper':
                _q = q.view(-1, head_size, q.shape[1], q.shape[-1])
                _k = k.view(-1, head_size, k.shape[1], k.shape[-1])
                _v = v.view(-1, head_size, v.shape[1], v.shape[-1])
                # print(_q.shape, _k.shape, _v.shape)
                hyper_attention(
                    _q, _k, _v,
                    causal=True, 
                    scale=1
                )
            elif METHOD == 'hip1.1':
                assert is_causal
                if state is None:
                    _, mask = hip_attention_11(
                        q,
                        k,
                        v,
                        # attention_mask=hip_attention_mask,
                        mask_k=args.k,
                        block_size_q=args.block_size_q,
                        block_stride_q=args.block_stride_q if args.block_stride_q is not None else 2,
                        block_size_k=args.block_size_k,
                        block_stride_k=args.block_stride_k if args.block_stride_k is not None else max(2, args.block_size_k//2),
                        q_quant=q_quant,
                        k_quant=k_quant,
                        sample_method='center',
                        mask_only=False,
                    )
                else:
                    _, mask = hip_attention_11(
                        q,
                        k,
                        v,
                        # attention_mask=hip_attention_mask,
                        mask_k=args.k,
                        block_size_q=args.block_size_q,
                        block_size_k=args.block_size_k,
                        block_stride_k=max(2, args.block_size_k//2),
                        q_quant=q_quant,
                        k_quant=k_quant,
                        sample_method='center',
                        previous_metadata=state,
                    )
                if mask is None:
                    return None
                return mask
            elif METHOD == 'hip1.0':
                if state is None:
                    _, mask = hip_attention(
                        q,
                        k,
                        v,
                        # attention_mask=hip_attention_mask,
                        mask_k=args.k,
                        block_size_q=args.block_size_q,
                        block_size_k=args.block_size_k,
                        scale_up=args.scale_up,
                        is_causal=is_causal,
                    )
                else:
                    indices, ks = state
                    
                    _, mask = hip_attention(
                        q,
                        k,
                        v,
                        # attention_mask=hip_attention_mask,
                        mask_k=args.k,
                        block_size_q=args.block_size_q,
                        block_size_k=args.block_size_k,
                        scale_up=args.scale_up,
                        is_causal=is_causal,
                        
                        using_precomputed_mask=True,
                        precomputed_indices=indices,
                        precomputed_ks=ks,
                    )
                if mask is None:
                    return None, None
                return mask[0], mask[1]
            else:
                raise Exception()
    
    s = torch.cuda.Stream()
    graph = None
    graph_stateful = None
    samples = []
    for i in tqdm.tqdm(range(n_samples), dynamic_ncols=True):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        if i < 5:
            s.wait_stream(torch.cuda.current_stream())
            state = sample()
            if args.refresh_interval > 0:
                sample(state)
            torch.cuda.current_stream().wait_stream(s)
        elif args.trace:
            sample()
        elif graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                state = sample()
            if args.refresh_interval > 0:
                graph_stateful = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_stateful):
                    sample(state)
        else:
            if args.refresh_interval > 0:
                if (i % args.refresh_interval) == 0:
                    graph.replay()
                else:
                    graph_stateful.replay()
            else:
                graph.replay()
        
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / args.batch_size
        
        if i > n_samples * 0.1:
            if not started:
                get_bench().reset_measures()
                get_bench().reset_trace()
                started = True
            samples.append(elapsed)
    
    if TRACE:
        print(get_bench().format_tracetree())
    
    samples = np.array(samples)
    print(f'({METHOD}) {np.mean(samples):.8f} ms +- {np.std(samples):.4f} ms (q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)})')
    
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
    # block = 256
    q, k, v, out = load_checkouts(
        dtype=torch.float32, 
        seq_len=block * 4, 
        idx=6, 
        window=1
    )
    
    reps = 1
    q = q.repeat(1, reps, 1)
    k = k.repeat(1, reps, 1)
    v = v.repeat(1, reps, 1)
    out = out.repeat(1, reps, 1)
    
    q = q[:, -k.shape[1]//4:, :]
    out = out[:, -k.shape[1]//4:, :]
    
    print('q', q.shape)
    print('k', k.shape)
    print('v', v.shape)
    print('out', out.shape)
    
    mask_k = 512
    block_size_q = 32
    block_size_k = 2
    
    maximum_ks = torch.where(
        torch.rand((q.shape[0], q.shape[1] // block_size_q), device=q.device) < 0.5,
        mask_k,
        mask_k // 4,
    ).to(torch.int32)
    # maximum_ks = None
    
    context, (
        atten_indices, 
        atten_ks, 
        atten_scores
    ) = hip_attention(
        q,
        k,
        v,
        mask_k=mask_k,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        dense_queries_exp=0,
        is_flash=True,
        using_sliding_window=True,
        maximum_ks=maximum_ks,
        maximum_ks_config=[128, 512],
        force_return_scores=True,
    )
    
    print(maximum_ks[0])
    print(atten_indices[0])
    print(atten_ks[0])
    
    atten_probs = atten_scores.softmax(-1)
    plt.figure(figsize=(4, 5.5))
    plt.imshow(atten_probs[0].cpu().numpy() ** 0.2)
    plt.axvline(32, color='red')
    plt.axvline(32 + 512, color='red')
    for idx_tdst in range(0, atten_probs.shape[1], block_size_q):
        plt.axhline(idx_tdst, color='magenta', linestyle=':')
    plt.colorbar()
    path = 'saves/models/hip_attention/atten_probs.png'
    plt.savefig(path, dpi=200)
    print('saved', path)
    print(atten_probs.shape)
    
    print(atten_ks, atten_ks.min(), atten_ks.max())
    
    stderr = (out - context).abs().mean().item()
    stdcontext = torch.std_mean(out)[0].item()
    
    root = 'cache/attention1_block_gpu'
    os.makedirs(root, exist_ok=True)
    torch.save({
        'indices': atten_indices,
        'ks': atten_ks,
    }, os.path.join(root, 'checkout_mask.pth'))
    
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
    
    context, (atten_indices, atten_ks, atten_probs) = hip_attention(
        q,
        k,
        v,
        attention_mask=mask,
    )
    
    stderr = (out - context).abs().mean().item()
    stdcontext = torch.std_mean(out)[0].item()
    
    print(f'err = {stderr:.6f} ({stderr/stdcontext:.4f} sigma), out_std = {stdcontext:.6f}')

def main_debug_max_ks():
    import nvtx
    global DEBUG
    DEBUG = False
    
    block = 1024
    # block = 256
    q, k, v, out = load_checkouts(
        dtype=torch.float32, 
        seq_len=block * 32, 
        idx=6, 
        window=1
    )
    
    reps = 1
    q = q.repeat(1, reps, 1)
    k = k.repeat(1, reps, 1)
    v = v.repeat(1, reps, 1)
    out = out.repeat(1, reps, 1)
    
    # q = q[:, -k.shape[1]//4:, :]
    # out = out[:, -k.shape[1]//4:, :]
    
    print('q', q.shape)
    print('k', k.shape)
    print('v', v.shape)
    print('out', out.shape)
    
    low_k = 128
    mask_k = 2048
    # low_k = 512
    # mask_k = 512
    block_size_q = 32
    block_size_k = 2
    
    def sample(ratio: float, n: int = 100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        maximum_ks = torch.where(
            torch.rand((q.shape[0], q.shape[1] // block_size_q), device=q.device) < ratio,
            mask_k,
            low_k
        ).to(torch.int32)
        
        def fn():
            hip_attention(
                q,
                k,
                v,
                mask_k=mask_k,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                dense_queries_exp=0,
                is_flash=True,
                using_sliding_window=True,
                maximum_ks=maximum_ks,
                maximum_ks_config=[low_k, mask_k],
                force_return_scores=False,
            )
        
        start.record()
        graph = None
        for i in range(n + 10):
            if graph is None:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                for i in range(3):
                    fn()
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()
            
            start.record()
            graph.replay()
            end.record()
            
            if i > 10:
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
                samples.append(elapsed)
        
        return sum(samples) / len(samples)

    for ratio in range(0, 10):
        ratio = ratio / 20
        samples = []
        with nvtx.annotate(f'ratio={ratio}'):
            for i in range(10):
                t = sample(ratio)
                if i > 3:
                    samples.append(t)
        
        if len(samples) > 0:
            latency = sum(samples) / len(samples)
            print(ratio, ratio * mask_k + (1-ratio) * low_k, latency)

if __name__ == '__main__':
    import sys
    if sys.argv[-1] == 'debug':
        main_debug()
    elif sys.argv[-1] == 'debug_mask':
        main_debug_mask()
    elif sys.argv[-1] == 'debug_max_ks':
        main_debug_max_ks()
    else:
        main_latency_benchmark()