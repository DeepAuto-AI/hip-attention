# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
#
# This code is a conversion of a Triton-based Block-Sparse Attention kernel
# to the TileLang paradigm, adapted by Gemini based on user-provided examples.

import torch
import torch.nn.functional as F
from torch import Tensor # CORRECTED: Added this import
from typing import Optional # CORRECTED: Added this import for Optional type hint
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
from functools import partial

from hip_attn.v1_2.attention_metadata import HiPAttentionArgs # From original code

# =================================================================
# Main JIT-Compiled Kernel Definition
# =================================================================

def block_sparse_attention_func(
    # Shapes of input tensors, passed as static values to the kernel
    q_shape, k_shape, v_shape, pos_shape, indices_shape, ks_se_shape,
    k_cache_shape, v_cache_shape, block_table_shape, cache_seq_lens_shape,
    cos_shape, sin_shape,
    # Static kernel parameters
    HEAD, KV_HEAD_REPEAT, MAX_TDST, MAX_TSRC,
    USING_PAGES, PAGE_SIZE,
    USING_EXTEND, NEED_APPLY_ROPE, rope_range_begin, rope_range_end, rope_is_neox_style,
    IS_CAUSAL,
    # Tunable parameters
    BLOCK_SIZE_Q, BLOCK_SIZE_K, BLOCK_BK, HID_BLOCK_0, HID_BLOCK_V,
    num_stages, threads
):
    """
    This function defines the entire TileLang computation for block-sparse attention.
    It's structured to be called by the main tilelang.jit decorator.
    """
    # Derived static values
    HID = q_shape[3]
    HID_V = v_shape[3]
    dtype = "float16"
    accum_dtype = "float"

    # =================================================================
    # 1. Macros for Reusable Sub-Computations
    # =================================================================

    @T.macro
    def ApplyRoPE(
        # Buffers
        q_shared: T.SharedBuffer, # The Q tile in shared memory to modify
        # RoPE parameters
        pos_tdst: T.FragmentBuffer, # Position indices for the Q block
        # Global RoPE tables
        Q_global: T.Tensor(q_shape, dtype),
        COS: T.Tensor(cos_shape, dtype),
        SIN: T.Tensor(sin_shape, dtype),
        # Grid indices
        bz: T.int32, by: T.int32, q_start_idx: T.int32
    ):
        """ Applies Rotary Position Embedding to the Q tile in shared memory. """
        # This macro is complex because RoPE requires rotating pairs of elements.
        # We first load the original Q tile, then compute the indices of the rotated
        # pairs and load them separately.
        
        # Allocate fragment buffers for RoPE calculations
        q_frag = T.alloc_fragment([BLOCK_SIZE_Q, HID_BLOCK_0], dtype)
        q_rot_frag = T.alloc_fragment([BLOCK_SIZE_Q, HID_BLOCK_0], dtype)
        cos_frag = T.alloc_fragment([BLOCK_SIZE_Q, HID_BLOCK_0], dtype)
        sin_frag = T.alloc_fragment([BLOCK_SIZE_Q, HID_BLOCK_0], dtype)

        T.copy(q_shared, q_frag) # Copy from shared to fragment

        ROPE_DIM = rope_range_end - rope_range_begin
        
        # This loop represents the element-wise RoPE logic.
        # A fully optimized version might use more advanced vectorization.
        for i in T.Parallel(BLOCK_SIZE_Q):
            for j in T.Parallel(HID_BLOCK_0):
                hid_idx = j
                is_in_rope_range = (hid_idx >= rope_range_begin) and (hid_idx < rope_range_end)
                
                if is_in_rope_range:
                    hid_rope_range = hid_idx - rope_range_begin
                    if rope_is_neox_style:
                        # For neox style, the rotated pair is at `j + ROPE_DIM / 2`
                        rot_j = (hid_rope_range + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin
                        cos_sin_idx = hid_rope_range % (ROPE_DIM // 2)
                        rope_mult = T.if_then_else(hid_rope_range + ROPE_DIM // 2 < ROPE_DIM, -1.0, 1.0)
                    else:
                        # For standard style, the rotated pair is at `j +/- 1`
                        flip = T.if_then_else(hid_rope_range % 2 == 0, 1, -1)
                        rot_j = hid_idx + flip
                        cos_sin_idx = hid_rope_range // 2
                        rope_mult = T.if_then_else(hid_rope_range % 2 == 0, -1.0, 1.0)

                    # Load rotated value, sin, and cos
                    q_rot_frag[i, j] = Q_global[bz, q_start_idx + i, by, rot_j] * rope_mult
                    pos = pos_tdst[i] - 1 # RoPE is applied based on position
                    cos_frag[i, j] = COS[pos, cos_sin_idx]
                    sin_frag[i, j] = SIN[pos, cos_sin_idx]

        # Apply the transformation in fragments
        for i, j in T.Parallel(BLOCK_SIZE_Q, HID_BLOCK_0):
            is_in_rope_range = (j >= rope_range_begin) and (j < rope_range_end)
            if is_in_rope_range:
                q_frag[i, j] = q_frag[i, j] * cos_frag[i, j] + q_rot_frag[i, j] * sin_frag[i, j]

        # Copy the result back to shared memory
        T.copy(q_frag, q_shared)


    @T.macro
    def QK_MMA_and_Softmax(
        Q_shared: T.SharedBuffer, K_shared: T.SharedBuffer,
        acc_s: T.FragmentBuffer, acc_s_cast: T.FragmentBuffer,
        scores_max: T.FragmentBuffer, scores_max_prev: T.FragmentBuffer,
        scores_scale: T.FragmentBuffer, scores_sum: T.FragmentBuffer,
        logsum: T.FragmentBuffer, scale: T.float32,
        # Causal masking info
        bx: T.int32, idx_tsrc: T.int32, pos_tdst: T.FragmentBuffer,
        mask_bk: T.bool
    ):
        """ Fused QK GEMM and Softmax update. """
        # Clear and compute Q @ K.T
        T.clear(acc_s)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
        
        # Apply causal and padding masks
        for i, j in T.Parallel(BLOCK_SIZE_Q, BLOCK_SIZE_K):
            is_valid = T.if_then_else(
                IS_CAUSAL,
                pos_tdst[i] > idx_tsrc + j,
                True
            )
            # mask_bk is used to prevent computation on padding blocks
            acc_s[i, j] = T.if_then_else(is_valid and mask_bk, acc_s[i,j], -T.infinity(accum_dtype))

        # Standard softmax update logic from the GQA example
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        
        for i in T.Parallel(BLOCK_SIZE_Q):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(BLOCK_SIZE_Q, BLOCK_SIZE_K):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(BLOCK_SIZE_Q):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def PV_MMA(
        V_shared: T.SharedBuffer, acc_s_cast: T.FragmentBuffer,
        acc_o: T.FragmentBuffer, scores_scale: T.FragmentBuffer
    ):
        """ Rescales accumulator and computes P @ V. """
        for i, j in T.Parallel(BLOCK_SIZE_Q, HID_BLOCK_V):
            acc_o[i, j] *= scores_scale[i]
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)


    # =================================================================
    # 2. Main Primitive Function (The CUDA Kernel)
    # =================================================================
    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype), K: T.Tensor(k_shape, dtype), V: T.Tensor(v_shape, dtype),
        POS: T.Tensor(pos_shape, "int32"), INDICES: T.Tensor(indices_shape, "int32"),
        KS_START_END: T.Tensor(ks_se_shape, "int32"),
        CONTEXT: T.Tensor(q_shape, dtype),
        # Paged KV args
        K_CACHE: T.Tensor(k_cache_shape, dtype), V_CACHE: T.Tensor(v_cache_shape, dtype),
        BLOCK_TABLE: T.Tensor(block_table_shape, "int32"), CACHE_SEQ_LENS: T.Tensor(cache_seq_lens_shape, "int32"),
        # RoPE args
        COS: T.Tensor(cos_shape, dtype), SIN: T.Tensor(sin_shape, dtype)
    ):
        # Grid definition: iterate over destination blocks (bx), heads (by), and batch (bz)
        with T.Kernel(T.ceildiv(MAX_TDST, BLOCK_SIZE_Q), HEAD, q_shape[0], threads=threads) as (bx, by, bz):
            # --- Memory Allocation ---
            Q_shared = T.alloc_shared([BLOCK_SIZE_Q, HID_BLOCK_0], dtype)
            K_shared = T.alloc_shared([BLOCK_SIZE_K, HID_BLOCK_0], dtype)
            V_shared = T.alloc_shared([BLOCK_SIZE_K, HID_BLOCK_V], dtype)

            acc_s = T.alloc_fragment([BLOCK_SIZE_Q, BLOCK_SIZE_K], accum_dtype)
            acc_s_cast = T.alloc_fragment([BLOCK_SIZE_Q, BLOCK_SIZE_K], dtype)
            acc_o = T.alloc_fragment([BLOCK_SIZE_Q, HID_BLOCK_V], accum_dtype)
            
            scores_max = T.alloc_fragment([BLOCK_SIZE_Q], accum_dtype)
            scores_max_prev = T.alloc_fragment([BLOCK_SIZE_Q], accum_dtype)
            scores_scale = T.alloc_fragment([BLOCK_SIZE_Q], accum_dtype)
            scores_sum = T.alloc_fragment([BLOCK_SIZE_Q], accum_dtype)
            logsum = T.alloc_fragment([BLOCK_SIZE_Q], accum_dtype)
            pos_tdst_frag = T.alloc_fragment([BLOCK_SIZE_Q], "int32")

            # --- Initialization ---
            scale_factor = (1.0 / HID)**0.5 * 1.44269504
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # --- Load Q Tile and its positions ---
            q_start_idx = bx * BLOCK_SIZE_Q
            q_offsets = T.arange(0, BLOCK_SIZE_Q)
            mask_tdst = (q_start_idx + q_offsets) < MAX_TDST
            
            T.copy(Q[bz, q_start_idx : q_start_idx + BLOCK_SIZE_Q, by, :], Q_shared, mask=mask_tdst)
            T.copy(POS[bz, q_start_idx : q_start_idx + BLOCK_SIZE_Q], pos_tdst_frag, mask=mask_tdst)
            
            # --- In-kernel RoPE on Q (if needed) ---
            if NEED_APPLY_ROPE:
                ApplyRoPE(Q_shared, pos_tdst_frag, Q, COS, SIN, bz, by, q_start_idx)

            # --- Main Pipelined Loop for Block-Sparse Attention ---
            bdst = bx # blockIdx.x corresponds to destination blocks
            range_start = KS_START_END[bz, bdst, 0]
            range_end = KS_START_END[bz, bdst, 1]

            # CRITICAL: This loop assumes TileLang can handle dynamic bounds.
            for i_bk in T.Pipelined(T.ceildiv(range_end - range_start, BLOCK_BK), num_stages=num_stages):
                
                current_block_idx = range_start + i_bk * BLOCK_BK
                mask_bk = current_block_idx < range_end
                
                idx_tsrc_start = INDICES[bz, bdst, current_block_idx]
                idx_kv_head = by // KV_HEAD_REPEAT
                
                # --- Pipelined Stages ---
                # 1. Load K and V for the current sparse block
                if USING_PAGES:
                    # This logic is complex and might be inefficient. It assumes no page crossing within a block.
                    # A more robust implementation would handle that case.
                    k_offsets = T.arange(0, BLOCK_SIZE_K)
                    idx_tsrc_block = idx_tsrc_start + k_offsets
                    seq_len = CACHE_SEQ_LENS[bz, 0]
                    mask_k_load = mask_bk & (idx_tsrc_block < seq_len)
                    
                    page_indices = idx_tsrc_block // PAGE_SIZE
                    page_offsets = idx_tsrc_block % PAGE_SIZE
                    block_ptr = BLOCK_TABLE[bz, page_indices[0]] # Simplification: assumes one page per block

                    T.copy(K_CACHE[block_ptr, page_offsets, idx_kv_head, :], K_shared, mask=mask_k_load)
                    T.copy(V_CACHE[block_ptr, page_offsets, idx_kv_head, :], V_shared, mask=mask_k_load)
                else: # Dense path
                    T.copy(K[bz, idx_tsrc_start:idx_tsrc_start+BLOCK_SIZE_K, idx_kv_head, :], K_shared, mask=mask_bk)
                    T.copy(V[bz, idx_tsrc_start:idx_tsrc_start+BLOCK_SIZE_K, idx_kv_head, :], V_shared, mask=mask_bk)

                # 2. Fused QK GEMM and Softmax update
                QK_MMA_and_Softmax(Q_shared, K_shared, acc_s, acc_s_cast,
                                   scores_max, scores_max_prev, scores_scale,
                                   scores_sum, logsum, scale_factor,
                                   bx, idx_tsrc_start, pos_tdst_frag, mask_bk)

                # 3. Compute P @ V
                PV_MMA(V_shared, acc_s_cast, acc_o, scores_scale)
            
            # --- Finalization and Store Output ---
            for i, j in T.Parallel(BLOCK_SIZE_Q, HID_BLOCK_V):
                acc_o[i, j] /= logsum[i]
            
            T.copy(acc_o, CONTEXT[bz, q_start_idx:q_start_idx+BLOCK_SIZE_Q, by, :], mask=mask_tdst)

    return main


# =================================================================
# 4. PyTorch Wrapper
# =================================================================
def block_sparse_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    seq_lens: Tensor,
    indices: Tensor,
    ks: Tensor,
    ks_count: Tensor,
    ks_start_end: Tensor,
    args: "HiPAttentionArgs",
    access_counter: Tensor,
    cache_miss_counter: Tensor,
    EXTEND_BACKEND: str = "streaming",
    model_context_length: int = 131072,
    extend_context_length: int = 131072,
    offload_update_cache: bool = False,
    return_running_statistics: bool = False,
    tune: bool = False
):
    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        _, TSRC, KV_HEAD, _ = k.shape
        HID_V = v.shape[-1]
    else: # Paged attention case
        KV_HEAD = args.k_cache.shape[2]
        HID_V = args.v_cache.shape[3]
        TSRC = args.k_cache.shape[0] * args.k_cache.shape[1]

    context = torch.empty_like(q)
    KV_HEAD_REPEAT = HEAD // KV_HEAD

    # Define shapes to pass to the kernel
    q_shape = list(q.shape)
    k_shape = list(k.shape) if k is not None else [0,0,0,0]
    v_shape = list(v.shape) if v is not None else [0,0,0,0]
    pos_shape = list(seq_lens.shape)
    indices_shape = list(indices.shape)
    ks_se_shape = list(ks_start_end.shape)
    k_cache_shape = list(args.k_cache.shape) if args.k_cache is not None else [0,0,0,0]
    v_cache_shape = list(args.v_cache.shape) if args.v_cache is not None else [0,0,0,0]
    block_table_shape = list(args.block_table.shape) if args.block_table is not None else [0,0]
    cache_seq_lens_shape = list(args.cache_seq_lens.shape) if args.cache_seq_lens is not None else [0,0]
    cos_shape = list(args.rope_cos.shape) if args.rope_cos is not None else [0,0]
    sin_shape = list(args.rope_sin.shape) if args.rope_sin is not None else [0,0]

    # Define static parameters
    static_params = {
        'q_shape': q_shape, 'k_shape': k_shape, 'v_shape': v_shape,
        'pos_shape': pos_shape, 'indices_shape': indices_shape, 'ks_se_shape': ks_se_shape,
        'k_cache_shape': k_cache_shape, 'v_cache_shape': v_cache_shape,
        'block_table_shape': block_table_shape, 'cache_seq_lens_shape': cache_seq_lens_shape,
        'cos_shape': cos_shape, 'sin_shape': sin_shape,
        'HEAD': HEAD, 'KV_HEAD_REPEAT': KV_HEAD_REPEAT, 'MAX_TDST': TDST, 'MAX_TSRC': TSRC,
        'USING_PAGES': args.using_paged_cache,
        'PAGE_SIZE': args.k_cache.shape[1] if args.using_paged_cache and args.k_cache is not None else -1,
        'USING_EXTEND': args.using_extend, 'NEED_APPLY_ROPE': args.need_apply_rope,
        'rope_range_begin': args.rope_range[0], 'rope_range_end': args.rope_range[1],
        'rope_is_neox_style': args.rope_is_neox_style,
        'IS_CAUSAL': args.is_causal,
    }

    # Define tunable parameters
    # These should be tuned for optimal performance
    # For now, using some default values
    tunable_params = {
        'BLOCK_SIZE_Q': args.block_size_q,
        'BLOCK_SIZE_K': args.block_size_k,
        'BLOCK_BK': 32 // args.block_size_k if args.block_size_k > 0 else 1,
        'HID_BLOCK_0': HID, # Assuming no splitting of HID for simplicity
        'HID_BLOCK_V': HID_V,
        'num_stages': 2,
        'threads': 256
    }
    
    # The tilelang.jit decorator will handle autotuning if `tune=True`
    # Here we define the kernel function with all parameters
    @tilelang.jit(out_idx=[6]) # CONTEXT is the 7th argument
    def _kernel(
        _q, _k, _v, _pos, _indices, _ks_se, _context,
        _k_cache, _v_cache, _block_table, _cache_seq_lens,
        _cos, _sin
    ):
        return block_sparse_attention_func(**static_params, **tunable_params)

    # Launch the kernel
    _kernel(
        q, k, v, seq_lens, indices, ks_start_end, context,
        args.k_cache, args.v_cache, args.block_table, args.cache_seq_lens,
        args.rope_cos, args.rope_sin
    )

    return context
