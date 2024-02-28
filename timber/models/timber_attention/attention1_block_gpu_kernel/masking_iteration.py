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
def _triton_kth_ascending(
    scores: tl.tensor, k: tl.tensor,
    BLOCK_SCORES: tl.constexpr,
) -> tl.tensor:
    sorted_score = tl.sort(scores)
    # tl.debug_barrier()
    sorted_score_mask = tl.arange(0, BLOCK_SCORES) < k
    return tl.max(sorted_score * sorted_score_mask + (-32000.0) * (~sorted_score_mask))

@triton.jit
def _masking_iteration_compute(
    # input matrices
    QUERIES, stride_queries_n, stride_queries_tdst, stride_queries_hid,
    KEYS, stride_keys_n, stride_keys_tsrc, stride_keys_hid,
    ATTEN_MASK, stride_atten_mask_n, stride_atten_mask_tsrc,
    SPARQ_INDICES, stride_sparq_indices_n, stride_sparq_indices_bdst, stride_sparq_indices_hid,
    
    # input / temp metrices (blocked)
    MASK, stride_mask_n, stride_mask_bdst, stride_mask_k,
    TMASK, stride_tmask_n, stride_tmask_bdst, stride_tmask_k,
    
    # temp vectors (blocked)
    WS, stride_ws_n, stride_ws_bdst,
    KS, stride_ks_n, stride_ks_bdst,
    TSRCS, stride_tsrcs_n, stride_tsrcs_bdst,
    SCORES, stride_scores_n, stride_scores_b, stride_scores_k,
    
    # operation variables (blocked)
    SCALE_UP: tl.constexpr, 
    N_PATCHES: tl.constexpr, 
    MASK_K: tl.constexpr, 
    TMASK_K: tl.constexpr, 
    IS_CAUSAL: tl.constexpr,
    
    # input variables
    KV_REPEAT_INTERLEAVE: int,
    N: int, 
    T_DST: int, 
    T_SRC: int, 
    B_DST: int, 
    B_SRC: int, 
    HID: int, 
    SPARQ_HID: int,
    N_COMPLETED: int,
    N_ITERATION: int,
    
    # vLLM compat inputs
    stride_keys_vllm_num_blcoks, 
    stride_keys_vllm_num_kv_heads,
    stride_keys_vllm_head_size_x,
    stride_keys_vllm_block_size,
    stride_keys_vllm_x,
    
    VLLM_NUM_BLOCKS: int, 
    VLLM_NUM_KV_HEADS: int,
    VLLM_HEAD_SIZE_X: int,
    VLLM_BLOCK_SIZE: int,
    VLLM_X: int, 
    VLLM_HEAD_SIZE: int,
    
    BLOCK_TABLES, 
    stride_block_tables_num_seqs, 
    stride_block_tables_max_num_blocks_per_seq,
    
    CONTEXT_LENGTH,
    stride_context_length_num_seqs,
    
    # block constant
    USING_SCORE_CACHE: tl.constexpr,
    KEY_CACHE_METHOD: tl.constexpr,
    SPARQ: tl.constexpr,
    REDUCE_METHOD: tl.constexpr,
    BLOCK_MASK_K: tl.constexpr, 
    BLOCK_TMASK_K: tl.constexpr, 
    BLOCK_MAX_DUP: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    REDUCE_STRDIE: tl.constexpr,
):
    idx_n = tl.program_id(2).to(tl.int64)
    
    idx_bdst = tl.program_id(1).to(tl.int64) + N_COMPLETED
    
    idx_kstride = tl.program_id(0).to(tl.int64)
    grid_kstride = tl.num_programs(0).to(tl.int64)
    
    """ non blocked
    # for each query
    w_old = ws[i, j, 0]
    t_src = t_srcs[i, j, 0]
    w_new = min(torch.round(w_old * scale_up), t_src)
    """
    
    if CONTEXT_LENGTH is not None:
        context_length = tl.load(
            CONTEXT_LENGTH +\
                ((idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS) * stride_context_length_num_seqs,
        ).to(tl.int64)
    
    w_old = tl.load(
        WS + \
            idx_n * stride_ws_n + \
            idx_bdst * stride_ws_bdst,
    )
    
    t_src = tl.load(
        TSRCS + \
            idx_n * stride_tsrcs_n + \
            idx_bdst * stride_tsrcs_bdst,
    )
    if CONTEXT_LENGTH is not None:
        t_src = tl.minimum(context_length, t_src)
    
    k_old = tl.load(
        KS + \
            idx_n * stride_ks_n +\
            idx_bdst * stride_ks_bdst,
    ).to(tl.int64)
    
    for idx_iteration in range(N_ITERATION):
        tl.debug_barrier()
        # tl.device_print("dd", idx_bdst)
        
        w_new = tl.minimum(
            tl.math.round(w_old.to(tl.float32) * SCALE_UP).to(tl.float32), 
            t_src
        ).to(tl.int64)
        
        """
        if w_old != w_new:
        """
        # if w_old == w_new:
        #     return

        tl.debug_barrier()
        mask_w = w_old != w_new
        tl.debug_barrier()
        
        """
        k_old = ks[i, j, 0]
        k_new = max(n_patches, int(min(mask_k * BLOCK_SIZE / t_src, 1.0) * w_new) c/ BLOCK_SIZE)
        k_new = min(t_src c/ BLOCK_SIZE, max(n_patches, k_new))
        """
        
        # """
        k_new = tl.maximum(
            N_PATCHES,
            (
                tl.minimum(
                    MASK_K / tl.cdiv(t_src, BLOCK_SIZE_K).to(tl.float32),
                    1.0
                ) * tl.cdiv(w_new, BLOCK_SIZE_K)
            ).to(tl.int64),
        )
        # """
            # k_new = tl.maximum(
            #     N_PATCHES,
            #     tl.cdiv(
            #         (tl.minimum((MASK_K * BLOCK_SIZE).to(tl.float32) / t_src.to(tl.float32), 1.0) * w_new.to(tl.float32)).to(tl.int64),
            #         BLOCK_SIZE
            #     ),
            # )
        # tl.device_print("before", t_src)
        k_new = tl.minimum(tl.cdiv(t_src, BLOCK_SIZE_K), tl.maximum(N_PATCHES, k_new))
        
        """
        # mask -> t_mask
        num_pixels = 0
        for k in range(k_old):
            loc = mask[i, j, k]
            loc_idx_start = int(loc * w_old)
            loc_idx_end = loc_idx_start + 1
            loc_idx_start = int(loc_idx_start / w_old * w_new)
            loc_idx_end = int(loc_idx_end / w_old * w_new)
            dup_pixels = loc_idx_end - loc_idx_start
            for l in range(dup_pixels):
                t_mask[i, j, num_pixels + l] = (loc_idx_start + l) / w_new
            num_pixels += dup_pixels
        """
        
        k_old_range = tl.arange(0, BLOCK_MASK_K).to(tl.int64)
        k_old_mask = tl.arange(0, BLOCK_MASK_K) < tl.cdiv(k_old, grid_kstride)
        # tl.debug_barrier()
        loc_vec = tl.load(
            MASK +\
                idx_n * stride_mask_n +\
                idx_bdst * stride_mask_bdst +\
                (k_old_range * grid_kstride + idx_kstride) * stride_mask_k,
            mask = mask_w & k_old_mask,
            other = 0
        )
        k_old_mask = k_old_mask & (loc_vec < 1.0)
        
        # w_old_fp = w_old.to(tl.float32)
        # w_new_fp = w_new.to(tl.float32)
        b_old_fp = tl.cdiv(w_old, BLOCK_SIZE_K).to(tl.float32)
        b_new_fp = tl.cdiv(w_new, BLOCK_SIZE_K).to(tl.float32)
        loc_idx_start_vec = (loc_vec * b_old_fp).to(tl.int64)
        loc_idx_end_vec = loc_idx_start_vec + 1
        loc_idx_start_vec = (loc_idx_start_vec.to(tl.float32) / b_old_fp * b_new_fp).to(tl.int64)
        loc_idx_end_vec = (loc_idx_end_vec.to(tl.float32) / b_old_fp * b_new_fp).to(tl.int64)
        
        dup_pixels_vec = loc_idx_end_vec - loc_idx_start_vec
        dup_pixels_vec = dup_pixels_vec * k_old_mask
        num_pixels_vec = tl.cumsum(dup_pixels_vec)
        dup_pixels_first = tl.min(num_pixels_vec)
        num_pixels_scalar = tl.max(num_pixels_vec)
        
        # num_pixels_scalar_exceed = tl.maximum(num_pixels_scalar - tl.cdiv(TMASK_K, grid_kstride), 0)
        # num_pixels_vec = tl.maximum(0, num_pixels_vec - num_pixels_scalar_exceed)
        dup_pixels_first = tl.min(num_pixels_vec)
        num_pixels_scalar = tl.max(num_pixels_vec)
        
        # NOTE: compiler bug?
        
        """
        dup_pixels_range = tl.arange(0, BLOCK_MAX_DUP)
        dup_pixels_mask = (dup_pixels_range[None, :] <= dup_pixels_vec[:, None]) & k_old_mask[:, None]
        
        tl.store(
            TMASK + \
                idx_n * stride_tmask_n +\
                idx_bdst * stride_tmask_bdst +\
                ((num_pixels_vec - dup_pixels_first)[:, None] + dup_pixels_range[None, :]) * stride_tmask_k,
            mask=dup_pixels_mask,
            value=(
                (loc_idx_start_vec[:, None] + tl.arange(0, BLOCK_MAX_DUP)[None, :]).to(tl.float32) / w_new.to(tl.float32)
            )
            # value = num_pixels_scalar=
        )
        """
        
        # interp_loc_vec_padded = (loc_idx_start_vec[:, None] + tl.arange(0, BLOCK_MAX_DUP)[None, :]).to(tl.float32) / w_new.to(tl.float32)
        # mask_interp_loc_vec_padded = tl.arange(0, BLOCK_MAX_DUP)[None, :] < dup_pixels_vec[:, None]
        # interp_loc_vec_padded = tl.reshape(interp_loc_vec_padded, BLOCK_MASK_K * BLOCK_MAX_DUP)
        
        
        # idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
        # mask_block_k = idx_block_k < BLOCK_SIZE_K
        idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED).to(tl.int64) * REDUCE_STRDIE
        mask_block_q = idx_block_q < BLOCK_SIZE_Q
        
        """
        # t_mask -> mask (using scores)
        if k_new < num_pixels:
        """
        if ((k_new < num_pixels_scalar) or (grid_kstride > 1)) or (REDUCE_STRDIE > 1):
        # if True:
            """
            # need top_k, so compute scores
            vec_q = queries[i, j, :]
            for k in range(num_pixels):
                # NOTE: nearest
                loc = t_mask[i, j, k]
                vec_k = keys[i, int(loc * t_src), :]
                
                score = torch.dot(vec_q, vec_k)
                scores[i, j, k] = -score # NOTE: store negative store
            """
            
            for _idx in range(BLOCK_MAX_DUP):
                # _idx = BLOCK_MAX_DUP - _idx - 1
                _value = (loc_idx_start_vec + _idx).to(tl.float32) / tl.cdiv(w_new, BLOCK_SIZE_K).to(tl.float32)
                if (_idx == 0) and (SCORES is not None):
                    _value *= -1.0
                
                # tl.atomic_and(
                #     TMASK + \
                #         idx_n * stride_tmask_n +\
                #         idx_bdst * stride_tmask_bdst +\
                #         (((num_pixels_vec - dup_pixels_first) + _idx).to(tl.int64) + grid_kstride * idx_kstride) * stride_tmask_k,
                #     mask=mask_w & (_idx <= dup_pixels_vec) & k_old_mask,
                #     val=0
                # )
                tl.atomic_xchg(
                    TMASK + \
                        idx_n * stride_tmask_n +\
                        idx_bdst * stride_tmask_bdst +\
                        (((num_pixels_vec - dup_pixels_first) + _idx).to(tl.int64) + grid_kstride * idx_kstride) * stride_tmask_k,
                    mask=mask_w & (_idx <= dup_pixels_vec) & k_old_mask,
                    val=_value
                )
                # tl.store(
                #     TMASK + \
                #         idx_n * stride_tmask_n +\
                #         idx_bdst * stride_tmask_bdst +\
                #         (((num_pixels_vec - dup_pixels_first) + _idx).to(tl.int64) + grid_kstride * idx_kstride) * stride_tmask_k,
                #     mask=mask_w & (_idx <= dup_pixels_vec) & k_old_mask,
                #     value=_value
                # )
            tl.debug_barrier()
            
            assert REDUCE_METHOD == 'max'
            scores = tl.zeros((BLOCK_TMASK_K,), dtype=tl.float32)
            scores += float("inf")
            
            idx_tdst = (idx_bdst * BLOCK_SIZE_Q + idx_block_q).to(tl.int64)
            mask_tdst = (idx_tdst < T_DST) & mask_block_q
            
            if ATTEN_MASK is not None:
                query_mask = tl.load(
                    ATTEN_MASK +\
                        idx_n * stride_atten_mask_n +\
                        (idx_tdst + T_SRC - T_DST) * stride_atten_mask_tsrc,
                    mask = mask_w & mask_tdst,
                    other = False
                ).to(tl.int1)
            
            num_pixels_range = tl.arange(0, BLOCK_TMASK_K).to(tl.int64)
            num_pixels_mask = num_pixels_range < num_pixels_scalar
            idx_tsrc_block = tl.load(
                TMASK +\
                    idx_n * stride_tmask_n +\
                    idx_bdst * stride_tmask_bdst +\
                    (num_pixels_range + grid_kstride * idx_kstride) * stride_tmask_k,
                mask = mask_w & num_pixels_mask,
                other = 0,
            )
            # NOTE: random key selection with in the block
            idx_tsrc_block = (idx_tsrc_block.to(tl.float32) * t_src.to(tl.float32)).to(tl.int64)
            mask_tsrc_block = num_pixels_mask
            
            if USING_SCORE_CACHE:
                mask_tsrc_block_reuse = idx_tsrc_block < 0
                if idx_iteration > 0:
                    mask_tsrc_block = (~mask_tsrc_block_reuse) & mask_tsrc_block
                # tl.device_print('ff', tl.sum(mask_tsrc_block_reuse.to(tl.float32)) / BLOCK_TMASK_K)
                idx_tsrc_block = tl.math.abs(idx_tsrc_block)
            
            for _idx_block_k in range(0, BLOCK_SIZE_K, 1):
                scores_partial = tl.zeros((BLOCK_SIZE_Q_PADDED, BLOCK_TMASK_K), dtype=tl.float32)
                
                # [BLOCK_TMASK_K, ]
                idx_tsrc = (idx_tsrc_block + _idx_block_k).to(tl.int64)
                mask_tsrc = (idx_tsrc < T_SRC) & (_idx_block_k < BLOCK_SIZE_K) & mask_tsrc_block
                
                # if CONTEXT_LENGTH is not None:
                #     mask_tsrc = mask_tsrc & (idx_tsrc < context_length)
                
                # [BLOCK_TMASK_K, ]
                if ATTEN_MASK is not None:
                    key_mask = tl.load(
                        ATTEN_MASK +\
                            idx_n * stride_atten_mask_n +\
                            idx_tsrc * stride_atten_mask_tsrc,
                        mask = mask_w & mask_tsrc,
                        other = False,
                    ).to(tl.int1)
                # mask_tsrc = mask_tsrc & key_mask
                
                mask_strided_block_q = True #(idx_block_q % REDUCE_STRDIE) == 0
                hidden_size = SPARQ_HID if SPARQ else HID
                for pid_hid in range(tl.cdiv(hidden_size, BLOCK_HID)):
                    idx_hid = (tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID).to(tl.int64)
                    mask_hid = idx_hid < hidden_size
                    
                    if SPARQ:
                        idx_hid = tl.load(
                            SPARQ_INDICES +\
                                idx_n * stride_sparq_indices_n +\
                                idx_bdst * stride_sparq_indices_bdst +\
                                idx_hid * stride_sparq_indices_hid,
                            mask = mask_w & mask_hid,
                            other = HID,
                        )
                    mask_hid = idx_hid < HID
                    
                    # [BLOCK_SIZE_PADDED: tdst, BLOCK_HID: hid]
                    mask_vec_q = (
                        mask_hid[None, :] &
                        mask_tdst[:, None] &
                        mask_block_q[:, None] &
                        mask_strided_block_q[:, None] &
                        True
                    )
                    if ATTEN_MASK is not None:
                        mask_vec_q = mask_vec_q & query_mask[:, None]
                    vec_q = tl.load(
                        QUERIES +\
                            idx_n * stride_queries_n +\
                            idx_tdst[:, None] * stride_queries_tdst +\
                            idx_hid[None, :] * stride_queries_hid,
                        mask = mask_w & mask_vec_q,
                        other = 0,
                    )
                    
                    # [BLOCK_HID: hid, BLOCK_TMASK_K: tsrc]
                    vec_k_mask = (
                        num_pixels_mask[None, :] &
                        mask_hid[:, None] &
                        mask_tsrc[None, :] &
                        # key_mask[None, :] &
                        True
                    )
                    if CONTEXT_LENGTH is not None:
                        vec_k_mask &= (
                            (idx_tsrc < context_length)[None, :]
                        )
                    if KEY_CACHE_METHOD == 'cont':
                        # [BLOCK_HID: hid, BLOCK_TMASK_K: tsrc]
                        vec_k = tl.load(
                            KEYS +\
                                (idx_n // KV_REPEAT_INTERLEAVE) * stride_keys_n +\
                                idx_tsrc[None, :] * stride_keys_tsrc + \
                                idx_hid[:, None] * stride_keys_hid,
                            mask = mask_w & vec_k_mask,
                            other = 0,
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
                            mask = mask_w & mask_tsrc,
                        ).to(tl.int64)
                        offset_block = (idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)).to(tl.int64)
                        
                        # [BLOCK_HID: hid, BLOCK_TMASK_K: tsrc]
                        vec_k = tl.load(
                            KEYS +\
                                idx_block[None, :] * stride_keys_vllm_num_blcoks +\
                                idx_head * stride_keys_vllm_num_kv_heads +\
                                (idx_hid[:, None] // VLLM_X) * stride_keys_vllm_head_size_x +\
                                offset_block[None, :] * stride_keys_vllm_block_size +\
                                (idx_hid[:, None] % VLLM_X) * stride_keys_vllm_x,
                            mask = mask_w & vec_k_mask,
                            other = 0,
                        )
                    else:
                        raise Exception()
                    
                    # [BLOCK_SIZE_PADDED: tdst, BLOCK_TMASK_K: tsrc]
                    if vec_k.dtype == tl.uint8:
                        vec_k = vec_k.to(tl.float8e5, bitcast=True).to(vec_q.dtype)
                    scores_micro = -tl.dot(vec_q, vec_k)
                    scores_partial += scores_micro.to(scores_partial.dtype)
                
                # [BLOCK_SIZE_PADDED: tdst, BLOCK_TMASK_K: tsrc]
                scores_partial_ignore_mask = (
                    (~num_pixels_mask[None, :]) |
                    (~mask_tdst[:, None]) |
                    (~mask_tsrc[None, :]) |
                    (~mask_block_q[:, None]) |
                    (~mask_strided_block_q[:, None]) |
                    (scores_partial == 0) |
                    False
                )
                
                if IS_CAUSAL:
                    scores_partial_ignore_mask |= (
                        ((idx_tdst[:, None] + T_SRC - T_DST) < idx_tsrc[None, :]) |
                        False
                    )
                
                if ATTEN_MASK is not None:
                    scores_partial_ignore_mask |= (
                        (~key_mask[None, :]) |
                        (~query_mask[:, None]) |
                        False
                    )
                
                if CONTEXT_LENGTH is not None:
                    scores_partial_ignore_mask |= (
                        (idx_tsrc[None, :] >= context_length)
                    )
                
                # NOTE: owo powerful dark magic. select first / last block always. testing sink attention.
                # scores_partial_force_mask = (
                #     (
                #         (idx_tsrc[None, :] == 0) | 
                #         (num_pixels_range[None, :] >= (num_pixels_scalar - 1)) |
                #         # ((idx_tdst[:, None]) <= idx_tsrc[None, :]) |
                #         False
                #     ) &
                #     ((idx_tdst[:, None] + T_SRC - T_DST) >= idx_tsrc[None, :]) &
                #     (mask_tsrc[None, :] & mask_tdst[:, None]) &
                #     (scores_partial != 0) &
                #     True
                # )
                scores_partial_force_mask = False
                
                scores_partial_ignore_mask = scores_partial_ignore_mask & (~scores_partial_force_mask)
                
                # NOTE: reduce
                scores_partial = scores_partial + scores_partial_ignore_mask * 32000.0
                scores_partial = scores_partial + scores_partial_force_mask * (-32000.0)
                # scores_partial = scores_partial * (~scores_partial_force_mask)
                scores_partial = tl.min(scores_partial, axis=0)
                scores = tl.minimum(scores, scores_partial)
            
            if USING_SCORE_CACHE:
                if idx_iteration > 0:
                    idx_cache_score = tl.cumsum(mask_tsrc_block_reuse.to(tl.int64)) - 1
                    tl.debug_barrier()
                    scores = tl.load(
                        SCORES +\
                            idx_n * stride_mask_n +\
                            idx_bdst * stride_mask_bdst +\
                            idx_cache_score * stride_mask_k,
                        mask=mask_tsrc_block_reuse & num_pixels_mask,
                        other=scores,
                    ).to(scores.dtype)
                    tl.debug_barrier()
            
            # done compute reduced scores
            
            """
            _, topk_indices = torch.topk(scores[i, j, :num_pixels], k=k_new, largest=False)
            for k in range(k_new):
                mask[i, j, k] = t_mask[i, j, topk_indices[k]]
            """
            
            # tl.device_print("", scores)
            
            # select min-k from negative scores -> select top-k
            masked_scores = scores
            
            kth = tl.cdiv(k_new, grid_kstride)
            scores_kth_large = _triton_kth_ascending(masked_scores, kth, BLOCK_TMASK_K)
            # scores_avg = tl.sum(masked_scores * (masked_scores < 1.0)) / num_pixels_scalar
            # scores_min = tl.min(masked_scores)
            # scores_kth_large = scores_avg # - (scores_min * 0.1)
            topk_mask = masked_scores <= scores_kth_large
            
            topk_mask_cumsum = tl.cumsum(topk_mask.to(tl.int64))
            topk_range = tl.minimum((topk_mask_cumsum - 1) * topk_mask, kth - 1).to(tl.int64)
            
            temp_range = tl.arange(0, BLOCK_TMASK_K).to(tl.int64)
            temp_mask = temp_range < num_pixels_scalar
            temp = tl.load(
                TMASK +\
                    idx_n * stride_tmask_n +\
                    idx_bdst * stride_tmask_bdst +\
                    (temp_range + grid_kstride * idx_kstride) * stride_tmask_k,
                mask=mask_w & temp_mask,
                other=0
            )
            # tl.atomic_and(
            #     MASK +\
            #         idx_n * stride_mask_n +\
            #         idx_bdst * stride_mask_bdst +\
            #         (topk_range * grid_kstride + idx_kstride) * stride_mask_k,
            #     mask=mask_w & topk_mask & temp_mask,
            #     val=0,
            # )
            tl.atomic_xchg(
                MASK +\
                    idx_n * stride_mask_n +\
                    idx_bdst * stride_mask_bdst +\
                    (topk_range * grid_kstride + idx_kstride) * stride_mask_k,
                mask=mask_w & topk_mask & temp_mask,
                val=temp,
            )
            # tl.store(
            #     MASK +\
            #         idx_n * stride_mask_n +\
            #         idx_bdst * stride_mask_bdst +\
            #         (topk_range * grid_kstride + idx_kstride) * stride_mask_k,
            #     mask=mask_w & topk_mask & temp_mask,
            #     value=temp,
            #     # value=0.1,
            # )
            if SCORES is not None:
                # tl.atomic_and(
                #     SCORES +\
                #         idx_n * stride_mask_n +\
                #         idx_bdst * stride_mask_bdst +\
                #         (topk_range * grid_kstride + idx_kstride) * stride_mask_k,
                #     mask=mask_w & topk_mask & temp_mask,
                #     val=0,
                # )
                tl.atomic_xchg(
                    SCORES +\
                        idx_n * stride_mask_n +\
                        idx_bdst * stride_mask_bdst +\
                        (topk_range * grid_kstride + idx_kstride) * stride_mask_k,
                    mask=mask_w & topk_mask & temp_mask,
                    val=scores,
                )
            # tl.debug_barrier()
        else:
            """
            else:
                mask[i, j, :num_pixels] = t_mask[i, j, :num_pixels]
            """
            for _idx in range(BLOCK_MAX_DUP):
                idx_mask_out = ((num_pixels_vec - dup_pixels_first) + _idx).to(tl.int64)
                mask_mask_out = (idx_mask_out < BLOCK_MASK_K) & (idx_mask_out < num_pixels_scalar) & (_idx <= dup_pixels_vec) & k_old_mask
                value_mask_out = (loc_idx_start_vec + _idx).to(tl.float64)
                value_mask_out = value_mask_out / tl.cdiv(w_new, BLOCK_SIZE_K).to(tl.float64)
                
                tl.atomic_xchg(
                    MASK +\
                        idx_n * stride_mask_n +\
                        idx_bdst * stride_mask_bdst +\
                        (idx_mask_out * grid_kstride + idx_kstride) * stride_mask_k,
                    mask=mask_w & mask_mask_out,
                    val=value_mask_out,
                )
                # tl.store(
                #     MASK +\
                #         idx_n * stride_mask_n +\
                #         idx_bdst * stride_mask_bdst +\
                #         (idx_mask_out * grid_kstride + idx_kstride) * stride_mask_k,
                #     mask=mask_w & mask_mask_out,
                #     value=value_mask_out,
                # )
            tl.debug_barrier()
        
        """
        ws[i, j, 0] = w_new
        ks[i, j, 0] = min(k_new, num_pixels)
        """
        if mask_w:
            w_old = w_new
            k_old = tl.minimum(k_new, num_pixels_scalar * grid_kstride)
        tl.debug_barrier()
    tl.debug_barrier()
    if idx_kstride == (grid_kstride - 1):
        tl.atomic_xchg(
            WS +\
                idx_n * stride_ws_n +\
                idx_bdst * stride_ws_bdst,
            # mask = mask_w,
            val = w_old
        )
        tl.atomic_xchg(
            KS +\
                idx_n * stride_ks_n +\
                idx_bdst * stride_ks_bdst,
            # mask = mask_w,
            val = k_old
        )

def masking_iteration(
    # input matrices
    queries: Tensor, keys: Union[Tensor, "PagedKeyCacheVllmCompat"], attention_mask: Tensor,
    # input metrices (blocked) 
    mask: Tensor, t_mask: Tensor, sparq_indices, sparq_indices_strides,
    # temp vectors (blocked)
    ws: Tensor, ks: Tensor, t_srcs: Tensor, 
    # operator variables
    scale_up: float, n_patches: int, mask_k: int, is_causal: bool,
    # iteration controls
    i_iteration: int, n_iteration: int,
    # input constant
    KV_REPEAT_INTERLEAVE: int,
    N: int, 
    T_DST: int, 
    T_SRC: int, 
    B_DST: int, 
    B_SRC: int, 
    HID: int, 
    SPARQ: bool, 
    SPARQ_HID: int,
    N_COMPLETED: int,
    # kernel constant
    BLOCK_SIZE_Q: int, 
    BLOCK_SIZE_K: int, 
    REDUCE_METHOD: str,
    REDUCE_STRIDE: int,
    DEBUG: bool = False,
):  
    if DEBUG:
        # print(ws)
        # print(ks[0, 10])
        # print(mask[0, 10])
        # print(t_srcs)
        print(
            'masking_iteration', 
            queries.shape, queries.data_ptr(), 
            keys.shape, keys.data_ptr(), 
            mask.shape, mask.data_ptr(),
            t_mask.shape, t_mask.data_ptr(),
            ws.shape, ws.data_ptr(),
            ks.shape, ks.data_ptr(),
            t_srcs.shape, t_srcs.data_ptr(),
            N, T_DST, T_SRC, B_DST, B_SRC, HID,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
            REDUCE_METHOD,
        )
        K = mask.shape[-1]
        assert t_srcs.min() > 0
        assert t_srcs.max() <= T_SRC
        assert ks.min() >= 0
        assert ks.max() <= K
        assert keys.shape[1] == T_SRC
        assert queries.shape[1] == T_DST
        assert mask.min() >= 0
        # assert mask.max() < 1
        assert t_mask.min() >= 0
        # assert t_mask.max() < 1
    
    BLOCK_MASK_K = triton.next_power_of_2(mask.shape[-1])
    BLOCK_TMASK_K = triton.next_power_of_2(t_mask.shape[-1])
    # print(BLOCK_MASK_K, BLOCK_TMASK_K)
    
    # if i_iteration == 0 or i_iteration == (n_iteration - 1):
    #     pass
    # else:
    #     if i_iteration > 1:
    #         BLOCK_MASK_K = BLOCK_MASK_K // scale_up
    #     BLOCK_TMASK_K = BLOCK_TMASK_K // scale_up
    
    BLOCK_HID = triton.next_power_of_2(HID)
    if SPARQ:
        BLOCK_HID = triton.next_power_of_2(max(16, SPARQ_HID))
    if BLOCK_TMASK_K >= 1024:
        BLOCK_HID = min(BLOCK_HID, 16)
    elif BLOCK_TMASK_K >= 512:
        BLOCK_HID = min(BLOCK_HID, 32)
    elif BLOCK_TMASK_K >= 256:
        BLOCK_HID = min(BLOCK_HID, 64)
    elif BLOCK_TMASK_K >= 128:
        BLOCK_HID = min(BLOCK_HID, 128)
    # print(BLOCK_HID, BLOCK_TMASK_K)
    
    if isinstance(keys, Tensor):
        KEY_CACHE_METHOD = 'cont'
        stride_keys_vllm = (0, 0, 0, 0, 0)
        VLLM_NUM_BLOCKS = 0
        VLLM_NUM_KV_HEADS = 0
        VLLM_HEAD_SIZE_X = 0
        VLLM_BLOCK_SIZE = 0
        VLLM_X = 0
        VLLM_HEAD_SIZE = 0
        block_tables = keys
        block_tables_stride = (0, 0)
        context_length = None
        context_length_stride = (0,)
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
        stride_keys_vllm = keys.key_cache.stride()
        (
            VLLM_NUM_BLOCKS, 
            VLLM_NUM_KV_HEADS, 
            VLLM_HEAD_SIZE_X, 
            VLLM_BLOCK_SIZE, 
            VLLM_X
        ) = keys.key_cache.shape
        VLLM_HEAD_SIZE = VLLM_HEAD_SIZE_X * VLLM_X
        block_tables = keys.block_table
        block_tables_stride = block_tables.stride()
        assert len(block_tables_stride) == 2
        
        context_length = keys.context_length
        context_length_stride = context_length.stride()
        assert len(context_length_stride) == 1
        
        # context_length = keys.context_length
        # context_length = context_length.unsqueeze(-1).repeat_interleave(VLLM_NUM_KV_HEADS, dim=0)
        # assert t_srcs.shape == context_length.shape, f"{t_srcs.shape} == {context_length.shape}"
        # t_srcs = context_length
    else:
        raise Exception()
    
    # NOTE: may improve latency, but hurt performance too much
    GRID_KSTRIDE = 1
    
    # NOTE: may improve latency, but hurt performance too much
    USING_SCORE_CACHE = False
    if USING_SCORE_CACHE:
        scores = torch.zeros_like(mask, dtype=torch.float32)
    else:
        scores = None
    
    grid = (GRID_KSTRIDE, B_DST - N_COMPLETED, N)
    
    # HID cannot be chunked if use reduce
    # if REDUCE_METHOD in ['max', 'sum']:
    #     assert HID <= BLOCK_HID
    assert REDUCE_METHOD in ['max', 'sum', 'first']
    
    assert queries.ndim == 3
    assert keys.ndim == 3
    if attention_mask is not None:
        assert attention_mask.ndim == 2
    assert mask.ndim == 3
    assert t_mask.ndim == 3
    assert ws.ndim == 2
    assert ks.ndim == 2
    assert t_srcs.ndim == 2
    _masking_iteration_compute[grid](
        # input matrices
        queries, *queries.stride(),
        keys, *keys.stride(),
        attention_mask, *(attention_mask.stride() if attention_mask is not None else (0, 0)),
        sparq_indices, *sparq_indices_strides,
        
        # input matrices (blocked)
        mask, *mask.stride(),
        t_mask, *t_mask.stride(),
        
        # temp vectors (blocked)
        ws, *ws.stride(),
        ks, *ks.stride(),
        t_srcs, *t_srcs.stride(),
        scores, *(scores.stride() if scores is not None else (0, 0, 0)),
        
        # operation variables
        float(scale_up), int(n_patches), int(mask_k), int(t_mask.shape[-1]), is_causal,
        
        # input variables
        KV_REPEAT_INTERLEAVE, 
        N, 
        T_DST, 
        T_SRC, 
        int(B_DST), 
        int(B_SRC), 
        HID, 
        SPARQ_HID, 
        N_COMPLETED,
        n_iteration,
        
        # vLLM compat inputs
        *stride_keys_vllm,
        
        VLLM_NUM_BLOCKS,
        VLLM_NUM_KV_HEADS,
        VLLM_HEAD_SIZE_X,
        VLLM_BLOCK_SIZE,
        VLLM_X,
        VLLM_HEAD_SIZE,
        
        block_tables, *block_tables_stride,
        
        context_length, *context_length_stride,
        
        # block constant
        USING_SCORE_CACHE,
        KEY_CACHE_METHOD,
        SPARQ,
        REDUCE_METHOD,
        triton.cdiv(BLOCK_MASK_K, GRID_KSTRIDE),
        triton.cdiv(BLOCK_TMASK_K, GRID_KSTRIDE),
        triton.next_power_of_2(math.ceil(scale_up)),
        int(BLOCK_HID),
        int(BLOCK_SIZE_Q),
        next_multiple_of(triton.cdiv(BLOCK_SIZE_Q, REDUCE_STRIDE), 16),
        int(BLOCK_SIZE_K),
        next_multiple_of(BLOCK_SIZE_K, 1),
        REDUCE_STRIDE,
        
        # num_warps=max(2, (min(8, max(BLOCK_TMASK_K//32, 1)) if SPARQ else 4) // GRID_KSTRIDE),
        num_warps=16,
        num_stages=2,
        enable_warp_specialization=False,
    )