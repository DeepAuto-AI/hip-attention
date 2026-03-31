"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import math
import os
import sys
import warnings
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np

# import pytest
import torch
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.extra import libdevice
from triton.language.standard import _log2, sum, zeros_like

from hip_attn.v1_2.attention_metadata import safe_stride
from hip_attn.v1_2.triton_argsort import argsort
from hip_attn.v1_2.utils import capture, triton_jit


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


@triton.jit
def convert_fp8_to_bf16(k: tl.tensor):
    if k is None:
        return None

    if k.dtype == tl.uint8:
        k = k.to(tl.float8e5, bitcast=True)
    if (
        (k.dtype == tl.float8e5)
        | (k.dtype == tl.float8e4nv)
        | (k.dtype == tl.float8e4b8)
        | (k.dtype == tl.float8e4b15)
    ):
        k = k.to(tl.bfloat16)

    return k


@triton.jit
def winner_update_inline(
    update_exp_sum,  # (M,) float32 – candidate scores per query (lanes)
    start_n,  # (1,) int64   – candidate key-block ids per query
    BSA_BLOCK_SUMS,  # pointer to values
    stride_bsk,
    stride_bsm,
    BSA_HEAP_INDICES,  # pointer to winner tree indices
    stride_hik,
    stride_him,
    BSA_INDICES,  # pointer to bsa block indices
    stride_bik,
    stride_bim,
    BSA_K,  # K in top-K
    BSA_LOGK,  # log2(K)
    BLOCK_M,  # query block size
    start_m,
    mask_m,  # query mask
    root_v,  # root min value stored in smem
    root_lf,  # root min index stored in smem
    root_idx,
    MASKING,
    mask,
    REVERSE_ITER,
):

    row_bs = ((start_m * BLOCK_M) + tl.arange(0, BLOCK_M)) * stride_bsm
    row_hi = ((start_m * BLOCK_M) + tl.arange(0, BLOCK_M)) * stride_him
    row_bi = ((start_m * BLOCK_M) + tl.arange(0, BLOCK_M)) * stride_bim
    # lanes that actually update (top-k max → min-winner tree)
    # we update when the sum is greater than root and the index is smaller
    # and query overflow mask allows it (mask_m)
    # and there is at least one unmasked element in this row (mask from causal and window)

    beat = update_exp_sum > root_v
    beat |= (update_exp_sum == root_v) & (start_n < root_idx)
    beat &= mask_m & (update_exp_sum > float("-inf"))
    if MASKING:
        beat &= tl.sum(mask, 1) > 0

    if tl.max(beat) > 0:
        root_leaf = BSA_K + root_lf
        leaf_off_bs = root_leaf * stride_bsk + row_bs
        tl.store(
            BSA_BLOCK_SUMS + leaf_off_bs.to(tl.int64),
            update_exp_sum.to(root_v.dtype),
            mask=beat,
        )
        leaf_off_bi = root_leaf * stride_bik + row_bi
        tl.store(BSA_INDICES + leaf_off_bi, start_n.to(tl.int32), mask=beat)

        # --- climb: keep the path child (value, pointer, index) in registers ---
        path_v = tl.where(beat, update_exp_sum, root_v).to(root_v.dtype)  # fp32
        path_p = root_lf  # int64 leaf-id [0..BSA_K-1]
        path_idx = tl.where(beat, start_n.to(tl.int32), root_idx)
        child = root_leaf  # int64 node idx [BSA_K..2*BSA_K-1]
        node = child >> 1  # parent idx [1..BSA_K-1]

        for i in tl.static_range(
            0, BSA_LOGK
        ):  # exactly BSA_LOGK steps; last write is root (node==1)
            # which side is our path child?
            sib = (
                child ^ 1
            )  # even == left odd == right, so just flip the bit to get sibling

            # load ONLY sibling from memory (unmasked loads)
            sib_v = tl.load(BSA_BLOCK_SUMS + (sib * stride_bsk + row_bs).to(tl.int64))
            sib_p = tl.load(BSA_HEAP_INDICES + (sib * stride_hik + row_hi).to(tl.int64))
            sib_idx = tl.load(BSA_INDICES + (sib * stride_bik + row_bi).to(tl.int64))

            # path with lower value bubles up as min. If there is a tie, break in favor
            # of keeping the older token
            take_path = (path_v < sib_v) | ((path_v == sib_v) & (path_idx < sib_idx))

            # print(f"path v, sib v, take path: ", path_v, sib_v, take_path)
            win_v = tl.where(take_path, path_v, sib_v)
            win_p = tl.where(take_path, path_p, sib_p)
            win_idx = tl.where(take_path, path_idx, sib_idx)

            tl.store(
                BSA_BLOCK_SUMS + (node * stride_bsk + row_bs).to(tl.int64),
                win_v,
                mask=beat,
            )
            tl.store(
                BSA_HEAP_INDICES + (node * stride_hik + row_hi).to(tl.int64),
                win_p,
                mask=beat,
            )
            tl.store(
                BSA_INDICES + (node * stride_bik + row_bi).to(tl.int64),
                win_idx,
                mask=beat,
            )

            # move up one level
            child = node
            node = node >> 1
            path_v = win_v
            path_p = win_p
            path_idx = win_idx

        root_v = tl.where(beat, path_v, root_v)
        root_lf = tl.where(beat, path_p, root_lf)
        root_idx = tl.where(beat, path_idx, root_idx)
    return root_v, root_lf, root_idx, beat


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    l_bsa,
    m_i,
    q,
    q_nope,
    K_ZH,
    stride_kn,
    stride_kk,
    V_ZH,
    stride_vn,
    stride_vk,
    mask_idx,
    start_m,
    qk_scale,
    k_descale,
    v_descale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_NOPE: tl.constexpr,
    HEAD_ROPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    offs_m,
    offs_n,
    mask_m,
    N_CTX,
    N_KV,
    fp8_v: tl.constexpr,
    USING_PAGED_CACHE: tl.constexpr,
    K_CACHE,
    stride_k_cache_t,
    stride_k_cache_page,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_t,
    stride_v_cache_page,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_tsrc,
    RETURN_BSA_MASK: tl.constexpr,
    BSA_MASK_SINK_TOKEN_SIZE: tl.constexpr,
    BSA_MASK_SW_SIZE,
    BSA_K: tl.constexpr,
    BSA_BLOCK_SIZE_K: tl.constexpr,
    BSA_INDICES,
    stride_bik,
    stride_bim,
    BSA_BLOCK_SUMS,
    stride_bsk,
    stride_bsm,
    BSA_HEAP_INDICES,
    stride_hik,
    stride_him,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    K_ROT,
    stride_k_rot_tsrc,
    stride_k_rot_hid,
    lo,
    hi,
    REVERSE_ITER: tl.constexpr,
    MASKING: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    MODEL_CONTEXT_LENGTH,
    SELF_EXTEND_SCALE,
    SELF_EXTEND_WINDOW,
    ONLINE_TOPK_METHOD: tl.constexpr = "online",
    running_mean=None,
    running_m2=None,
    num_tracking=None,
    topk_idx=None,
    EXACT_K: tl.constexpr = None,
    THRESHOLD_REFRESH_INTERVAL: tl.constexpr = None,
):
    # range of values handled by this stage
    # lo, hi = 0, N_KV
    # lo, hi = 0, tl.max(mask_idx) + 1

    SQRT2: tl.constexpr = 1.4142135623730951  # math.sqrt(2.0)
    EXACT_LOGK: tl.constexpr = _log2(EXACT_K)
    EST_K: tl.constexpr = BSA_K - EXACT_K

    if RETURN_BSA_MASK:
        if ONLINE_TOPK_METHOD == "tree":
            bs_idx = (
                1 * stride_bsk
                + (start_m * BLOCK_M).to(tl.int64)
                + tl.arange(0, BLOCK_M).to(tl.int64) * stride_bsm
            )
            root_v = tl.load(BSA_BLOCK_SUMS + bs_idx.to(tl.int64))  # node 1
            hi_idx = (
                1 * stride_hik
                + (start_m * BLOCK_M).to(tl.int64)
                + tl.arange(0, BLOCK_M).to(tl.int64) * stride_him
            )
            root_lf = tl.load(BSA_HEAP_INDICES + hi_idx.to(tl.int64))
            bi_idx = (
                1 * stride_bik
                + (start_m * BLOCK_M).to(tl.int64)
                + tl.arange(0, BLOCK_M).to(tl.int64) * stride_bim
            )
            root_idx = tl.load(BSA_INDICES + bi_idx.to(tl.int64))
        else:
            # THESE STRIDES LOOK LIKE THEY ARE REVERSED BUT THEY ARE NOT
            # IT IS DUE TO HEAP AND PLAIN NEEDING DIFFERENT MEMORY LAYOUT.
            # THE VARIABLE NAMES SUCK, BUT IT WAS THE EASIEST THING TO DO.
            bi_idx = (
                start_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
            ) * stride_bik + tl.arange(0, EXACT_K)[None, :].to(tl.int64) * stride_bim
            block_idx = tl.load(BSA_INDICES + bi_idx, mask=mask_m[:, None])
            bs_idx = (
                start_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
            ) * stride_bsk + tl.arange(0, EXACT_K)[None, :].to(tl.int64) * stride_bsm
            block_sums = tl.load(BSA_BLOCK_SUMS + bs_idx, mask=mask_m[:, None])
            block_sums_min, block_sums_min_idx = tl.min(
                block_sums, axis=-1, return_indices=True
            )  # (M, K) -> (M,)
            col_idx = tl.arange(0, EXACT_K)[None, :]  # (1, K) will be used later

    nblocks = tl.cdiv(hi - lo, BLOCK_N)
    last_start = lo + (nblocks - 1) * BLOCK_N
    if not USING_PAGED_CACHE:
        advance_init = lo
        advance = BLOCK_N
        if REVERSE_ITER:
            advance_init = last_start
            advance = -BLOCK_N

        V_ZHN = (
            V_ZH
            + (advance_init + tl.arange(0, BLOCK_N))[:, None].to(tl.int64) * stride_vk
            + tl.arange(0, HEAD_DIM)[None, :].to(tl.int64) * stride_vn
        )

        if HEAD_DIM == HEAD_ROPE:
            K_ZHN = (
                K_ZH
                + (advance_init + tl.arange(0, BLOCK_N))[None, :].to(tl.int64)
                * stride_kn
                + tl.arange(0, HEAD_DIM)[:, None].to(tl.int64) * stride_kk
            )
            K_ZHN_NOPE = None
        else:
            K_ZHN = (
                K_ZH
                + (advance_init + tl.arange(0, BLOCK_N))[None, :].to(tl.int64)
                * stride_kn
                + tl.arange(0, HEAD_ROPE)[:, None].to(tl.int64) * stride_kk
            )

            K_ZHN_NOPE = (
                K_ZH
                + (advance_init + tl.arange(0, BLOCK_N))[None, :].to(tl.int64)
                * stride_kn
                + tl.arange(0, HEAD_NOPE)[:, None].to(tl.int64) * stride_kk
            )

    # probability of being below the top-k threshold
    erf_pr = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # loop over k, v and update accumulator
    counter = tl.zeros((), tl.int32)
    for _start_n in tl.range(lo, hi, BLOCK_N, num_stages=3):
        start_n = _start_n
        if REVERSE_ITER:
            # if we start at something nonzero, we need to subtract lo to get the right block
            start_n = last_start - (_start_n - lo)

        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        idx_tsrc = offs_n + start_n
        mask_tsrc = idx_tsrc < hi

        if not USING_PAGED_CACHE:
            tl.static_assert(EXTEND_BACKEND == "none")
            k = tl.load(K_ZHN, mask=mask_tsrc[None, :])
        else:
            idx_t = tl.load(
                BLOCK_TABLE + idx_tsrc.to(tl.int64) * stride_block_table_tsrc,
                mask=mask_tsrc,
            ).to(tl.int64)
            k = tl.load(
                K_CACHE
                + idx_t[None, :] * stride_k_cache_t
                + 0 * stride_k_cache_page
                + (tl.arange(0, HEAD_ROPE) + HEAD_NOPE)[:, None] * stride_k_cache_hid,
                mask=mask_tsrc[None, :],
                other=0.0,
            )
        if HEAD_DIM != HEAD_ROPE:
            if not USING_PAGED_CACHE:
                tl.static_assert(EXTEND_BACKEND == "none")
                k_nope = tl.load(K_ZHN_NOPE, mask=mask_tsrc[None, :])
            else:
                idx_t = tl.load(
                    BLOCK_TABLE + idx_tsrc.to(tl.int64) * stride_block_table_tsrc,
                    mask=mask_tsrc,
                ).to(tl.int64)
                k_nope = tl.load(
                    K_CACHE
                    + idx_t[None, :] * stride_k_cache_t
                    + 0 * stride_k_cache_page
                    + (tl.arange(0, HEAD_NOPE))[:, None] * stride_k_cache_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                )
        else:
            k_nope = None

        k = convert_fp8_to_bf16(k)
        if k_nope is not None:
            k_nope = convert_fp8_to_bf16(k_nope)

        if EXTEND_BACKEND == "none":
            pass
        elif EXTEND_BACKEND == "self_extend":
            idx_hid = tl.arange(0, HEAD_ROPE)
            idx_hid_rot = (idx_hid + HEAD_ROPE // 2) % HEAD_ROPE
            idx_hid_cos_sin = idx_hid % (HEAD_ROPE // 2)
            rope_mult = tl.where((idx_hid + HEAD_ROPE // 2) < HEAD_ROPE, -1.0, 1.0)

            # max_pos_tsrc = tl.max(tl.where(mask_m, mask_idx, 0))

            # offset = idx_tsrc.to(tl.int64) - max_pos_tsrc
            # offset = tl.minimum(offset, 0)
            # idx_rope = tl.where(
            #     offset > (-SELF_EXTEND_WINDOW),
            #     offset + MODEL_CONTEXT_LENGTH - 1,
            #     (offset + SELF_EXTEND_WINDOW) // SELF_EXTEND_SCALE
            #     + MODEL_CONTEXT_LENGTH
            #     - 1
            #     - SELF_EXTEND_WINDOW,
            # )
            # # idx_rope = idx_tsrc

            max_pos_tsrc = tl.max(tl.where(mask_m, mask_idx, 0))
            min_pos_tsrc = tl.min(tl.where(mask_m, mask_idx, 987654321))

            self_sliding_window = tl.maximum(
                1024 + max_pos_tsrc - min_pos_tsrc, SELF_EXTEND_WINDOW
            )

            offset = idx_tsrc.to(tl.int64) - max_pos_tsrc
            idx_rope = tl.where(
                offset > (-self_sliding_window),
                offset + (MODEL_CONTEXT_LENGTH - 1),
                (offset + self_sliding_window) // SELF_EXTEND_SCALE
                + (MODEL_CONTEXT_LENGTH - 1)
                - self_sliding_window,
            )
            # idx_rope = idx_tsrc

            if not USING_PAGED_CACHE:
                k_rot = tl.load(
                    K_ROT
                    + idx_tsrc[None, :] * stride_k_rot_tsrc
                    + (idx_hid_rot + HEAD_NOPE)[:, None] * stride_k_rot_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                )
            else:
                idx_t = tl.load(
                    BLOCK_TABLE + idx_tsrc.to(tl.int64) * stride_block_table_tsrc,
                    mask=mask_tsrc,
                ).to(tl.int64)
                k_rot = tl.load(
                    K_CACHE
                    + idx_t[None, :] * stride_k_cache_t
                    + 0 * stride_k_cache_page
                    + (idx_hid_rot + HEAD_NOPE)[:, None] * stride_k_cache_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                )

            k_rot = convert_fp8_to_bf16(k_rot)

            cos = tl.load(
                COS
                + idx_rope[None, :] * stride_cos_t
                + idx_hid_cos_sin[:, None] * stride_cos_hid,
                mask=mask_tsrc[None, :],
                other=0.0,
            )
            sin = tl.load(
                SIN
                + idx_rope[None, :] * stride_sin_t
                + idx_hid_cos_sin[:, None] * stride_sin_hid,
                mask=mask_tsrc[None, :],
                other=0.0,
            )

            k = (
                k.to(tl.float32) * cos.to(tl.float32)
                + k_rot.to(tl.float32)
                * rope_mult.to(tl.float32)[:, None]
                * sin.to(tl.float32)
            ).to(k.dtype)
        else:
            raise Exception(EXTEND_BACKEND)

        if k_descale is not None:
            k *= k_descale
            k_nope *= k_descale

        q_dtype = q.dtype

        cq = tl.sqrt(HEAD_DIM * 1.0) / tl.sqrt(tl.sqrt(HEAD_DIM * 1.0))
        ck = 1 / tl.sqrt(tl.sqrt(HEAD_DIM * 1.0))

        # qk = tl.dot(
        #     (q * cq).to(q_dtype),
        #     (k.to(q_dtype) * ck).to(q_dtype)
        # ).to(tl.float32)
        # if HEAD_DIM != HEAD_ROPE:
        #     qk = qk + tl.dot(
        #         (q_nope * cq).to(q_dtype),
        #         (k_nope.to(q_dtype) * ck).to(q_dtype)
        #     ).to(tl.float32)
        qk = tl.dot((q * cq).to(q_dtype), (k * ck).to(q_dtype))
        if HEAD_DIM != HEAD_ROPE:
            qk = tl.dot(q_nope, k_nope, acc=qk)

        qk = qk * 1.44269504  # 1 / ln(2)

        if MASKING:
            mask = (mask_idx[:, None]) >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))

        # qk = tl.where(qk == 0, float("-inf"), qk)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # if we go backwards through the keys, some blocks may be totally masked
        # if block_m > block_n and the max will be at the init value -inf.
        # if we don't set it to 0 here, -inf - -inf will cause nans.
        qk -= tl.where(m_ij[:, None] == float("-inf"), 0, m_ij[:, None])

        p = tl.math.exp2(qk)  # 2^(x / ln(2)) == exp(x)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        # see note above about iterating backwards through keys
        alpha = tl.where(m_ij == float("-inf"), 1, tl.math.exp2(m_i - m_ij))
        l_i = (l_i * alpha + l_ij).to(l_i.dtype)

        # -- update block sums and indices for block sparse attention
        # if RETURN_BSA_MASK:
        if RETURN_BSA_MASK and (EXACT_K < BSA_K or start_n >= BSA_MASK_SINK_TOKEN_SIZE):
            # FIXME How can i this thing more dynamic?
            tl.static_assert(BLOCK_N >= BSA_BLOCK_SIZE_K)
            tl.static_assert(BLOCK_N <= (BSA_BLOCK_SIZE_K * 4))
            tl.static_assert(
                (BSA_BLOCK_SIZE_K == BLOCK_N)
                | ((BSA_BLOCK_SIZE_K * 2) == BLOCK_N)
                | ((BSA_BLOCK_SIZE_K * 4) == BLOCK_N)
            )

            mask = None
            if MASKING:
                mask = (mask_idx[:, None] - BSA_MASK_SW_SIZE) >= (
                    start_n + offs_n[None, :]
                )

            if MASKING:
                p_split = tl.where(mask, p, 0.0)
            else:
                p_split = p

            if start_n >= BSA_MASK_SINK_TOKEN_SIZE:
                l_bsa = (l_bsa * alpha + tl.sum(p_split, 1)).to(l_bsa.dtype)
            else:
                l_bsa = (l_bsa * alpha).to(l_bsa.dtype)

            if BLOCK_N == BSA_BLOCK_SIZE_K:
                l_ij_0 = tl.math.log2(l_ij) + tl.where(m_ij == float("-inf"), 0, m_ij)
            elif BLOCK_N == (BSA_BLOCK_SIZE_K * 2):
                p_split = tl.reshape(p_split, BLOCK_M, 2, BSA_BLOCK_SIZE_K)
                l_ij_all = tl.math.log2(tl.sum(p_split, 2)) + tl.where(
                    m_ij[:, None] == float("-inf"), 0, m_ij[:, None]
                )
                l_ij_0, l_ij_1 = tl.split(l_ij_all)
            elif BLOCK_N == (BSA_BLOCK_SIZE_K * 4):
                p_split = tl.reshape(p_split, BLOCK_M, 2, 2, BSA_BLOCK_SIZE_K)
                l_ij_all = tl.math.log2(tl.sum(p_split, 3)) + tl.where(
                    m_ij[:, None, None] == float("-inf"), 0, m_ij[:, None, None]
                )
                l_ij_01, l_ij_23 = tl.split(l_ij_all)
                l_ij_0, l_ij_1 = tl.split(l_ij_01)
                l_ij_2, l_ij_3 = tl.split(l_ij_23)
            else:
                raise Exception()

            if EXACT_K < BSA_K:
                for i_offset in tl.static_range(0, BLOCK_N, BSA_BLOCK_SIZE_K):
                    if i_offset == (0 * BSA_BLOCK_SIZE_K):
                        update_exp_sum = l_ij_0
                    elif i_offset == (1 * BSA_BLOCK_SIZE_K):
                        update_exp_sum = l_ij_1
                    elif i_offset == (2 * BSA_BLOCK_SIZE_K):
                        update_exp_sum = l_ij_2
                    elif i_offset == (3 * BSA_BLOCK_SIZE_K):
                        update_exp_sum = l_ij_3

                    causal_mask = l_ij > 0
                    log_score = update_exp_sum

                    # update running mean and std
                    num_tracking += causal_mask
                    delta = log_score - running_mean
                    running_mean = tl.where(
                        causal_mask,
                        running_mean + delta / num_tracking,
                        running_mean,
                    )
                    delta2 = log_score - running_mean
                    running_m2 = tl.where(
                        causal_mask,
                        running_m2 + delta * delta2,
                        running_m2,
                    )

            if start_n >= BSA_MASK_SINK_TOKEN_SIZE:
                for i_offset in tl.static_range(0, BLOCK_N, BSA_BLOCK_SIZE_K):
                    if i_offset == (0 * BSA_BLOCK_SIZE_K):
                        update_exp_sum = l_ij_0
                    elif i_offset == (1 * BSA_BLOCK_SIZE_K):
                        update_exp_sum = l_ij_1
                    elif i_offset == (2 * BSA_BLOCK_SIZE_K):
                        update_exp_sum = l_ij_2
                    elif i_offset == (3 * BSA_BLOCK_SIZE_K):
                        update_exp_sum = l_ij_3

                    if ONLINE_TOPK_METHOD == "tree":
                        root_v, root_lf, root_idx, block_update = winner_update_inline(
                            update_exp_sum,
                            start_n + i_offset,
                            BSA_BLOCK_SUMS,
                            stride_bsk,
                            stride_bsm,
                            BSA_HEAP_INDICES,
                            stride_hik,
                            stride_him,
                            BSA_INDICES,
                            stride_bik,
                            stride_bim,
                            EXACT_K,
                            EXACT_LOGK,
                            BLOCK_M,
                            start_m,
                            mask_m,
                            root_v,
                            root_lf,
                            root_idx,
                            MASKING,
                            mask,
                            REVERSE_ITER,
                        )
                    else:
                        # update if current max greater than stored max
                        # if they are equal, only take current one if the current index is smaller (older)
                        # and query overflow mask allows it (mask_m)
                        # and there is at least one unmasked element in this row (mask from causal and window)
                        min_mask = col_idx == block_sums_min_idx[:, None]
                        idx_at_min = tl.sum(block_idx * min_mask, 1)

                        block_update = update_exp_sum > block_sums_min
                        block_update |= (update_exp_sum == block_sums_min) & (
                            (start_n + i_offset) < idx_at_min
                        )
                        block_update &= mask_m & (update_exp_sum > float("-inf"))
                        if MASKING:
                            block_update &= tl.max(mask, 1) > 0

                        if tl.max(block_update) > 0:
                            block_sums_max = tl.maximum(
                                block_sums_min, update_exp_sum
                            )  # (M,)
                            # make a mask of the minimum indices
                            bsa_mask = block_update[:, None] & (
                                col_idx == block_sums_min_idx[:, None]
                            )  # (M, K)
                            block_sums = tl.where(
                                bsa_mask,
                                block_sums_max[:, None].to(block_sums.dtype),
                                block_sums,
                            )
                            block_idx = tl.where(
                                bsa_mask, start_n + i_offset, block_idx
                            )
                            # calculate the new block sums min for the next iteration
                            block_sums_min, block_sums_min_idx = tl.min(
                                block_sums, axis=-1, return_indices=True
                            )  # (M, K) -> (M,)

                    if EXACT_K < BSA_K:
                        new_block_idx = start_n + i_offset
                        causal_mask = (mask_idx - BSA_MASK_SW_SIZE) >= new_block_idx
                        log_score = update_exp_sum

                        est_std = tl.sqrt(running_m2 / (num_tracking - 1))

                        # recompute every THRESHOLD_REFRESH_INTERVAL steps because it's expensive
                        if counter == 0:
                            # probability of being below the top-k threshold
                            OVERESTIMATE_FACTOR = 1.5
                            remaining_slots = tl.maximum(
                                OVERESTIMATE_FACTOR * (EST_K - topk_idx),
                                0,
                            )
                            remaining_blocks = tl.maximum(
                                tl.cdiv(
                                    mask_idx - (start_n + i_offset), BSA_BLOCK_SIZE_K
                                ),
                                1,
                            )
                            pr_below_th = (
                                1 - remaining_slots / remaining_blocks
                            )  # [BLOCK_M,]
                            erf_pr = tl.where(
                                pr_below_th > 0,
                                libdevice.erfinv(2 * pr_below_th - 1) * SQRT2,
                                -3000.0,
                            )
                        counter = tl.where(
                            counter + 1 >= THRESHOLD_REFRESH_INTERVAL, 0, counter + 1
                        )

                        # estimated threshold for top-k
                        log_est_th = erf_pr * est_std + running_mean

                        # 1. always add if num_tracking < EST_K
                        if ONLINE_TOPK_METHOD == "tree":
                            upd_idx_0_bi = (
                                start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                            ) * stride_bim
                            upd_idx_0_bi += (EXACT_K * 2) * stride_bik
                            upd_idx_0_bs = (
                                start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                            ) * stride_bsm
                            upd_idx_0_bs += (EXACT_K * 2) * stride_bsk
                        else:
                            upd_idx_0_bi = (
                                start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                            ) * stride_bik
                            upd_idx_0_bi += EXACT_K * stride_bim
                            upd_idx_0_bs = (
                                start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                            ) * stride_bsk
                            upd_idx_0_bs += EXACT_K * stride_bsm

                        # 2. otherwise, update if the new value is larger than the estimated threshold
                        do_update = (
                            (~block_update)
                            & causal_mask  # skip masked values
                            & (log_score > log_est_th)
                            # & (topk_idx < EST_K)
                            & mask_m
                        )  # (M,)
                        if ONLINE_TOPK_METHOD == "tree":
                            upd_idx_bi = upd_idx_0_bi + topk_idx * stride_bik  # (M,)
                            upd_idx_bs = upd_idx_0_bs + topk_idx * stride_bsk
                        else:
                            upd_idx_bi = upd_idx_0_bi + topk_idx * stride_bim  # (M,)
                            upd_idx_bs = upd_idx_0_bs + topk_idx * stride_bsm

                        # only update if the new value is larger
                        prev_val = tl.load(
                            BSA_BLOCK_SUMS + upd_idx_bs,
                            mask=do_update & (topk_idx >= EST_K),
                            other=float("-inf"),
                        )
                        do_update &= update_exp_sum > prev_val

                        tl.store(
                            BSA_BLOCK_SUMS + upd_idx_bs,
                            value=update_exp_sum,
                            mask=do_update,
                        )
                        tl.store(
                            BSA_INDICES + upd_idx_bi,
                            value=new_block_idx,
                            mask=do_update,
                        )

                        topk_idx += do_update.to(
                            topk_idx.dtype
                        )  # increment end pointer
                        topk_idx = tl.where(topk_idx >= EST_K, 0, topk_idx)
        else:
            l_bsa = (l_bsa * alpha).to(l_bsa.dtype)

        # -- update output accumulator --
        acc = acc * alpha.to(acc.dtype)[:, None]
        # update acc
        if not USING_PAGED_CACHE:
            v = tl.load(V_ZHN, mask=mask_tsrc[:, None])
        else:
            v = tl.load(
                V_CACHE
                + idx_t[:, None] * stride_v_cache_t
                + 0 * stride_v_cache_page
                + tl.arange(0, HEAD_DIM)[None, :] * stride_v_cache_hid,
                mask=mask_tsrc[:, None],
                other=0.0,
            )

        if v_descale is not None:
            v *= v_descale

        # NOTE FIXME why this conversion needed?
        # if fp8_v:
        #     p = p.to(tl.float8e5)
        # else:
        #     p = p.to(v.dtype)

        acc = acc + tl.dot(
            p.to(q_dtype),
            v.to(q_dtype),
            out_dtype=tl.float32,
            allow_tf32=True,
        )
        # update m_i and l_i
        m_i = m_ij
        if not USING_PAGED_CACHE:
            K_ZHN += advance.to(tl.int64) * stride_kn
            V_ZHN += advance.to(tl.int64) * stride_vk

            if K_ZHN_NOPE is not None:
                K_ZHN_NOPE += advance.to(tl.int64) * stride_kn
        else:
            # idx_tsrc = idx_tsrc + BLOCK_N
            # mask_tsrc = idx_tsrc < hi
            pass

    if RETURN_BSA_MASK and ONLINE_TOPK_METHOD != "tree":
        tl.store(BSA_INDICES + bi_idx, value=block_idx, mask=mask_m[:, None])
        tl.store(BSA_BLOCK_SUMS + bs_idx, value=block_sums, mask=mask_m[:, None])

    return acc, l_i, l_bsa, m_i, running_mean, running_m2, num_tracking, topk_idx


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
if os.getenv("HIP_DISABLE_AUTOTUNE", "0") == "1":
    configs = [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
        for BM in [
            128,
        ]
        for BN in [
            64,
        ]
        for s in [
            3,
        ]
        for w in [
            4,
        ]
    ]
else:
    # NOTE: BSA GQ = BQ // DeltaW
    qsa_mask_group_size_q = int(os.getenv("BSA_GROUP_Q", "64"))
    qsa_mask_block_size_k = int(os.getenv("BSA_BLOCK_K", "64"))

    configs = [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
        for BM in filter(lambda x: x >= qsa_mask_group_size_q, [32, 64, 128, 256])
        for BN in [qsa_mask_block_size_k]
        for s in ([1] if is_hip() else [1, 3, 7])
        for w in [4, 8]
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


# @triton_jit(
#     configs=list(filter(keep, configs)),
#     key=[
#         "N_CTX",
#         "N_KV",
#         "HEAD_DIM",
#         "USING_PAGED_CACHE",
#     ],
#     do_autotune=True,
# )

# @triton.autotune(
#     configs=configs,
#     key=["N_CTX_AUTOTUNE", "N_KV_AUTOTUNE"],
#     restore_value=["BSA_INDICES", "BSA_BLOCK_SUMS", "BSA_HEAP_INDICES"],
# )


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    K_DESCALE,
    V_DESCALE,
    SOFTMAX_SINK,
    sm_scale,
    M,
    M_BSA,
    MX,
    NC,
    Out,
    MaskIdx,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_mz,
    stride_mm,
    USING_PAGED_CACHE: tl.constexpr,
    HEAD_REPEAT: tl.constexpr,
    K_CACHE,
    stride_k_cache_t,
    stride_k_cache_page,
    stride_k_cache_head_kv,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_t,
    stride_v_cache_page,
    stride_v_cache_head_kv,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_tsrc,
    RETURN_POOLED_SCORES: tl.constexpr,
    SCORE_POOLING_BQ: tl.constexpr,
    SCORE_POOLING_BK: tl.constexpr,
    SCORES,
    stride_scores_bsz,
    stride_scores_head,
    stride_scores_bdst,
    stride_scores_bsrc,
    ACC,
    stride_acc_bsz,
    stride_acc_head,
    stride_acc_split,
    stride_acc_tdst,
    stride_acc_hid,
    MI,
    stride_mi_bsz,
    stride_mi_head,
    stride_mi_split,
    strdie_mi_tdst,
    LI,
    stride_li_bsz,
    stride_li_head,
    stride_li_split,
    stride_li_tdst,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    RETURN_BSA_MASK: tl.constexpr,
    BSA_MASK_SINK_TOKEN_SIZE: tl.constexpr,
    BSA_MASK_SW_SIZE,
    BSA_K: tl.constexpr,
    BSA_BLOCK_SIZE_K: tl.constexpr,
    BSA_INDICES,
    stride_biz,
    stride_bih,
    stride_bik,
    stride_bim,
    BSA_BLOCK_SUMS,
    stride_bsz,
    stride_bsh,
    stride_bsk,
    stride_bsm,
    BSA_HEAP_INDICES,
    stride_hiz,
    stride_hih,
    stride_hik,
    stride_him,
    Z,
    H,
    N_CTX,
    N_KV,
    HEAD_DIM: tl.constexpr,
    HEAD_NOPE: tl.constexpr,
    HEAD_ROPE: tl.constexpr,
    N_SPLIT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    REVERSE_ITER: tl.constexpr,
    V_FP8: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    MODEL_CONTEXT_LENGTH=32768,
    SELF_EXTEND_SCALE=12,
    SELF_EXTEND_WINDOW=1024,
    ONLINE_TOPK_METHOD: tl.constexpr = "online",
    EXACT_K_: tl.constexpr = None,
    THRESHOLD_REFRESH_INTERVAL: tl.constexpr = None,
    N_KV_AUTOTUNE=0,
    N_CTX_AUTOTUNE=0,
):
    EXACT_K: tl.constexpr = BSA_K if EXACT_K_ is None else EXACT_K_
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    pid = tl.program_id(0)

    pid_bdst = pid % tl.cdiv(N_CTX, BLOCK_M)
    pid = pid // tl.cdiv(N_CTX, BLOCK_M)
    pid_n_split = pid % N_SPLIT
    pid_bsz_head = pid // N_SPLIT

    start_m = pid_bdst
    off_hz = pid_bsz_head.to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh

    idx_split = pid_n_split.to(tl.int64)

    Q_ZH = Q + q_offset
    if HEAD_DIM == HEAD_ROPE:
        Q_ZHT = (
            Q_ZH
            + (
                (start_m * BLOCK_M).to(tl.int64)
                + tl.arange(0, BLOCK_M)[:, None].to(tl.int64)
            )
            * stride_qm
            + tl.arange(0, HEAD_DIM)[None, :].to(tl.int64) * stride_qk
        )

        Q_ZHT_NOPE = None
    else:
        Q_ZHT = (
            Q_ZH
            + (
                (start_m * BLOCK_M).to(tl.int64)
                + tl.arange(0, BLOCK_M)[:, None].to(tl.int64)
            )
            * stride_qm
            + tl.arange(0, HEAD_ROPE)[None, :].to(tl.int64) * stride_qk
        )

        Q_ZHT_NOPE = (
            Q_ZH
            + (
                (start_m * BLOCK_M).to(tl.int64)
                + tl.arange(0, BLOCK_M)[:, None].to(tl.int64)
            )
            * stride_qm
            + tl.arange(0, HEAD_NOPE)[None, :].to(tl.int64) * stride_qk
        )

    if RETURN_BSA_MASK:
        BSA_INDICES += off_z.to(tl.int64) * stride_biz + off_h.to(tl.int64) * stride_bih
        BSA_BLOCK_SUMS += (
            off_z.to(tl.int64) * stride_bsz + off_h.to(tl.int64) * stride_bsh
        )
        if ONLINE_TOPK_METHOD == "tree":
            BSA_HEAP_INDICES += (
                off_z.to(tl.int64) * stride_hiz + off_h.to(tl.int64) * stride_hih
            )

    if not USING_PAGED_CACHE:
        # WARNING: If you are using float8e5, this might need to change.
        # v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
        V_ZH = V + kv_offset
        K_ZH = K + kv_offset
    else:
        K_CACHE = K_CACHE + (off_h.to(tl.int64) // HEAD_REPEAT) * stride_k_cache_head_kv
        V_CACHE = V_CACHE + (off_h.to(tl.int64) // HEAD_REPEAT) * stride_v_cache_head_kv
        BLOCK_TABLE = BLOCK_TABLE + off_z.to(tl.int64) * stride_block_table_bsz
        K_ZH = None
        V_ZH = None

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    mask_m = offs_m < N_CTX
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)

    mask_idx = tl.load(
        MaskIdx + off_z.to(tl.int64) * stride_mz + offs_m.to(tl.int64) * stride_mm,
        mask=mask_m,
        other=0,
    )
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], dtype=tl.float32, value=float("-inf"))
    l_i = tl.full([BLOCK_M], dtype=tl.float32, value=1.0)
    l_bsa = tl.full([BLOCK_M], dtype=tl.float32, value=1.0)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    if K_DESCALE is not None:
        k_descale = tl.load(K_DESCALE + off_z * H + off_h)
        v_descale = tl.load(V_DESCALE + off_z * H + off_h)
    else:
        k_descale = None
        v_descale = None

    # load q: it will stay in SRAM throughout
    q = tl.load(Q_ZHT, mask=mask_m[:, None])
    if Q_ZHT_NOPE is not None:
        q_nope = tl.load(Q_ZHT_NOPE, mask=mask_m[:, None])
    else:
        q_nope = None

    _K = K_CACHE if USING_PAGED_CACHE else K
    if (
        (_K.dtype.element_ty == tl.float8e5)
        | (_K.dtype.element_ty == tl.float8e4nv)
        | (_K.dtype.element_ty == tl.float8e4b8)
        | (_K.dtype.element_ty == tl.float8e4b15)
        | (_K.dtype.element_ty == tl.uint8)
        | (_K.dtype.element_ty == tl.int8)
    ):
        q = q.to(tl.bfloat16)
        if q_nope is not None:
            q_nope = q_nope.to(tl.bfloat16)

    if EXTEND_BACKEND == "none":
        q_rot = None
    elif EXTEND_BACKEND == "self_extend":
        idx_hid = tl.arange(0, HEAD_ROPE)
        idx_hid_rot = (idx_hid + HEAD_ROPE) % HEAD_ROPE
        idx_hid_cos_sin = idx_hid % (HEAD_ROPE // 2)
        rope_mult = tl.where((idx_hid + HEAD_ROPE // 2) < HEAD_ROPE, -1.0, 1.0)

        max_pos_tdst = tl.max(tl.where(mask_m, mask_idx, 0))
        idx_rope = mask_idx.to(tl.int64) - max_pos_tdst + MODEL_CONTEXT_LENGTH - 1 + 1
        # idx_rope = mask_idx.to(tl.int64)

        q_rot = tl.load(
            Q
            + off_z.to(tl.int64) * stride_qz
            + off_h.to(tl.int64) * stride_qh
            + offs_m[:, None].to(tl.int64) * stride_qm
            + (idx_hid_rot + HEAD_NOPE)[None, :].to(tl.int64) * stride_qk,
            mask=mask_m[:, None],
            other=0.0,
        )

        cos = tl.load(
            COS
            + idx_rope[:, None] * stride_cos_t
            + idx_hid_cos_sin[None, :] * stride_cos_hid,
            mask=mask_m[:, None],
            other=0.0,
        )
        sin = tl.load(
            SIN
            + idx_rope[:, None] * stride_sin_t
            + idx_hid_cos_sin[None, :] * stride_sin_hid,
            mask=mask_m[:, None],
            other=0.0,
        )

        q = (
            q.to(tl.float32) * cos.to(tl.float32)
            + q_rot.to(tl.float32)
            * rope_mult.to(tl.float32)[None, :]
            * sin.to(tl.float32)
        ).to(q.dtype)
    else:
        raise Exception()

    lo = 0
    mid = tl.maximum(
        0,
        tl.min(
            tl.where(
                mask_m,
                tl.maximum(0, mask_idx - (BSA_MASK_SW_SIZE if RETURN_BSA_MASK else 0)),
                987654321,
            )
        )
        // BLOCK_N
        * BLOCK_N,
    ).to(tl.int32)
    tl.multiple_of(mid, BLOCK_N)
    hi = (tl.max(tl.where(mask_m, mask_idx, 0)) + 1).to(tl.int32)

    running_mean = tl.zeros((BLOCK_M,), dtype=tl.float32)
    running_m2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
    num_tracking = tl.zeros((BLOCK_M,), dtype=tl.int32)
    topk_idx = tl.zeros((BLOCK_M,), dtype=tl.int16)

    if (N_SPLIT > 1) and False:
        k_chunk_size = tl.cdiv(hi, N_SPLIT)
        start_k = k_chunk_size * idx_split
        end_k = k_chunk_size * (idx_split + 1)

        # (start_k, end_k) (lo, mid)
        if tl.maximum(start_k, lo) < tl.minimum(end_k, mid):
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_ZH,
                stride_kn,
                stride_kk,
                V_ZH,
                stride_vn,
                stride_vk,
                mask_idx,
                start_m,
                qk_scale,
                k_descale,
                v_descale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                offs_m,
                offs_n,
                mask_m,
                N_CTX,
                N_KV,
                V_FP8,
                USING_PAGED_CACHE=USING_PAGED_CACHE,
                K_CACHE=K_CACHE,
                stride_k_cache_t=stride_k_cache_t,
                stride_k_cache_page=stride_k_cache_page,
                stride_k_cache_hid=stride_k_cache_hid,
                V_CACHE=V_CACHE,
                stride_v_cache_t=stride_v_cache_t,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_hid=stride_v_cache_hid,
                BLOCK_TABLE=BLOCK_TABLE,
                stride_block_table_tsrc=stride_block_table_tsrc,
                RETURN_BSA_MASK=RETURN_BSA_MASK,
                BSA_MASK_SINK_TOKEN_SIZE=BSA_MASK_SINK_TOKEN_SIZE,
                BSA_K=BSA_K,
                BSA_BLOCK_SIZE_K=BSA_BLOCK_SIZE_K,
                BSA_INDICES=BSA_INDICES,
                stride_bim=stride_bim,
                stride_bik=stride_bik,
                BSA_BLOCK_SUMS=BSA_BLOCK_SUMS,
                stride_bsk=stride_bsk,
                stride_bsm=stride_bsm,
                BSA_HEAP_INDICES=BSA_HEAP_INDICES,
                stride_hik=stride_hik,
                stride_him=stride_him,
                COS=COS,
                stride_cos_t=stride_cos_t,
                stride_cos_hid=stride_cos_hid,
                SIN=SIN,
                stride_sin_t=stride_sin_t,
                stride_sin_hid=stride_sin_hid,
                K_ROT=K + kv_offset if K is not None else None,
                stride_k_rot_tsrc=stride_kn,
                stride_k_rot_hid=stride_kk,
                lo=tl.maximum(start_k, lo).to(tl.int32),
                hi=tl.minimum(end_k, mid).to(tl.int32),
                REVERSE_ITER=REVERSE_ITER,
                MASKING=False,
                EXTEND_BACKEND=EXTEND_BACKEND,
                MODEL_CONTEXT_LENGTH=MODEL_CONTEXT_LENGTH,
                SELF_EXTEND_SCALE=SELF_EXTEND_SCALE,
                ONLINE_TOPK_METHOD=ONLINE_TOPK_METHOD,
                running_mean=running_mean,
                running_m2=running_m2,
                num_tracking=num_tracking,
                topk_idx=topk_idx,
                EXACT_K=EXACT_K,
                THRESHOLD_REFRESH_INTERVAL=THRESHOLD_REFRESH_INTERVAL,
            )
        # (start_k, end_k) (mid, hi)
        if tl.maximum(start_k, mid) < tl.minimum(end_k, hi):
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_ZH,
                stride_kn,
                stride_kk,
                V_ZH,
                stride_vn,
                stride_vk,
                mask_idx,
                start_m,
                qk_scale,
                k_descale,
                v_descale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                offs_m,
                offs_n,
                mask_m,
                N_CTX,
                N_KV,
                V_FP8,
                USING_PAGED_CACHE=USING_PAGED_CACHE,
                K_CACHE=K_CACHE,
                stride_k_cache_t=stride_k_cache_t,
                stride_k_cache_page=stride_k_cache_page,
                stride_k_cache_hid=stride_k_cache_hid,
                V_CACHE=V_CACHE,
                stride_v_cache_t=stride_v_cache_t,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_hid=stride_v_cache_hid,
                BLOCK_TABLE=BLOCK_TABLE,
                stride_block_table_tsrc=stride_block_table_tsrc,
                RETURN_BSA_MASK=RETURN_BSA_MASK,
                BSA_MASK_SINK_TOKEN_SIZE=BSA_MASK_SINK_TOKEN_SIZE,
                BSA_K=BSA_K,
                BSA_BLOCK_SIZE_K=BSA_BLOCK_SIZE_K,
                BSA_INDICES=BSA_INDICES,
                stride_bim=stride_bim,
                stride_bik=stride_bik,
                BSA_BLOCK_SUMS=BSA_BLOCK_SUMS,
                stride_bsk=stride_bsk,
                stride_bsm=stride_bsm,
                BSA_HEAP_INDICES=BSA_HEAP_INDICES,
                stride_hik=stride_hik,
                stride_him=stride_him,
                COS=COS,
                stride_cos_t=stride_cos_t,
                stride_cos_hid=stride_cos_hid,
                SIN=SIN,
                stride_sin_t=stride_sin_t,
                stride_sin_hid=stride_sin_hid,
                K_ROT=K + kv_offset if K is not None else None,
                stride_k_rot_tsrc=stride_kn,
                stride_k_rot_hid=stride_kk,
                lo=tl.maximum(start_k, mid).to(tl.int32),
                hi=tl.minimum(end_k, hi).to(tl.int32),
                REVERSE_ITER=REVERSE_ITER,
                MASKING=True,
                EXTEND_BACKEND=EXTEND_BACKEND,
                MODEL_CONTEXT_LENGTH=MODEL_CONTEXT_LENGTH,
                SELF_EXTEND_SCALE=SELF_EXTEND_SCALE,
                ONLINE_TOPK_METHOD=ONLINE_TOPK_METHOD,
                running_mean=running_mean,
                running_m2=running_m2,
                num_tracking=num_tracking,
                topk_idx=topk_idx,
                EXACT_K=EXACT_K,
                THRESHOLD_REFRESH_INTERVAL=THRESHOLD_REFRESH_INTERVAL,
            )
    else:
        acc, l_i, l_bsa, m_i, running_mean, running_m2, num_tracking, topk_idx = (
            _attn_fwd_inner(
                acc,
                l_i,
                l_bsa,
                m_i,
                q,
                q_nope,
                K_ZH,
                stride_kn,
                stride_kk,
                V_ZH,
                stride_vn,
                stride_vk,
                mask_idx,
                start_m,
                qk_scale,
                k_descale,
                v_descale,
                BLOCK_M,
                HEAD_DIM,
                HEAD_NOPE,
                HEAD_ROPE,
                BLOCK_N,
                offs_m,
                offs_n,
                mask_m,
                N_CTX,
                N_KV,
                V_FP8,
                USING_PAGED_CACHE=USING_PAGED_CACHE,
                K_CACHE=K_CACHE,
                stride_k_cache_t=stride_k_cache_t,
                stride_k_cache_page=stride_k_cache_page,
                stride_k_cache_hid=stride_k_cache_hid,
                V_CACHE=V_CACHE,
                stride_v_cache_t=stride_v_cache_t,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_hid=stride_v_cache_hid,
                BLOCK_TABLE=BLOCK_TABLE,
                stride_block_table_tsrc=stride_block_table_tsrc,
                RETURN_BSA_MASK=RETURN_BSA_MASK,
                BSA_MASK_SINK_TOKEN_SIZE=BSA_MASK_SINK_TOKEN_SIZE,
                BSA_MASK_SW_SIZE=BSA_MASK_SW_SIZE,
                BSA_K=BSA_K,
                BSA_BLOCK_SIZE_K=BSA_BLOCK_SIZE_K,
                BSA_INDICES=BSA_INDICES,
                stride_bim=stride_bim,
                stride_bik=stride_bik,
                BSA_BLOCK_SUMS=BSA_BLOCK_SUMS,
                stride_bsk=stride_bsk,
                stride_bsm=stride_bsm,
                BSA_HEAP_INDICES=BSA_HEAP_INDICES,
                stride_hik=stride_hik,
                stride_him=stride_him,
                COS=COS,
                stride_cos_t=stride_cos_t,
                stride_cos_hid=stride_cos_hid,
                SIN=SIN,
                stride_sin_t=stride_sin_t,
                stride_sin_hid=stride_sin_hid,
                K_ROT=K + kv_offset if K is not None else None,
                stride_k_rot_tsrc=stride_kn,
                stride_k_rot_hid=stride_kk,
                lo=lo,
                hi=mid,
                REVERSE_ITER=REVERSE_ITER,
                MASKING=False,
                EXTEND_BACKEND=EXTEND_BACKEND,
                MODEL_CONTEXT_LENGTH=MODEL_CONTEXT_LENGTH,
                SELF_EXTEND_SCALE=SELF_EXTEND_SCALE,
                SELF_EXTEND_WINDOW=SELF_EXTEND_WINDOW,
                ONLINE_TOPK_METHOD=ONLINE_TOPK_METHOD,
                running_mean=running_mean,
                running_m2=running_m2,
                num_tracking=num_tracking,
                topk_idx=topk_idx,
                EXACT_K=EXACT_K,
                THRESHOLD_REFRESH_INTERVAL=THRESHOLD_REFRESH_INTERVAL,
            )
        )

        acc, l_i, l_bsa, m_i, running_mean, running_m2, num_tracking, topk_idx = (
            _attn_fwd_inner(
                acc,
                l_i,
                l_bsa,
                m_i,
                q,
                q_nope,
                K_ZH,
                stride_kn,
                stride_kk,
                V_ZH,
                stride_vn,
                stride_vk,
                mask_idx,
                start_m,
                qk_scale,
                k_descale,
                v_descale,
                BLOCK_M,
                HEAD_DIM,
                HEAD_NOPE,
                HEAD_ROPE,
                BLOCK_N,
                offs_m,
                offs_n,
                mask_m,
                N_CTX,
                N_KV,
                V_FP8,
                USING_PAGED_CACHE=USING_PAGED_CACHE,
                K_CACHE=K_CACHE,
                stride_k_cache_t=stride_k_cache_t,
                stride_k_cache_page=stride_k_cache_page,
                stride_k_cache_hid=stride_k_cache_hid,
                V_CACHE=V_CACHE,
                stride_v_cache_t=stride_v_cache_t,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_hid=stride_v_cache_hid,
                BLOCK_TABLE=BLOCK_TABLE,
                stride_block_table_tsrc=stride_block_table_tsrc,
                RETURN_BSA_MASK=RETURN_BSA_MASK,
                BSA_MASK_SINK_TOKEN_SIZE=BSA_MASK_SINK_TOKEN_SIZE,
                BSA_MASK_SW_SIZE=BSA_MASK_SW_SIZE,
                BSA_K=BSA_K,
                BSA_BLOCK_SIZE_K=BSA_BLOCK_SIZE_K,
                BSA_INDICES=BSA_INDICES,
                stride_bim=stride_bim,
                stride_bik=stride_bik,
                BSA_BLOCK_SUMS=BSA_BLOCK_SUMS,
                stride_bsk=stride_bsk,
                stride_bsm=stride_bsm,
                BSA_HEAP_INDICES=BSA_HEAP_INDICES,
                stride_hik=stride_hik,
                stride_him=stride_him,
                COS=COS,
                stride_cos_t=stride_cos_t,
                stride_cos_hid=stride_cos_hid,
                SIN=SIN,
                stride_sin_t=stride_sin_t,
                stride_sin_hid=stride_sin_hid,
                K_ROT=K + kv_offset if K is not None else None,
                stride_k_rot_tsrc=stride_kn,
                stride_k_rot_hid=stride_kk,
                lo=mid,
                hi=hi,
                REVERSE_ITER=REVERSE_ITER,
                MASKING=True,
                EXTEND_BACKEND=EXTEND_BACKEND,
                MODEL_CONTEXT_LENGTH=MODEL_CONTEXT_LENGTH,
                SELF_EXTEND_SCALE=SELF_EXTEND_SCALE,
                SELF_EXTEND_WINDOW=SELF_EXTEND_WINDOW,
                ONLINE_TOPK_METHOD=ONLINE_TOPK_METHOD,
                running_mean=running_mean,
                running_m2=running_m2,
                num_tracking=num_tracking,
                topk_idx=topk_idx,
                EXACT_K=EXACT_K,
                THRESHOLD_REFRESH_INTERVAL=THRESHOLD_REFRESH_INTERVAL,
            )
        )

    # if ONLINE_TOPK_METHOD == "estimate":
    #     if start_m == tl.cdiv(N_CTX, BLOCK_M) - 1 and pid_bsz_head == 0 and idx_split == 0:
    #         tl.device_print("final running_mean, running_std",
    #                         (start_m * BLOCK_M).to(tl.int64) + tl.arange(0, BLOCK_M).to(tl.int64),
    #                         running_mean, tl.sqrt(running_m2 / (num_tracking - 1)))

    # epilogue
    if N_SPLIT > 1:
        # checkout acc, l_i, m_i
        tl.store(
            ACC
            + off_z.to(tl.int64) * stride_acc_bsz
            + off_h.to(tl.int64) * stride_acc_head
            + idx_split.to(tl.int64) * stride_acc_split
            + offs_m.to(tl.int64)[:, None] * stride_acc_tdst
            + tl.arange(0, HEAD_DIM).to(tl.int64)[None, :] * stride_acc_hid,
            mask=mask_m[:, None],
            value=acc,
        )
        tl.store(
            MI
            + off_z.to(tl.int64) * stride_mi_bsz
            + off_h.to(tl.int64) * stride_mi_head
            + idx_split.to(tl.int64) * stride_mi_split
            + offs_m.to(tl.int64) * strdie_mi_tdst,
            mask=mask_m,
            value=m_i,
        )
        tl.store(
            LI
            + off_z.to(tl.int64) * stride_li_bsz
            + off_h.to(tl.int64) * stride_li_head
            + idx_split.to(tl.int64) * stride_li_split
            + offs_m.to(tl.int64) * stride_li_tdst,
            mask=mask_m,
            value=l_i,
        )

    if N_SPLIT <= 1:
        if SOFTMAX_SINK is not None:
            curr_sink = tl.load(SOFTMAX_SINK + off_h)
            l_i += tl.exp(curr_sink - m_i)

        if MX is not None:
            m_ptrs = MX + off_hz * N_CTX + offs_m
            tl.store(m_ptrs, m_i, mask=mask_m)

        if NC is not None:
            l_ptrs = NC + off_hz * N_CTX + offs_m
            tl.store(l_ptrs, l_i, mask=mask_m)

        if M is not None:
            m_ptrs = M + off_hz * N_CTX + offs_m
            rowsum = tl.math.exp2(m_i + tl.math.log2(l_i))
            tl.store(m_ptrs, rowsum, mask=mask_m)

        if M_BSA is not None:
            m_ptrs = M_BSA + off_hz * N_CTX + offs_m
            rowsum_bsa = tl.math.exp2(m_i + tl.math.log2(l_bsa))
            tl.store(m_ptrs, rowsum_bsa, mask=mask_m)

        acc = acc / l_i[:, None]

        tl.store(
            Out
            + q_offset
            + (
                start_m.to(tl.int64) * BLOCK_M
                + tl.arange(0, BLOCK_M)[:, None].to(tl.int64)
            )
            * stride_om
            + tl.arange(0, HEAD_DIM)[None, :].to(tl.int64) * stride_on,
            acc.to(Out.type.element_ty),
            mask=mask_m[:, None],
        )
    else:
        tl.static_assert(M is None)
        tl.static_assert(MX is None)
        tl.static_assert(NC is None)


_ATTN_FWD_NO_RESTORE = triton.autotune(
    configs=configs,
    key=["N_CTX_AUTOTUNE", "N_KV_AUTOTUNE"],
)(_attn_fwd)

_ATTN_FWD_RESTORE_BSA_IND_SUM = triton.autotune(
    configs=configs,
    key=["N_CTX_AUTOTUNE", "N_KV_AUTOTUNE"],
    restore_value=[
        "BSA_INDICES",
        "BSA_BLOCK_SUMS",
    ],
)(_attn_fwd)

_ATTN_FWD_RESTORE_BSA_HEAP = triton.autotune(
    configs=configs,
    key=["N_CTX_AUTOTUNE", "N_KV_AUTOTUNE"],
    restore_value=["BSA_INDICES", "BSA_BLOCK_SUMS", "BSA_HEAP_INDICES"],
)(_attn_fwd)


@triton.jit
def _attn_merge(
    O,
    stride_o_bsz,
    stride_o_head,
    stride_o_tdst,
    stride_o_hid,
    ACC,
    stride_acc_bsz,
    stride_acc_head,
    stride_acc_split,
    stride_acc_tdst,
    stride_acc_hid,
    MI,
    stride_mi_bsz,
    stride_mi_head,
    stride_mi_split,
    stride_mi_tdst,
    LI,
    stride_li_bsz,
    stride_li_head,
    stride_li_split,
    stride_li_tdst,
    TDST,
    HEAD,
    HID: tl.constexpr,
    N_SPLIT,
    BLOCK_TDST: tl.constexpr,
):
    idx_tdst_start = tl.program_id(0).to(tl.int64) * BLOCK_TDST
    idx_tdst = tl.arange(0, BLOCK_TDST) + idx_tdst_start
    mask_tdst = idx_tdst < TDST
    idx_bsz_head = tl.program_id(1).to(tl.int64)
    idx_bsz = idx_bsz_head // HEAD
    idx_head = idx_bsz_head % HEAD
    idx_hid = tl.arange(0, HID)

    ACC = ACC + idx_bsz * stride_acc_bsz + idx_head * stride_acc_head
    MI = MI + idx_bsz * stride_mi_bsz + idx_head * stride_mi_head
    LI = LI + idx_bsz * stride_li_bsz + idx_head * stride_li_head

    m_i = tl.full([BLOCK_TDST], dtype=tl.float32, value=float("-inf"))
    l_i = tl.zeros([BLOCK_TDST], dtype=tl.float32)
    acc = tl.zeros([BLOCK_TDST, HID], dtype=tl.float32)

    for idx_split in range(N_SPLIT):
        m_split = tl.load(
            MI + idx_split * stride_mi_split + idx_tdst * stride_mi_tdst,
            mask=mask_tdst,
        )
        l_split = tl.load(
            LI + idx_split * stride_li_split + idx_tdst * stride_li_tdst,
            mask=mask_tdst,
        )
        acc_split = tl.load(
            ACC
            + idx_split * stride_acc_split
            + idx_tdst[:, None] * stride_acc_tdst
            + idx_hid[None, :] * stride_acc_hid,
            mask=mask_tdst[:, None],
        )

        tv = acc_split / l_split[:, None]
        tlogic = m_split + tl.math.log2(l_split)

        n_e_max = tl.maximum(tlogic, m_i)

        old_scale = tl.math.exp2(m_i - n_e_max)
        exp_logic = tl.math.exp2(tlogic - n_e_max)
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv

        l_i = l_i * old_scale + exp_logic
        m_i = n_e_max

    acc = acc / l_i[:, None]

    tl.store(
        O
        + idx_bsz * stride_o_bsz
        + idx_head * stride_o_head
        + idx_tdst[:, None] * stride_o_tdst
        + idx_hid[None, :] * stride_o_hid,
        value=acc.to(O.type.element_ty),
        mask=mask_tdst[:, None],
    )


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        softmax_sink: torch.Tensor,
        mask: torch.Tensor,
        sm_scale: float,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        return_running_statistics: bool,
        return_bsa_indices: bool,
        bsa_mask_sink_token_size: int,
        bsa_mask_sliding_window_size: int,
        return_pooled_scores: bool,
        score_pooling_block_size_q: int,
        score_pooling_block_size_k: int,
        score_pooling_max_seq_len: int,
        extend_backend: Literal["self_extend", "none"],
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
        model_context_length: int,
        self_extend_scale: int,
        bsa_top_block_k: int,
        bsa_block_size_k: int,
        online_topk_method: Literal["online", "tree"],
        reverse_iter: bool,
        exact_k: Optional[int],
        threshold_refresh_interval: int,
        return_row_sums: bool,
    ):
        exact_k = bsa_top_block_k if exact_k in [None, -1] else exact_k
        est_k = bsa_top_block_k - exact_k
        q = (q * sm_scale).to(q.dtype)

        USING_PAGED_CACHE = k_cache is not None
        if not USING_PAGED_CACHE:
            HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        else:
            HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k_cache.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        if not USING_PAGED_CACHE:
            HEAD_DIM_V = v.shape[-1]
        else:
            HEAD_DIM_V = v_cache.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 1
        extra_kern_args = {}
        # Tuning fo
        # r AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        N_CTX = q.shape[2]
        N_HEAD = q.shape[1]
        N_BATCH = q.shape[0]
        V_FP8 = (v if not USING_PAGED_CACHE else v_cache).dtype in (
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        )

        # NOTE: this is for backward
        NC = MX = None
        if return_running_statistics:
            # assert not return_bsa_indices
            MX = torch.empty(
                (q.shape[0], q.shape[1], q.shape[2]),
                device=q.device,
                dtype=torch.float32,
            )
            NC = torch.empty(
                (q.shape[0], q.shape[1], q.shape[2]),
                device=q.device,
                dtype=torch.float32,
            )

        M = M_BSA = None
        if return_row_sums:
            M = torch.empty(
                (q.shape[0], q.shape[1], q.shape[2]),
                device=q.device,
                dtype=torch.float32,
            )
            M_BSA = torch.empty(
                (q.shape[0], q.shape[1], q.shape[2]),
                device=q.device,
                dtype=torch.float32,
            )

        bsa_indices = bsa_block_sums = bsa_heap_indices = None
        if return_bsa_indices:
            # assert not return_running_statistics
            assert not return_pooled_scores
            BSZ, HEAD, TDST = q.shape[:3]

            k_factor = 1
            if online_topk_method == "tree":
                k_factor = 2

                # FIXME: this doesn't need to be 2K but I could not find the
                # energy to split the strides and add more arguments :(
                bsa_indices = torch.full(  # for real block indices
                    (BSZ, HEAD, exact_k * k_factor + est_k, TDST),
                    987654321,
                    device=q.device,
                    dtype=torch.int32,
                )
                bsa_block_sums = torch.full(  # for real block indices
                    (exact_k * k_factor + est_k,),
                    float("-inf"),
                    device=q.device,
                    dtype=torch.float32,
                )

                # need to initialize the heap in the proper order
                bsa_heap_indices = torch.zeros(
                    exact_k * 2, device=q.device, dtype=torch.int32
                )
                bsa_heap_indices[exact_k:] = torch.arange(
                    0, exact_k, device=q.device, dtype=torch.int32
                )

                for node in range(exact_k - 1, 0, -1):
                    l = node << 1
                    r = l + 1

                    lv = bsa_block_sums[l]
                    rv = bsa_block_sums[r]
                    lp = bsa_heap_indices[l]
                    rp = bsa_heap_indices[r]

                    take_left = lv < rv
                    bsa_heap_indices[node] = torch.where(take_left, lp, rp)
                    bsa_block_sums[node] = torch.where(take_left, lv, rv)

                bsa_heap_indices = bsa_heap_indices.view(1, 1, -1, 1).repeat(
                    BSZ, HEAD, 1, TDST
                )
                bsa_block_sums = bsa_block_sums.view(1, 1, -1, 1).repeat(
                    BSZ, HEAD, 1, TDST
                )
                bsa_heap_indices = bsa_heap_indices.contiguous()
                bsa_block_sums = bsa_block_sums.contiguous()

            else:
                # energy to split the strides and add more arguments :(
                bsa_heap_indices = torch.empty(
                    (BSZ, HEAD, TDST, bsa_top_block_k * k_factor),
                    device=q.device,
                    dtype=q.dtype,
                )
                bsa_indices = torch.full(  # for real block indices
                    (BSZ, HEAD, TDST, bsa_top_block_k * k_factor),
                    987654321,
                    device=q.device,
                    dtype=torch.int32,
                )
                bsa_block_sums = torch.full(  # for
                    (BSZ, HEAD, TDST, bsa_top_block_k * k_factor),
                    fill_value=float("-inf"),
                    device=q.device,
                    dtype=torch.float32,
                )

        if return_pooled_scores:
            assert not return_running_statistics
            assert not return_bsa_indices
            warnings.warn(
                "Pooled score should not be returned for efficient inference."
            )

            if k is not None:
                MAX_TSRC = k.shape[2]
            else:
                assert score_pooling_max_seq_len is not None
                MAX_TSRC = score_pooling_max_seq_len

            scores = torch.full(
                (
                    q.shape[0],
                    q.shape[1],
                    triton.cdiv(q.shape[2], score_pooling_block_size_q),
                    triton.cdiv(MAX_TSRC, score_pooling_block_size_k),
                ),
                fill_value=-3200.0,
                dtype=torch.float32,
                device=q.shape,
            )
        else:
            scores = None

        assert (
            q.shape[1] <= 128
        )  # N HEAD should be smaller than 128. this could be adjusted.
        assert len(mask.size()) == 2, "expecting mask to be 2D"

        if extend_backend != "none":
            assert isinstance(rope_sin, torch.Tensor)
            assert isinstance(rope_cos, torch.Tensor)
            assert rope_sin.ndim == 2
            assert rope_cos.ndim == 2
            assert extend_backend in ["self_extend", "nope"]

        if rope_sin is not None:
            HEAD_DIM_K_ROPE = rope_sin.shape[-1]
            HEAD_DIM_K_NOPE = HEAD_DIM_K - HEAD_DIM_K_ROPE
        else:
            HEAD_DIM_K_ROPE = HEAD_DIM_K
            HEAD_DIM_K_NOPE = 0

        N_CTX_BLOCK = 128
        N_PROGRAM = triton.cdiv(N_CTX, N_CTX_BLOCK) * N_HEAD * N_BATCH
        N_SM = 256  # TODO make a good solution to get this without init CUDA context on GPU 0
        N_SPLIT = triton.cdiv(N_SM, N_PROGRAM)
        ignore_n_split = os.getenv("HIP_DEBUG_RECOMPUTE_SPLIT", "0") == "0"
        if return_running_statistics or ignore_n_split:
            if N_SPLIT > 1:
                warnings.warn("N_SPLIT is ignored. this should be fixed")
            N_SPLIT = 1

        if return_bsa_indices and (N_SPLIT > 1):
            # BUG FIXME this warning should be activated later
            # warnings.warn("N_SPLIT is ignored when returning bsa indices. this should be fixed")
            N_SPLIT = 1
        else:
            # warnings.warn("N_SPLIT is ignored during researching. this should be fixed")
            N_SPLIT = 1

        assert safe_stride(k, 4)[:2] == safe_stride(v, 4)[:2]
        assert safe_stride(q, 4)[:2] == safe_stride(o, 4)[:2]
        if bsa_indices is not None:
            assert bsa_block_sums is not None
            assert bsa_indices.stride() == bsa_block_sums.stride()

        N_KV = (
            k.shape[2] if not USING_PAGED_CACHE else k_cache.shape[0] * k_cache.shape[1]
        )
        N_KV_AUTOTUNE = 1024 if N_KV > 1024 else 1
        N_CTX_AUTOTUNE = 128 if N_CTX > 128 else 1

        _attn_fwd = _ATTN_FWD_NO_RESTORE
        if bsa_indices is not None and bsa_heap_indices is not None:
            _attn_fwd = _ATTN_FWD_RESTORE_BSA_HEAP
        elif bsa_indices is not None:
            _attn_fwd = _ATTN_FWD_RESTORE_BSA_IND_SUM

        if (N_SPLIT > 1) and (not ignore_n_split):
            raise Exception("WIP: QSA-BSA masking, fill argument correctly after work.")
            # N_SPLIT = 1

            grid = lambda args: (
                triton.cdiv(N_CTX, args["BLOCK_M"]) * N_SPLIT * N_BATCH * N_HEAD,
            )

            acc = torch.zeros(
                (N_BATCH, N_HEAD, N_SPLIT, N_CTX, HEAD_DIM_V),
                dtype=torch.float32,
                device=q.device,
            )
            m_i = torch.zeros(
                (N_BATCH, N_HEAD, N_SPLIT, N_CTX), dtype=torch.float32, device=q.device
            )
            l_i = torch.zeros(
                (N_BATCH, N_HEAD, N_SPLIT, N_CTX), dtype=torch.float32, device=q.device
            )

            _attn_fwd[grid](
                q,
                k,
                v,
                k_descale,
                v_descale,
                sm_scale,
                M,
                MX,
                NC,
                bsa_indices,
                bsa_block_sums,
                o,
                mask,
                *safe_stride(q, 4),
                *safe_stride(k, 4),
                *safe_stride(v, 4),
                *safe_stride(bsa_indices, 4),
                *safe_stride(o, 4),
                *safe_stride(mask, 2),
                k_cache is not None,
                (
                    q.shape[1] // k_cache.shape[2]
                    if k_cache is not None
                    else q.shape[1] // k.shape[1]
                ),
                k_cache,
                *safe_stride(k_cache, 4),
                v_cache,
                *safe_stride(v_cache, 4),
                block_table,
                *safe_stride(block_table, 2),
                return_pooled_scores,
                score_pooling_block_size_q,
                score_pooling_block_size_k,
                scores,
                *safe_stride(scores, 4),
                acc,
                *safe_stride(acc, 5),
                m_i,
                *safe_stride(m_i, 4),
                l_i,
                *safe_stride(l_i, 4),
                rope_cos,
                *safe_stride(rope_cos, 2),
                rope_sin,
                *safe_stride(rope_sin, 2),
                q.shape[0],
                q.shape[1],
                N_CTX=N_CTX,
                N_KV=(
                    k.shape[2]
                    if not USING_PAGED_CACHE
                    else k_cache.shape[0] * k_cache.shape[1]
                ),
                HEAD_DIM=HEAD_DIM_K,
                N_SPLIT=N_SPLIT,
                REVERSE_ITER=reverse_iter,
                V_FP8=V_FP8,
                EXTEND_BACKEND=extend_backend,
                MODEL_CONTEXT_LENGTH=model_context_length,
                SELF_EXTEND_SCALE=self_extend_scale,
                BSA_K=bsa_top_block_k,
                ONLINE_TOPK_METHOD=online_topk_method,
                THRESHOLD_REFRESH_INTERVAL=threshold_refresh_interval,
                **extra_kern_args,
            )

            BLOCK_M = 128
            grid = (
                triton.cdiv(N_CTX, BLOCK_M),
                N_BATCH * N_HEAD,
                1,
            )

            _attn_merge[grid](
                o,
                *safe_stride(o, 4),
                acc,
                *safe_stride(acc, 5),
                m_i,
                *safe_stride(m_i, 4),
                l_i,
                *safe_stride(l_i, 4),
                TDST=N_CTX,
                HEAD=N_HEAD,
                HID=HEAD_DIM_V,
                N_SPLIT=N_SPLIT,
                BLOCK_TDST=BLOCK_M,
            )

            """
            # NOTE sanity check code for merge. do not delete for later debugging.
            def sanity_check(t: torch.Tensor):
                assert t.isnan().nonzero().shape[0] == 0
                assert t.isinf().nonzero().shape[0] == 0
                return t

            l_i = sanity_check(l_i)
            m_i = sanity_check(m_i)
            acc = sanity_check(acc)

            # l_i = torch.where(l_i <= (1.0 + 1e-4), l_i + 1e-4, l_i)

            logits = acc / l_i[:, :, :, :, None]
            logits = sanity_check(logits)
            stats = m_i + torch.log2(l_i)
            stats = sanity_check(stats)

            e_sum = torch.zeros_like(l_i[:, :, 0, :].contiguous())
            e_max = torch.full_like(m_i[:, :, 0, :].contiguous(), fill_value=float('-inf'))
            acc = torch.zeros_like(o, dtype=torch.float32)

            for i_split in range(N_SPLIT):
                tv = logits[:, :, i_split, :, :]
                tv = sanity_check(tv)
                tlogic = stats[:, :, i_split, :]
                tlogic = sanity_check(tlogic)
                n_e_max = torch.maximum(tlogic, e_max)
                n_e_max = sanity_check(n_e_max)

                old_scale = torch.exp2(e_max - n_e_max)
                old_scale = sanity_check(old_scale)
                exp_logic = torch.exp2(tlogic - n_e_max)
                exp_logic = sanity_check(exp_logic)
                acc = acc * old_scale[:, :, :, None] + exp_logic[:, :, :, None] * tv
                acc = sanity_check(acc)

                e_sum = e_sum * old_scale + exp_logic
                e_sum = sanity_check(e_sum)
                e_max = n_e_max
                e_max = sanity_check(e_max)

            acc = acc / e_sum[:, :, :, None]
            acc = sanity_check(acc)

            o = acc.to(o.dtype)
            """
        else:

            def grid(args):
                return (triton.cdiv(N_CTX, args["BLOCK_M"]) * 1 * N_BATCH * N_HEAD,)

            assert math.log2(bsa_top_block_k) == int(math.log2(bsa_top_block_k))

            N_CTX = N_CTX
            N_KV = (
                k.shape[2]
                if not USING_PAGED_CACHE
                else k_cache.shape[0] * k_cache.shape[1]
            )
            N_CTX_AUTOTUNE = 128 if N_CTX > 128 else 1
            N_KV_AUTOTUNE = 1024 if N_KV > 1024 else 1

            _attn_fwd[grid](
                q,
                k,
                v,
                k_descale,
                v_descale,
                softmax_sink.contiguous() if softmax_sink is not None else None,
                sm_scale,
                M,
                M_BSA,
                MX,
                NC,
                o,
                mask,
                *safe_stride(q, 4),
                *safe_stride(k, 4),
                *safe_stride(v, 4),
                *safe_stride(o, 4),
                *safe_stride(mask, 2),
                k_cache is not None,
                (
                    q.shape[1] // k_cache.shape[2]
                    if k_cache is not None
                    else q.shape[1] // k.shape[1]
                ),
                k_cache,
                *safe_stride(k_cache, 4),
                v_cache,
                *safe_stride(v_cache, 4),
                block_table,
                *safe_stride(block_table, 2),
                return_pooled_scores,
                score_pooling_block_size_q,
                score_pooling_block_size_k,
                scores,
                *safe_stride(scores, 4),
                # acc, m_i, l_i
                None,
                *safe_stride(None, 5),
                None,
                *safe_stride(None, 4),
                None,
                *safe_stride(None, 4),
                rope_cos,
                *safe_stride(rope_cos, 2),
                rope_sin,
                *safe_stride(rope_sin, 2),
                (bsa_indices is not None) and (bsa_block_sums is not None),
                bsa_mask_sink_token_size,
                bsa_mask_sliding_window_size,
                bsa_top_block_k,
                bsa_block_size_k,
                bsa_indices,
                *safe_stride(bsa_indices, 4),
                bsa_block_sums,
                *safe_stride(bsa_block_sums, 4),
                bsa_heap_indices,
                *safe_stride(bsa_heap_indices, 4),
                q.shape[0],
                q.shape[1],
                N_CTX=N_CTX,
                N_KV=(
                    k.shape[2]
                    if not USING_PAGED_CACHE
                    else k_cache.shape[0] * k_cache.shape[1]
                ),
                HEAD_DIM=HEAD_DIM_K,
                HEAD_NOPE=HEAD_DIM_K_NOPE,
                HEAD_ROPE=HEAD_DIM_K_ROPE,
                N_SPLIT=1,
                REVERSE_ITER=reverse_iter,
                V_FP8=V_FP8,
                # BLOCK_M=64,
                # BLOCK_N=32,
                EXTEND_BACKEND=("none" if extend_backend == "nope" else extend_backend),
                # EXTEND_BACKEND=extend_backend,
                MODEL_CONTEXT_LENGTH=model_context_length,
                SELF_EXTEND_SCALE=self_extend_scale,
                ONLINE_TOPK_METHOD=online_topk_method,
                N_CTX_AUTOTUNE=N_CTX_AUTOTUNE,
                N_KV_AUTOTUNE=N_KV_AUTOTUNE,
                EXACT_K_=exact_k,
                THRESHOLD_REFRESH_INTERVAL=threshold_refresh_interval,
                **extra_kern_args,
            )

        outputs = (o,)
        if return_running_statistics:
            outputs = outputs + ((MX, NC),)
        if return_row_sums:
            outputs = outputs + ((M, M_BSA),)
        if return_bsa_indices:
            if online_topk_method != "tree":
                outputs = outputs + ((bsa_indices, bsa_block_sums),)
            else:
                outputs = outputs + (
                    (
                        bsa_indices[:, :, exact_k:, :].transpose(-1, -2),
                        bsa_block_sums[:, :, exact_k:, :].transpose(-1, -2),
                    ),
                )
        return o if len(outputs) == 1 else outputs

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("bwd not implemented for query sparse kernel")


# for typing wrapper and provide kwargs
@capture
def query_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    sm_scale: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    return_running_statistics: bool = False,
    return_pooled_scores: bool = False,
    score_pooling_block_size_q: int = 64,
    score_pooling_block_size_k: int = 64,
    score_pooling_max_seq_len: int = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_sink: Optional[torch.Tensor] = None,
    extend_backend: Literal["self_extend", "none"] = "none",
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    model_context_length: int = 131072,
    self_extend_scale: int = 12,
    return_bsa_indices: bool = False,
    bsa_mask_sink_token_size: int = 64,
    bsa_mask_sliding_window_size: int = 0,
    bsa_top_block_k: int = 128,
    bsa_block_size_k: int = 32,
    online_topk_method: Literal["online", "tree"] = "online",
    reverse_iter: bool = True,
    exact_k: int = None,
    threshold_refresh_interval: int = 4,
    return_row_sums: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    assert online_topk_method in {"online", "tree"}
    return _attention.apply(
        q,
        k,
        v,
        k_descale,
        v_descale,
        softmax_sink,
        mask,
        sm_scale,
        k_cache,
        v_cache,
        block_table,
        return_running_statistics,
        return_bsa_indices,
        bsa_mask_sink_token_size,
        bsa_mask_sliding_window_size,
        return_pooled_scores,
        score_pooling_block_size_q,
        score_pooling_block_size_k,
        score_pooling_max_seq_len,
        extend_backend,
        rope_cos,
        rope_sin,
        model_context_length,
        self_extend_scale,
        bsa_top_block_k,
        bsa_block_size_k,
        online_topk_method,
        reverse_iter,
        exact_k,
        threshold_refresh_interval,
        return_row_sums,
    )
