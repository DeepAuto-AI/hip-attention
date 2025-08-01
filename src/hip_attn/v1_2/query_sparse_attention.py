"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import os
import warnings
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np

# import pytest
import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor

from hip_attn.v1_2.attention_metadata import safe_stride
from hip_attn.v1_2.utils import capture

# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = "cuda:0"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    mask_idx,
    start_m,
    qk_scale,
    k_descale,
    v_descale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
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
    MASKING: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    MODEL_CONTEXT_LENGTH,
    SELF_EXTEND_SCALE,
):
    # range of values handled by this stage
    # lo, hi = 0, N_KV
    # lo, hi = 0, tl.max(mask_idx) + 1

    if not USING_PAGED_CACHE:
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    else:
        idx_hid = tl.arange(0, HEAD_DIM)
        # idx_tsrc = tl.arange(0, BLOCK_N) + lo
        # mask_tsrc = idx_tsrc < hi

    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, num_stages=1):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        idx_tsrc = offs_n + start_n
        mask_tsrc = idx_tsrc < hi

        if not USING_PAGED_CACHE:
            tl.static_assert(EXTEND_BACKEND == "none")
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        else:
            idx_t = tl.load(
                BLOCK_TABLE + idx_tsrc.to(tl.int64) * stride_block_table_tsrc,
                mask=mask_tsrc,
            ).to(tl.int64)
            k = tl.load(
                K_CACHE
                + idx_t[None, :] * stride_k_cache_t
                + 0 * stride_k_cache_page
                + idx_hid[:, None] * stride_k_cache_hid,
                mask=mask_tsrc[None, :],
                other=0.0,
            )
            if k.dtype == tl.uint8:
                k = k.to(tl.float8e5, bitcast=True)
            if (
                (k.dtype == tl.float8e5)
                | (k.dtype == tl.float8e4b8)
                | (k.dtype == tl.float8e4b15)
                | (k.dtype == tl.float8e4nv)
            ):
                k = k.to(tl.bfloat16)

        if EXTEND_BACKEND == "none":
            pass
        elif EXTEND_BACKEND == "self_extend":
            idx_hid = tl.arange(0, HEAD_DIM)
            idx_hid_rot = (idx_hid + HEAD_DIM // 2) % HEAD_DIM
            idx_hid_cos_sin = idx_hid % (HEAD_DIM // 2)
            rope_mult = tl.where((idx_hid + HEAD_DIM // 2) < HEAD_DIM, -1.0, 1.0)

            SELF_EXTEND_WINDOW = 4096

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

            offset = idx_tsrc.to(tl.int64) - max_pos_tsrc
            idx_rope = tl.where(
                offset > (-SELF_EXTEND_WINDOW),
                offset + MODEL_CONTEXT_LENGTH - 1,
                (offset + SELF_EXTEND_WINDOW) // SELF_EXTEND_SCALE
                + MODEL_CONTEXT_LENGTH
                - 1
                - SELF_EXTEND_WINDOW,
            )

            if not USING_PAGED_CACHE:
                k_rot = tl.load(
                    K_ROT
                    + idx_tsrc[None, :] * stride_k_rot_tsrc
                    + idx_hid_rot[:, None] * stride_k_rot_hid,
                    mask=mask_tsrc,
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
                    + idx_hid_rot[:, None] * stride_k_cache_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                )

            if k_rot.dtype == tl.uint8:
                k_rot = k_rot.to(tl.float8e5, bitcast=True)
            if (
                (k_rot.dtype == tl.float8e5)
                | (k_rot.dtype == tl.float8e4nv)
                | (k_rot.dtype == tl.float8e4b8)
                | (k_rot.dtype == tl.float8e5b16)
                | (k_rot.dtype == tl.float8e4b15)
            ):
                k_rot = k_rot.to(tl.bfloat16)

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

        # qk = tl.dot(q, k)

        q_dtype = q.dtype

        cq = tl.sqrt(HEAD_DIM * 1.0) / tl.sqrt(tl.sqrt(HEAD_DIM * 1.0))
        ck = 1 / tl.sqrt(tl.sqrt(HEAD_DIM * 1.0))

        qk = tl.dot((q * cq).to(q_dtype), (k.to(q_dtype) * ck).to(q_dtype)).to(
            tl.float32
        )

        qk = qk * 1.44269504

        if MASKING:
            mask = (mask_idx[:, None]) >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = (l_i * alpha + l_ij).to(l_i.dtype)
        # -- update output accumulator --
        acc = acc * alpha.to(acc.dtype)[:, None]
        # update acc
        if not USING_PAGED_CACHE:
            v = tl.load(
                V_block_ptr,
                boundary_check=(0,),
                padding_option="zero",
            )
        else:
            v = tl.load(
                V_CACHE
                + idx_t[:, None] * stride_v_cache_t
                + 0 * stride_v_cache_page
                + idx_hid[None, :] * stride_v_cache_hid,
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
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        else:
            # idx_tsrc = idx_tsrc + BLOCK_N
            # mask_tsrc = idx_tsrc < hi
            pass
    return acc, l_i, m_i


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
    configs = [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [32, 64]
        for s in ([1] if is_hip() else [3, 4, 7])
        for w in [4, 8]
        # for BM in [128,]
        # for BN in [64,]
        # for s in [3, ]
        # for w in [4, ]
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(
    list(filter(keep, configs)),
    key=[
        # "N_CTX",
        # "N_KV",
        "HEAD_DIM",
        "USING_PAGED_CACHE",
    ],
)
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    K_DESCALE,
    V_DESCALE,
    sm_scale,
    M,
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
    Z,
    H,
    N_CTX,
    N_KV,
    HEAD_DIM: tl.constexpr,
    N_SPLIT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    V_FP8: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    MODEL_CONTEXT_LENGTH=32768,
    SELF_EXTEND_SCALE=12,
):
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

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    if not USING_PAGED_CACHE:
        v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_offset,
            shape=(N_KV, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_offset,
            shape=(HEAD_DIM, N_KV),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
    else:
        K_CACHE = K_CACHE + (off_h.to(tl.int64) // HEAD_REPEAT) * stride_k_cache_head_kv
        V_CACHE = V_CACHE + (off_h.to(tl.int64) // HEAD_REPEAT) * stride_v_cache_head_kv
        BLOCK_TABLE = BLOCK_TABLE + off_z.to(tl.int64) * stride_block_table_bsz
        K_block_ptr = None
        V_block_ptr = None
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_CTX
    offs_n = tl.arange(0, BLOCK_N)

    mask_idx = tl.load(
        MaskIdx + off_z.to(tl.int64) * stride_mz + offs_m.to(tl.int64) * stride_mm,
        mask=mask_m,
        other=0,
    )
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], dtype=tl.float32, value=float("-inf"))
    l_i = tl.full([BLOCK_M], dtype=tl.float32, value=1.0)
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
    q = tl.load(
        Q_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
    )

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

    if EXTEND_BACKEND == "none":
        q_rot = None
    elif EXTEND_BACKEND == "self_extend":
        idx_hid = tl.arange(0, HEAD_DIM)
        idx_hid_rot = (idx_hid + HEAD_DIM // 2) % HEAD_DIM
        idx_hid_cos_sin = idx_hid % (HEAD_DIM // 2)
        rope_mult = tl.where((idx_hid + HEAD_DIM // 2) < HEAD_DIM, -1.0, 1.0)

        max_pos_tdst = tl.max(tl.where(mask_m, mask_idx + 1, 0) - 1)
        idx_rope = mask_idx.to(tl.int64) - max_pos_tdst + MODEL_CONTEXT_LENGTH - 1

        q_rot = tl.load(
            Q
            + off_z.to(tl.int64) * stride_qz
            + off_h.to(tl.int64) * stride_qh
            + offs_m[:, None].to(tl.int64) * stride_qm
            + idx_hid_rot[None, :].to(tl.int64) * stride_qk,
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

    if (not USING_PAGED_CACHE) and False:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
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
            N_CTX,
            N_KV,
            V_FP8,
        )
    else:
        lo = 0
        mid = (tl.min(tl.where(mask_m, mask_idx, 987654321)) // BLOCK_N * BLOCK_N).to(
            tl.int32
        )
        tl.multiple_of(mid, BLOCK_N)
        hi = (tl.max(tl.where(mask_m, mask_idx, 0)) + 1).to(tl.int32)

        if N_SPLIT > 1:
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
                    K_block_ptr,
                    V_block_ptr,
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
                    MASKING=False,
                    EXTEND_BACKEND=EXTEND_BACKEND,
                    MODEL_CONTEXT_LENGTH=MODEL_CONTEXT_LENGTH,
                    SELF_EXTEND_SCALE=SELF_EXTEND_SCALE,
                )
            # (start_k, end_k) (mid, hi)
            if tl.maximum(start_k, mid) < tl.minimum(end_k, hi):
                acc, l_i, m_i = _attn_fwd_inner(
                    acc,
                    l_i,
                    m_i,
                    q,
                    K_block_ptr,
                    V_block_ptr,
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
                    MASKING=True,
                    EXTEND_BACKEND=EXTEND_BACKEND,
                    MODEL_CONTEXT_LENGTH=MODEL_CONTEXT_LENGTH,
                    SELF_EXTEND_SCALE=SELF_EXTEND_SCALE,
                )
        else:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
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
                MASKING=False,
                EXTEND_BACKEND=EXTEND_BACKEND,
                MODEL_CONTEXT_LENGTH=MODEL_CONTEXT_LENGTH,
                SELF_EXTEND_SCALE=SELF_EXTEND_SCALE,
            )

            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                K_block_ptr,
                V_block_ptr,
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
                MASKING=True,
                EXTEND_BACKEND=EXTEND_BACKEND,
                MODEL_CONTEXT_LENGTH=MODEL_CONTEXT_LENGTH,
                SELF_EXTEND_SCALE=SELF_EXTEND_SCALE,
            )

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
        if MX is not None:
            m_ptrs = MX + off_hz * N_CTX + offs_m
            tl.store(m_ptrs, m_i, mask=mask_m)

        if NC is not None:
            l_ptrs = NC + off_hz * N_CTX + offs_m
            tl.store(l_ptrs, l_i, mask=mask_m)

        if M is not None:
            m_i += tl.math.log2(l_i)
            m_ptrs = M + off_hz * N_CTX + offs_m
            tl.store(m_ptrs, m_i, mask=mask_m)

        acc = acc / l_i[:, None]
        tl.store(
            O_block_ptr,
            acc.to(Out.type.element_ty),
            boundary_check=(0,),
        )
    else:
        tl.static_assert(M is None)
        tl.static_assert(MX is None)
        tl.static_assert(NC is None)


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

    @capture
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        mask: torch.Tensor,
        sm_scale: float,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        return_running_statistics: bool,
        return_pooled_scores: bool,
        score_pooling_block_size_q: int,
        score_pooling_block_size_k: int,
        score_pooling_max_seq_len: int,
        extend_backend: Literal["self_extend", "none"],
        rope_cos: Optional[torch.Tensor],
        rope_sin: Optional[torch.Tensor],
        model_context_length: int,
        self_extend_scale: int,
    ):
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
        # M = torch.empty(
        #     (q.shape[0], q.shape[1], q.shape[2]),
        #     device=q.device,
        #     dtype=torch.float32,
        # )
        NC = MX = M = None
        if return_running_statistics:
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

        if return_pooled_scores:
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
                fill_value=float("-inf"),
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
            assert extend_backend in ["self_extend"]

        N_CTX_BLOCK = 128
        N_PROGRAM = triton.cdiv(N_CTX, N_CTX_BLOCK) * N_HEAD * N_BATCH
        N_SM = 256  # TODO make a good solution to get this without init CUDA context on GPU 0
        N_SPLIT = triton.cdiv(N_SM, N_PROGRAM)
        ignore_n_split = os.getenv("HIP_DEBUG_RECOMPUTE_SPLIT", "0") == "0"
        if return_running_statistics or ignore_n_split:
            if N_SPLIT > 1:
                warnings.warn("N_SPLIT is ignored. this should be fixed")
            N_SPLIT = 1

        if (N_SPLIT > 1) and (not ignore_n_split):
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
                V_FP8=V_FP8,
                EXTEND_BACKEND=extend_backend,
                MODEL_CONTEXT_LENGTH=model_context_length,
                SELF_EXTEND_SCALE=self_extend_scale,
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

            # def sanity_check(t: torch.Tensor):
            #     assert t.isnan().nonzero().shape[0] == 0
            #     assert t.isinf().nonzero().shape[0] == 0
            #     return t

            # l_i = sanity_check(l_i)
            # m_i = sanity_check(m_i)
            # acc = sanity_check(acc)

            # # l_i = torch.where(l_i <= (1.0 + 1e-4), l_i + 1e-4, l_i)

            # logits = acc / l_i[:, :, :, :, None]
            # logits = sanity_check(logits)
            # stats = m_i + torch.log2(l_i)
            # stats = sanity_check(stats)

            # e_sum = torch.zeros_like(l_i[:, :, 0, :].contiguous())
            # e_max = torch.full_like(m_i[:, :, 0, :].contiguous(), fill_value=float('-inf'))
            # acc = torch.zeros_like(o, dtype=torch.float32)

            # for i_split in range(N_SPLIT):
            #     tv = logits[:, :, i_split, :, :]
            #     tv = sanity_check(tv)
            #     tlogic = stats[:, :, i_split, :]
            #     tlogic = sanity_check(tlogic)
            #     n_e_max = torch.maximum(tlogic, e_max)
            #     n_e_max = sanity_check(n_e_max)

            #     old_scale = torch.exp2(e_max - n_e_max)
            #     old_scale = sanity_check(old_scale)
            #     exp_logic = torch.exp2(tlogic - n_e_max)
            #     exp_logic = sanity_check(exp_logic)
            #     acc = acc * old_scale[:, :, :, None] + exp_logic[:, :, :, None] * tv
            #     acc = sanity_check(acc)

            #     e_sum = e_sum * old_scale + exp_logic
            #     e_sum = sanity_check(e_sum)
            #     e_max = n_e_max
            #     e_max = sanity_check(e_max)

            # acc = acc / e_sum[:, :, :, None]
            # acc = sanity_check(acc)

            # o = acc.to(o.dtype)
        else:
            grid = lambda args: (
                triton.cdiv(N_CTX, args["BLOCK_M"]) * 1 * N_BATCH * N_HEAD,
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
                q.shape[0],
                q.shape[1],
                N_CTX=N_CTX,
                N_KV=(
                    k.shape[2]
                    if not USING_PAGED_CACHE
                    else k_cache.shape[0] * k_cache.shape[1]
                ),
                HEAD_DIM=HEAD_DIM_K,
                N_SPLIT=1,
                V_FP8=V_FP8,
                EXTEND_BACKEND=extend_backend,
                MODEL_CONTEXT_LENGTH=model_context_length,
                SELF_EXTEND_SCALE=self_extend_scale,
                **extra_kern_args,
            )

        if return_running_statistics:
            return o, (MX, NC)
        else:
            return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("bwd not implemented for recompute kernel")


# for typing wrapper and provide kwargs
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
    extend_backend: Literal["self_extend", "none"] = "none",
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    model_context_length: int = 131072,
    self_extend_scale: int = 12,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return _attention.apply(
        q,
        k,
        v,
        k_descale,
        v_descale,
        mask,
        sm_scale,
        k_cache,
        v_cache,
        block_table,
        return_running_statistics,
        return_pooled_scores,
        score_pooling_block_size_q,
        score_pooling_block_size_k,
        score_pooling_max_seq_len,
        extend_backend,
        rope_cos,
        rope_sin,
        model_context_length,
        self_extend_scale,
    )
