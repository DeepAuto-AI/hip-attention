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
from typing import Callable, Tuple, Union

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
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    mask_idx,
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    N_KV: tl.constexpr,
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
    lo,
    hi,
    MASKING: tl.constexpr,
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
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if not USING_PAGED_CACHE:
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        else:
            idx_tsrc = tl.arange(0, BLOCK_N) + start_n
            mask_tsrc = idx_tsrc < hi

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
                other=0,
            )

        # qk = tl.dot(q, k)

        q_dtype = q.dtype

        cq = tl.sqrt(HEAD_DIM * 1.0) / tl.sqrt(tl.sqrt(HEAD_DIM * 1.0))
        ck = 1 / tl.sqrt(tl.sqrt(HEAD_DIM * 1.0))

        qk = tl.dot(
            (q * cq).to(q_dtype),
            (k.to(q_dtype) * ck).to(q_dtype),
            out_dtype=tl.float32,
            allow_tf32=True,
        ).to(tl.float32)

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
                other=0,
            )
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(v.dtype)

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


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "N_KV", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
    MX,
    NC,
    Out,  #
    MaskIdx,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
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
    Z,
    H,
    N_CTX,  #
    N_KV,
    HEAD_DIM: tl.constexpr,  #
    N_SPLIT,
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    V_FP8: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh

    idx_split = tl.program_id(2).to(tl.int64)

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
    # load q: it will stay in SRAM throughout
    q = tl.load(
        Q_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
    )

    if not USING_PAGED_CACHE:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,  #
            mask_idx,
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            offs_m,
            offs_n,
            N_CTX,
            N_KV,
            V_FP8,  #
        )
    else:
        lo = 0
        mid = tl.min(tl.where(mask_m, mask_idx, 987654321)) // BLOCK_N * BLOCK_N
        hi = tl.max(mask_idx) + 1

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
                    None,
                    None,
                    mask_idx,
                    start_m,
                    qk_scale,
                    BLOCK_M,
                    HEAD_DIM,
                    BLOCK_N,
                    offs_m,
                    offs_n,
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
                    lo=tl.maximum(start_k, lo),
                    hi=tl.minimum(end_k, mid),
                    MASKING=False,
                )
            # (start_k, end_k) (mid, hi)
            if tl.maximum(start_k, mid) < tl.minimum(end_k, hi):
                acc, l_i, m_i = _attn_fwd_inner(
                    acc,
                    l_i,
                    m_i,
                    q,
                    None,
                    None,
                    mask_idx,
                    start_m,
                    qk_scale,
                    BLOCK_M,
                    HEAD_DIM,
                    BLOCK_N,
                    offs_m,
                    offs_n,
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
                    lo=tl.maximum(start_k, mid),
                    hi=tl.minimum(end_k, hi),
                    MASKING=True,
                )
        else:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                None,
                None,
                mask_idx,
                start_m,
                qk_scale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                offs_m,
                offs_n,
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
                lo=lo,
                hi=mid,
                MASKING=False,
            )

            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                None,
                None,
                mask_idx,
                start_m,
                qk_scale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                offs_m,
                offs_n,
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
                lo=mid,
                hi=hi,
                MASKING=True,
            )

    # epilogue
    if N_SPLIT > 1:
        # checkout acc, l_i, m_i
        tl.store(
            ACC
            + off_z * stride_acc_bsz
            + off_h * stride_acc_head
            + idx_split * stride_acc_split
            + offs_m[:, None] * stride_acc_tdst
            + tl.arange(0, HEAD_DIM)[None, :] * stride_acc_hid,
            mask=mask_m[:, None],
            value=acc,
        )
        tl.store(
            MI
            + off_z * stride_mi_bsz
            + off_h * stride_mi_head
            + idx_split * stride_mi_split
            + offs_m * strdie_mi_tdst,
            mask=mask_m,
            value=m_i,
        )
        tl.store(
            LI
            + off_z * stride_li_bsz
            + off_h * stride_li_head
            + idx_split * stride_li_split
            + offs_m * stride_li_tdst,
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
        V_FP8 = (
            v.dtype == torch.float8_e5m2
            if not USING_PAGED_CACHE
            else v_cache.dtype == torch.float8_e5m2
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

        N_CTX_BLOCK = 128
        N_PROGRAM = triton.cdiv(N_CTX, N_CTX_BLOCK) * N_HEAD * N_BATCH
        N_SM = 256  # TODO make a good solution to get this without init CUDA context on GPU 0
        N_SPLIT = triton.cdiv(N_SM, N_PROGRAM)
        if return_running_statistics:
            if N_SPLIT > 1:
                warnings.warn("N_SPLIT is ignored. this should be fixed")
            N_SPLIT = 1

        if (N_SPLIT > 1) and (os.getenv("HIP_DEBUG_RECOMPUTE_SPLIT", "1") == "1"):
            # N_SPLIT = 1

            grid = lambda args: (
                triton.cdiv(N_CTX, args["BLOCK_M"]),
                N_BATCH * N_HEAD,
                N_SPLIT,
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
                sm_scale,
                M,
                MX,
                NC,
                o,  #
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
                acc,
                *safe_stride(acc, 5),
                m_i,
                *safe_stride(m_i, 4),
                l_i,
                *safe_stride(l_i, 4),
                return_pooled_scores,
                score_pooling_block_size_q,
                score_pooling_block_size_k,
                scores,
                *safe_stride(scores, 4),
                q.shape[0],
                q.shape[1],  #
                N_CTX=N_CTX,  #
                N_KV=(
                    k.shape[2]
                    if not USING_PAGED_CACHE
                    else k_cache.shape[0] * k_cache.shape[1]
                ),
                HEAD_DIM=HEAD_DIM_K,  #
                N_SPLIT=N_SPLIT,
                V_FP8=V_FP8,
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
                triton.cdiv(N_CTX, args["BLOCK_M"]),
                N_BATCH * N_HEAD,
                1,
            )

            _attn_fwd[grid](
                q,
                k,
                v,
                sm_scale,
                M,
                MX,
                NC,
                o,  #
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
                # acc, m_i, l_i
                None,
                *safe_stride(None, 5),
                None,
                *safe_stride(None, 4),
                None,
                *safe_stride(None, 4),
                return_pooled_scores,
                score_pooling_block_size_q,
                score_pooling_block_size_k,
                scores,
                *safe_stride(scores, 4),
                q.shape[0],
                q.shape[1],  #
                N_CTX=N_CTX,  #
                N_KV=(
                    k.shape[2]
                    if not USING_PAGED_CACHE
                    else k_cache.shape[0] * k_cache.shape[1]
                ),
                HEAD_DIM=HEAD_DIM_K,  #
                N_SPLIT=1,
                V_FP8=V_FP8,
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
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return _attention.apply(
        q,
        k,
        v,
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
    )
