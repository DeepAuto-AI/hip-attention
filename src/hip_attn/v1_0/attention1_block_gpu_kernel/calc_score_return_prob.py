import os
from typing import Union

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from transformers.utils import logging

from hip_attn.utils.benchmarking import get_bench

logger = logging.get_logger(__name__)
timer = lambda x: get_bench().region(x)

DEBUG = os.environ.get("hip_DEBUG", "0") == "1"


def next_multiple_of(x: int, multiple_by: int = 16):
    return triton.next_power_of_2(max(x, multiple_by))


@triton.jit
def _calc_score_compute(
    # input matrix
    QUERIES,
    stride_queries_n,
    stride_queries_tdst,
    stride_queries_hid,
    KEYS,
    stride_keys_n,
    stride_keys_tsrc,
    stride_keys_hid,
    ATTEN_MASK,
    stride_atten_mask_n,
    stride_atten_mask_tsrc,
    # block indices
    INDICES,
    stride_indices_n,
    stride_indices_bdst,
    stride_indices_bk,
    KS,
    stride_ks_n,
    stride_ks_bdst,
    # out matrix
    SCORES,
    stride_scores_n,
    stride_scores_tdst,
    stride_scores_k,
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
        KS + idx_n * stride_ks_n + idx_bdst * stride_ks_bdst,
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
        INDICES
        + idx_n * stride_indices_n
        + idx_bdst * stride_indices_bdst
        + idx_bk * stride_indices_bk,
        mask=mask_bk,
    )
    # [BLOCK_BK: bk, BLOCK_SIZE_K_PADDED]
    idx_tsrc = idx_tsrc[:, None] + idx_block_k[None, :]
    mask_tsrc = (idx_tsrc < TSRC) & mask_block_k[None, :] & mask_bk[:, None]

    # [BLOCK_BK: bk, BLOCK_SIZE_K_PADDED]
    if ATTEN_MASK is not None:
        key_mask = tl.load(
            ATTEN_MASK
            + idx_n * stride_atten_mask_n
            + idx_tsrc * stride_atten_mask_tsrc,
            mask=mask_tsrc,
            other=False,
        ).to(tl.int1)
        mask_tsrc = mask_tsrc & key_mask

    idx_tdst = idx_bdst * BLOCK_SIZE_Q + idx_block_q
    mask_tdst = (idx_tdst < TDST) & mask_block_q
    if ATTEN_MASK is not None:
        query_mask = tl.load(
            ATTEN_MASK
            + idx_n * stride_atten_mask_n
            + (idx_tdst + TSRC - TDST) * stride_atten_mask_tsrc,
            mask=mask_tdst,
            other=False,
        ).to(tl.int1)
        mask_tdst = mask_tdst & query_mask

    # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_BK: bk, BLOCK_SIZE_K_PADDED: tsrc]
    scores = tl.zeros(
        (BLOCK_SIZE_Q_PADDED, BLOCK_BK, BLOCK_SIZE_K_PADDED), dtype=tl.float32
    )
    for pid_hid in range(tl.cdiv(HID, BLOCK_HID)):
        idx_hid = (tl.arange(0, BLOCK_HID) + pid_hid * BLOCK_HID).to(tl.int64)
        mask_hid = idx_hid < HID

        # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_HID: hid]
        queries = tl.load(
            QUERIES
            + idx_n * stride_queries_n
            + idx_tdst[:, None] * stride_queries_tdst
            + idx_hid[None, :] * stride_queries_hid,
            mask=mask_tdst[:, None] & mask_hid[None, :],
            other=0,
        )

        if KEY_CACHE_METHOD == "cont":
            # [BLOCK_HID: hid, BLOCK_BK: bk, BLOCK_SIZE_K_PADDED: tsrc]
            keys = tl.load(
                KEYS
                + (idx_n // KV_REPEAT_INTERLEAVE) * stride_keys_n
                + idx_tsrc[None, :, :] * stride_keys_tsrc
                + idx_hid[:, None, None] * stride_keys_hid,
                mask=mask_tsrc[None, :, :] & mask_hid[:, None, None],
                other=0,
            )
        elif KEY_CACHE_METHOD == "vllm":
            """
            idx_block = block_tables[idx_batch, idx_tsrc // block_size]
            offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
            key = key_cache[idx_block, idx_head, :, offset_block, :].reshape(-1)
            """
            idx_batch = ((idx_n // KV_REPEAT_INTERLEAVE) // VLLM_NUM_KV_HEADS).to(
                tl.int64
            )
            idx_head = ((idx_n // KV_REPEAT_INTERLEAVE) % VLLM_NUM_KV_HEADS).to(
                tl.int64
            )
            idx_block = tl.load(
                BLOCK_TABLES
                + idx_batch * stride_block_tables_num_seqs
                + (idx_tsrc // VLLM_BLOCK_SIZE)
                * stride_block_tables_max_num_blocks_per_seq,
                mask=mask_tsrc,
            ).to(tl.int64)
            offset_block = (
                idx_tsrc - ((idx_tsrc // VLLM_BLOCK_SIZE) * VLLM_BLOCK_SIZE)
            ).to(tl.int64)

            # [BLOCK_HID: hid, BLOCK_BK: bk, BLOCK_SIZE_K_PADDED: tsrc]
            keys = tl.load(
                KEYS
                + idx_block[None, :, :] * stride_keys_vllm_num_bocks
                + idx_head * stride_keys_vllm_num_kv_heads
                + (idx_hid[:, None, None] // VLLM_X) * stride_keys_vllm_head_size_x
                + offset_block[None, :, :] * stride_keys_vllm_block_size
                + (idx_hid[:, None, None] % VLLM_X) * stride_keys_vllm_x,
                mask=mask_tsrc[None, :, :] & mask_hid[:, None, None],
                other=0,
            )
        else:
            raise Exception()
        keys = tl.reshape(keys, (BLOCK_HID, BLOCK_BK * BLOCK_SIZE_K_PADDED))

        # TOOD: WIP

        if keys.dtype == tl.uint8:
            keys = keys.to(tl.float8e5, bitcast=True).to(queries.dtype)
        scores_mini = tl.dot(queries, keys)
        scores_mini = tl.reshape(
            scores_mini, (BLOCK_SIZE_Q_PADDED, BLOCK_BK, BLOCK_SIZE_K_PADDED)
        )

        scores += scores_mini.to(scores.dtype)

    idx_scorek = idx_bk[:, None] * BLOCK_SIZE_K + idx_block_k[None, :]
    mask_scorek = (idx_scorek < K) & mask_block_k[None, :] & mask_bk[:, None]

    scores_mask = (
        (mask_tdst[:, None, None] & mask_tsrc[None, :, :]) & mask_scorek[None, :] & True
    )

    if IS_CAUSAL:
        scores_mask = scores_mask & (
            (idx_tdst[:, None, None] + (TSRC - TDST)) >= idx_tsrc[None, :, :]
        )

    tl.store(
        SCORES
        + idx_n * stride_scores_n
        + idx_tdst[:, None, None] * stride_scores_tdst
        + idx_scorek[None, :, :] * stride_scores_k,
        mask=scores_mask,
        value=scores,
    )


@triton.jit
def _calc_score_compute_bwd_queries(
    # input matrices
    KS,
    stride_ks_n,
    stride_ks_bdst,
    INDICES,
    stride_indices_n,
    stride_indices_bdst,
    stride_indices_bk,
    KEYS,
    stride_keys_n,
    stride_keys_tsrc,
    stride_keys_hid,
    # grad output (read)
    GRAD_SCORES,
    stride_grad_scores_n,
    stride_grad_scores_tdst,
    stride_grad_scores_k,
    # grad input (write)
    GRAD_QUERIES,
    stride_grad_queries_n,
    stride_grad_queries_tdst,
    stride_grad_queries_hid,
    # input variables
    N,
    TDST,
    TSRC,
    HID,
    BLOCK_K,
    K,
    # block constant
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_Q_PADDED: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_K_PADDED: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
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
    idx_query_block = tl.program_id(1)

    idx_block_q = tl.arange(0, BLOCK_SIZE_Q_PADDED)
    idx_block_k = tl.arange(0, BLOCK_SIZE_K_PADDED)
    idx_hid = tl.arange(0, BLOCK_HID)

    scalar_ks = tl.load(
        KS
        + idx_n.to(tl.int64) * stride_ks_n
        + idx_query_block.to(tl.int64) * stride_ks_bdst
    )

    accumulator = tl.zeros(
        (
            BLOCK_SIZE_Q_PADDED,
            BLOCK_HID,
        ),
        dtype=tl.float32,
    )
    for idx_key_block in range(scalar_ks):
        idx_key_start = tl.load(
            INDICES
            + idx_n.to(tl.int64) * stride_indices_n
            + idx_query_block.to(tl.int64) * stride_indices_bdst
            + idx_key_block.to(tl.int64) * stride_indices_bk,
        )

        if IS_CAUSAL:
            causal_mask = (idx_key_start + idx_block_k)[None, :] <= (
                idx_query_block * BLOCK_SIZE_Q + idx_block_q
            )[:, None]
        else:
            causal_mask = True

        # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_SIZE_K_PADDED: score]
        grad_score = tl.load(
            GRAD_SCORES
            + idx_n.to(tl.int64) * stride_grad_scores_n
            + (idx_query_block * BLOCK_SIZE_Q + idx_block_q)[:, None].to(tl.int64)
            * stride_grad_scores_tdst
            + (idx_key_block * BLOCK_SIZE_K + idx_block_k)[None, :].to(tl.int64)
            * stride_grad_scores_k,
            mask=((idx_query_block * BLOCK_SIZE_Q + idx_block_q)[:, None] < TDST)
            & (idx_block_q[:, None] < BLOCK_SIZE_Q)
            & ((idx_key_block * BLOCK_SIZE_K + idx_block_k)[None, :] < K)
            & (idx_block_k[None, :] < BLOCK_SIZE_K)
            & causal_mask,
            other=0,
        )

        # [BLOCK_SIZE_K_PADDED: score, BLOCK_HID: hid]
        key = tl.load(
            KEYS
            + idx_n.to(tl.int64) * stride_keys_n
            + (idx_key_start + idx_block_k)[:, None].to(tl.int64) * stride_keys_tsrc
            + idx_hid[None, :].to(tl.int64) * stride_keys_hid,
            mask=((idx_key_start + idx_block_k)[:, None] < TSRC)
            & (idx_block_k[:, None] < BLOCK_SIZE_K)
            & (idx_hid[None, :] < HID),
            other=0,
        )

        # tl.device_print("", idx_tsrc)
        accumulator += tl.dot(grad_score, key).to(accumulator.dtype)

    tl.store(
        GRAD_QUERIES
        + idx_n.to(tl.int64) * stride_grad_queries_n
        + (idx_query_block * BLOCK_SIZE_Q + idx_block_q)[:, None].to(tl.int64)
        * stride_grad_queries_tdst
        + idx_hid[None, :].to(tl.int64) * stride_grad_queries_hid,
        mask=((idx_query_block * BLOCK_SIZE_Q + idx_block_q)[:, None] < TDST)
        & (idx_block_q[:, None] < BLOCK_SIZE_Q)
        & (idx_hid[None, :] < HID),
        value=accumulator,
    )


@triton.jit
def _calc_score_compute_bwd_keys(
    # input matrices
    ks,
    stride_ks_n,
    stride_ks_bdst,
    indices,
    stride_indices_n,
    stride_indices_bdst,
    stride_indices_bk,
    queries,
    stride_queries_n,
    stride_queries_tdst,
    stride_queries_hid,
    # grad output (read)
    grad_scores,
    stride_grad_scores_n,
    stride_grad_scores_tdst,
    stride_grad_scores_k,
    # grad input (write)
    grad_keys,
    stride_grad_keys_n,
    stride_grad_keys_tsrc,
    stride_grad_keys_hid,
    # input variables
    N,
    TDST,
    TSRC,
    HID,
    BK,
    K,
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
        ks + idx_n * stride_ks_n + idx_bdst * stride_ks_bdst,
    )
    # mask_job = idx_bk < scalar_ks
    if idx_bk >= scalar_ks:
        return

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

    # [BLOCK_SIZE_K_PADDED: tsrc, BLOCK_SIZE_Q_PADDED: tdst]
    grad_score = tl.load(
        grad_scores
        + idx_n * stride_grad_scores_n
        + idx_tdst[None, :] * stride_grad_scores_tdst
        + idx_k[:, None] * stride_grad_scores_k,
        mask=mask_tdst[None, :] & mask_k[:, None],
        other=0,
    )
    # [BLOCK_SIZE_Q_PADDED: tdst, BLOCK_HID: hid]
    query = tl.load(
        queries
        + idx_n * stride_queries_n
        + idx_tdst[:, None] * stride_queries_tdst
        + idx_hid[None, :] * stride_queries_hid,
        mask=mask_tdst[:, None] & mask_hid[None, :],
        other=0,
    )
    # [BLOCK_SIZE_K_PADDED: tsrc, BLOCK_HID: hid]
    scores = tl.dot(grad_score, query)

    idx_tsrc = tl.load(
        indices
        + idx_n * stride_indices_n
        + idx_bdst * stride_indices_bdst
        + idx_bk * stride_indices_bk,
    )
    idx_tsrc = idx_tsrc + idx_block_k
    mask_tsrc = (idx_tsrc < TSRC) & mask_block_k
    tl.atomic_add(
        grad_keys
        + idx_n * stride_grad_keys_n
        + idx_tsrc[:, None] * stride_grad_keys_tsrc
        + idx_hid[None, :] * stride_grad_keys_hid,
        val=scores,
        mask=mask_tsrc[:, None] & mask_hid[None, :],
    )


# NOTE: you have to perform softmax after this
class CalcScoreAutoGradFn(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(  # noqa
        ctx,
        # matrices
        queries: Tensor,
        keys: Union[Tensor, "PagedKeyCacheVllmCompat"],
        attention_mask: Tensor,
        # indices matrices
        indices: Tensor,
        ks: Tensor,
        # block constant
        KV_REPEAT_INTERLEAVE: int,
        BLOCK_SIZE_Q: int,
        BLOCK_SIZE_K: int,
        IS_CAUSAL: bool,
    ):
        ctx.save_for_backward(queries, keys, indices, ks)
        ctx.BLOCK_SIZE_Q = BLOCK_SIZE_Q
        ctx.BLOCK_SIZE_K = BLOCK_SIZE_K
        ctx.IS_CAUSAL = IS_CAUSAL

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
            dtype=queries.dtype,
        )

        BLOCK_SIZE_Q_PADDED = next_multiple_of(BLOCK_SIZE_Q, 16)
        BLOCK_SIZE_K_PADDED = next_multiple_of(BLOCK_SIZE_K, 1)
        BLOCK_BK = next_multiple_of(128 // BLOCK_SIZE_K_PADDED, 1)
        # BLOCK_BK = 1
        BLOCK_HID = triton.next_power_of_2(HID)
        # BLOCK_HID = max(BLOCK_SIZE_Q_PADDED, BLOCK_SIZE_K_PADDED)
        BLOCK_HID = 32

        if isinstance(keys, Tensor):
            KEY_CACHE_METHOD = "cont"

            VLLM_NUM_BLOCKS = VLLM_NUM_KV_HEADS = VLLM_HEAD_SIZE_X = VLLM_BLOCK_SIZE = (
                VLLM_X
            ) = VLLM_HEAD_SIZE = 0

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

            KEY_CACHE_METHOD = "vllm"

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
            orig_device = torch.cuda.current_device()
            torch.cuda.set_device(queries.device)
            _calc_score_compute[grid](
                # input matrix
                queries,
                *queries.stride(),
                keys,
                *keys.stride(),
                attention_mask,
                *(attention_mask.stride() if attention_mask is not None else (0, 0)),
                # block indices
                indices,
                *indices.stride(),
                ks,
                *ks.stride(),
                # out matrix
                scores,
                *scores.stride(),
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
                block_tables,
                *block_tables_strides,
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
            torch.cuda.set_device(orig_device)

        # print(scores[0, 300, :])
        return scores

    @staticmethod
    def backward(ctx, grad_scores):  # noqa
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
                assert grad_queries.ndim == 3

                _calc_score_compute_bwd_queries[grid](
                    ks,
                    ks.stride(0),
                    ks.stride(1),
                    indices,
                    indices.stride(0),
                    indices.stride(1),
                    indices.stride(2),
                    keys,
                    keys.stride(0),
                    keys.stride(1),
                    keys.stride(2),
                    grad_scores,
                    grad_scores.stride(0),
                    grad_scores.stride(1),
                    grad_scores.stride(2),
                    grad_queries,
                    grad_queries.stride(0),
                    grad_queries.stride(1),
                    grad_queries.stride(2),
                    N,
                    T_DST,
                    T_SRC,
                    HID,
                    BK,
                    K,
                    BLOCK_SIZE_Q,
                    next_multiple_of(BLOCK_SIZE_Q, 16),
                    BLOCK_SIZE_K,
                    next_multiple_of(BLOCK_SIZE_K, 16),
                    BLOCK_HID,
                    ctx.IS_CAUSAL,
                )

        # for keys
        if ctx.needs_input_grad[1]:
            grid = (N, triton.cdiv(T_DST, BLOCK_SIZE_Q), BK)
            BLOCK_HID = triton.next_power_of_2(HID)

            grad_keys = torch.zeros_like(keys, dtype=torch.float32)

            if ENABLED:
                _calc_score_compute_bwd_keys[grid](
                    ks,
                    ks.stride(0),
                    ks.stride(1),
                    indices,
                    indices.stride(0),
                    indices.stride(1),
                    indices.stride(2),
                    queries,
                    queries.stride(0),
                    queries.stride(1),
                    queries.stride(2),
                    grad_scores,
                    grad_scores.stride(0),
                    grad_scores.stride(1),
                    grad_scores.stride(2),
                    grad_keys,
                    grad_keys.stride(0),
                    grad_keys.stride(1),
                    grad_keys.stride(2),
                    N,
                    T_DST,
                    T_SRC,
                    HID,
                    BK,
                    K,
                    BLOCK_SIZE_Q,
                    next_multiple_of(BLOCK_SIZE_Q, 16),
                    BLOCK_SIZE_K,
                    next_multiple_of(BLOCK_SIZE_K, 16),
                    BLOCK_HID,
                )

            grad_keys = grad_keys.to(keys.dtype)

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
    queries: Tensor,
    keys: Tensor,
    attention_mask: Tensor,
    indices: Tensor,
    ks: Tensor,
    KV_REPEAT_INTERLEAVE: int,
    BLOCK_SIZE_Q: int,
    BLOCK_SIZE_K: int,
    IS_CAUSAL: bool,
):
    scores = CalcScoreAutoGradFn.apply(
        queries,
        keys,
        attention_mask,
        indices,
        ks,
        KV_REPEAT_INTERLEAVE,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
        IS_CAUSAL,
    )  # type: Tensor

    with timer("calc_score_return_prob.softmax"):
        probs = scores.softmax(-1).to(scores.dtype)

    assert probs.dtype == queries.dtype

    N, TDST, K = scores.shape
    if attention_mask is not None:
        _, TSRC = attention_mask.shape
        if probs.requires_grad:
            probs = probs * attention_mask[:, TSRC - TDST :, None]
        else:
            probs.masked_fill_(~attention_mask[:, TSRC - TDST :, None], 0)

    assert scores.dtype == queries.dtype
    assert probs.dtype == queries.dtype

    return scores, probs
