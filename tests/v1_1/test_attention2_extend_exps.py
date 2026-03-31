import os
import unittest

import numba
import numpy as np
import torch
import triton
from hip_research.utils.load_checkouts import load_checkouts

from hip_attn.v1_1.attention2_draft_sampling_extend import (
    HiPAttentionArgs,
    dual_stage_quadratic_hip_attention,
)


class TestAttention2ExtendExps(unittest.TestCase):

    def test_attention_extend_exps(self):
        seq_len = 131072
        seq_dups = int(os.getenv("DUPS", "1"))
        batch_size = int(os.getenv("BATCH_SIZE", "1"))

        assert seq_dups > 0

        q, k, v, _, cos, sin = load_checkouts(
            idx=0,
            window=40,
            seq_len=seq_len,
            return_cos_sin=True,
            derope=True,
            dtype=torch.bfloat16,
        )
        HEAD = q.shape[0]
        HEAD_KV = k.shape[0]
        seq_len = seq_len * seq_dups

        q = q.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
        k = (
            k.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
        )  # .to(torch.float8_e5m2)
        v = (
            v.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
        )  # .to(torch.float8_e5m2)
        if cos is not None:
            cos = cos.repeat(seq_dups, 1)  # .to(torch.float8_e5m2)
            sin = sin.repeat(seq_dups, 1)  # .to(torch.float8_e5m2)

        if batch_size > 1:
            q = q[:, :1, :, :].contiguous()
            q = q.expand(batch_size, -1, -1, -1)
            k = k.expand(batch_size, -1, -1, -1)
            v = v.expand(batch_size, -1, -1, -1)

        print(q.shape, k.shape, v.shape)

        _, high_res_metadata = dual_stage_quadratic_hip_attention(
            q=q,
            k=k,
            v=v,
            args=HiPAttentionArgs(
                mask_k=128,
                block_size_q=64,
                block_stride_q=1,
                block_size_k=64,  # BLOCK_CHUNK
                block_stride_k=1,
                sliding_window_size=1024,
                sink_token_size=256,
                # position_ids=position_ids,
                using_extend=True,
                rope_cos=cos,
                rope_sin=sin,
                need_apply_rope=True,
            ),
            second_stage_k=1024,
            stages=[
                (1, 32, 32768),
                (1, 1, 8192),
            ],
            scan_stride=1,
            stage_stride=1,
            block_sparse_block_size_q=64,
            model_context_length=131072,
            scan_early_terminate=1,
            stage_early_terminate=1,
            mask_only=False,
            scan_extend_backend="streaming",
            sa_extend_backend="streaming",
        )

        _, low_res_metadata = dual_stage_quadratic_hip_attention(
            q=q,
            k=k,
            v=v,
            args=HiPAttentionArgs(
                mask_k=256,
                block_size_q=64,
                block_stride_q=1,
                block_size_k=64,  # BLOCK_CHUNK
                block_stride_k=1,
                sliding_window_size=1024,
                sink_token_size=256,
                # position_ids=position_ids,
                using_extend=True,
                rope_cos=cos,
                rope_sin=sin,
                need_apply_rope=True,
            ),
            second_stage_k=2048,
            stages=[
                (1, 32, 32768),
                (1, 1, 8192),
            ],
            scan_stride=16,
            stage_stride=8,
            block_sparse_block_size_q=64,
            model_context_length=131072,
            scan_early_terminate=1,
            stage_early_terminate=1,
            mask_only=False,
            scan_extend_backend="streaming",
            sa_extend_backend="streaming",
        )


@numba.njit
def convert_to_dense_numba(
    indices,
    mask,
    TDST,
    TSRC,
    BQ,
    BK,
):
    target_batch = 0
    BDST = indices.shape[1]
    KS = indices.shape[2]
    for ibdst in numba.prange(BDST):
        for iks in range(KS):
            k = indices[target_batch, ibdst, iks]
            mask[ibdst]


def convert_to_dense(
    indices,
    TDST,
    TSRC,
    block_size_q,
    block_size_k,
):
    mask = np.zeros(
        (triton.cdiv(TDST, block_size_q), triton.cdiv(TSRC, block_size_k)),
        dtype=np.uint8,
    )

    convert_to_dense_numba(indices, mask, TDST, TSRC, block_size_q, block_size_k)

    return mask
