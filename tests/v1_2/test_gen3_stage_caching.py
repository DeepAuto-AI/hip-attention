import os
import unittest

import torch
from hip_research.utils.load_checkouts import load_checkouts

from hip_attn.v1_2.attention_extend import (
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
    ScanStage,
    dual_stage_quadratic_hip_attention,
)


class TestGen3StageCaching(unittest.TestCase):
    def test_gen3_stage_caching(self):
        checkout = load_checkouts()

        seq_len = int(os.getenv("SEQ_LEN", "131072"))
        seq_dups = int(os.getenv("DUPS", "1"))

        assert seq_dups > 0

        q, k, v, out, cos, sin = load_checkouts(
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

        preset = os.getenv("HIP_PRESET", "mid")
        config_stage = {
            "high": [
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=1,
                    stage_chunk_size=64,
                    stage_k=None,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=1,
                    stage_chunk_size=16,
                    stage_k=65536,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=1,
                    stage_chunk_size=1,
                    stage_k=16384,
                    stage_stride=1,
                ),
            ],
            "mid": [
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=4,
                    stage_chunk_size=256,
                    stage_k=None,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=4,
                    stage_chunk_size=32,
                    stage_k=32768,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=1,
                    stage_chunk_size=16,
                    stage_k=8192,
                    stage_stride=1,
                ),
            ],
            "low": [
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=4,
                    stage_chunk_size=256,
                    stage_k=None,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=4,
                    stage_chunk_size=32,
                    stage_k=32768,
                    stage_stride=1,
                ),
            ],
            "debug": [
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=4,
                    stage_chunk_size=16,
                    stage_k=None,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=2,
                    stage_chunk_size=4,
                    stage_k=512,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=1,
                    stage_chunk_size=1,
                    stage_k=256,
                    stage_stride=1,
                ),
            ],
        }[preset]
        config_second_k = {
            "high": 4096,
            "mid": 2048,
            "low": 2048,
            "debug": 128,
        }[preset]
        config_sa_extend_backend = {
            "high": "streaming",
            "mid": "streaming",
            "low": "streaming",
            "debug": "streaming",
        }[preset]

        dual_stage_kwargs = dict(
            q=q,
            k=k,
            v=v,
            args=HiPAttentionArgs(
                sliding_window_size=128 if preset == "debug" else 1024,
                sink_token_size=64 if preset == "debug" else 256,
                need_apply_rope=True,
                using_extend=True,
                rope_cos=cos,
                rope_sin=sin,
                second_stage_k=config_second_k,
                stages=config_stage,
                model_context_length=64 * 1024,
                sa_extend_backend=config_sa_extend_backend,
            ),
        )

        output, metadata = dual_stage_quadratic_hip_attention(**dual_stage_kwargs)

        assert metadata is not None

        for i_cached_stage in range(len(metadata.stage_caches)):
            new_metadata = HiPAttentionOutputMetadata(
                indices=None,
                ks=None,
                ks_count=None,
                ks_start_end=None,
                mask_cache_statistics=None,
                sa_cache_statistics=None,
                stage_caches=metadata.stage_caches[: i_cached_stage + 1],
            )
            output_cached, _ = dual_stage_quadratic_hip_attention(
                **dual_stage_kwargs,
                cached_metadata=new_metadata,
            )
            mse = ((output - output_cached) ** 2).sum()
            print(mse)
            assert mse < 1e-8, mse
