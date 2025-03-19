import os
import unittest

import torch
from hip_research.utils.load_checkouts import load_checkouts

import hip_attn.v1_1.attention2_draft_sampling
from hip_attn.v1_1.attention2_draft_prefetch import HiPAttentionArgs, hip_attention
from hip_attn.v1_1.attention2_draft_sampling import dual_stage_quadratic_hip_attention


class TestAttention2DraftSampling(unittest.TestCase):

    def test_attention(self):
        seq_len = 131072
        seq_dups = int(os.getenv("DUPS", "1"))

        q, k, v, out, cos, sin = load_checkouts(
            idx=0, window=40, seq_len=seq_len, return_cos_sin=True, dtype=torch.bfloat16
        )
        HEAD = q.shape[0]
        HEAD_KV = k.shape[0]
        seq_len = seq_len * seq_dups

        q = q.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
        k = k.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
        v = v.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)

        from flash_attn import flash_attn_func

        print("-" * 20)

        for i in range(3):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            flash_attn_func(q, k, v, causal=True)
            end.record()

            end.synchronize()
            print(start.elapsed_time(end))

        print("-" * 20)

        for i in range(10):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            hip_attention(
                q,
                k,
                v,
                args=HiPAttentionArgs(
                    mask_k=512,
                    block_size_q=64,
                    block_stride_q=2,
                    block_size_k=2,
                    block_stride_k=1,
                ),
                mask_only=False,
            )
            end.record()

            end.synchronize()
            print(start.elapsed_time(end))

        print("-" * 20)

        for i in range(10):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()
            if i == 0:
                hip_attn.v1_1.attention2_draft_sampling.DEBUG = (
                    os.getenv("DEBUG", "0") == "1"
                )

            # dual_stage_hip_attention(
            #     q, k, v,
            #     first_stage_args=HiPAttentionArgs(
            #         mask_k=seq_len // 128,
            #         block_size_q=128,
            #         block_stride_q=4,
            #         block_size_k=128,
            #         sliding_window_size=512,
            #         sink_token_size=512,
            #     ),
            #     second_stage_init_chunk=16,
            #     second_stage_init_k=512,
            #     second_stage_args=HiPAttentionArgs(
            #         mask_k=512,
            #         block_size_q=128,
            #         block_stride_q=2,
            #         block_size_k=8,
            #         block_stride_k=4,
            #         sliding_window_size=512,
            #         sink_token_size=512,
            #     )
            # )

            # sampling_only_attention(
            #     q, k, v,
            #     args=HiPAttentionArgs(
            #         mask_k=512,
            #         block_size_q=64,
            #         block_stride_q=2,
            #         block_size_k=64,
            #         sliding_window_size=512,
            #         sink_token_size=512,
            #     ),
            # )

            dual_stage_quadratic_hip_attention(
                q,
                k,
                v,
                args=HiPAttentionArgs(
                    mask_k=256,
                    block_size_q=64,
                    block_stride_q=4,
                    block_size_k=64,  # BLOCK_CHUNK
                    sliding_window_size=256,
                    sink_token_size=256,
                    # position_ids=position_ids,
                ),
                second_stage_k=2048,
                stages=[
                    # (128, 8192),
                    # (128, 65536),
                    # (64, 16384),
                    (64, 8192),
                    # (16, 1024),
                ],
            )

            # dual_stage_quadratic_scan_hip_attention(
            #     q, k, v,
            #     scan_chunk_size=512,
            #     scan_k=32768,
            #     args=HiPAttentionArgs(
            #         mask_k=512,
            #         block_size_q=64,
            #         block_stride_q=2,
            #         block_size_k=2, # BLOCK_CHUNK
            #         block_stride_k=1,
            #         sliding_window_size=256,
            #         sink_token_size=256,
            #         # position_ids=position_ids,
            #     )
            # )

            if i == 0:
                hip_attn.v1_1.attention2_draft_sampling.DEBUG = False
            end.record()

            end.synchronize()
            print(start.elapsed_time(end))
