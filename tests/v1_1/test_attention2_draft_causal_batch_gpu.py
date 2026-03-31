import os
import unittest

import torch
from hip_research.utils.load_checkouts import load_checkouts

from hip_attn.v1_1.attention2_draft_causal_batch_gpu import hip_attention


class TestAttention2DraftCausalBatchGpu(unittest.TestCase):

    def test_attention(self):
        seq_len = 4096
        if os.getenv("HIP_DEBUG", "1") == "0":
            seq_len = 32768

        q, k, v, out, cos, sin = load_checkouts(
            idx=0, window=40, seq_len=seq_len, return_cos_sin=True, dtype=torch.bfloat16
        )

        # q = q[:, -32:, :]
        # out = out[:, -32:, :]

        def fn():
            return hip_attention(
                q,
                k,
                v,
                mask_k=512,
                block_size_q=32,
                block_size_k=2,
                block_size_k_group=1,
                sliding_window_size=128,
                sink_token_size=16,
                using_extend=False,
                rope_cos=cos,
                rope_sin=sin,
                self_extend_neighboor_window=1024,
                self_extend_group_size=4,
                topk_head_group_size=2,
                sample_method="first",
                branch_method="half",
                traverse_from_last_step=False,
                step_size=64,
                num_samples=1,
                chunk_size=None,
                num_unions=1,
                score_head_group_size=1,
                using_sparq=False,
                sparq_hid=64,
                low_res_sample_scale=1,
                low_res_oversample_rate=1,
                low_res_oversample_block_stride_k=2,
            )

        if "HIP_DEBUG" not in os.environ:
            os.environ["HIP_DEBUG"] = "1"

        context, _ = fn()

        os.environ["HIP_DEBUG"] = "0"

        torch.cuda.synchronize()

        graph = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        sample = 0
        elapsed = 0
        for i in range(50):
            if graph is None:
                for _ in range(3):
                    fn()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()

                print("graph compiled")

            start.record()
            graph.replay()
            end.record()

            if i > 3:
                torch.cuda.synchronize()
                elapsed += start.elapsed_time(end)
                sample += 1

        if sample > 0:
            print(f"latency: {elapsed / sample:.6f} ms")

        if context is not None:
            stderr = (out - context).abs().mean().item()
            stdcontext = torch.std_mean(out)[0].item()

            print(
                f"err = {stderr:.8f} ({stderr / stdcontext:.6f} sigma), out_std = {stdcontext:.8f}"
            )
