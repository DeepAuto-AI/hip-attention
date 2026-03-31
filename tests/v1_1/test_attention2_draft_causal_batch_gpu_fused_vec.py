import os
import unittest

import torch
from hip_research.utils.load_checkouts import load_checkouts

from hip_attn.v1_1.attention2_draft_causal_batch_gpu_fused_vec import hip_attention


class TestAttention2DraftCausalBatchGpuFusedVec(unittest.TestCase):

    def test_attention(self):
        debug_only = True
        seq_len = 8192 * 4
        seq_repeat = 1
        batch_repeat = 1
        if os.getenv("HIP_DEBUG", "1") == "0":
            seq_len = 32768
            # seq_len = 16384
            # seq_len = 131072
            seq_repeat = 1
            batch_repeat = 1
            debug_only = False

        q, k, v, out, cos, sin = load_checkouts(
            idx=0, window=40, seq_len=seq_len, return_cos_sin=True, dtype=torch.bfloat16
        )
        HEAD = q.shape[0]
        HEAD_KV = k.shape[0]

        if seq_repeat > 1 or batch_repeat > 1:
            q = q.repeat(batch_repeat, seq_repeat, 1)
            k = k.repeat(batch_repeat, seq_repeat, 1)
            v = v.repeat(batch_repeat, seq_repeat, 1)
            out = out.repeat(batch_repeat, seq_repeat, 1)
            cos = cos.repeat(seq_repeat, 1)
            sin = sin.repeat(seq_repeat, 1)

        def reshape(x, HEAD):
            N, T, H = x.shape
            x = (
                x.contiguous()
                .view(N // HEAD, HEAD, T, H)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            assert x.shape == (N // HEAD, T, HEAD, H)
            assert x.is_contiguous()
            return x

        q = reshape(q, HEAD)
        k = reshape(k, HEAD_KV)
        v = reshape(v, HEAD_KV)
        out = reshape(out, HEAD)
        q_quant = q.to(torch.float8_e5m2).view(torch.uint8)  # [...,::2]
        k_quant = k.to(torch.float8_e5m2).view(torch.uint8)  # [...,::2]
        # q_quant = q
        # k_quant = k

        # num_queries = 1
        # q = q[:, -num_queries:]
        # q_quant = q_quant[:, -num_queries:]
        # out = out[:, -num_queries:,]

        print(q.shape, k.shape, v.shape)

        def fn():
            return hip_attention(
                q,
                k,
                v,
                mask_k=128,
                block_size_q=32,
                block_stride_q=2,
                block_size_k=2,
                block_stride_k=1,
                block_size_k_group=1,
                sliding_window_size=32,
                sink_token_size=4,
                using_extend=False,
                rope_cos=cos,
                rope_sin=sin,
                self_extend_neighboor_window=1024,
                self_extend_group_size=4,
                topk_head_group_size=1,
                sample_method="center",
                branch_method="half",
                traverse_from_last_step=False,
                step_size=None,
                num_samples=1,
                chunk_size=None,
                num_unions=1,
                score_head_group_size=1,
                using_sparq=False,
                sparq_hid=64,
                low_res_sample_scale=1,
                low_res_oversample_rate=1,
                low_res_oversample_block_stride_k=1,
                q_quant=q_quant,
                k_quant=k_quant,
                # NOTE: change this to True to simulate key cache algorithms
                output_key_access_log=True,
            )

        if "HIP_DEBUG" not in os.environ:
            os.environ["HIP_DEBUG"] = "1"

        context, metadata = fn()

        if context is not None:
            stderr = (out - context).abs().mean().item()
            stdcontext = torch.std_mean(out)[0].item()

            print(
                f"err = {stderr:.8f} ({stderr / stdcontext:.6f} sigma), out_std = {stdcontext:.8f}"
            )

        if debug_only:
            return

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

            if i > 3:
                start.record()
            graph.replay()
            if i > 3:
                end.record()

            if i > 3:
                torch.cuda.synchronize()
                elapsed += start.elapsed_time(end)
                sample += 1

        if sample > 0:
            print(f"latency: {elapsed / sample:.6f} ms")
