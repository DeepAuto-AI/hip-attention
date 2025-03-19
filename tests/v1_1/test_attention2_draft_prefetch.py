import os
import unittest

import torch
from hip_research.utils.load_checkouts import load_checkouts

from hip_attn.v1_1.attention2_draft_prefetch import HiPAttentionArgs, hip_attention


class TestAttention2DraftPrefetch(unittest.TestCase):

    def test_attention(self):
        debug_only = True
        seq_len = 1024 * 128
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
        # q_quant = q.to(torch.float8_e5m2).view(torch.uint8)#[...,::2]
        # k_quant = k.to(torch.float8_e5m2).view(torch.uint8)#[...,::2]
        q_quant = q
        k_quant = k

        # bidirectional out
        # bi_probs = torch.softmax(q.permute(0, 2, 1, 3) @ k.repeat(1, 1, 4, 1).permute(0, 2, 3, 1), dim=-1)
        # plt.imshow(bi_probs[0, 0].cpu().float().numpy() ** 0.2)
        # plt.savefig('dummy_biprob.png')
        # out = (bi_probs @ v.repeat(1, 1, 4, 1).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        # print(out.shape)

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
                args=HiPAttentionArgs(
                    mask_k=2048,
                    block_size_q=64,
                    block_stride_q=2,
                    block_size_k=2,
                    block_stride_k=1,
                    block_size_k_group=1,
                    block_size_k_after_masking=-1,
                    group_size_q=1,
                    add_snap_kv=True,
                    snap_kv_vert_k=2048,
                    snap_kv_diag_k=2048,
                    is_causal=True,
                    sliding_window_size=1024,
                    sink_token_size=16,
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
                    low_res_oversample_block_stride_k=4,
                    q_quant=q_quant,
                    k_quant=k_quant,
                    randomize_mask=False,
                    # NOTE: change this to True to simulate key cache algorithms
                    output_key_access_log=False,
                ),
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
