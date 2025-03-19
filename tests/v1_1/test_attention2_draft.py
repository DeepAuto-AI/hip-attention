import unittest

import torch
from hip_research.utils.load_checkouts import load_checkouts

from hip_attn.v1_1.attention2_draft import hip_attention


class TestAttention2Draft(unittest.TestCase):

    def test_attention(self):
        q, k, v, out, cos, sin = load_checkouts(
            idx=1, window=16, seq_len=4096, return_cos_sin=True, dtype=torch.float32
        )

        # q = q[:, -32:, :]
        # out = out[:, -32:, :]

        context, _ = hip_attention(
            q,
            k,
            v,
            mask_k=512,
            block_size_k=4,
            block_size_k_group=1,
            block_size_q=32,
            num_sifter=1,
            recursive_sifter=False,
            using_sliding_window=True,
            sliding_window_size=128,
            using_extend=False,
            rope_cos=cos,
            rope_sin=sin,
            self_extend_neighboor_window=1024,
            self_extend_group_size=8,
            topk_head_group_size=8,
        )

        stderr = (out - context).abs().mean().item()
        stdcontext = torch.std_mean(out)[0].item()

        print(
            f"err = {stderr:.6f} ({stderr / stdcontext:.4f} sigma), out_std = {stdcontext:.6f}"
        )
