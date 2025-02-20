import unittest

import torch

from hip_attn import HiPAttentionArgs11, paged_hip_attention_11


class TestAttention2PagedDecode(unittest.TestCase):
    def test_paged_decode(self):
        state = torch.load(
            "cache/vllm/hip_attn/decode_example.pth", map_location="cuda:0"
        )

        out, _ = paged_hip_attention_11(
            q=state["q"],
            softmax_scale=state["softmax_scale"],
            args=HiPAttentionArgs11(
                k_cache=state["k"],
                v_cache=state["v"],
                block_table=state["block_table"],
                cache_seq_lens=state["cache_seq_lens"],
            ),
        )

        truth = state["out"]

        print(out)
        print(truth)

        print(out.shape, out.dtype, truth.shape, truth.dtype)
        print(torch.nn.functional.mse_loss(out, truth))
