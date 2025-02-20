import unittest

import torch

from hip_attn.v1_0.attention1 import attention


class TestAttention1(unittest.TestCase):

    def test_attention1(self):
        data_source = "llama"
        if data_source == "llama":
            state = torch.load("./cache/llama/qkvout.pth", map_location="cpu")
            q = state["q"]
            k = state["k"]
            v = state["v"]
            N, H, T_DST, HID = q.shape
            N, H, T_SRC, HID = k.shape
            idx = 7
            q = q.view(N * H, T_DST, HID)[idx : idx + 1].contiguous()
            k = k.view(N * H, T_SRC, HID)[idx : idx + 1].contiguous()
            v = v.view(N * H, T_SRC, HID)[idx : idx + 1].contiguous()
        else:
            q = torch.randn((1, 64, 4))
            k = torch.randn((1, 64, 4))
            v = k.clone()

        print(q.shape, k.shape, v.shape)
        out = attention(q, k, v)
