import time
import unittest

import torch

from hip_attn.utils.attention_norm import attention_norm


class TestAttentionNorm(unittest.TestCase):

    def test_attention_norm(self):
        q = torch.randn((32, 8192, 128), dtype=torch.float16, device=0)
        k = torch.randn((32, 8192, 128), dtype=torch.float16, device=0)
        block_size_q = 32
        block_size_k = 2

        num_samples = 50

        elapsed_truth = 0
        for i in range(num_samples):
            torch.cuda.synchronize()
            t = time.time()
            truth = q @ k.transpose(-1, -2)
            truth = torch.where(
                torch.arange(0, k.shape[1], device=q.device)[None, None, :]
                <= torch.arange(0, q.shape[1], device=q.device)[None, :, None],
                truth,
                float("-inf"),
            )
            truth = truth.view(
                q.shape[0],
                q.shape[1] // block_size_q,
                block_size_q,
                k.shape[1] // block_size_k,
                block_size_k,
            )
            truth = truth.amax(-1).amax(-2)
            truth = torch.norm(torch.softmax(truth, dim=-1), dim=-1)
            torch.cuda.synchronize()
            if i > 3:
                elapsed_truth += time.time() - t

        elapsed_triton = 0
        for i in range(num_samples):
            torch.cuda.synchronize()
            t = time.time()
            norm = attention_norm(q, k)
            torch.cuda.synchronize()
            if i > 3:
                elapsed_triton += time.time() - t

        print("torch", truth[-1, -10:].tolist())
        print("triton", norm[-1, -10:].tolist())

        elapsed_truth /= num_samples - 4
        elapsed_triton /= num_samples - 4

        print("torch", elapsed_truth * 1000)
        print("triton", elapsed_triton * 1000)
