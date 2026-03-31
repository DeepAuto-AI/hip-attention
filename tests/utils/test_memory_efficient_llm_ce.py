import time
import unittest

import torch

from hip_attn.utils.memory_efficient_llm_ce import memory_efficient_llm_ce


class TestMemoryEfficientLLMCE(unittest.TestCase):

    def test_memory_efficient_llm_ce(self):
        HID = 4096
        NUM_VOCAB = 32001
        N = 8192

        hidden = torch.randn((N, HID), dtype=torch.float16, device=0)
        labels = torch.randint(0, NUM_VOCAB, (N,), dtype=torch.long, device=0)
        weight = torch.randn((NUM_VOCAB, HID), dtype=torch.float16, device=0)

        t = time.time()
        for i in range(100):
            if i == 3:
                t = time.time()
            logits = torch.nn.functional.linear(
                hidden,
                weight,
                None,
            )
            losses_torch = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)
        elapsed_torch = time.time() - t

        t = time.time()
        for i in range(100):
            if i == 3:
                t = time.time()
            losses_triton = memory_efficient_llm_ce(hidden, weight, labels, "none")
        elapsed_triton = time.time() - t

        print(losses_torch[:10])
        print(losses_triton[:10])

        print(elapsed_torch * 1000, elapsed_triton * 1000)
