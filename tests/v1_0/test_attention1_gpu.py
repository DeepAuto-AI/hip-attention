import unittest

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from hip_research.utils.load_checkouts import load_checkouts
from hip_research.utils.seed import seed
from torch import Tensor

import hip_attn.v1_0.attention1_gpu
from hip_attn.utils.benchmarking import get_bench
from hip_attn.v1_0.attention1_gpu import hip_attention


class TestAttention1GPU(unittest.TestCase):

    def test_main_debug(self):
        main_debug()

    def test_main_latency_benchmark(self):
        main_latency_benchmark()


def main_debug():
    hip_attn.v1_0.attention1_gpu.DEBUG = True

    q, k, v, out = load_checkouts(window=1)

    # bsize = 64
    # dups = 4
    # q = q.repeat(bsize, dups, 1)
    # k = k.repeat(bsize, dups, 1)
    # v = v.repeat(bsize, dups, 1)
    # skip = 7500
    # out = out[:, skip:, :]
    # q = q[:, skip:, :]

    context, (atten_indices, atten_ks, atten_probs) = hip_attention(
        q,
        k,
        v,
        # w_start=64,
        # n_patches=32,
        # mask_k=128,
        # scale_up=2,
    )

    stderr = (out - context).abs().mean().item()
    stdcontext = torch.std_mean(context)[0].item()

    print(
        f"err = {stderr:.6f} ({stderr / stdcontext:.4f} sigma), context_std = {stdcontext:.6f}"
    )


def torch_attention(q: Tensor, k: Tensor, v: Tensor):
    scores = torch.bmm(q, k.transpose(-1, -2))
    probs = torch.softmax(scores, dim=-1)
    context = torch.bmm(probs, v)
    return context, probs


def flash_attention(q: Tensor, k: Tensor, v: Tensor, attention_mask: Tensor = None):
    context = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=attention_mask is None,
        scale=1.0,
        attn_mask=attention_mask,
    )
    return context, None


def main_latency_benchmark():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dups", type=int, default=2)
    parser.add_argument("--query_size", type=int, default=1)
    parser.add_argument("--method", type=str, default="hip")
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    hip_attn.v1_0.attention1_gpu.DEBUG = args.debug
    TRACE = args.trace
    BSIZE = args.batch_size
    DUPS = args.dups
    QUERY_SIZE = args.query_size
    METHOD = args.method
    n_samples = args.samples

    if hip_attn.v1_0.attention1_gpu.DEBUG:
        seed()

    get_bench().disabled = not TRACE
    get_bench().synchronize = True

    q, k, v, out = load_checkouts(idx=0, window=40, seq_len=1024)

    q = q.repeat(BSIZE, DUPS, 1)[:, :QUERY_SIZE, :].contiguous()
    k = k.repeat(BSIZE, DUPS, 1)
    v = v.repeat(BSIZE, DUPS, 1)
    started = False

    samples = []
    for i in tqdm.tqdm(range(n_samples)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            if METHOD in ["torch", "none", "default"]:
                torch_attention(q, k, v)
            elif METHOD == "flash":
                flash_attention(q, k, v)
            elif METHOD == "hip":
                hip_attention(
                    q,
                    k,
                    v,
                )
            else:
                raise Exception()
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)

        if i > n_samples * 0.1:
            if not started:
                get_bench().reset_measures()
                get_bench().reset_trace()
                started = True
            samples.append(elapsed)

    if TRACE:
        print(get_bench().format_tracetree())

    samples = np.array(samples)
    print(
        f"[{METHOD}] {np.mean(samples):.4f}ms +- {np.std(samples):.4f}ms (q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)})"
    )


if __name__ == "__main__":
    import sys

    if sys.argv[-1] == "debug":
        main_debug()
    else:
        main_latency_benchmark()
