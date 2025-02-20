from __future__ import annotations

import unittest

import torch

from hip_attn.v1_2.attention_decode_bsa import decode_block_sparse_attention
from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention


class TestAttentionDecodeBSA(unittest.TestCase):
    def test_correctness(self):
        file = "../bsa_args_2.pth"
        test_correctness(file, use_cuda_graph=False)

    def test_performance(self):
        file = "../bsa_args_2.pth"
        benchmark(file, use_cuda_graph=False)


def test_correctness(file_path, use_cuda_graph=False):
    args = load_saved_tensors(file_path)

    def run_orig(output=None):
        result = block_sparse_attention(
            args["q"],
            args["k"],
            args["v"],
            args["seq_lens"],
            args["indices"],
            args["ks"],
            args["ks_count"],
            args["ks_start_end"],
            args["args"],
            args["access_counter"],
            args["cache_miss_counter"],
            args["EXTEND_BACKEND"],
            args["model_context_length"],
            args["extend_context_length"],
        )
        if output is not None:
            output.copy_(result)
        return result

    def run_flash(output=None):
        result = decode_block_sparse_attention(
            args["q"],
            args["k"],
            args["v"],
            args["seq_lens"],
            args["indices"],
            args["ks"],
            args["ks_count"],
            args["ks_start_end"],
            args["args"],
            args["access_counter"],
            args["cache_miss_counter"],
            args["EXTEND_BACKEND"],
            args["model_context_length"],
            args["extend_context_length"],
        )
        if output is not None:
            output.copy_(result)
        return result

    if use_cuda_graph:
        gt_output = torch.zeros_like(args["q"])

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run_orig(gt_output)
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g_orig = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_orig):
            run_orig(gt_output)

        gt_output.zero_()
        g_orig.replay()

        output = torch.zeros_like(args["q"])

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run_flash(output)
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g_flash = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_flash):
            run_flash(output)

        output.zero_()
        g_flash.replay()

    else:
        gt_output = run_orig()
        output = run_flash()

    print("context diff", (output - gt_output).abs().mean() / gt_output.abs().mean())


@torch.no_grad()
def benchmark(file_path, use_cuda_graph=False):
    from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention

    args = load_saved_tensors(file_path)

    def run_orig():
        return block_sparse_attention(
            args["q"],
            args["k"],
            args["v"],
            args["seq_lens"],
            args["indices"],
            args["ks"],
            args["ks_count"],
            args["ks_start_end"],
            args["args"],
            args["access_counter"],
            args["cache_miss_counter"],
            args["EXTEND_BACKEND"],
            args["model_context_length"],
            args["extend_context_length"],
        )

    def run_flash():
        return decode_block_sparse_attention(
            args["q"],
            args["k"],
            args["v"],
            args["seq_lens"],
            args["indices"],
            args["ks"],
            args["ks_count"],
            args["ks_start_end"],
            args["args"],
            args["access_counter"],
            args["cache_miss_counter"],
            args["EXTEND_BACKEND"],
            args["model_context_length"],
            args["extend_context_length"],
        )

    if use_cuda_graph:
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run_orig()
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g_orig = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_orig):
            run_orig()

        run_orig = lambda: g_orig.replay()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    start_mem = torch.cuda.max_memory_allocated(0)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    run_orig()
    end_time.record()
    torch.cuda.synchronize()
    peak_mem_orig = torch.cuda.max_memory_allocated(0) - start_mem
    elapsed_orig = start_time.elapsed_time(end_time)

    if use_cuda_graph:
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                run_flash()
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g_flash = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_flash):
            run_flash()

        run_flash = lambda: g_flash.replay()

    else:
        # Warmup
        for _ in range(2):
            run_flash()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)
    start_mem = torch.cuda.max_memory_allocated(0)
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    run_flash()
    end_time.record()
    torch.cuda.synchronize()
    peak_mem_flash = torch.cuda.max_memory_allocated(0) - start_mem
    elapsed_flash = start_time.elapsed_time(end_time)

    print(f"Time: orig={elapsed_orig:.2f}ms, flash={elapsed_flash:.2f}ms")
    print(
        f"Peak memory: orig={peak_mem_orig / 10 ** 6:.3f}MB, flash={peak_mem_flash / 10 ** 6:.3f}MB"
    )


# Test
def load_saved_tensors(file_path):
    args = torch.load(file_path, map_location="cuda:0")

    # FIXME: current implementation is incorrect across heads
    for i in range(args["indices"].shape[0]):
        args["indices"][i] = args["indices"][0]
        args["ks"][i] = args["ks"][0]
        args["ks_count"][i] = args["ks_count"][0]
        args["ks_start_end"][i] = args["ks_start_end"][0]

    return args
