import json
import math
import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import triton
from hip_research.utils.load_checkouts import load_checkouts
from hip_research.utils.seed import seed
from torch import Tensor

import hip_attn.v1_0.attention1_block_gpu
from hip_attn.utils.benchmarking import get_bench
from hip_attn.v1_0.attention1_block_gpu import flash_attention, hip_attention
from hip_attn.v1_1.attention2_draft_prefetch import hip_attention as hip_attention_11


class TestAttention1BlockGpu(unittest.TestCase):

    def test_main_debug(self):
        main_debug()

    def test_main_debug_mask(self):
        main_debug_mask()

    def test_main_debug_max_ks(self):
        main_debug_max_ks()

    def test_main_latency_benchmark(self):
        main_latency_benchmark()


def torch_attention(q: Tensor, k: Tensor, v: Tensor):
    scores = torch.bmm(q, k.transpose(-1, -2))
    probs = torch.softmax(scores, dim=-1)
    context = torch.bmm(probs, v)
    return context, probs


def landmark_attention(q: Tensor, k: Tensor, v: Tensor):
    """
    https://arxiv.org/pdf/2305.16300.pdf
    this paper claimed, they are faster than original attetnion... but seems not?
    """
    from hip_research.models.landmark_attention import fused_landmark_attention

    seqlen_k = k.shape[1]
    block_size = 64
    is_mem = torch.arange(0, seqlen_k, device=q.device) % block_size == (block_size - 1)
    return fused_landmark_attention(q, k, v, is_mem, block_size=block_size)


@torch.inference_mode(True)
def streaming_attention(
    q: Tensor, k: Tensor, v: Tensor, cos: Tensor, sin: Tensor, window_size: int
):
    from hip_research.models.sink_attention import sink_attention

    return sink_attention(q, k, v, cos, sin, window_size=window_size)


def main_latency_benchmark():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--dups", type=int, default=2)
    parser.add_argument("--query_size", type=int, default=1)
    parser.add_argument("--method", type=str, default="hip1.1")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--block_size_q", type=int, default=32)
    parser.add_argument("--block_stride_q", type=int, default=None)
    parser.add_argument("--block_size_k", type=int, default=1)
    parser.add_argument("--block_stride_k", type=int, default=None)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--scale_up", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=-1)
    parser.add_argument("--head_size", type=int, default=-1)
    parser.add_argument("--refresh_interval", type=int, default=8)
    parser.add_argument("--not_causal", action="store_true")
    parser.add_argument("--head_groups", type=int, default=1)
    args = parser.parse_args()

    if args.query_size > 1:
        args.refresh_interval = -1

    hip_attn.v1_0.attention1_block_gpu.DEBUG = args.debug
    TRACE = args.trace
    BSIZE = args.batch_size
    DUPS = args.dups
    QUERY_SIZE = args.query_size
    METHOD = args.method
    n_samples = args.samples
    is_causal = not args.not_causal

    if hip_attn.v1_0.attention1_block_gpu.DEBUG:
        seed()

    get_bench().disabled = not TRACE
    get_bench().synchronize = True

    CHUNK_LEN = 1024
    q, k, v, out, cos, sin = load_checkouts(idx=0, window=40, seq_len=CHUNK_LEN)
    HID = q.shape[-1]

    q = q.cpu()
    k = k.cpu()
    v = v.cpu()

    if args.head_size > 0 and args.head_size != q.shape[0]:
        head_reps = int(math.ceil(args.head_size / q.shape[0]))
        q = q.repeat(head_reps, 1, 1)[: args.head_size, :, :].contiguous()
        k = k.repeat(head_reps, 1, 1)[: args.head_size, :, :].contiguous()
        v = v.repeat(head_reps, 1, 1)[: args.head_size, :, :].contiguous()

    q = q.repeat(BSIZE, max(1, triton.cdiv(QUERY_SIZE, 1024)), 1)[
        :, :QUERY_SIZE, :
    ].contiguous()
    k = k.repeat(BSIZE, DUPS, 1)
    v = v.repeat(BSIZE, DUPS, 1)
    started = False

    if args.hidden_size > 0 and args.hidden_size != HID:
        hid_reps = int(math.ceil(args.hidden_size / HID))
        q = q.repeat(1, 1, hid_reps)[:, :, : args.hidden_size].contiguous()
        k = k.repeat(1, 1, hid_reps)[:, :, : args.hidden_size].contiguous()
        v = v.repeat(1, 1, hid_reps)[:, :, : args.hidden_size].contiguous()
        HID = args.hidden_size

    head_size = q.shape[0]
    cos = sin = torch.randn((k.shape[1], k.shape[2]), dtype=k.dtype, device=k.device)

    if METHOD in ["flash", "fa2", "hip1.1"]:
        q = q.view(BSIZE, -1, QUERY_SIZE, HID).permute(0, 2, 1, 3).contiguous()
        k = (
            k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID)[:, :: args.head_groups, :, :]
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        v = (
            v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID)[:, :: args.head_groups, :, :]
            .permute(0, 2, 1, 3)
            .contiguous()
        )
    elif METHOD in ["landmark"]:
        q = q.view(BSIZE, -1, QUERY_SIZE, HID).contiguous()
        k = k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).contiguous()
        v = v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID).contiguous()
    elif METHOD in ["streaming", "hyper"]:
        k = (
            k.view(BSIZE, -1, CHUNK_LEN * DUPS, HID)
            .repeat(q.shape[0] // k.shape[0], 1, 1, 1)
            .view(q.shape[0], -1, q.shape[2])
        )
        v = (
            v.view(BSIZE, -1, CHUNK_LEN * DUPS, HID)
            .repeat(q.shape[0] // v.shape[0], 1, 1, 1)
            .view(q.shape[0], -1, q.shape[2])
        )

    q = q.cuda()
    k = k.cuda()
    v = v.cuda()
    cos = cos.cuda()
    sin = sin.cuda()
    if METHOD in ["hip1.1"]:
        # q_quant = q.to(torch.float8_e5m2).view(torch.uint8)
        # k_quant = k.to(torch.float8_e5m2).view(torch.uint8)
        q_quant = q
        k_quant = k

    hip_attention_mask = torch.full(
        (q.shape[0], k.shape[1]), True, dtype=torch.bool, device=q.device
    )

    from hip_research.models.hyper_attention.hyper_attn import HyperAttention

    hyper_attention = (
        HyperAttention(
            input_dim=q.shape[-1],
            lsh_num_projs=7,  # not very meaningful after 7
            block_size=64,  # smaller better
            sample_size=1024,  # larger better
            min_seq_len=32,  # this factor is kind of random. usually smaller better
            cuda=True,
        )
        .to("cuda")
        .to(q.dtype)
    )

    def sample(state=None):
        with torch.no_grad():
            if METHOD in ["torch", "none", "default"]:
                torch_attention(q, k, v)
            elif METHOD in ["flash", "fa2"]:
                flash_attention(q, k, v, is_causal=is_causal)
            elif METHOD == "landmark":
                landmark_attention(q, k, v)
            elif METHOD == "streaming":
                streaming_attention(q, k, v, cos, sin, window_size=args.k)
            elif METHOD == "hyper":
                _q = q.view(-1, head_size, q.shape[1], q.shape[-1])
                _k = k.view(-1, head_size, k.shape[1], k.shape[-1])
                _v = v.view(-1, head_size, v.shape[1], v.shape[-1])
                # print(_q.shape, _k.shape, _v.shape)
                hyper_attention(_q, _k, _v, causal=True, scale=1)
            elif METHOD == "hip1.1":
                assert is_causal
                if state is None:
                    _, mask = hip_attention_11(
                        q,
                        k,
                        v,
                        # attention_mask=hip_attention_mask,
                        mask_k=args.k,
                        block_size_q=args.block_size_q,
                        block_stride_q=(
                            args.block_stride_q
                            if args.block_stride_q is not None
                            else 2
                        ),
                        block_size_k=args.block_size_k,
                        block_stride_k=(
                            args.block_stride_k
                            if args.block_stride_k is not None
                            else max(2, args.block_size_k // 2)
                        ),
                        q_quant=q_quant,
                        k_quant=k_quant,
                        sample_method="center",
                        mask_only=False,
                    )
                else:
                    _, mask = hip_attention_11(
                        q,
                        k,
                        v,
                        # attention_mask=hip_attention_mask,
                        mask_k=args.k,
                        block_size_q=args.block_size_q,
                        block_size_k=args.block_size_k,
                        block_stride_k=max(2, args.block_size_k // 2),
                        q_quant=q_quant,
                        k_quant=k_quant,
                        sample_method="center",
                        previous_metadata=state,
                    )
                if mask is None:
                    return None
                return mask
            elif METHOD == "hip1.0":
                if state is None:
                    _, mask = hip_attention(
                        q,
                        k,
                        v,
                        # attention_mask=hip_attention_mask,
                        mask_k=args.k,
                        block_size_q=args.block_size_q,
                        block_size_k=args.block_size_k,
                        scale_up=args.scale_up,
                        is_causal=is_causal,
                    )
                else:
                    indices, ks = state

                    _, mask = hip_attention(
                        q,
                        k,
                        v,
                        # attention_mask=hip_attention_mask,
                        mask_k=args.k,
                        block_size_q=args.block_size_q,
                        block_size_k=args.block_size_k,
                        scale_up=args.scale_up,
                        is_causal=is_causal,
                        using_precomputed_mask=True,
                        precomputed_indices=indices,
                        precomputed_ks=ks,
                    )
                if mask is None:
                    return None, None
                return mask[0], mask[1]
            else:
                raise Exception()

    s = torch.cuda.Stream()
    graph = None
    graph_stateful = None
    samples = []
    for i in tqdm.tqdm(range(n_samples), dynamic_ncols=True):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        if i < 5:
            s.wait_stream(torch.cuda.current_stream())
            state = sample()
            if args.refresh_interval > 0:
                sample(state)
            torch.cuda.current_stream().wait_stream(s)
        elif args.trace:
            sample()
        elif graph is None:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                state = sample()
            if args.refresh_interval > 0:
                graph_stateful = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_stateful):
                    sample(state)
        else:
            if args.refresh_interval > 0:
                if (i % args.refresh_interval) == 0:
                    graph.replay()
                else:
                    graph_stateful.replay()
            else:
                graph.replay()

        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / args.batch_size

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
        f"({METHOD}) {np.mean(samples):.8f} ms +- {np.std(samples):.4f} ms (q: {tuple(q.shape)}, k: {tuple(k.shape)}, v: {tuple(v.shape)})"
    )

    os.makedirs("./cache/attention1_block_gpu/", exist_ok=True)
    with open("./cache/attention1_block_gpu/result.json", "w") as f:
        json.dump(
            {
                "method": METHOD,
                "mean": np.mean(samples),
                "std": np.std(samples),
                "query_length": q.shape[-2],
                "keyvalue_length": k.shape[-2],
            },
            f,
            indent=2,
        )


def main_debug():
    hip_attn.v1_0.attention1_block_gpu.DEBUG = True

    block = 1024
    # block = 256
    q, k, v, out = load_checkouts(
        dtype=torch.float32, seq_len=block * 4, idx=6, window=1
    )

    reps = 1
    q = q.repeat(1, reps, 1)
    k = k.repeat(1, reps, 1)
    v = v.repeat(1, reps, 1)
    out = out.repeat(1, reps, 1)

    q = q[:, -k.shape[1] // 4 :, :]
    out = out[:, -k.shape[1] // 4 :, :]

    print("q", q.shape)
    print("k", k.shape)
    print("v", v.shape)
    print("out", out.shape)

    mask_k = 512
    block_size_q = 32
    block_size_k = 2

    maximum_ks = torch.where(
        torch.rand((q.shape[0], q.shape[1] // block_size_q), device=q.device) < 0.5,
        mask_k,
        mask_k // 4,
    ).to(torch.int32)
    # maximum_ks = None

    context, (atten_indices, atten_ks, atten_scores) = hip_attention(
        q,
        k,
        v,
        mask_k=mask_k,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        dense_queries_exp=0,
        is_flash=True,
        using_sliding_window=True,
        maximum_ks=maximum_ks,
        maximum_ks_config=[128, 512],
        force_return_scores=True,
    )

    print(maximum_ks[0])
    print(atten_indices[0])
    print(atten_ks[0])

    atten_probs = atten_scores.softmax(-1)
    plt.figure(figsize=(4, 5.5))
    plt.imshow(atten_probs[0].cpu().numpy() ** 0.2)
    plt.axvline(32, color="red")
    plt.axvline(32 + 512, color="red")
    for idx_tdst in range(0, atten_probs.shape[1], block_size_q):
        plt.axhline(idx_tdst, color="magenta", linestyle=":")
    plt.colorbar()
    path = "saves/models/hip_attention/atten_probs.png"
    plt.savefig(path, dpi=200)
    print("saved", path)
    print(atten_probs.shape)

    print(atten_ks, atten_ks.min(), atten_ks.max())

    stderr = (out - context).abs().mean().item()
    stdcontext = torch.std_mean(out)[0].item()

    root = "cache/attention1_block_gpu"
    os.makedirs(root, exist_ok=True)
    torch.save(
        {
            "indices": atten_indices,
            "ks": atten_ks,
        },
        os.path.join(root, "checkout_mask.pth"),
    )

    print(
        f"err = {stderr:.6f} ({stderr / stdcontext:.4f} sigma), out_std = {stdcontext:.6f}"
    )


def main_debug_mask():
    hip_attn.v1_0.attention1_block_gpu.DEBUG = True

    seed()
    q, k, v, out = load_checkouts(
        dtype=torch.float16, seq_len=1024 * 2, idx=24, window=1
    )

    q = q[:, 512:, :]
    out = out[:, 512:, :]

    N, TSRC, HID = k.shape
    mask = torch.full((N, TSRC), 1, dtype=torch.float32, device=k.device)
    for i in range(N):
        mask[i, :1024] = 0

    context, (atten_indices, atten_ks, atten_probs) = hip_attention(
        q,
        k,
        v,
        attention_mask=mask,
    )

    stderr = (out - context).abs().mean().item()
    stdcontext = torch.std_mean(out)[0].item()

    print(
        f"err = {stderr:.6f} ({stderr / stdcontext:.4f} sigma), out_std = {stdcontext:.6f}"
    )


def main_debug_max_ks():
    import nvtx

    hip_attn.v1_0.attention1_block_gpu.DEBUG = False

    block = 1024
    # block = 256
    q, k, v, out = load_checkouts(
        dtype=torch.float32, seq_len=block * 32, idx=6, window=1
    )

    reps = 1
    q = q.repeat(1, reps, 1)
    k = k.repeat(1, reps, 1)
    v = v.repeat(1, reps, 1)
    out = out.repeat(1, reps, 1)

    # q = q[:, -k.shape[1]//4:, :]
    # out = out[:, -k.shape[1]//4:, :]

    print("q", q.shape)
    print("k", k.shape)
    print("v", v.shape)
    print("out", out.shape)

    low_k = 128
    mask_k = 2048
    # low_k = 512
    # mask_k = 512
    block_size_q = 32
    block_size_k = 2

    def sample(ratio: float, n: int = 100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        maximum_ks = torch.where(
            torch.rand((q.shape[0], q.shape[1] // block_size_q), device=q.device)
            < ratio,
            mask_k,
            low_k,
        ).to(torch.int32)

        def fn():
            hip_attention(
                q,
                k,
                v,
                mask_k=mask_k,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                dense_queries_exp=0,
                is_flash=True,
                using_sliding_window=True,
                maximum_ks=maximum_ks,
                maximum_ks_config=[low_k, mask_k],
                force_return_scores=False,
            )

        start.record()
        graph = None
        for i in range(n + 10):
            if graph is None:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                for i in range(3):
                    fn()
                torch.cuda.current_stream().wait_stream(s)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn()

            start.record()
            graph.replay()
            end.record()

            if i > 10:
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
                samples.append(elapsed)

        return sum(samples) / len(samples)

    for ratio in range(0, 10):
        ratio = ratio / 20
        samples = []
        with nvtx.annotate(f"ratio={ratio}"):
            for i in range(10):
                t = sample(ratio)
                if i > 3:
                    samples.append(t)

        if len(samples) > 0:
            latency = sum(samples) / len(samples)
            print(ratio, ratio * mask_k + (1 - ratio) * low_k, latency)
