import os
import warnings
from typing import Callable, Tuple, Union

import numpy as np
import pytest

# import pytest
import torch
import triton

from hip_attn.v1_2.query_sparse_attention import query_sparse_attention

# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = "cuda:0"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.bfloat16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )
    k = (
        torch.empty((Z, H, N_CTX * 2, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )
    v = (
        torch.empty((Z, H, N_CTX * 2, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )

    mask = torch.randint(0, N_CTX * 2, size=(1, N_CTX)).cuda()
    # mask = torch.full((1, N_CTX), N_CTX * 2 - 1).cuda()
    # mask = torch.full((1, N_CTX), 35).cuda()

    sm_scale = 1 / np.sqrt(HEAD_DIM)

    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    M = torch.zeros_like(p)
    for i in range(N_CTX):
        M[:, :, i, mask[0, i] + 1 :] = torch.finfo(M.dtype).min

    p = p + M
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)

    # triton implementation
    tri_out = query_sparse_attention(q, k, v, mask, sm_scale).to(dtype)

    diff = (ref_out - tri_out).abs()
    # print(f"{diff.amax()=} {diff.mean()=}")
    # print(f"{ref_out - tri_out=}")
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    print(f"[PASS] rectangle")


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op_causal_block_irregular(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.bfloat16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )
    k = (
        torch.empty((Z, H, N_CTX * 4, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )
    v = (
        torch.empty((Z, H, N_CTX * 4, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )

    mask = torch.randperm(N_CTX).cuda().unsqueeze(0)

    sm_scale = 1 / np.sqrt(HEAD_DIM)

    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    M = torch.zeros_like(p)
    for i in range(N_CTX):
        M[:, :, i, mask[0, i] + 1 :] = torch.finfo(M.dtype).min

    p = p + M
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)

    # triton implementation
    tri_out = query_sparse_attention(q, k, v, mask, sm_scale).to(dtype)

    diff = (ref_out - tri_out).abs()
    # print(f"{diff.amax()=} {diff.mean()=}")
    # print(f"{ref_out - tri_out=}")
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    print(f"[PASS] causal block irregular")


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op_causal(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.bfloat16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )
    k = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )
    v = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )

    mask = torch.arange(0, N_CTX).cuda().unsqueeze(0)

    sm_scale = 1 / np.sqrt(HEAD_DIM)

    # reference implementation
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    M = torch.zeros_like(p)
    for i in range(N_CTX):
        M[:, :, i, mask[0, i] + 1 :] = torch.finfo(M.dtype).min

    p = p + M
    p = torch.softmax(p.float(), dim=-1).to(dtype)
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)

    # triton implementation
    tri_out = query_sparse_attention(q, k, v, mask, sm_scale).to(dtype)

    diff = (ref_out - tri_out).abs()
    # print(f"{diff.amax()=} {diff.mean()=}")
    # print(f"{ref_out - tri_out=}")
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    print(f"[PASS] causal eager")


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])
def test_op_flash(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.bfloat16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )
    k = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )
    v = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .cuda()
    )

    mask = torch.arange(0, N_CTX).cuda().unsqueeze(0)

    sm_scale = 1 / np.sqrt(HEAD_DIM)

    # reference implementation
    from flash_attn import flash_attn_func

    ref_out = flash_attn_func(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True
    )
    ref_out = ref_out.transpose(1, 2)

    # triton implementation
    tri_out = query_sparse_attention(q, k, v, mask, sm_scale).to(dtype)

    diff = (ref_out - tri_out).abs()
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    print(f"[PASS] flash")


try:
    from flash_attn.flash_attn_interface import (
        flash_attn_qkvpacked_func as flash_attn_func,
    )

    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
BATCH, N_HEADS, HEAD_DIM = 4, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [True, False]:
        if mode == "bwd" and not causal:
            continue
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(10, 15)],
                line_arg="provider",
                line_vals=["triton-fp16"]
                + (["triton-fp8"] if TORCH_HAS_FP8 else [])
                + (["flash"] if HAS_FLASH else []),
                line_names=["Triton [FP16]"]
                + (["Triton [FP8]"] if TORCH_HAS_FP8 else [])
                + (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="TFLOPS",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            )
        )


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device=DEVICE
):
    assert mode in ["fwd"]
    dtype = torch.bfloat16
    if "triton" in provider:
        q = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )
        k = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )
        v = torch.randn(
            (BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True
        )
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2).contiguous()
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: query_sparse_attention(q, k, v, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    if provider == "flash":
        qkv = torch.randn(
            (BATCH, N_CTX, 3, H, HEAD_DIM),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    # print(f"{ref_out - tri_out=}")
    # compare
    # assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    # print(f"[PASS] causal block irregular")

    for i in range(128):
        print(f"{i}")
        ctx = torch.randint(128, 1024, size=(1,)).item()
        test_op(1, 32, ctx, 64, False)
        test_op_causal(1, 32, ctx, 64, False)
        test_op_causal_block_irregular(1, 32, ctx, 64, False)
        ctx = torch.randint(4096, 16384, size=(1,)).item()
        test_op_flash(1, 32, ctx, 64, False, dtype=torch.bfloat16)
    # bench_flash_attention.run(save_path=".", print_data=True)
