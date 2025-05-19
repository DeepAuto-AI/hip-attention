"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor

# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = "cuda:0"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    K_block_ptr,
    V_block_ptr,  #
    mask_idx,
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    N_CTX: tl.constexpr,
    N_KV: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # range of values handled by this stage
    # lo, hi = 0, N_KV
    lo, hi = 0, tl.max(mask_idx) + 1

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k)

        mask = (mask_idx[:, None]) >= (start_n + offs_n[None, :])

        qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(
            V_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(v.dtype)

        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
    for BM in [64, 128]
    for BN in [32, 64]
    for s in ([1] if is_hip() else [3, 4, 7])
    for w in [4, 8]
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "N_KV", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    sm_scale,
    M,
    Out,  #
    MaskIdx,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    stride_mz,
    stride_mm,
    Z,
    H,
    N_CTX,  #
    N_KV,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_KV, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(HEAD_DIM, N_KV),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_CTX
    offs_n = tl.arange(0, BLOCK_N)

    mask_idx = tl.load(
        MaskIdx + off_z.to(tl.int64) * stride_mz + offs_m.to(tl.int64) * stride_mm,
        mask=mask_m,
    )
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(
        Q_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
    )

    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q,
        K_block_ptr,
        V_block_ptr,  #
        mask_idx,
        start_m,
        qk_scale,  #
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,  #
        offs_m,
        offs_n,
        N_CTX,
        N_KV,
        V.dtype.element_ty == tl.float8e5,  #
    )

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i, mask=mask_m)
    tl.store(
        O_block_ptr,
        acc.to(Out.type.element_ty),
        boundary_check=(0,),
    )


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, mask, sm_scale):

        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )

        # assert q.shape[1] in (1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 48, 64, 80, 96,)
        assert q.shape[1] <= 128
        grid = lambda args: (
            triton.cdiv(q.shape[2], args["BLOCK_M"]),
            q.shape[0] * q.shape[1],
            1,
        )

        assert len(mask.size()) == 2, "expecting mask to be 2D"
        _attn_fwd[grid](
            q,
            k,
            v,
            sm_scale,
            M,
            o,  #
            mask,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),  #
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),  #
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),  #
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),  #
            mask.stride(0),
            mask.stride(1),
            q.shape[0],
            q.shape[1],  #
            N_CTX=q.shape[2],  #
            N_KV=k.shape[2],
            HEAD_DIM=HEAD_DIM_K,  #
            **extra_kern_args,
        )

        return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("bwd not implemented for recompute kernel")


attention = _attention.apply


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
    tri_out = attention(q, k, v, mask, sm_scale).to(dtype)

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
    tri_out = attention(q, k, v, mask, sm_scale).to(dtype)

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
    tri_out = attention(q, k, v, mask, sm_scale).to(dtype)

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
    tri_out = attention(q, k, v, mask, sm_scale).to(dtype)

    diff = (ref_out - tri_out).abs()
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    print(f"[PASS] flash")


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func

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
        fn = lambda: attention(q, k, v, causal, sm_scale)
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
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    print(f"[PASS] causal block irregular")

    for i in range(128):
        print(f"{i}")
        ctx = torch.randint(128, 1024, size=(1,)).item()
        test_op(1, 32, ctx, 64, False)
        test_op_causal(1, 32, ctx, 64, False)
        test_op_causal_block_irregular(1, 32, ctx, 64, False)
        ctx = torch.randint(4096, 16384, size=(1,)).item()
        test_op_flash(1, 32, ctx, 64, False, dtype=torch.bfloat16)
    # bench_flash_attention.run(save_path=".", print_data=True)
