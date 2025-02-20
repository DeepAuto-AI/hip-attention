import unittest

import numpy as np
import torch
import triton

from hip_attn.v1_0.attention1_block_gpu_kernel.calc_score_return_prob import (
    CalcScoreAutoGradFn,
)


class TestScoreProb(unittest.TestCase):

    def test_CalcScoreAutoGradFn(self):
        test_CalcScoreAutoGradFn()

    def test_CalcScoreAutoGradFn_perf(self):
        test_CalcScoreAutoGradFn_perf()


def test_CalcScoreAutoGradFn():
    BSZ = 2
    QUERY_LEN = 32768
    KEY_LEN = 32768
    QUERY_DIM = 64
    BLOCK_SIZE_Q = 32
    BLOCK_SIZE_K = 2
    K = 512
    IS_CAUSAL = True
    KV_REPEAT_INTERLEAVE = 1
    compute_backward = True

    BLOCK_K = K // BLOCK_SIZE_K

    queries = torch.randn(BSZ, QUERY_LEN, QUERY_DIM).cuda().requires_grad_(True)
    keys = torch.randn(BSZ, KEY_LEN, QUERY_DIM).cuda().requires_grad_(True)
    attention_mask = None
    dout = torch.randn(BSZ, QUERY_LEN, K).cuda()

    # indices to key for each query block
    indices = torch.randint(
        0, KEY_LEN, (BSZ, triton.cdiv(QUERY_LEN, BLOCK_SIZE_Q), BLOCK_K)
    )
    indices.copy_(
        torch.load("exmaple_indices.pth", map_location="cpu")[
            : indices.shape[0], : indices.shape[1], : indices.shape[2]
        ]
    )
    indices = indices.clone().cuda()

    # number of key blocks to attend to for each query block
    ks = torch.randint(0, BLOCK_K, (BSZ, triton.cdiv(QUERY_LEN, BLOCK_SIZE_Q)))
    ks.copy_(
        torch.load("example_ks.pth", map_location="cpu")[: ks.shape[0], : ks.shape[1]]
    )
    ks = ks.clone().cuda()

    scores = CalcScoreAutoGradFn.apply(
        queries,
        keys,
        attention_mask,
        indices,
        ks,
        KV_REPEAT_INTERLEAVE,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
        IS_CAUSAL,
    )
    # scores: [BSZ, QUERY_LEN, K]
    if compute_backward:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.softmax(scores, dim=-1).backward(dout)
        torch.cuda.synchronize()
        print(
            "Triton Backward pass Peak memory allocated: ",
            torch.cuda.max_memory_allocated() / 1024 / 1024,
            "MB",
        )
        tri_dq, queries.grad = queries.grad.clone(), None
        tri_dk, keys.grad = keys.grad.clone(), None

    ref_scores = reference_impl(
        queries,
        keys,
        attention_mask,
        indices,
        ks,
        KV_REPEAT_INTERLEAVE,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
        IS_CAUSAL,
    )
    if compute_backward:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.softmax(ref_scores, dim=-1).backward(dout)
        torch.cuda.synchronize()
        print(
            "Ref Backward pass Peak memory allocated: ",
            torch.cuda.max_memory_allocated() / 1024 / 1024,
            "MB",
        )
        ref_dq, queries.grad = queries.grad.clone(), None
        ref_dk, keys.grad = keys.grad.clone(), None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ref2_dq = reference_dQ_impl(
        queries,
        keys,
        scores,
        indices,
        ks,
        KV_REPEAT_INTERLEAVE,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
        IS_CAUSAL,
    )
    torch.cuda.synchronize()
    print(
        "Ref2 Backward pass Peak memory allocated: ",
        torch.cuda.max_memory_allocated() / 1024 / 1024,
        "MB",
    )

    compare("scores", ref_scores[ref_scores > -1e10], scores[ref_scores > -1e10])
    if compute_backward:
        compare("dQ", ref_dq, tri_dq)
        compare("dQ2", ref2_dq, tri_dq)
        compare("dK", ref_dk, tri_dk)


def test_CalcScoreAutoGradFn_perf(warmup=25, rep=100):
    BSZ = 2
    QUERY_LEN = 32768
    KEY_LEN = 32768
    QUERY_DIM = 64
    BLOCK_SIZE_Q = 32
    BLOCK_SIZE_K = 2
    K = 512
    IS_CAUSAL = True
    KV_REPEAT_INTERLEAVE = 1
    compute_backward = True

    BLOCK_K = K // BLOCK_SIZE_K

    queries = torch.randn(BSZ, QUERY_LEN, QUERY_DIM).cuda().requires_grad_(True)
    keys = torch.randn(BSZ, KEY_LEN, QUERY_DIM).cuda().requires_grad_(True)
    attention_mask = None
    dout = torch.randn(BSZ, QUERY_LEN, K).cuda()

    # indices to key for each query block
    indices = torch.randint(
        0, KEY_LEN, (BSZ, triton.cdiv(QUERY_LEN, BLOCK_SIZE_Q), BLOCK_K)
    )
    indices.copy_(
        torch.load("exmaple_indices.pth", map_location="cpu")[
            : indices.shape[0], : indices.shape[1], : indices.shape[2]
        ]
    )
    indices = indices.clone().cuda()

    # number of key blocks to attend to for each query block
    ks = torch.randint(0, BLOCK_K, (BSZ, triton.cdiv(QUERY_LEN, BLOCK_SIZE_Q)))
    ks.copy_(
        torch.load("example_ks.pth", map_location="cpu")[: ks.shape[0], : ks.shape[1]]
    )
    ks = ks.clone().cuda()

    scores = CalcScoreAutoGradFn.apply(
        queries,
        keys,
        attention_mask,
        indices,
        ks,
        KV_REPEAT_INTERLEAVE,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
        IS_CAUSAL,
    )
    # scores: [BSZ, QUERY_LEN, K]
    if compute_backward:
        fn = lambda: torch.softmax(scores, dim=-1).backward(dout, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        print("CalcScoreAutoGradFn backward: ", ms)
        tri_dq, queries.grad = queries.grad.clone(), None
        tri_dk, keys.grad = keys.grad.clone(), None

    ref_scores = reference_impl(
        queries,
        keys,
        attention_mask,
        indices,
        ks,
        KV_REPEAT_INTERLEAVE,
        BLOCK_SIZE_Q,
        BLOCK_SIZE_K,
        IS_CAUSAL,
    )
    if compute_backward:
        fn = lambda: torch.softmax(ref_scores, dim=-1).backward(dout, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        print("reference_impl backward: ", ms)
        ref_dq, queries.grad = queries.grad.clone(), None
        ref_dk, keys.grad = keys.grad.clone(), None


def reference_dQ_impl(
    queries,
    keys,
    d_scores,
    indices,
    ks,
    KV_REPEAT_INTERLEAVE,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_K,
    IS_CAUSAL,
):
    assert KV_REPEAT_INTERLEAVE == 1

    BSZ, QUERY_LEN, QUERY_DIM = queries.shape
    _, KEY_LEN, _ = keys.shape
    _, _, BLOCK_K = indices.shape
    K = BLOCK_K * BLOCK_SIZE_K
    QUERY_BLOCKS = triton.cdiv(QUERY_LEN, BLOCK_SIZE_Q)
    KEY_BLOCKS = triton.cdiv(KEY_LEN, BLOCK_SIZE_K)

    keys = keys.reshape(BSZ, KEY_BLOCKS, BLOCK_SIZE_K, QUERY_DIM)

    # indices: (BSZ, QUERY_BLOCKS, BLOCK_K)
    block_indices = indices // BLOCK_SIZE_K
    real_indices = (
        (indices.unsqueeze(-1) + torch.arange(BLOCK_SIZE_K, device=indices.device))
        .unsqueeze(2)
        .expand(-1, -1, BLOCK_SIZE_Q, -1, -1)
        .reshape(BSZ, QUERY_LEN, K)
    )

    keys_gathered = keys.gather(
        1,
        block_indices.reshape(BSZ, QUERY_BLOCKS * BLOCK_K, 1, 1).expand(
            -1, -1, BLOCK_SIZE_K, QUERY_DIM
        ),
    ).reshape(BSZ, QUERY_BLOCKS, BLOCK_K, BLOCK_SIZE_K, QUERY_DIM)

    # mask off >= num_k_blocks indices
    mask = (
        torch.arange(BLOCK_K, device=indices.device)[None, None, :] >= ks[:, :, None]
    )  # (BSZ, QUERY_BLOCKS, BLOCK_K)
    keys_gathered = keys_gathered.masked_fill(mask[:, :, :, None, None], 0)

    if IS_CAUSAL:
        # mask off future tokens
        mask = (
            real_indices
            > torch.arange(QUERY_LEN, device=real_indices.device)[None, :, None]
        )
        d_scores = d_scores.masked_fill(mask, 0)

    d_scores = d_scores.reshape(BSZ, QUERY_BLOCKS, BLOCK_SIZE_Q, BLOCK_K, BLOCK_SIZE_K)
    d_queries = torch.einsum("bQqKk,bQKkd->bQqd", d_scores, keys_gathered).reshape(
        BSZ, QUERY_LEN, QUERY_DIM
    )

    return d_queries


def reference_impl(
    queries,
    keys,
    attention_mask,
    indices,
    ks,
    KV_REPEAT_INTERLEAVE,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_K,
    IS_CAUSAL,
):
    assert attention_mask is None
    assert KV_REPEAT_INTERLEAVE == 1

    BSZ, QUERY_LEN, QUERY_DIM = queries.shape
    _, KEY_LEN, _ = keys.shape
    _, _, BLOCK_K = indices.shape
    K = BLOCK_K * BLOCK_SIZE_K
    QUERY_BLOCKS = triton.cdiv(QUERY_LEN, BLOCK_SIZE_Q)
    KEY_BLOCKS = triton.cdiv(KEY_LEN, BLOCK_SIZE_K)

    queries = queries.reshape(BSZ, QUERY_BLOCKS, BLOCK_SIZE_Q, QUERY_DIM)
    keys = keys.reshape(BSZ, KEY_BLOCKS, BLOCK_SIZE_K, QUERY_DIM)

    # indices: (BSZ, QUERY_BLOCKS, BLOCK_K)
    block_indices = indices // BLOCK_SIZE_K
    real_indices = (
        (indices.unsqueeze(-1) + torch.arange(BLOCK_SIZE_K, device=indices.device))
        .unsqueeze(2)
        .expand(-1, -1, BLOCK_SIZE_Q, -1, -1)
        .reshape(BSZ, QUERY_LEN, K)
    )

    keys_gathered = keys.gather(
        1,
        block_indices.reshape(BSZ, QUERY_BLOCKS * BLOCK_K, 1, 1).expand(
            -1, -1, BLOCK_SIZE_K, QUERY_DIM
        ),
    ).reshape(BSZ, QUERY_BLOCKS, BLOCK_K, BLOCK_SIZE_K, QUERY_DIM)

    qk = torch.einsum("bQqd,bQKkd->bQqKk", queries, keys_gathered)
    # qk: (BSZ, QUERY_BLOCKS, BLOCK_SIZE_Q, BLOCK_K, BLOCK_SIZE_K)

    # mask off >= num_k_blocks indices
    mask = (
        torch.arange(BLOCK_K, device=indices.device)[None, None, :] >= ks[:, :, None]
    )  # (BSZ, QUERY_BLOCKS, BLOCK_K)
    qk = qk.masked_fill(mask[:, :, None, :, None], float("-inf"))

    scores = qk.reshape(BSZ, QUERY_LEN, K)
    if IS_CAUSAL:
        # mask off future tokens
        mask = (
            real_indices
            > torch.arange(QUERY_LEN, device=real_indices.device)[None, :, None]
        )
        scores = scores.masked_fill(mask, float("-inf"))

    return scores


@torch.no_grad()
def compare(name, ref, tri):
    print(name)
    assert ref.shape == tri.shape
    print("Mean Absoulte Error: ", torch.abs(ref - tri).mean().item())
    print("Max Absoulte Error: ", torch.abs(ref - tri).max().item())
    abserr = np.quantile(torch.abs(ref - tri).flatten().float().cpu().numpy(), 0.99)
    print("99% Quantile Absoulte Error: ", abserr)
    print(
        "Mean Relative Error: ",
        (torch.abs(ref - tri) / (torch.abs(ref) + 1e-6)).mean().item(),
    )
    print(
        "Max Relative Error: ",
        (torch.abs(ref - tri) / (torch.abs(ref) + 1e-6)).max().item(),
    )
    relerr = np.quantile(
        (torch.abs(ref - tri) / (torch.abs(ref) + 1e-6))
        .flatten()
        .float()
        .cpu()
        .numpy(),
        0.99,
    )
    print("99% Quantile Relative Error: ", relerr)
    assert abserr < 0.05 and relerr < 0.1
