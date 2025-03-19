import os
import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch
from hip_research.utils.load_checkouts import load_checkouts
from torch import Tensor

from hip_attn.v1_0.attention1_block_gpu import (
    hip_attention,
    hip_attention_mask,
    sparse_attention,
)


class TestAttention1BlockGPUFwd(unittest.TestCase):

    def test_main(self):
        print("=" * 80)
        main()
        print("=" * 80)


def to_dense(
    indices: np.ndarray,
    ks: np.ndarray,
    value: np.ndarray,
    N,
    T_DST,
    T_SRC,
    BLOCK_SIZE_Q,
    BLOCK_SIZE_K,
):
    # print(indices.shape, ks.shape, value.shape, T_DST, T_SRC)
    out = torch.zeros((N, T_DST, T_SRC), device=indices.device, dtype=value.dtype)
    for idx_n in range(N):
        for idx_bdst in range(indices.shape[1]):
            for idx_k in range(indices.shape[2]):
                if idx_k < ks[idx_n, idx_bdst]:
                    idx_tsrc = indices[idx_n, idx_bdst, idx_k]
                    out[
                        idx_n,
                        idx_bdst * BLOCK_SIZE_Q : (idx_bdst + 1) * BLOCK_SIZE_Q,
                        idx_tsrc : idx_tsrc + BLOCK_SIZE_K,
                    ] = value[
                        idx_n,
                        idx_bdst * BLOCK_SIZE_Q : (idx_bdst + 1) * BLOCK_SIZE_Q,
                        idx_k * BLOCK_SIZE_K : (idx_k + 1) * BLOCK_SIZE_K,
                    ]
    return out


def imsave(im: Tensor, name: str, gamma: float = 0.2, idx_batch: int = -1):
    im = im[idx_batch].cpu().detach().numpy() ** gamma
    plt.clf()
    plt.title(name)
    plt.imshow(im)
    plt.colorbar()
    os.makedirs("./saves/models/test_hip_block_fwd", exist_ok=True)
    path = f"./saves/models/test_hip_block_fwd/{name}.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.clf()
    print("saved", path)


def main():
    q, k, v, out = load_checkouts(idx=4, window=4, seq_len=4096, dtype=torch.float16)

    q = q[:, 2048:, :].contiguous()

    q_backup = q.clone()
    k_backup = k.clone()
    v_backup = v.clone()

    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape
    BLOCKSIZE_Q = 16
    BLOCKSIZE_K = 1
    mask_k = 512
    scale_up = 2
    w_start = mask_k * scale_up
    n_patches = mask_k / scale_up

    indices, ks, probs, scores = hip_attention_mask(
        q,
        k,
        torch.ones((N, TSRC), dtype=torch.bool, device=q.device),
        w_start,
        n_patches,
        mask_k,
        scale_up,
        block_size_q=BLOCKSIZE_Q,
        block_size_k=BLOCKSIZE_K,
    )

    print(q.shape, indices.shape, ks.shape, probs.shape, scores.shape)
    print(indices.min(), indices.max())

    probs_dense = to_dense(
        indices.cpu(),
        ks.cpu(),
        probs.cpu(),
        N,
        TDST,
        TSRC,
        BLOCKSIZE_Q,
        BLOCKSIZE_K,
    ).to(indices.device)

    scores_dense = to_dense(
        indices.cpu(),
        ks.cpu(),
        scores.cpu(),
        N,
        TDST,
        TSRC,
        BLOCKSIZE_Q,
        BLOCKSIZE_K,
    ).to(indices.device)

    mask_dense = probs_dense <= 1e-7
    mask_dense = mask_dense.to(probs.dtype) * torch.finfo(probs.dtype).min

    scores_truth = torch.bmm(q, k.transpose(-1, -2))
    print(scores_truth.dtype, mask_dense.dtype)
    probs_truth = (scores_truth + mask_dense).softmax(dim=-1)

    probs_error_map = (probs_truth - probs_dense).abs()
    probs_error = probs_error_map[:].max()
    print((probs_error_map == probs_error).nonzero())

    scores_dense = scores_dense * (probs_dense > 1e-12)
    scores_truth = scores_truth * (probs_dense > 1e-12)
    scores_error_map = (scores_truth - scores_dense).abs()
    scores_error = scores_error_map.max()

    imsave(probs, "probs")
    imsave(probs_truth, "probs_truth")
    imsave(probs_dense, "probs_dense")
    imsave(probs_error_map, "probs_error_map")
    imsave(scores_truth.abs(), "scores_truth", gamma=1.0)
    imsave(scores_dense.abs(), "scores_dense", gamma=1.0)
    imsave(scores_error_map, "scores_error_map", gamma=1.0)
    print("scores_error", scores_error)
    print("probs_error", probs_error)

    context_dense = sparse_attention(
        v.contiguous(),
        indices,
        ks,
        probs,
        BLOCKSIZE_Q,
        BLOCKSIZE_K,
    )
    context_truth = torch.bmm(probs_dense, v)

    context_error_map = (context_dense - context_truth).abs()
    # print(context_error_map)

    context_error = context_error_map.max()
    context_error_loc = (context_error_map == context_error).nonzero()

    imsave(context_error_map, "context_error_map", gamma=1.0, idx_batch=2)
    print(
        "std_mean context_dense, context_truth",
        torch.std_mean(context_dense),
        torch.std_mean(context_truth),
    )
    print("cte", context_error, context_error_loc)

    # print(context_error_map[1, 2, :])
    # print(context_dense[1, 2, :])
    # print(context_truth[1, 2, :])

    for i in range(3):
        context_hip, (indices_hip, ks_hip, probs_hip) = hip_attention(
            q,
            k,
            v,
            torch.ones((N, TSRC), dtype=torch.bool, device=q.device),
            w_start,
            n_patches,
            mask_k,
            scale_up,
            BLOCKSIZE_Q,
            BLOCKSIZE_K,
        )
    context_hip_error_map = (context_hip - context_dense).abs()
    context_hip_error = context_hip_error_map.max()
    print((q - q_backup).abs().sum())
    print((k - k_backup).abs().sum())
    print((v - v_backup).abs().sum())
    print("error between hip", context_hip_error)

    probs_dense_hip = to_dense(
        indices_hip.cpu(),
        ks_hip.cpu(),
        probs_hip.cpu(),
        N,
        TDST,
        TSRC,
        BLOCKSIZE_Q,
        BLOCKSIZE_K,
    ).to(indices.device)
    imsave(probs_dense_hip, "probs_dense_hip")
