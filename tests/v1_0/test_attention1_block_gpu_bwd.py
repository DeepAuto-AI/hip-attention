import unittest
import warnings

import torch
from hip_research.utils.load_checkouts import load_checkouts
from torch import nn

from hip_attn.v1_0.attention1_block_gpu import hip_attention_mask, sparse_attention


class TestAttention1BlockGPUBwd(unittest.TestCase):

    def test_sparse_attention(self):
        test_sparse_attention()

    def test_attention_mask(self):
        test_attention_mask()


def test_sparse_attention():
    q, k, v, out = load_checkouts(idx=0, window=40)

    w_start = 512
    n_patches = 128
    mask_k = 256
    scale_up = 2
    block_size_q = 16
    block_size_k = 4

    if q.dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        warnings.warn("hip attention does not support 32 bits right now.")

    mask = torch.ones((q.shape[0], k.shape[1]), dtype=torch.bool, device=q.device)

    with torch.autocast("cuda", torch.float32):
        indices, ks, probs, scores = hip_attention_mask(
            q,
            k,
            v,
            mask,
            1,
            w_start=w_start,
            n_patches=n_patches,
            mask_k=mask_k,
            scale_up=scale_up,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            is_causal=False,
            is_flash=False,
        )

    v = nn.Parameter(v)
    probs = nn.Parameter(probs)

    # exam GD
    for i in range(1000):
        context = sparse_attention(
            v,
            indices,
            ks,
            probs,
            1,
            block_size_q,
            block_size_k,
        )

        loss = context.square().sum() * 0.01
        loss.backward()

        v.data -= 0.1 * v.grad
        probs.data -= 0.1 * probs.grad

        v.grad = None
        probs.grad = None

        # print(loss.item())

    assert loss.item() < 1.0
    print("[pass] test_sparse_attention")


def test_attention_mask():
    q, k, v, out = load_checkouts(idx=0, window=40)

    w_start = 512
    n_patches = 128
    mask_k = 256
    scale_up = 2
    block_size_q = 16
    block_size_k = 4

    if q.dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        warnings.warn("hip attention does not support 32 bits right now.")

    mask = torch.ones((q.shape[0], k.shape[1]), dtype=torch.bool, device=q.device)

    q = nn.Parameter(q)
    k = nn.Parameter(k)

    # exam GD
    for i in range(1000):
        indices, ks, probs, scores = hip_attention_mask(
            q,
            k,
            v,
            mask,
            1,
            w_start=w_start,
            n_patches=n_patches,
            mask_k=mask_k,
            scale_up=scale_up,
            block_size_q=block_size_q,
            block_size_k=block_size_k,
            is_causal=False,
            is_flash=False,
        )

        loss = probs.std() * 1000
        loss.backward()

        # print(q.grad.abs().sum())
        q.data -= 50 * q.grad
        k.data -= 50 * k.grad

        q.grad = None
        k.grad = None

        # print(loss.item())

    assert loss.item() < 5.0
    print("[pass] test_attention_mask")
