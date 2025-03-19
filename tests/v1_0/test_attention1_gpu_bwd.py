import unittest
import warnings

import torch
from hip_research.utils.load_checkouts import load_checkouts
from torch import nn

from hip_attn.v1_0.attention1_gpu import attention_matrix, sparse_attention


class TestAttention1GPUBwd(unittest.TestCase):

    def test_sparse_attention(self):
        test_sparse_attention()

    def test_attention_mask(self):
        test_attention_mask()


def test_sparse_attention():
    q, k, v, out = load_checkouts()

    w_start = 512
    n_patches = 128
    mask_k = 256
    scale_up = 2

    if q.dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        warnings.warn("hip attention does not support 32 bits right now.")

    with torch.autocast("cuda", torch.float32):
        indices, ks, probs = attention_matrix(
            q,
            k,
            w_start,
            n_patches,
            mask_k,
            scale_up,
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
        )

        loss = context.square().sum()
        loss.backward()

        v.data -= 0.1 * v.grad
        probs.data -= 0.001 * probs.grad

        v.grad = None
        probs.grad = None

        # print(loss.item())

    assert loss.item() < 0.01
    print("[pass] test_sparse_attention")


def test_attention_mask():
    q, k, v, out = load_checkouts()

    w_start = 512
    n_patches = 128
    mask_k = 256
    scale_up = 2

    if q.dtype != torch.float32:
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        warnings.warn("hip attention does not support 32 bits right now.")

    q = nn.Parameter(q)
    k = nn.Parameter(k)

    # exam GD
    for i in range(1000):
        indices, ks, probs = attention_matrix(
            q,
            k,
            w_start,
            n_patches,
            mask_k,
            scale_up,
        )

        loss = probs.std() * 1000
        loss.backward()

        # print(q.grad.abs().sum())
        q.data -= 0.1 * q.grad
        k.data -= 0.1 * k.grad

        q.grad = None
        k.grad = None

        # print(loss.item())

    assert loss.item() < 3.5
    print("[pass] test_attention_mask")
