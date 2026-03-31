import unittest

import torch
from hip_research.utils.load_checkouts import load_checkouts

from hip_attn.v1_1.attention2_draft_prefetch import (
    HiPAttentionArgs,
    block_sparse_attention,
)


class TestAttention2PrefetchBsa(unittest.TestCase):

    def test_indices(self):
        _, _, _, _, cos, sin = load_checkouts(
            idx=0,
            window=40,
            seq_len=seq_len,
            return_cos_sin=True,
            derope=True,
            dtype=torch.bfloat16,
        )

        for i in range(0, seq_len, 371):
            test_index(i, cos, sin)


seq_len = 131072
head_dim = 128
num_heads = 32
num_heads_kv = 8
bsz = 2


def test_index(target_idx, cos, sin):
    device = 0
    dtype = torch.bfloat16
    q = torch.zeros((bsz, 1, num_heads, head_dim), device=device, dtype=dtype)
    k = torch.zeros((bsz, seq_len, num_heads, head_dim), device=device, dtype=dtype)
    v = torch.zeros((bsz, seq_len, num_heads, head_dim), device=device, dtype=dtype)
    seq_lens = torch.tensor([[seq_len], [seq_len]], dtype=torch.int64, device=device)

    args = HiPAttentionArgs(
        mask_k=512,
        block_size_q=1,
        block_stride_q=1,
        block_size_k=8,
        block_stride_k=1,
        sink_token_size=1024,
        sliding_window_size=1024,
        rope_cos=cos,
        rope_sin=sin,
    )

    indices = torch.zeros((bsz * num_heads, 1, 1), device=device, dtype=torch.int32)
    indices.fill_(target_idx // 8 * 8)
    ks = torch.zeros((bsz * num_heads, 1), device=device, dtype=torch.int32)
    ks.fill_(1)
    ks_count = ks.unsqueeze(-1)
    ks_start_end = torch.cat(
        [torch.zeros_like(ks.unsqueeze(-1)), ks.unsqueeze(-1)], dim=-1
    )

    q[:, :, :, :] = (torch.arange(0, head_dim, device=device, dtype=dtype) / head_dim)[
        None, None, None, :
    ]
    k[:, target_idx : target_idx + 1, :, :] = (
        torch.arange(0, head_dim, device=device, dtype=dtype) / head_dim
    )[None, None, None, :]
    v[:, :, :, 0] = torch.arange(0, seq_len, device=device, dtype=dtype)[None, :, None]
    v[:, target_idx, :, -1] = 1

    out = block_sparse_attention(
        q=q,
        k=k,
        v=v,
        seq_lens=seq_lens,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args,
        EXTEND_BACKEND="streaming",
        model_context_length=131072,
    )

    lookup_idx = out[0, 0, 0, 0].item()
    lookup_canary = out[0, 0, 0, -1].item()
    print(target_idx, lookup_idx, lookup_canary)
