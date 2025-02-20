import math

import torch
from einops import rearrange

from .angular_lsh import AngularLSH
from .utils import add_self_attentions, exact_attention, exact_attention_cuda, indexing


class HyperAttention(torch.nn.Module):

    def __init__(
        self,
        input_dim=64,
        lsh_num_projs=7,
        block_size=256,
        sample_size=256,
        min_seq_len=4096,
        cuda=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.lsh_num_projs = lsh_num_projs
        self.block_size = block_size
        self.sample_size = sample_size
        self.min_seq_len = min_seq_len
        self.cuda = cuda
        self.lsh = AngularLSH(num_projs=self.lsh_num_projs, dim=(1, 1, input_dim))

    def forward(
        self,
        query: torch.tensor,
        key: torch.tensor,
        value: torch.tensor,
        scale=None,
        causal=False,
        return_lse=False,
    ):
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        n_query = query.shape[2]
        batch_size, n_heads, n_key, dim = key.shape
        scale = dim ** (-0.5) if scale is None else scale

        # Without causal masking
        if not causal:
            attn, lse = self.forward_no_causal_mask(query, key, value, scale)

        # With causal masking
        else:
            if n_key <= self.min_seq_len:
                if self.cuda:
                    attn, lse = exact_attention_cuda(
                        query, key, value, scale, causal=True
                    )
                else:
                    attn, lse = exact_attention(query, key, value, scale, causal=True)
            else:

                # If n_query is odd we pad inputs by adding all-zero rows
                if n_query % 2:
                    query = torch.nn.functional.pad(
                        query, (0, 0, 0, 1), mode="constant", value=0.0
                    )
                    key = torch.nn.functional.pad(
                        key, (0, 0, 0, 1), mode="constant", value=0.0
                    )
                    value = torch.nn.functional.pad(
                        value, (0, 0, 0, 1), mode="constant", value=0.0
                    )

                q_bd = query.view(
                    batch_size, 2 * n_heads, query.shape[2] // 2, query.shape[-1]
                )
                k_bd = key.view(
                    batch_size, 2 * n_heads, key.shape[2] // 2, key.shape[-1]
                )
                v_bd = value.view(
                    batch_size, 2 * n_heads, key.shape[2] // 2, value.shape[-1]
                )

                # print(q_bd.shape)
                attn_bd, lse_bd = self.forward(q_bd, k_bd, v_bd, scale, True, True)

                if attn_bd.shape[2] not in attn_bd.stride():
                    attn_bd = attn_bd.contiguous()
                attn_bd = attn_bd.view(batch_size, n_heads, -1, dim)

                if lse_bd.shape[2] not in lse_bd.stride():
                    lse_bd = lse_bd.contiguous()
                lse_bd = lse_bd.view(batch_size, n_heads, -1, 1)

                attn_unmasked, lse_unmasked = self.forward_no_causal_mask(
                    query[:, :, key.shape[2] // 2 :, :],
                    key[:, :, : key.shape[2] // 2, :],
                    value[:, :, : key.shape[2] // 2, :],
                    scale,
                )

                attn_up, lse_up = (
                    attn_bd[:, :, : query.shape[2] // 2, :],
                    lse_bd[:, :, : query.shape[2] // 2, :],
                )
                attn_down, lse_down = add_self_attentions(
                    attn_bd[:, :, query.shape[2] // 2 :, :],
                    lse_bd[:, :, query.shape[2] // 2 :, :],
                    attn_unmasked,
                    lse_unmasked,
                )

                attn = torch.cat((attn_up, attn_down), dim=-2)
                lse = torch.cat((lse_up, lse_down), dim=-2)

                # If n_query was odd exclude the last rows
                if n_query % 2:
                    attn = attn[:, :, :-1, :]
                    lse = lse[:, :, :-1, :]

        if not return_lse:
            return attn
        else:
            return attn, lse

    def forward_no_causal_mask(self, query, key, value, scale):

        batch_size, head_size, n_query, dim = query.shape
        n_key = key.shape[2]

        if self.min_seq_len > n_query:
            if self.cuda:
                return exact_attention_cuda(query, key, value, scale, causal=False)
            else:
                return exact_attention(query, key, value, scale, causal=False)

        # 1. Sorted block-diagonal via sortLSH
        _, query_sort_idx = torch.sort(
            self.lsh.hash(query), dim=2, stable=True
        )  # batch_size x head_size x n
        _, key_sort_idx = torch.sort(self.lsh.hash(key), dim=2, stable=True)
        query_sort_idx_inv = torch.argsort(
            query_sort_idx, dim=2, stable=True
        )  # for recovering the row order

        key_block_size = self.block_size

        query_sorted = indexing(query, query_sort_idx, key_block_size)
        key_sorted = indexing(key, key_sort_idx, key_block_size)
        value_sorted = indexing(value, key_sort_idx, key_block_size)

        if key_block_size > 0:

            num_blocks = key_sorted.shape[2] // key_block_size
            query_block_size = query_sorted.shape[2] // num_blocks

            # Reshape tensors to [batch_size*head_size, 1, block_size, dim] as Flash-attn only allows 4d-tensors
            query_split_per_block = query_sorted.view(-1, 1, query_block_size, dim)
            key_split_per_block = key_sorted.view(-1, 1, key_block_size, dim)
            value_split_per_block = value_sorted.view(-1, 1, key_block_size, dim)

            if self.cuda:
                attn_block, lse_block = exact_attention_cuda(
                    query_split_per_block,
                    key_split_per_block,
                    value_split_per_block,
                    softmax_scale=scale,
                    causal=False,
                )
            else:
                attn_block, lse_block = exact_attention(
                    query_split_per_block,
                    key_split_per_block,
                    value_split_per_block,
                    softmax_scale=scale,
                    causal=False,
                )

            if attn_block.shape[2] not in attn_block.stride():
                attn_block = attn_block.contiguous()
            attn_block = attn_block.view(
                batch_size, head_size, query_sorted.shape[2], -1
            )

            if lse_block.shape[2] not in lse_block.stride():
                lse_block = lse_block.contiguous()
            lse_block = lse_block.view(batch_size, head_size, query_sorted.shape[2], -1)

            # When inputs are padded, then unpad them
            if query_sorted.shape[2] != n_query:  # query.shape[2]:
                attn_block, lse_block = (
                    attn_block[:, :, :n_query, :],
                    lse_block[:, :, :n_query, :],
                )
                query_sorted = query_sorted[:, :, :n_query, :]
                key_sorted = key_sorted[:, :, :n_key, :]
                value_sorted = value_sorted[:, :, :n_key, :]

        else:
            query_block_size = -1
            query_block_size = -1
            attn_block, lse_block = 0, 0

        # 2. Residual low-rank part via uniform sampling
        # Sample indices uniformly at random
        sample_size = self.sample_size
        if (
            sample_size > 0
            and (n_query > query_block_size)
            and (n_key > key_block_size)
        ):
            sampled_set = torch.randint(
                n_key,
                size=(batch_size, head_size, sample_size),
                device=query_sorted.device,
            )

            # Compute mask for hiding A_ij computed in block-diagonal attention
            offset_n = rearrange(
                torch.arange(n_query, device=query_sorted.device), "n -> 1 n 1"
            )
            weights = n_key / sample_size
            value_subset = indexing(value_sorted, sampled_set)
            key_subset = indexing(key_sorted, sampled_set)

            if not self.cuda:
                block_mask = (offset_n // query_block_size) == (
                    sampled_set // key_block_size
                ).view(-1, 1, sample_size)
                block_mask = block_mask.view(batch_size, head_size, -1, sample_size)
                block_mask = block_mask.to(query_sorted.dtype)
                block_mask *= torch.finfo(
                    query_sorted.dtype
                ).min  # adding -inf added to QK^T

                attn_res, lse_res = exact_attention(
                    query_sorted,
                    key_subset,
                    value_subset,
                    scale,
                    causal=False,
                    bias=block_mask,
                )
            else:
                attn_res, lse_res = exact_attention_cuda(
                    query_sorted, key_subset, value_subset, scale, causal=False
                )
            lse_res = lse_res + math.log(weights)

            # Add two attentions
            if key_block_size > 0:
                attn, lse = add_self_attentions(
                    attn_block, lse_block, attn_res, lse_res
                )
            else:
                attn, lse = attn_res, lse_res
        else:
            attn, lse = attn_block, lse_block

        # Re-order rows with the inverse order for query_sorted -> query
        attn = indexing(attn, query_sort_idx_inv)
        lse = indexing(lse, query_sort_idx_inv)
        return attn, lse
