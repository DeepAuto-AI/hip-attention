import unittest

import numpy as np
import torch
import tqdm
from vllm.model_executor.layers.attention import _paged_attention

from hip_attn.v1_0.attention1_block_gpu import paged_hip_attention


class TestPageAttention(unittest.TestCase):

    def test_vllm(self):
        test_vllm()

    def test_vllm_compat(self):
        test_vllm_compat()

    def test_vllm_compat_cpu(self):
        test_vllm_compat_cpu()


def load_states():
    return torch.load("./cache/llama/vllmout.pth", map_location="cuda:0")


def test_vllm():
    state = load_states()
    query = state["query"]
    key_cache = state["key_cache"]
    value_cache = state["value_cache"]
    input_metadata = state["input_metadata"]
    num_kv_heads = state["num_kv_heads"]
    scale = state["scale"]
    alibi_slopes = state["alibi_slopes"]
    assert alibi_slopes is None
    output_truth = state["output"]

    output = _paged_attention(
        query,
        key_cache,
        value_cache,
        input_metadata,
        num_kv_heads,
        scale,
        alibi_slopes,
    )

    # print(output_truth)

    error = torch.abs(output - output_truth).mean()
    print(torch.std_mean(output_truth), torch.std_mean(output), error, sep="\n")


def cpu_kernel(
    query: np.ndarray,
    query_scale: float,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    block_tables: np.ndarray,
    context_lens: np.ndarray,
    max_context_len: int,
):
    """
    vLLM compatible paged attention

    q: [num_seqs, num_heads, head_size]
    k: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    v: [num_blocks, num_kv_heads, head_size, block_size]
    block_tables: [num_seqs, max_num_blocks_per_seq]
    context_lens: [num_seqs]
    """

    query = query * query_scale

    output = np.zeros_like(query)

    block_size = value_cache.shape[-1]
    batch_size, num_heads, head_size = query.shape

    print(query.shape, key_cache.shape, value_cache.shape)

    for idx_n in range(batch_size):
        for idx_h in range(num_heads):
            context_length = context_lens[idx_n]
            scores = np.zeros((context_length,))
            # q @ k
            for idx_tsrc in tqdm.tqdm(
                range(context_length), dynamic_ncols=True, desc=f"score_{idx_n}_{idx_h}"
            ):
                idx_block = block_tables[idx_n, idx_tsrc // block_size]
                offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
                assert key_cache.ndim == 5, key_cache.shape
                try:
                    key = key_cache[idx_block, idx_h, :, offset_block, :].reshape(-1)
                except:
                    raise Exception(
                        f"{idx_block}, {idx_h}, :, {offset_block}, : {key_cache.shape}"
                    )
                qvec = query[idx_n, idx_h, :]
                score = np.sum(key * qvec)
                scores[idx_tsrc] = score
            # softmax
            scores = scores - np.max(scores)
            scores = np.exp(scores) / np.sum(np.exp(scores))
            # s @ v
            for idx_tsrc in tqdm.tqdm(
                range(context_length), dynamic_ncols=True, desc=f"value_{idx_n}_{idx_h}"
            ):
                idx_block = block_tables[idx_n, idx_tsrc // block_size]
                offset_block = idx_tsrc - ((idx_tsrc // block_size) * block_size)
                value = value_cache[idx_block, idx_h, :, offset_block].reshape(-1)
                prob = scores[idx_tsrc]
                output[idx_n, idx_h, :] += prob * value

    return output


def test_vllm_compat_cpu():
    state = load_states()
    query = state["query"]
    key_cache = state["key_cache"]
    value_cache = state["value_cache"]
    input_metadata = state["input_metadata"]
    num_kv_heads = state["num_kv_heads"]
    scale = state["scale"]
    alibi_slopes = state["alibi_slopes"]
    assert alibi_slopes is None
    output_truth = state["output"]

    output = cpu_kernel(
        query=query.cpu().to(torch.float32).numpy(),
        query_scale=scale,
        key_cache=key_cache.cpu().to(torch.float32).numpy(),
        value_cache=value_cache.cpu().to(torch.float32).numpy(),
        block_tables=input_metadata.block_tables.cpu().numpy(),
        context_lens=input_metadata.context_lens.cpu().numpy(),
        max_context_len=input_metadata.max_context_len,
    )
    output = torch.tensor(output, device=output_truth.device, dtype=output_truth.dtype)

    error = torch.abs(output - output_truth).mean()
    print(torch.std_mean(output_truth), torch.std_mean(output), error, sep="\n")


def test_vllm_compat():
    state = load_states()
    query = state["query"]
    key_cache = state["key_cache"]
    value_cache = state["value_cache"]
    input_metadata = state["input_metadata"]
    num_kv_heads = state["num_kv_heads"]
    scale = state["scale"]
    alibi_slopes = state["alibi_slopes"]
    assert alibi_slopes is None
    output_truth = state["output"]

    with torch.no_grad():
        output, _ = paged_hip_attention(
            q=query,
            q_scale=scale,
            k=key_cache,
            v=value_cache,
            block_tables=input_metadata.block_tables,
            context_lens=input_metadata.context_lens,
            max_context_len=input_metadata.max_context_len,
            mask_k=1024,
        )
        N_H, _, HID = output.shape
        N = query.shape[0]
        H = N_H // N
        output = output.view(N, H, HID)

    print(output.shape, output.dtype, output_truth.shape, output_truth.dtype)

    error = torch.abs(output - output_truth).mean()
    print(torch.std_mean(output_truth), torch.std_mean(output), error, sep="\n")
