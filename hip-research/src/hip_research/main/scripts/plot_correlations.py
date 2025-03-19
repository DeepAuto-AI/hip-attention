from typing import Literal

import numpy as np
import torch
import triton
from hip_research.utils.load_checkouts import load_checkouts
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

from hip_attn.v1_2.attention_extend import (
    chunk_controllable_sampling_mask_cuda,
    safe_stride,
)

# X: chunk size
# Y: correlation
# Legend: method {gen2-left, gen2-center, gen2-right, infllm style, gen3}


def compute_oracle(q: torch.Tensor, k: torch.Tensor, chunk_size: int):
    q = q[0, -1, 0, :]
    k = k[0, :, 0, :]
    scores = (q[None, :] @ k.T)[0]
    scores = scores.view(-1, chunk_size)
    scores = scores.max(dim=-1).values
    return scores.float().cpu().numpy()


def compute_gen2(
    q: torch.Tensor,
    k: torch.Tensor,
    chunk_size: int,
    location: Literal["left", "right", "center"],
):
    q = q[0, -1, 0, :]
    k = k[0, :, 0, :]
    scores = (q[None, :] @ k.T)[0]
    scores = scores.view(-1, chunk_size)

    if location == "left":
        scores = scores[:, 0]
    elif location == "center":
        scores = scores[:, chunk_size // 2]
    elif location == "right":
        scores = scores[:, -1]
    else:
        raise Exception()

    return scores.float().cpu().numpy()


def compute_gen3(q: torch.Tensor, k: torch.Tensor, chunk_size: int):
    BLOCK_SIZE_Q = 16
    q = q[:, -1:, :, :]
    BSZ = 1
    chunk_count = k.shape[1] // chunk_size
    BLOCK_CHUNK = 16
    TDST = 1
    STAGE_STRIDE = 1
    HEAD = 1
    MAX_TSRC = 196608  # k.shape[1]
    BDST_SCAN = 1

    indices_left = torch.zeros(
        (BSZ, BDST_SCAN, HEAD, chunk_count), device=q.device, dtype=torch.int64
    )

    indices_left[:, :, :, :] = (
        torch.floor(
            torch.arange(0, chunk_count, device=q.device, dtype=torch.float64)
            * chunk_size
            + 0
        ).to(indices_left.dtype)
    )[None, None, None, :]
    indices_right = indices_left + chunk_size
    indices_right.clamp_max_(MAX_TSRC - 0)

    out_scores = torch.full(
        (BSZ, BDST_SCAN, HEAD, triton.next_power_of_2(chunk_count)),
        device=q.device,
        dtype=torch.float32,
        fill_value=-32000.0,
    )

    position_ids = torch.full(
        (BSZ, TDST), device=q.device, dtype=torch.int32, fill_value=k.shape[1] - 1
    )

    grid = (
        BSZ
        * triton.cdiv(chunk_count, BLOCK_CHUNK)
        * triton.cdiv(triton.cdiv(TDST, BLOCK_SIZE_Q), STAGE_STRIDE)
        * HEAD,
    )
    njobs = grid[0]
    group_jobs = 1

    chunk_controllable_sampling_mask_cuda[grid](
        q,
        *q.stride(),
        k,
        *safe_stride(k, 4),
        position_ids,
        *position_ids.stride(),
        *(
            False,
            1,
            None,
            *safe_stride(None, 4),
            None,
            *safe_stride(None, 4),
            None,
            *safe_stride(None, 2),
            None,
            *safe_stride(None, 1),
        ),
        *(
            False,
            False,
            0,
            None,
            0,
            0,
            None,
            0,
            0,
            None,
            0,
            0,
            None,
            0,
            0,
            None,
            0,
            0,
            0,
        ),
        indices_left,
        *indices_left.stride(),
        indices_right,
        *indices_right.stride(),
        out_scores,
        *out_scores.stride(),
        cos,
        *safe_stride(cos, 2),
        sin,
        *safe_stride(sin, 2),
        None,
        *safe_stride(None, 3),
        None,
        *safe_stride(None, 3),
        chunk_count,
        MAX_TSRC,
        TDST,
        HEAD,
        0,
        0,
        # model_context_length if (not scan_extend_backend == 'streaming') else 0,
        131072,
        group_jobs,
        njobs,
        BLOCK_HID=q.shape[-1],
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        STRIDE_Q=1,
        BLOCK_CHUNK=BLOCK_CHUNK,
        HEAD_GROUP=HEAD // 1,
        USING_EXTEND=False,
        EXTEND_BACKEND="relative",
        NEED_APPLY_ROPE=False,
        TERMINATE_SIZE=1,
        SCAN_STRIDE=STAGE_STRIDE,
        UPDATE_CACHE=False,
    )

    return out_scores.float().view(-1).cpu().numpy()


# def compute_infllm(q: torch.Tensor, k: torch.Tensor, chunk_size: int):
#     num_repr = min(chunk_size, 4)
#     reprs = []
#     for i_start in range(0, k.shape[1], chunk_size):
#         i_end = i_start + chunk_size
#         tq = q[0, i_start: i_end, 0, :]
#         tk = k[0, i_start: i_end, 0, :]
#         cmask = torch.arange(0, chunk_size, device=q.device)[:, None] >= torch.arange(0, chunk_size, device=q.device)[None, :]
#         repr_loc = (((tq @ tk.T) * cmask).sum(0) / cmask.float().sum(0)).topk(k=num_repr).indices
#         reprs.append(repr_loc)
#     reprs = torch.stack(reprs, dim=0)

#     curr_q = q[0, -1, 0, :]
#     k = k[0, :, 0, :]
#     scores = (curr_q[None, :] @ k.T)[0]
#     scores = scores.view(-1, chunk_size)
#     scores = scores.gather(dim=1, index=reprs).max(dim=-1).values

#     return scores.float().cpu().numpy()


def compute_infllm(
    q: torch.Tensor,
    k: torch.Tensor,
    chunk_size: int,
    num_repr: int = 4,
):
    num_repr = min(chunk_size, num_repr)
    reprs = []
    for i_start in range(0, k.shape[1], chunk_size):
        i_end = i_start + chunk_size
        tq = q[0, i_start:i_end, 0, :]
        tk = k[0, i_start:i_end, 0, :]
        # cmask = torch.arange(0, chunk_size, device=q.device)[:, None] >= torch.arange(0, chunk_size, device=q.device)[None, :]
        # repr_loc = (((tq @ tk.T) * cmask).sum(0) / cmask.float().sum(0)).topk(k=num_repr).indices
        repr_loc = (tq.mean(dim=0, keepdim=True) @ tk.T)[0].topk(k=num_repr).indices
        reprs.append(repr_loc)
    reprs = torch.stack(reprs, dim=0)

    curr_q = q[0, -1, 0, :]
    k = k[0, :, 0, :]
    scores = (curr_q[None, :] @ k.T)[0]
    scores = scores.view(-1, chunk_size)
    scores = scores.gather(dim=1, index=reprs).max(dim=-1).values

    return scores.float().cpu().numpy()


def compute_all(
    q: torch.Tensor,
    k: torch.Tensor,
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    chunk_size: int,
):
    sink_chunk = 256 // chunk_size
    sw_chunk = 1024 // chunk_size

    oracle_values = compute_oracle(q_rope, k_rope, chunk_size)[sink_chunk:-sw_chunk]

    gen2_left = compute_gen2(q, k, chunk_size, "left")[sink_chunk:-sw_chunk]
    gen2_center = compute_gen2(q, k, chunk_size, "center")[sink_chunk:-sw_chunk]
    gen2_right = compute_gen2(q, k, chunk_size, "right")[sink_chunk:-sw_chunk]

    gen3_values = compute_gen3(q, k, chunk_size)[sink_chunk:-sw_chunk]

    infllm_values_4 = compute_infllm(q, k, chunk_size, 4)[sink_chunk:-sw_chunk]
    infllm_values_8 = compute_infllm(q, k, chunk_size, 8)[sink_chunk:-sw_chunk]
    infllm_values_16 = compute_infllm(q, k, chunk_size, 16)[sink_chunk:-sw_chunk]

    gen3_infllm_values_4 = np.maximum(infllm_values_4, gen3_values)
    gen3_infllm_values_8 = np.maximum(infllm_values_8, gen3_values)
    gen3_infllm_values_16 = np.maximum(infllm_values_16, gen3_values)

    gen2_left_corr = spearmanr(gen2_left, oracle_values).correlation
    gen2_center_corr = spearmanr(gen2_center, oracle_values).correlation
    gen2_right_corr = spearmanr(gen2_right, oracle_values).correlation

    gen3_corr = spearmanr(gen3_values, oracle_values).correlation

    infllm_corr_4 = spearmanr(infllm_values_4, oracle_values).correlation
    infllm_corr_8 = spearmanr(infllm_values_8, oracle_values).correlation
    infllm_corr_16 = spearmanr(infllm_values_16, oracle_values).correlation

    gen3_infllm_corr_4 = spearmanr(gen3_infllm_values_4, oracle_values).correlation
    gen3_infllm_corr_8 = spearmanr(gen3_infllm_values_8, oracle_values).correlation
    gen3_infllm_corr_16 = spearmanr(gen3_infllm_values_16, oracle_values).correlation

    return {
        "gen2_left": gen2_left_corr,
        "gen2_center": gen2_center_corr,
        "gen2_right": gen2_right_corr,
        "gen3": gen3_corr,
        "infllm_4": infllm_corr_4,
        "infllm_8": infllm_corr_8,
        "infllm_16": infllm_corr_16,
        "gen3_infllm_corr_4": gen3_infllm_corr_4,
        "gen3_infllm_corr_8": gen3_infllm_corr_8,
        "gen3_infllm_corr_16": gen3_infllm_corr_16,
    }


q, k, v, out, cos, sin = load_checkouts(
    idx=0,
    window=999,
    seq_len=131072,
    dtype=torch.bfloat16,
    return_cos_sin=True,
    derope=False,
)

q_rope, k_rope, _, _, _, _ = load_checkouts(
    idx=0,
    window=999,
    seq_len=131072,
    dtype=torch.bfloat16,
    return_cos_sin=True,
    derope=False,
)


def reshape(x: torch.Tensor):
    return x.unsqueeze(0).permute(0, 2, 1, 3)


q = reshape(q).to(0)
k = reshape(k).to(0)
v = reshape(v).to(0)
q_rope = reshape(q_rope).to(0)
k_rope = reshape(k_rope).to(0)

num_heads = 4
chunk_sizes = [4, 8, 16, 32, 64, 128, 256]
data = {}

for target_head in range(num_heads):
    current_q = q[:, :, target_head : target_head + 1, :]
    curr_k = k[
        :,
        :,
        target_head
        // (q.shape[2] // k.shape[2]) : target_head
        // (q.shape[2] // k.shape[2])
        + 1,
        :,
    ]
    current_q_rope = q_rope[:, :, target_head : target_head + 1, :]
    curr_k_rope = k_rope[
        :,
        :,
        target_head
        // (q.shape[2] // k.shape[2]) : target_head
        // (q.shape[2] // k.shape[2])
        + 1,
        :,
    ]

    for i_chunk, chunk_size in enumerate(chunk_sizes):
        results = compute_all(
            current_q, curr_k, current_q_rope, curr_k_rope, chunk_size
        )
        for key in results:
            if key not in data:
                data[key] = [
                    0,
                ] * len(chunk_sizes)
            data[key][i_chunk] += results[key]

plt.clf()

for legend in data:
    plt.plot(chunk_sizes, (np.array(data[legend]) / num_heads).tolist(), label=legend)

plt.grid()
plt.xlabel("Chunk Size")
plt.ylabel("Spearman Corr.")
# plt.xscale('log', base=2)
plt.legend()

plt.savefig("dummy.png")
