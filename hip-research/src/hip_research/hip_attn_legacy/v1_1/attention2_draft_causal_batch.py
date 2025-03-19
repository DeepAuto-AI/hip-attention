"""
HiP v1.1
TODO:
1. Masking iteration using integer to avoid aliasing and collision
 - Convert tmask into int32 (good!)
 - Reuse the computed dot products
2. Using QUEST method for b_k (not very good)
3. Maximum token location predictor
 - Test oracle (not very good)
 - Test estimators
4. sifters? (not very good)
5. masking -> allocate cells (not very good)
6. StreamLLM based traverse (use Self-Extend instead of SLLM)
7. causal-batch
8. 2d support
9. support backward across tree
10. chunk-wise BPTT
"""

# normal                    PPL: 9.7576
# bk 1 bkg 4                PPL: 9.3042
# bk 4 bkg 1                PPL: 9.1336
# bk 1 bkg 4 oracle_rep
# bk 4 bkg 1 oracle_rep     PPL: 9.1336
# bk 4 bkg 1 2 sifters      PPL: 9.2236
# bk 4 bkg 1 recurse 2 lv   PPL: 9.1364
# bk 4 bkg 1 recurse 3 lv   PPL: 9.1930

# topk_head_group
# g1    5.9297
# g4    5.5488
# g8    5.5473


import math
from typing import Optional

import numba
import numpy as np
import torch
from numpy import ndarray as NdArray
from torch import Tensor


@numba.njit
def cdiv(a, b):
    return math.ceil(a / b)


@numba.njit
def de_rope(vec_rope, cos, sin):
    assert len(vec_rope.shape) == 1
    assert vec_rope.shape == cos.shape
    assert cos.shape == sin.shape
    out = np.zeros_like(vec_rope)
    half = len(vec_rope) // 2
    c0 = cos[:half]
    ch = cos[half:]
    s0 = sin[:half]
    sh = sin[half:]
    vr0 = vec_rope[:half]
    vrh = vec_rope[half:]
    out[:half] = (vrh * s0 + vr0 * ch) / (c0 * ch + sh * s0 + 1e-20)
    out[half:] = (out[:half] * c0 - vr0) / (s0 + 1e-20)
    return out


@numba.njit
def rotate_half(vec):
    assert len(vec.shape) == 1
    out = np.zeros_like(vec)
    x1 = vec[: len(vec) // 2]
    x2 = vec[len(vec) // 2 :]
    out[: len(vec) // 2] = -x2
    out[len(vec) // 2 :] = x1
    return out


@numba.njit
def apply_rope(vec, cos, sin):
    assert vec.shape == cos.shape
    assert cos.shape == sin.shape
    vec_rope = (vec * cos) + (rotate_half(vec) * sin)
    return vec_rope


@numba.njit
def masking_iteration_draft_numba_kernel(
    # in
    q: NdArray,  # fp32[block_size_q, HID]
    k: NdArray,  # fp32[TSRC(sliced), HID]
    # out
    indices: NdArray,  # int32[MASK_K // B_K]
    # param
    mask_k: int,
    block_size_q: int,
    block_size_k: int,
    block_size_k_group: int,
    oracle_rep: bool,
    sliding_window_size: int,
    using_extend: bool,
    rope_cos: Optional[Tensor],
    rope_sin: Optional[Tensor],
    idx_bdst: int,
    TDST: int,
    MAX_TSRC: int,
    self_extend_neighboor_window: int,
    self_extend_group_size: int,
    topk_head_group_size: int,
) -> int:
    mask_block_k = cdiv(mask_k, block_size_k)
    TSRC = max(0, k.shape[1] - sliding_window_size)
    BSRC = cdiv(TSRC, block_size_k)
    MAX_BSRC = cdiv(MAX_TSRC, block_size_k)

    if TSRC <= mask_k:
        k_out = 0
        for i in range(topk_head_group_size):
            for j in range(BSRC):
                indices[k_out] = i * MAX_TSRC + j * block_size_k
                k_out += 1
        return k_out
    else:
        # initialize
        group_sizes = np.zeros_like(indices)
        for i in range(topk_head_group_size):
            for j in range(mask_block_k):
                indices[i * mask_block_k + j] = MAX_BSRC * i + int(
                    BSRC / mask_block_k * j
                )
                group_sizes[i * mask_block_k + j] = min(
                    BSRC, int(BSRC / mask_block_k * (i + 1))
                ) - int(BSRC / mask_block_k * i)

        group_size = BSRC / mask_block_k

        # until converge
        stage = 0
        while group_size > 1:
            # divide
            dupped_indices = indices.repeat(2).copy()
            dupped_indices[1::2] = (dupped_indices[1::2] + group_sizes * 0.5).astype(
                np.int32
            )
            dupped_group_sizes = group_sizes.repeat(2).copy()
            dupped_group_sizes[0::2] = dupped_indices[1::2] - dupped_indices[0::2]
            dupped_group_sizes[1::2] = (
                dupped_indices[0::2] + group_sizes - dupped_indices[1::2]
            )
            dupped_mask = dupped_group_sizes >= 1

            scores = np.zeros_like(dupped_indices, dtype=np.float32)
            for i in range(len(scores)):
                if not dupped_mask[i]:
                    continue

                idx_tgsrc = dupped_indices[i] * block_size_k
                idx_group = idx_tgsrc // MAX_TSRC
                idx_tsrc = idx_tgsrc % MAX_TSRC

                if block_size_k_group > 1:
                    assert not oracle_rep
                    assert not using_extend
                    queries = q[idx_group, 1::2, :].copy()
                    keys_min = k[
                        idx_group, idx_tsrc : idx_tsrc + block_size_k, : q.shape[-1]
                    ]
                    keys_max = k[
                        idx_group, idx_tsrc : idx_tsrc + block_size_k, q.shape[-1] :
                    ]
                    t_1 = np.ascontiguousarray(queries) @ np.ascontiguousarray(
                        keys_min.T
                    )
                    t_2 = np.ascontiguousarray(queries) @ np.ascontiguousarray(
                        keys_max.T
                    )
                    scores[i] = max(t_1.max(), t_2.max())
                else:
                    if not oracle_rep:
                        queries = q[idx_group, 1::2, :].copy()
                        keys = k[
                            idx_group, idx_tsrc : idx_tsrc + block_size_k, :
                        ].copy()

                        if using_extend:
                            assert rope_cos is not None
                            assert rope_sin is not None
                            for j in range(len(keys)):
                                old_idx = idx_tsrc + j

                                # StreamingLLM (not working well)
                                # new_idx = i * block_size_k + j

                                # Self Extend (working great)
                                if idx_tsrc >= (
                                    idx_bdst - self_extend_neighboor_window
                                ):
                                    new_idx = old_idx
                                else:
                                    new_idx = old_idx // self_extend_group_size

                                keys[j] = de_rope(
                                    keys[j], rope_cos[old_idx], rope_sin[old_idx]
                                )
                                keys[j] = apply_rope(
                                    keys[j], rope_cos[new_idx], rope_sin[new_idx]
                                )

                            for j in range(len(queries)):
                                old_idx = idx_bdst + j + TSRC - TDST

                                # new_idx = len(scores) * block_size_k - block_size_q + j

                                if idx_tsrc >= (
                                    idx_bdst - self_extend_neighboor_window
                                ):
                                    new_idx = old_idx
                                else:
                                    new_idx = old_idx // self_extend_group_size

                                queries[j] = de_rope(
                                    queries[j], rope_cos[old_idx], rope_sin[old_idx]
                                )
                                queries[j] = apply_rope(
                                    queries[j], rope_cos[new_idx], rope_sin[new_idx]
                                )

                        t = np.ascontiguousarray(queries) @ np.ascontiguousarray(keys.T)
                        scores[i] = t.max()
                    else:
                        assert not using_extend
                        queries = q[idx_group, 1::2, :].copy()
                        for shift in range(dupped_group_sizes[i]):
                            keys = k[
                                idx_group,
                                idx_tsrc
                                + shift * block_size_k : idx_tsrc
                                + shift * block_size_k
                                + block_size_k,
                                :,
                            ]
                            t = np.ascontiguousarray(queries) @ np.ascontiguousarray(
                                keys.T
                            )
                            scores[i] = max(scores[i], t.max())
            scores[:] += -32000.0 * ~dupped_mask

            # select
            topk_indices = np.argsort(-scores)[: mask_block_k * topk_head_group_size]
            indices[:] = dupped_indices[topk_indices]
            group_sizes[:] = dupped_group_sizes[topk_indices]
            # print(group_size, indices, topk_indices, group_sizes)

            group_size = group_size / 2
            stage += 1

        indices[:] = np.sort(indices) * block_size_k
        return mask_block_k * topk_head_group_size


@numba.njit(parallel=True)
def masking_iteration_draft_numba(
    # in
    q: NdArray,
    k: NdArray,
    # out
    indices: NdArray,
    ks: NdArray,
    # param
    mask_k: int,
    block_size_q: int,
    block_size_k: int,
    block_size_k_group: int,
    oracle_rep: bool,
    sliding_window_size: int,
    using_extend: bool,
    rope_cos: Optional[Tensor],
    rope_sin: Optional[Tensor],
    self_extend_neighboor_window: int,
    self_extend_group_size: int,
    topk_head_group_size: int,
):
    """
    grid = (N, TDST)
    """

    N, G, TDST, HID = q.shape
    _, _, TSRC, _ = k.shape

    for idx_n in numba.prange(N):
        for idx_bdst in numba.prange(cdiv(TDST, block_size_q)):
            q_chunk = q[
                idx_n, :, idx_bdst * block_size_q : (idx_bdst + 1) * block_size_q, :
            ]
            k_chunk = k[
                idx_n,
                :,
                : ((idx_bdst + 1) * block_size_q + TSRC * block_size_k_group - TDST)
                // block_size_k_group,
                :,
            ]
            final_k = masking_iteration_draft_numba_kernel(
                q_chunk,
                k_chunk,
                indices[idx_n, idx_bdst, :],
                mask_k=mask_k,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                block_size_k_group=block_size_k_group,
                oracle_rep=oracle_rep,
                sliding_window_size=sliding_window_size,
                using_extend=using_extend,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                idx_bdst=idx_bdst,
                TDST=TDST,
                MAX_TSRC=k.shape[-2],
                self_extend_neighboor_window=self_extend_neighboor_window,
                self_extend_group_size=self_extend_group_size,
                topk_head_group_size=topk_head_group_size,
            )
            ks[idx_n, idx_bdst] = final_k


def masking_iteration_draft(
    q: Tensor,
    k: Tensor,
    mask_k: int,
    block_size_q: int,
    block_size_k: int,
    block_size_k_group: int,
    oracle_rep: bool,
    sliding_window_size: int,
    using_extend: bool,
    rope_cos: Optional[Tensor],
    rope_sin: Optional[Tensor],
    self_extend_neighboor_window: int,
    self_extend_group_size: int,
    topk_head_group_size: int,
):
    device = q.device

    device = q.device
    q = q.cpu().float().numpy()
    k = k.cpu().float().numpy()

    if rope_cos is not None:
        assert rope_cos.ndim == 2
        assert rope_cos.shape[-1] == q.shape[-1]
        rope_cos = rope_cos.cpu().float().numpy()

    if rope_sin is not None:
        assert rope_sin.ndim == 2
        assert rope_sin.shape[-1] == q.shape[-1]
        rope_sin = rope_sin.cpu().float().numpy()

    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape

    assert (
        N % topk_head_group_size
    ) == 0, "batch*n_head should divisible by group size"

    q = q.reshape(N // topk_head_group_size, topk_head_group_size, TDST, HID)
    k = k.reshape(N // topk_head_group_size, topk_head_group_size, TSRC, HID)

    N, G, TDST, HID = q.shape
    _, _, TSRC, _ = k.shape

    if block_size_k_group > 1:
        k_group = np.reshape(
            k, (N, G, TSRC // block_size_k_group, block_size_k_group, HID)
        )
        k_group_min = np.min(k_group, axis=-2)
        k_group_max = np.max(k_group, axis=-2)
        k = np.concatenate([k_group_min, k_group_max], axis=-1)

    indices = np.zeros(
        (
            N,
            cdiv(TDST, block_size_q),
            # head group is merged as single sequence
            G * cdiv(mask_k, block_size_k * block_size_k_group),
        ),
        dtype=np.int32,
    )

    ks = np.zeros(
        (
            N,
            cdiv(TDST, block_size_q),
        ),
        dtype=np.int32,
    )

    masking_iteration_draft_numba(
        q,
        k,
        indices,
        ks,
        mask_k // block_size_k_group,
        block_size_q,
        block_size_k,
        block_size_k_group,
        oracle_rep,
        sliding_window_size,
        using_extend,
        rope_cos,
        rope_sin,
        self_extend_neighboor_window,
        self_extend_group_size,
        topk_head_group_size,
    )

    indices *= block_size_k_group

    return (
        torch.tensor(indices, device=device),
        torch.tensor(ks, device=device),
    )


@numba.njit(parallel=True)
def block_sparse_attention(
    q: NdArray,
    k: NdArray,
    v: NdArray,
    indices: NdArray,
    ks: NdArray,
    block_size_q: int,
    block_size_k: int,
    mask_k: int,
    topk_head_group_size: int,
) -> NdArray:
    output = np.zeros_like(q)

    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape

    G = topk_head_group_size
    B = N // G
    assert N == (B * G)

    _, BDST = ks.shape
    _, _, GKS = indices.shape

    for idx_n in numba.prange(B):
        for idx_bdst in numba.prange(BDST):
            # compute scores for each index
            idx_tdst = idx_bdst * block_size_q
            scores = (
                np.zeros((block_size_q, GKS * block_size_k), dtype=q.dtype) - 32000.0
            )
            for i in range(min(ks[idx_n, idx_bdst], GKS)):
                idx_index = indices[idx_n, idx_bdst, i]
                idx_group = idx_index // TSRC
                idx_tsrc = idx_index % TSRC
                queries = q[
                    idx_n * G + idx_group, idx_tdst : idx_tdst + block_size_q, :
                ]
                keys = k[idx_n * G + idx_group, idx_tsrc : idx_tsrc + block_size_k, :]
                t = np.ascontiguousarray(queries) @ np.ascontiguousarray(keys.T)
                for m in range(t.shape[0]):
                    for n in range(t.shape[1]):
                        if (idx_tsrc + n) > (idx_tdst + TSRC - TDST + m):
                            t[m, n] = -32000.0
                scores[:, i * block_size_k : (i + 1) * block_size_k] = t
            # compute exp
            scores_max = np.max(scores)
            scores = np.exp(scores - scores_max).astype(np.float32)
            # sum of each head
            scores_sum = np.zeros((topk_head_group_size, block_size_q), dtype=q.dtype)
            for i in range(min(ks[idx_n, idx_bdst], GKS)):
                idx_index = indices[idx_n, idx_bdst, i]
                idx_group = idx_index // TSRC
                scores_sum[idx_group, :] += np.sum(
                    scores[:, i * block_size_k : (i + 1) * block_size_k], axis=-1
                )
            # divide by sum of each
            for i in range(min(ks[idx_n, idx_bdst], GKS)):
                idx_index = indices[idx_n, idx_bdst, i]
                idx_group = idx_index // TSRC
                for j in range(block_size_k):
                    scores[:, i * block_size_k + j] /= scores_sum[idx_group, :] + 1e-12
            # multiply and acc for each head
            for i in range(min(ks[idx_n, idx_bdst], GKS)):
                idx_index = indices[idx_n, idx_bdst, i]
                idx_group = idx_index // TSRC
                idx_tsrc = idx_index % TSRC
                values = np.ascontiguousarray(
                    v[idx_n * G + idx_group, idx_tsrc : idx_tsrc + block_size_k, :]
                )
                t = (
                    np.ascontiguousarray(
                        scores[:, i * block_size_k : (i + 1) * block_size_k]
                    )
                    @ values
                )
                output[
                    idx_n * G + idx_group,
                    idx_bdst * block_size_q : (idx_bdst + 1) * block_size_q,
                    :,
                ] += t

    return output


@torch.inference_mode()
def hip_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask_k: int = 512,
    block_size_q: int = 32,
    block_size_k: int = 1,
    block_size_k_group: int = 8,
    using_sliding_window: bool = True,
    sliding_window_size: int = 128,
    oracle_rep: bool = False,
    using_extend: bool = False,
    rope_cos: Optional[Tensor] = None,
    rope_sin: Optional[Tensor] = None,
    self_extend_neighboor_window: int = 1024,
    self_extend_group_size: int = 8,
    topk_head_group_size: int = 8,
):
    indices, ks = masking_iteration_draft(
        q,
        k,
        mask_k=mask_k,
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        block_size_k_group=block_size_k_group,
        oracle_rep=oracle_rep,
        sliding_window_size=sliding_window_size if using_sliding_window else 0,
        using_extend=using_extend,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        self_extend_neighboor_window=self_extend_neighboor_window,
        self_extend_group_size=self_extend_group_size,
        topk_head_group_size=topk_head_group_size,
    )

    print(indices[0, 0], ks)

    # N, TDST, HID = q.shape
    # _, TSRC, _ = k.shape
    # debug_mask = to_dense(
    #     indices.cpu().numpy(),
    #     ks.cpu().numpy(),
    #     None,
    #     N,
    #     TDST,
    #     TSRC * topk_head_group_size,
    #     block_size_q,
    #     block_size_k * block_size_k_group,
    # )
    # # print(debug_mask)
    # import matplotlib.pyplot as plt
    # plt.imshow(debug_mask[0])
    # plt.savefig('dummy.png', dpi=200)
    # print('saved dummy.png')

    context = block_sparse_attention(
        q.cpu().float().numpy(),
        k.cpu().float().numpy(),
        v.cpu().float().numpy(),
        indices.cpu().long().numpy(),
        ks.cpu().long().numpy(),
        block_size_q=block_size_q,
        block_size_k=block_size_k,
        mask_k=mask_k,
        topk_head_group_size=topk_head_group_size,
    )
    context = torch.tensor(context, device=q.device)

    return context, None
