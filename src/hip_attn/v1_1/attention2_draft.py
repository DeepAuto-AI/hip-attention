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

import math
from typing import Optional, Union

import numba
import numpy as np
import torch
from numpy import ndarray as NdArray
from torch import Tensor

from hip_attn.v1_0.attention1_block_gpu import (
    calc_prob_return_context as block_sparse_flash_attention,
)
from hip_attn.v1_0.attention1_block_gpu import to_dense


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
    mask_k_init: int,
    block_size_q: int,
    block_size_k: int,
    block_size_k_group: int,
    oracle_rep: bool,
    sifter_shift: int,
    has_initial_indices: bool,
    has_initial_ks: bool,
    initial_group_sizes: Optional[NdArray],
    sliding_window_size: int,
    using_extend: bool,
    rope_cos: Optional[Tensor],
    rope_sin: Optional[Tensor],
    idx_tdst: int,
    TDST: int,
    self_extend_neighboor_window: int,
    self_extend_group_size: int,
    topk_head_group_size: int,
) -> int:
    mask_block_k = cdiv(mask_k, block_size_k)
    TSRC = max(0, k.shape[0] - sliding_window_size)
    BSRC = cdiv(TSRC, block_size_k)

    if TSRC <= mask_k:
        for i in range(len(indices)):
            indices[i] = i * block_size_k
        return BSRC
    else:
        # initialize
        if has_initial_indices or has_initial_ks:
            assert has_initial_indices
            assert has_initial_ks
            assert initial_group_sizes is not None

            group_sizes = initial_group_sizes.copy()
            group_size = np.max(group_sizes)
        else:
            group_sizes = np.zeros_like(indices)
            for i in range(len(indices)):
                indices[i] = int(BSRC / mask_block_k * i)
                group_sizes[i] = int(BSRC / mask_block_k * (i + 1)) - int(
                    BSRC / mask_block_k * i
                )
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
            dupped_indices_max = np.max(dupped_indices)

            scores = np.zeros_like(dupped_indices, dtype=np.float32)
            for i in range(len(scores)):
                if not dupped_mask[i]:
                    continue
                idx_tsrc = dupped_indices[i] * block_size_k
                queries = q[1::2, :]
                if block_size_k_group > 1:
                    assert not oracle_rep
                    assert not using_extend
                    keys_min = k[idx_tsrc : idx_tsrc + block_size_k, : q.shape[-1]]
                    keys_max = k[idx_tsrc : idx_tsrc + block_size_k, q.shape[-1] :]
                    t_1 = queries @ np.ascontiguousarray(keys_min.T)
                    t_2 = queries @ np.ascontiguousarray(keys_max.T)
                    scores[i] = max(t_1.max(), t_2.max())
                else:
                    if not oracle_rep:
                        keys = k[idx_tsrc : idx_tsrc + block_size_k, :].copy()
                        queries = queries.copy()

                        if using_extend:
                            for j in range(len(keys)):
                                old_idx = idx_tsrc + j

                                # StreamingLLM (not working well)
                                # new_idx = i * block_size_k + j

                                # Self Extend (working great)
                                if idx_tsrc >= (
                                    idx_tdst - self_extend_neighboor_window
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
                                old_idx = idx_tdst + j + TSRC - TDST

                                # new_idx = len(scores) * block_size_k - block_size_q + j

                                if idx_tsrc >= (
                                    idx_tdst - self_extend_neighboor_window
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
                        for shift in range(dupped_group_sizes[i]):
                            keys = k[
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
            # print(scores)
            scores[:] += -32000.0 * ~dupped_mask

            # select
            if stage == 0:
                topk_indices = np.argsort(-scores)[
                    mask_block_k * sifter_shift : mask_block_k * sifter_shift
                    + mask_block_k
                ]
            else:
                topk_indices = np.argsort(-scores)[:mask_block_k]
            indices[:] = dupped_indices[topk_indices]
            group_sizes[:] = dupped_group_sizes[topk_indices]
            # print(group_size, indices, topk_indices, group_sizes)

            group_size = group_size / 2
            stage += 1
        indices[:] = np.sort(indices) * block_size_k
        return mask_block_k


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
    mask_k_init: int,
    block_size_q: int,
    block_size_k: int,
    block_size_k_group: int,
    oracle_rep: bool,
    sifter_shift: int,
    has_initial_indices: bool,
    has_initial_ks: bool,
    initial_group_sizes: Optional[NdArray],
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

    N, TDST, _ = q.shape
    _, TSRC, _ = k.shape

    for idx_n in numba.prange(N):
        for idx_tdst in numba.prange(cdiv(TDST, block_size_q)):
            _initial_group_sizes = (
                initial_group_sizes[idx_n, idx_tdst, :]
                if initial_group_sizes is not None
                else None
            )
            final_k = masking_iteration_draft_numba_kernel(
                q[idx_n, idx_tdst * block_size_q : (idx_tdst + 1) * block_size_q, :],
                k[
                    idx_n,
                    : ((idx_tdst + 1) * block_size_q + TSRC * block_size_k_group - TDST)
                    // block_size_k_group,
                    :,
                ],
                indices[idx_n, idx_tdst, :],
                mask_k=mask_k,
                mask_k_init=mask_k_init,
                block_size_q=block_size_q,
                block_size_k=block_size_k,
                block_size_k_group=block_size_k_group,
                oracle_rep=oracle_rep,
                sifter_shift=sifter_shift,
                has_initial_indices=has_initial_indices,
                has_initial_ks=has_initial_ks,
                initial_group_sizes=_initial_group_sizes,
                sliding_window_size=sliding_window_size,
                using_extend=using_extend,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                idx_tdst=idx_tdst,
                TDST=TDST,
                self_extend_neighboor_window=self_extend_neighboor_window,
                self_extend_group_size=self_extend_group_size,
                topk_head_group_size=topk_head_group_size,
            )
            ks[idx_n, idx_tdst] = final_k


def masking_iteration_draft_sifter(
    q: Tensor,
    k: Tensor,
    mask_k: int,
    block_size_q: int,
    block_size_k: int,
    block_size_k_group: int,
    oracle_rep: bool,
    sifter_shift: int,
    initial_indices: Optional[Union[Tensor, NdArray]],
    initial_ks: Optional[Union[Tensor, NdArray]],
    initial_group_sizes: Optional[Union[Tensor, NdArray]],
    sliding_window_size: int,
    using_extend: bool,
    rope_cos: Optional[Tensor],
    rope_sin: Optional[Tensor],
    self_extend_neighboor_window: int,
    self_extend_group_size: int,
    topk_head_group_size: int,
):
    mask_k_init = mask_k // 2

    device = q.device
    q = q.cpu().float().numpy()
    k = k.cpu().float().numpy()
    N, TSRC, HID = k.shape
    if block_size_k_group > 1:
        k_group = np.reshape(
            k, (N, TSRC // block_size_k_group, block_size_k_group, HID)
        )
        k_group_min = np.min(k_group, axis=-2)
        k_group_max = np.max(k_group, axis=-2)
        k = np.concatenate([k_group_min, k_group_max], axis=-1)

    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape

    if initial_indices is None:
        indices = np.zeros(
            (
                N,
                cdiv(TDST, block_size_q),
                cdiv(mask_k, block_size_k * block_size_k_group),
            ),
            dtype=np.int32,
        )
    else:
        indices = (
            initial_indices
            if isinstance(initial_indices, NdArray)
            else initial_indices.cpu().numpy()
        )

    if initial_ks is None:
        ks = np.zeros(
            (
                N,
                cdiv(TDST, block_size_q),
            ),
            dtype=np.int32,
        )
    else:
        ks = initial_ks if isinstance(initial_ks, NdArray) else initial_ks.cpu().numpy()

    if rope_cos is not None:
        assert rope_cos.ndim == 2
        assert rope_cos.shape[-1] == q.shape[-1]
        rope_cos = rope_cos.cpu().float().numpy()

    if rope_sin is not None:
        assert rope_sin.ndim == 2
        assert rope_sin.shape[-1] == q.shape[-1]
        rope_sin = rope_sin.cpu().float().numpy()

    masking_iteration_draft_numba(
        q,
        k,
        indices,
        ks,
        mask_k // block_size_k_group,
        mask_k_init // block_size_k_group,
        block_size_q,
        block_size_k,
        block_size_k_group,
        oracle_rep,
        sifter_shift,
        initial_indices is not None,
        initial_ks is not None,
        (
            (
                initial_group_sizes
                if isinstance(initial_group_sizes, NdArray)
                else initial_group_sizes.cpu().numpy()
            )
            if initial_group_sizes is not None
            else None
        ),
        sliding_window_size,
        using_extend,
        rope_cos,
        rope_sin,
        self_extend_neighboor_window,
        self_extend_group_size,
        topk_head_group_size,
    )

    indices *= block_size_k_group

    return indices, ks


@numba.njit
def masking_iteration_initialize_mask(
    # in
    indices: NdArray,
    ks: NdArray,
    # out
    initial_indices: NdArray,
    initial_ks: NdArray,
    initial_group_sizes: NdArray,
    TDST: int,
    TSRC: int,
    block_size_k: int,
):
    N, BDST, MASK_BLOCK_K = indices.shape
    for idx_n in range(N):
        for idx_bdst in range(BDST):
            last_index = -1000
            group_size = 0
            j = 0
            for i in range(MASK_BLOCK_K):
                curr_index = indices[idx_n, idx_bdst, i]
                if last_index + block_size_k == curr_index:
                    group_size += block_size_k
                else:
                    last_index = max(0, last_index)
                    initial_indices[idx_n, idx_bdst, j] = last_index // block_size_k
                    initial_group_sizes[idx_n, idx_bdst, j] = group_size // block_size_k
                    if initial_group_sizes[idx_n, idx_bdst, j] > 0:
                        j += 1
                    initial_indices[idx_n, idx_bdst, j] = (
                        last_index + group_size
                    ) // block_size_k
                    initial_group_sizes[idx_n, idx_bdst, j] = (
                        max(0, curr_index - (last_index + group_size)) // block_size_k
                    )
                    if initial_group_sizes[idx_n, idx_bdst, j] > 0:
                        j += 1

                    last_index = curr_index
                    group_size = 0
            initial_ks[idx_n, idx_bdst] = j


def masking_iteration_draft(
    q: Tensor,
    k: Tensor,
    mask_k: int,
    block_size_q: int,
    block_size_k: int,
    block_size_k_group: int,
    oracle_rep: bool,
    num_sifter: int,
    recursive_sifter: bool,
    sliding_window_size: int,
    using_extend: bool,
    rope_cos: Optional[Tensor],
    rope_sin: Optional[Tensor],
    self_extend_neighboor_window: int,
    self_extend_group_size: int,
    topk_head_group_size: int,
):
    device = q.device

    if num_sifter == 1:
        indices, ks = masking_iteration_draft_sifter(
            q,
            k,
            mask_k,
            block_size_q,
            block_size_k,
            block_size_k_group,
            oracle_rep,
            0,
            None,
            None,
            None,
            sliding_window_size,
            using_extend,
            rope_cos,
            rope_sin,
            self_extend_neighboor_window,
            self_extend_group_size,
            topk_head_group_size,
        )
    else:
        if recursive_sifter:
            indices, ks = masking_iteration_draft_sifter(
                q,
                k,
                mask_k // 4,
                block_size_q,
                block_size_k,
                block_size_k_group,
                oracle_rep,
                0,
                None,
                None,
                None,
                sliding_window_size,
                using_extend,
                rope_cos,
                rope_sin,
                self_extend_neighboor_window,
                self_extend_group_size,
                topk_head_group_size,
            )

            # N, TDST, HID = q.shape
            # _, TSRC, _ = k.shape
            # debug_mask = to_dense(
            #     indices,
            #     ks,
            #     None,
            #     N, TDST, TSRC, block_size_q, block_size_k * block_size_k_group,
            # )
            # # print(debug_mask)
            # import matplotlib.pyplot as plt
            # plt.imshow(debug_mask[0])
            # plt.savefig('dummy1.png')
            # print('saved dummy.png')

            N, BDST, MASK_BLOCK_K = indices.shape

            initial_indices = np.zeros((N, BDST, MASK_BLOCK_K * 2), dtype=np.int32)
            initial_ks = np.zeros((N, BDST), dtype=np.int32)
            initial_group_sizes = np.zeros((N, BDST, MASK_BLOCK_K * 2), dtype=np.int32)

            masking_iteration_initialize_mask(
                indices,
                ks,
                initial_indices,
                initial_ks,
                initial_group_sizes,
                q.shape[1],
                k.shape[1],
                block_size_k,
            )

            # print(initial_indices[0, -1])
            # print(initial_group_sizes[0, -1])
            # print(initial_ks[0, -1])

            indices, ks = masking_iteration_draft_sifter(
                q,
                k,
                mask_k // 2,
                block_size_q,
                block_size_k,
                block_size_k_group,
                oracle_rep,
                0,
                initial_indices,
                initial_ks,
                initial_group_sizes,
                sliding_window_size,
                using_extend,
                rope_cos,
                rope_sin,
                self_extend_neighboor_window,
                self_extend_group_size,
                topk_head_group_size,
            )

            # N, TDST, HID = q.shape
            # _, TSRC, _ = k.shape
            # debug_mask = to_dense(
            #     indices,
            #     ks,
            #     None,
            #     N, TDST, TSRC, block_size_q, block_size_k * block_size_k_group,
            # )
            # # print(debug_mask)
            # import matplotlib.pyplot as plt
            # plt.imshow(debug_mask[0])
            # plt.savefig('dummy2.png')
            # print('saved dummy.png')

            N, BDST, MASK_BLOCK_K = indices.shape

            initial_indices = np.zeros((N, BDST, MASK_BLOCK_K * 2), dtype=np.int32)
            initial_ks = np.zeros((N, BDST), dtype=np.int32)
            initial_group_sizes = np.zeros((N, BDST, MASK_BLOCK_K * 2), dtype=np.int32)

            masking_iteration_initialize_mask(
                indices,
                ks,
                initial_indices,
                initial_ks,
                initial_group_sizes,
                q.shape[1],
                k.shape[1],
                block_size_k,
            )

            # print(initial_indices[0, -1])
            # print(initial_group_sizes[0, -1])
            # print(initial_ks[0, -1])

            indices, ks = masking_iteration_draft_sifter(
                q,
                k,
                mask_k,
                block_size_q,
                block_size_k,
                block_size_k_group,
                oracle_rep,
                0,
                initial_indices,
                initial_ks,
                initial_group_sizes,
                sliding_window_size,
                using_extend,
                rope_cos,
                rope_sin,
                self_extend_neighboor_window,
                self_extend_group_size,
                topk_head_group_size,
            )

            # N, TDST, HID = q.shape
            # _, TSRC, _ = k.shape
            # debug_mask = to_dense(
            #     indices,
            #     ks,
            #     None,
            #     N, TDST, TSRC, block_size_q, block_size_k * block_size_k_group,
            # )
            # # print(debug_mask)
            # import matplotlib.pyplot as plt
            # plt.imshow(debug_mask[0])
            # plt.savefig('dummy3.png')
            # print('saved dummy.png')
        else:
            indices_sifters = []
            ks_sifters = 0
            for idx_sifter in range(num_sifter):
                indices, ks = masking_iteration_draft_sifter(
                    q,
                    k,
                    mask_k // num_sifter,
                    block_size_q,
                    block_size_k,
                    block_size_k_group,
                    oracle_rep,
                    idx_sifter,
                    None,
                    None,
                    None,
                    sliding_window_size,
                    using_extend,
                    rope_cos,
                    rope_sin,
                    self_extend_neighboor_window,
                    self_extend_group_size,
                    topk_head_group_size,
                )
                indices_sifters.append(indices)
                ks_sifters += ks
            indices = np.concatenate(indices_sifters, axis=-1)
            indices = np.sort(indices, axis=-1)
            ks = ks_sifters

    return (
        torch.tensor(indices, device=device),
        torch.tensor(ks, device=device),
    )


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
    num_sifter: int = 1,
    recursive_sifter: bool = False,
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
        num_sifter=num_sifter,
        recursive_sifter=recursive_sifter,
        sliding_window_size=sliding_window_size if using_sliding_window else 0,
        using_extend=using_extend,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        self_extend_neighboor_window=self_extend_neighboor_window,
        self_extend_group_size=self_extend_group_size,
        topk_head_group_size=topk_head_group_size,
    )

    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape
    debug_mask = to_dense(
        indices.cpu().numpy(),
        ks.cpu().numpy(),
        None,
        N,
        TDST,
        TSRC,
        block_size_q,
        block_size_k * block_size_k_group,
    )
    # print(debug_mask)
    import matplotlib.pyplot as plt

    plt.imshow(debug_mask[0])
    plt.savefig("dummy.png")
    print("saved dummy.png")

    context = block_sparse_flash_attention(
        q,
        k,
        v,
        attention_mask=None,
        indices=indices,
        ks=ks,
        IS_CAUSAL=True,
        KV_REPEAT_INTERLEAVE=1,
        BLOCK_SIZE_Q=block_size_q,
        BLOCK_SIZE_K=block_size_k * block_size_k_group,
        USING_SLIDING_WINDOW=using_sliding_window,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        ROPE_METHOD="none",
        ROPE_COS=None,
        ROPE_SIN=None,
        POSITION_IDS=None,
        SELF_EXTEND_SCALE=1,
        SELF_EXTEND_WINDOW=1,
    )

    return context, None
