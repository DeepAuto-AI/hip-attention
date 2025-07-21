import os
from typing import Optional

import matplotlib.pyplot as plt
import torch
import triton
import triton.language as tl
from torch import Tensor

from .attention_metadata import HiPAttentionArgs, HiPAttentionState, safe_stride
from .utils import capture

if os.getenv("HIP_DISABLE_AUTOTUNE", "0") == "1":
    configs = [
        triton.Config(
            {"BLOCK_TSRC": BLOCK_TSRC, "BLOCK_TDST": BLOCK_TDST},
            num_stages=s,
            num_warps=w,
        )
        for BLOCK_TSRC in [128]
        for BLOCK_TDST in [128]
        for s in [
            3,
        ]
        for w in [
            4,
        ]
    ]
else:
    configs = [
        triton.Config(
            {"BLOCK_TSRC": BLOCK_TSRC, "BLOCK_TDST": BLOCK_TDST},
            num_stages=s,
            num_warps=w,
        )
        for BLOCK_TSRC in [64, 128]
        for BLOCK_TDST in [64, 128]
        for s in [
            1,
            3,
            4,
        ]
        for w in [4, 8]
    ]


def keep(conf):
    BLOCK_TSRC = conf.kwargs["BLOCK_TSRC"]
    BLOCK_TDST = conf.kwargs["BLOCK_TDST"]
    return True


@triton.autotune(list(filter(keep, configs)), key=["HID", "USING_PAGED_CACHE"])
@triton.jit
def _sw_score_sample(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head_kv,
    stride_k_hid,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    USING_PAGED_CACHE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_head_kv,
    stride_k_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_tsrc,
    SCORES,
    stride_scores_bsz,
    stride_scores_head,
    stride_scores_tdst,
    window_size,
    T,
    HEAD,
    HEAD_KV,
    HID: tl.constexpr,
    BLOCK_TDST: tl.constexpr,
    BLOCK_TSRC: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    idx_head = pid % HEAD
    idx_head_kv = idx_head // (HEAD // HEAD_KV)
    pid = pid // HEAD

    idx_bsrc = pid % tl.cdiv(T, BLOCK_TSRC)
    pid = pid // tl.cdiv(T, BLOCK_TSRC)

    idx_tsrc_start = idx_bsrc * BLOCK_TSRC
    idx_tsrc = tl.arange(0, BLOCK_TSRC) + idx_tsrc_start
    mask_tsrc = idx_tsrc < T

    idx_bsz = pid
    idx_hid = tl.arange(0, HID)

    pos_tdst_start = tl.load(
        POS + idx_bsz * stride_pos_bsz + idx_tsrc_start * stride_pos_tdst,
    )

    pos_tsrc = tl.arange(0, BLOCK_TSRC) + pos_tdst_start

    if USING_PAGED_CACHE:
        tl.static_assert(USING_PAGED_CACHE)
        idx_page = tl.load(
            BLOCK_TABLE
            + idx_bsz * stride_block_table_bsz
            + pos_tsrc * stride_block_table_tsrc,
            mask=mask_tsrc,
        )
        keys = tl.load(
            K_CACHE
            + idx_page[None, :] * stride_k_cache_page
            + 0 * stride_k_cache_offset
            + idx_head_kv * stride_k_cache_head_kv
            + idx_hid[:, None] * stride_k_cache_hid,
            mask=mask_tsrc[None, :],
            other=0.0,
        )
    else:
        keys = tl.load(
            K
            + idx_bsz * stride_k_bsz
            + pos_tsrc[None, :] * stride_k_tsrc
            + idx_head_kv * stride_k_head_kv
            + idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc[None, :],
            other=0.0,
        )

    dot_dtype = (
        torch.float16 if Q.dtype.element_ty == tl.float8e5 else Q.dtype.element_ty
    )
    keys = keys.to(dot_dtype)

    acc = tl.zeros((BLOCK_TSRC,), dtype=tl.float32) + 42

    for i_start in range(0, tl.maximum(BLOCK_TDST, BLOCK_TSRC), BLOCK_TDST):
        idx_tdst = idx_tsrc_start + tl.arange(0, BLOCK_TDST) + i_start
        mask_tdst = idx_tdst < T

        pos_tdst = tl.load(
            POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
            mask=mask_tdst,
        )

        queries = tl.load(
            Q
            + idx_bsz * stride_q_bsz
            + idx_tdst[:, None] * stride_q_tdst
            + idx_head * stride_q_head
            + idx_hid[None, :] * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0,
        ).to(dot_dtype)

        scores = tl.dot(queries, keys)

        mask = pos_tdst[:, None] >= pos_tsrc[None, :]
        acc = acc + tl.sum(scores * mask, axis=0)

    for i_start in range(
        tl.maximum(BLOCK_TDST, BLOCK_TSRC), window_size + BLOCK_TSRC, BLOCK_TDST
    ):
        idx_tdst = idx_tsrc_start + tl.arange(0, BLOCK_TDST) + i_start
        mask_tdst = idx_tdst < T

        queries = tl.load(
            Q
            + idx_bsz * stride_q_bsz
            + idx_tdst[:, None] * stride_q_tdst
            + idx_head * stride_q_head
            + idx_hid[None, :] * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0,
        ).to(dot_dtype)

        scores = tl.dot(queries, keys)

        acc = acc + tl.sum(scores, axis=0)

    weight = tl.minimum(T - idx_tsrc, window_size)
    acc = acc / weight

    tl.store(
        SCORES
        + idx_bsz * stride_scores_bsz
        + idx_head * stride_scores_head
        + idx_tsrc * stride_scores_tdst,
        mask=mask_tsrc,
        value=acc,
    )


@capture
def landmark_sample(
    q: Tensor,
    k: Optional[Tensor],
    state: Optional[HiPAttentionState],
    args: HiPAttentionArgs,
    BSZ,
    HEAD,
    HEAD_KV,
    BDST,
    DEBUG,
    __logall_index,
):
    landmark_chunk = 512
    landmark_derope = False

    HID = q.shape[-1]

    __fused = True

    if __fused:
        q_for_landmark = (
            args.query_for_landmark if args.query_for_landmark is not None else q
        )
        position_ids_for_landmark = (
            args.position_ids_for_landmark
            if args.position_ids_for_landmark is not None
            else args.position_ids
        )
        TDST = q_for_landmark.shape[1]
        assert q_for_landmark.shape[0] == BSZ
        assert q_for_landmark.shape[2] == HEAD
        assert position_ids_for_landmark.shape[0] == BSZ
        assert position_ids_for_landmark.shape[1] == TDST

        _using_k = (not args.using_paged_cache) and (k is not None)
        _using_paged_k = args.using_paged_cache and (k is None)
        assert _using_k or _using_paged_k, f"todo {_using_k} or {_using_paged_k}"
        assert not landmark_derope, "todo"

        TDST_PADDED = (
            TDST
            if (TDST % landmark_chunk) == 0
            else TDST + (landmark_chunk - TDST % landmark_chunk)
        )

        k_cache = args.get_k_cache()
        if k_cache is not None:
            k_cache = k_cache[..., : q.shape[-1]]

        landmark_scores = torch.full(
            (BSZ, HEAD, TDST_PADDED),
            fill_value=float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

        # NOTE: TDST should divided by TSRC, because TSRC == TDST here (only handle new chunk)
        grid = lambda kwargs: (BSZ * HEAD * triton.cdiv(TDST, kwargs["BLOCK_TSRC"]),)

        _sw_score_sample[grid](
            q_for_landmark,
            *safe_stride(q_for_landmark, 4),
            k,
            *safe_stride(k, 4),
            position_ids_for_landmark,
            *safe_stride(position_ids_for_landmark, 2),
            args.using_paged_cache,
            k_cache,
            *safe_stride(k_cache, 4),
            args.block_table,
            *safe_stride(args.block_table, 2),
            landmark_scores,
            *safe_stride(landmark_scores, 3),
            landmark_chunk,
            TDST,
            HEAD,
            HEAD_KV,
            HID,
            # BLOCK_TDST,
            # BLOCK_TSRC,
        )

        landmark_scores[:, :, -min(landmark_chunk, 32) :].fill_(0)

        if state is not None:
            if args.block_table is not None:
                q_block_index = args.block_table.gather(
                    dim=1, index=position_ids_for_landmark
                )
            else:
                assert args.position_ids.shape[0] == 1
                q_block_index = args.position_ids[0]
            # sanity_check = q_block_index.amax().item()
            # assert sanity_check < state.landmark_scores.shape[0], f'{sanity_check=} < {state.landmark_scores.shape=}[0]'
            state.landmark_scores[q_block_index] = (
                landmark_scores[:, :, : q_for_landmark.shape[1]]
                .contiguous()
                .permute(0, 2, 1)
            )
            if args.block_table is not None:
                landmark_scores = state.landmark_scores[
                    args.block_table[
                        :,
                        : args.block_table.shape[1]
                        - (args.block_table.shape[1] % landmark_chunk),
                    ]
                ]
            else:
                assert k is not None
                assert k.shape[0] == 1
                landmark_scores = state.landmark_scores[None, : k.shape[1], :]
            landmark_scores = landmark_scores.permute(0, 2, 1)

        # print(q_for_landmark.shape, HEAD, HEAD_KV, TDST, triton.cdiv(TDST, BLOCK_TSRC), position_ids_for_landmark.shape)
        if DEBUG:
            plt.clf()
            plt.plot(
                landmark_scores[
                    0,
                    0,
                ]
                .cpu()
                .numpy()
            )
            plt.savefig("dummy_landmark.png")
    else:

        def pad_seq(t: torch.Tensor):
            if (t.shape[1] % landmark_chunk) == 0:
                return t
            pad = landmark_chunk - t.shape[1] % landmark_chunk
            return torch.nn.functional.pad(t, pad=(0, 0, 0, 0, 0, pad))

        def split_half(x: Tensor):
            HID = x.shape[-1]
            return x[..., : HID // 2], x[..., HID // 2 :]

        def merge_half(x: Tensor, y: Tensor):
            return torch.cat([x, y], dim=-1)

        def de_rope(vec: Tensor, cos: Tensor, sin: Tensor):
            c0, ch = split_half(cos)
            s0, sh = split_half(sin)
            vr0, vrh = split_half(vec)

            out0 = (vrh * s0 + vr0 * ch) / (c0 * ch + sh * s0 + 1e-20)
            outh = (out0 * c0 - vr0) / (s0 + 1e-20)
            out = merge_half(out0, outh)
            return out

        if state is not None:
            q_for_landmark = (
                args.query_for_landmark if args.query_for_landmark is not None else q
            )
            position_ids_for_landmark = (
                args.position_ids_for_landmark
                if args.position_ids_for_landmark is not None
                else args.position_ids
            )
            k_chunk = args.gather_extend_k_from_paged_cache(
                disable_gqa=False,
                gqa_q=q_for_landmark,
                position_ids=position_ids_for_landmark,
            )

            q_tp = pad_seq(q_for_landmark)
            TDST_PADDED = q_tp.shape[1]
            k_tp = pad_seq(k_chunk)
            TSRC_PADDED = k_tp.shape[1]
            assert TDST_PADDED == TSRC_PADDED, f"{TDST_PADDED} == {TSRC_PADDED}"

            if landmark_derope:
                padded_position_ids_for_landmark = pad_seq(
                    position_ids_for_landmark[:, :, None, None]
                )[:, :, 0, 0]
                q_tp = de_rope(
                    q_tp,
                    args.rope_cos[padded_position_ids_for_landmark, :][:, :, None, :],
                    args.rope_sin[padded_position_ids_for_landmark, :][:, :, None, :],
                )
                k_tp = de_rope(
                    k_tp,
                    args.rope_cos[padded_position_ids_for_landmark, :][:, :, None, :],
                    args.rope_sin[padded_position_ids_for_landmark, :][:, :, None, :],
                )

            q_tp = q_tp.permute(0, 2, 1, 3).reshape(
                BSZ, HEAD, TDST_PADDED // landmark_chunk, landmark_chunk, HID
            )
            k_tp = (
                k_tp.permute(0, 2, 3, 1)
                .reshape(
                    BSZ, HEAD_KV, HID, TSRC_PADDED // landmark_chunk, landmark_chunk
                )
                .permute(0, 1, 3, 2, 4)
                .repeat_interleave(dim=1, repeats=HEAD // HEAD_KV)
            )

            landmark_scores = torch.matmul(q_tp, k_tp)  # .to(torch.float32)

            # TODO Need to handle chunked prefill scenario
            idx_t = torch.arange(0, landmark_chunk, device=q_for_landmark.device)
            mask = idx_t[:, None] >= idx_t[None, :]
            landmark_scores = landmark_scores * mask[None, None, None, :, :]
            assert landmark_scores.shape == (
                BSZ,
                HEAD,
                TSRC_PADDED // landmark_chunk,
                landmark_chunk,
                landmark_chunk,
            )
            landmark_scores = (
                landmark_scores.sum(dim=3, dtype=torch.float32)
                / mask.int().sum(dim=0)[None, None, None, :]
            )
            landmark_scores = landmark_scores.view(BSZ, HEAD, TSRC_PADDED)
            landmark_scores[:, :, q_for_landmark.shape[1] :].fill_(float("-inf"))

            q_block_index = args.block_table.gather(
                dim=1, index=position_ids_for_landmark
            )
            # sanity_check = q_block_index.amax().item()
            # assert sanity_check < state.landmark_scores.shape[0], f'{sanity_check=} < {state.landmark_scores.shape=}[0]'
            state.landmark_scores[q_block_index] = (
                landmark_scores[:, :, : q_for_landmark.shape[1]]
                .contiguous()
                .permute(0, 2, 1)
            )
            landmark_scores = state.landmark_scores[
                args.block_table[
                    :,
                    : args.block_table.shape[1]
                    - (args.block_table.shape[1] % landmark_chunk),
                ]
            ]
            landmark_scores = landmark_scores.permute(0, 2, 1)
            # print('landmark score extended', args.layer_id)
        else:
            q_for_landmark = (
                args.query_for_landmark if args.query_for_landmark is not None else q
            )
            position_ids_for_landmark = (
                args.position_ids_for_landmark
                if args.position_ids_for_landmark is not None
                else args.position_ids
            )

            q_tp = pad_seq(q_for_landmark)
            TDST_PADDED = q_tp.shape[1]
            k_tp = pad_seq(k)
            TSRC_PADDED = k_tp.shape[1]
            assert TDST_PADDED == TSRC_PADDED

            if landmark_derope:
                padded_position_ids_for_landmark = pad_seq(
                    position_ids_for_landmark[:, :, None, None]
                )[:, :, 0, 0]
                q_tp = de_rope(
                    q_tp,
                    args.rope_cos[padded_position_ids_for_landmark, :][:, :, None, :],
                    args.rope_sin[padded_position_ids_for_landmark, :][:, :, None, :],
                )
                k_tp = de_rope(
                    k_tp,
                    args.rope_cos[padded_position_ids_for_landmark, :][:, :, None, :],
                    args.rope_sin[padded_position_ids_for_landmark, :][:, :, None, :],
                )

            q_tp = q_tp.permute(0, 2, 1, 3).reshape(
                BSZ, HEAD, TDST_PADDED // landmark_chunk, landmark_chunk, HID
            )
            k_tp = (
                k_tp.permute(0, 2, 3, 1)
                .reshape(
                    BSZ, HEAD_KV, HID, TSRC_PADDED // landmark_chunk, landmark_chunk
                )
                .permute(0, 1, 3, 2, 4)
                .repeat_interleave(dim=1, repeats=HEAD // HEAD_KV)
            )
            # print(q_tp.shape, k_tp.shape)
            landmark_scores = torch.matmul(q_tp, k_tp)  # .to(torch.float32)
            # TODO Need to handle chunked prefill scenario
            # idx_tdst = args.position_ids[0]
            idx_t = torch.arange(0, landmark_chunk, device=q.device)
            mask = idx_t[:, None] >= idx_t[None, :]
            landmark_scores = landmark_scores * mask[None, None, None, :, :]
            assert landmark_scores.shape == (
                BSZ,
                HEAD,
                TSRC_PADDED // landmark_chunk,
                landmark_chunk,
                landmark_chunk,
            )
            landmark_scores = (
                landmark_scores.sum(dim=3) / mask.int().sum(dim=0)[None, None, None, :]
            )
            landmark_scores = landmark_scores.view(BSZ, HEAD, TSRC_PADDED)
            landmark_scores[:, :, k.shape[1] :].fill_(float("-inf"))

    if DEBUG and (BDST > 1):
        os.makedirs("./cache/mask_log", exist_ok=True)
        t = landmark_scores[0, 0, :].cpu().numpy()
        plt.clf()
        plt.plot(t)
        plt.savefig(f"./cache/mask_log/{__logall_index}_landmark_scores.png")

    return landmark_scores
