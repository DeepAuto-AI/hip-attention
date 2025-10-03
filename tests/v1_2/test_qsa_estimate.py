import math
import os
from typing import Any, Tuple

import einx
import numpy as np
import torch
from flash_attn import flash_attn_func

from hip_attn.v1_2.attention_extend import dual_stage_quadratic_hip_attention

# from hip_research.utils.load_checkouts import load_checkouts
from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention
from hip_attn.v1_2.attention_metadata import HiPAttentionArgs, ScanStage
from hip_attn.v1_2.delta.apply_delta import apply_delta
from hip_attn.v1_2.query_sparse_attention import query_sparse_attention


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    n = x.size(2) // 2
    return torch.cat((-x[:, :, :n], x[:, :, n:]), dim=2)


def latency(fn: Any, n_sample: int = 10) -> float:
    elapsed = []
    for i in range(n_sample):
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        start.record()
        _ = fn()
        end.record()
        end.synchronize()
        if i > 3:
            elapsed.append(start.elapsed_time(end))
    return float(sum(elapsed) / len(elapsed))


def test() -> None:
    q_block, k_block, exact_k = 16, 64, 8
    seq = 4096 * 32
    device = 0
    window_size, sink_tokens = 2048, 64

    d = torch.load(
        "/data/ainl/library/hip-attention/cache/llama/qkvout.pth", map_location="cpu"
    )
    q, k, v, cos, sin = (
        d["q"].cuda(device),
        d["k"].cuda(device),
        d["v"].cuda(device),
        d["cos"].cuda(device),
        d["sin"].cuda(device),
    )
    print(f"after making q: {q.size()=}")

    b, h, s, d = k.size()
    k = (
        k.view(b, h, 1, s, d)
        .repeat(1, 1, q.size(1) // k.size(1), 1, 1)
        .view(b, -1, s, d)
    )
    v = (
        v.view(b, h, 1, s, d)
        .repeat(1, 1, q.size(1) // v.size(1), 1, 1)
        .view(b, -1, s, d)
    )

    q = q * cos[:, None] + rotate_half(q) * sin[:, None]
    k = k * cos[:, None] + rotate_half(k) * sin[:, None]

    qp = q[:, :, :seq]
    kp = k[:, :, :seq]
    vp = v[:, :, :seq]
    print(f"after making q: {qp.size()=}")

    qp, kp, vp = qp.contiguous(), kp.contiguous(), vp.contiguous()

    b, h, s, d = qp.size()
    mask = torch.arange(seq).view(1, seq).repeat(b, 1).cuda(device)
    mask = mask.view(b, seq // q_block, q_block)[:, :, 0]
    qp = qp.view(b, h, s // q_block, q_block, d)[:, :, :, 0]

    def qsa(
        K,
        online_topk_method,
        exact_k=None,
        reverse=True,
        k_block=k_block,
        return_bsa_indices=True,
        return_running_statistics=False,
        return_row_sums=False,
    ) -> Tuple[torch.Tensor, ...]:
        results = query_sparse_attention(
            q=qp,
            k=kp,
            v=vp,
            mask=mask,
            k_cache=None,
            v_cache=None,
            block_table=None,
            return_bsa_indices=return_bsa_indices,
            sm_scale=math.sqrt(1 / q.size(-1)),
            bsa_top_block_k=K,
            bsa_block_size_k=k_block,
            bsa_mask_sliding_window_size=window_size,
            bsa_mask_sink_token_size=sink_tokens,
            online_topk_method=online_topk_method,
            exact_k=exact_k,
            reverse_iter=reverse,
            return_running_statistics=return_running_statistics,
            threshold_refresh_interval=4,
            return_row_sums=return_row_sums,
        )

        return results

    o = flash_attn_func(
        q[:, :, :seq].transpose(1, 2),
        k[:, :, :seq].transpose(1, 2),
        v[:, :, :seq].transpose(1, 2),
        causal=True,
    )
    o = o.transpose(1, 2)

    # warmup burn-in. autotune has s dirty init so this is necessary right now
    K, SMALL_K = 64, 32
    out, (row_sums, row_sums_bsa), (block_idx, block_sums) = qsa(
        K, "tree", reverse=False, return_row_sums=True
    )
    out, (row_sums, row_sums_bsa), (block_idx, block_sums) = qsa(
        K, "tree", reverse=False, return_row_sums=True
    )
    gt_exp_sc = check_topk_selection(
        block_idx, qp, kp, vp, row_sums, k_block, math.sqrt(1 / q.size(-1))
    )
    print(f"{block_idx=}")

    torch.save(
        {
            "row_sums": row_sums,
            "row_sums_bsa": row_sums_bsa,
            "block_idx": block_idx,
            "block_sums": block_sums,
        },
        "row_sums_exact_tree.pth",
    )

    out, (row_sums, row_sums_bsa), (block_idx, _) = qsa(
        SMALL_K, "tree", reverse=True, return_row_sums=True
    )
    out, (row_sums, row_sums_bsa), (block_idx, _) = qsa(
        SMALL_K, "tree", reverse=True, return_row_sums=True
    )
    small_exp_sc = check_topk_selection(
        block_idx, qp, kp, vp, row_sums, k_block, math.sqrt(1 / q.size(-1))
    )

    overestimate_K = 128
    test_k_block = 64
    out, (row_sums, row_sums_bsa), (block_idx, block_sums) = qsa(
        overestimate_K,
        "online",
        exact_k=exact_k,
        k_block=test_k_block,
        reverse=False,
        return_row_sums=True,
    )
    out, (row_sums, row_sums_bsa), (block_idx, block_sums) = qsa(
        overestimate_K,
        "online",
        exact_k=exact_k,
        k_block=test_k_block,
        reverse=False,
        return_row_sums=True,
    )
    est_exp_sc = check_topk_selection(
        block_idx, qp, kp, vp, row_sums, test_k_block, math.sqrt(1 / q.size(-1))
    )
    print(f"{block_idx=}")
    print(f"{row_sums.size()=} {row_sums=}")
    recall_rates = (est_exp_sc / (gt_exp_sc + 1e-6))[gt_exp_sc > 0]
    recall_rates_small = (est_exp_sc / (small_exp_sc + 1e-6))[small_exp_sc > 0]
    print(f"{recall_rates.mean()=} ({K=})")
    print(f"{recall_rates_small.mean()=} ({SMALL_K=})")

    torch.save(
        {
            "row_sums": row_sums,
            "row_sums_bsa": row_sums_bsa,
            "block_idx": block_idx,
            "block_sums": block_sums,
        },
        "row_sums_128.pth",
    )

    TEST_LATENCY = os.getenv("TEST_LATENCY", "0") == "1"
    if TEST_LATENCY:
        flash_latency = latency(
            lambda: flash_attn_func(
                q[:, :, :seq].transpose(1, 2),
                k[:, :, :seq].transpose(1, 2),
                v[:, :, :seq].transpose(1, 2),
                causal=True,
            )
        )
        print(f"flash: {flash_latency:.2f} ms took")

        # 6.1 test forward/reverse latency with topk estimation
        LATENCY_LOWER, LATENCY_UPPER = 4, 10
        for topk in [2**i for i in range(LATENCY_LOWER, LATENCY_UPPER)]:
            fwd_latency = latency(
                lambda: qsa(
                    topk,
                    return_bsa_indices=True,
                    reverse=False,
                    online_topk_method="online",
                    exact_k=exact_k,
                )
            )
            rev_latency = latency(
                lambda: qsa(
                    topk,
                    return_bsa_indices=True,
                    reverse=True,
                    online_topk_method="online",
                    exact_k=exact_k,
                )
            )
            print(f"estimate top-k latency {topk=} {fwd_latency=} {rev_latency=}")

        # 6.1 test forward/reverse latency with topk estimation
        LATENCY_LOWER, LATENCY_UPPER = 4, 9
        for topk in [2**i for i in range(LATENCY_LOWER, LATENCY_UPPER)]:
            fwd_latency = latency(
                lambda: qsa(
                    topk,
                    return_bsa_indices=True,
                    reverse=False,
                    online_topk_method="tree",
                    exact_k=exact_k,
                )
            )
            rev_latency = latency(
                lambda: qsa(
                    topk,
                    return_bsa_indices=True,
                    reverse=True,
                    online_topk_method="tree",
                    exact_k=exact_k,
                )
            )
            print(f"estimate+tree top-k latency {topk=} {fwd_latency=} {rev_latency=}")

        # 6.2 test forward/reverse latency with naive topk
        for topk in [2**i for i in range(LATENCY_LOWER, LATENCY_UPPER)]:
            fwd_latency = latency(
                lambda: qsa(
                    topk,
                    return_bsa_indices=True,
                    reverse=False,
                    online_topk_method="online",
                )
            )
            rev_latency = latency(
                lambda: qsa(
                    topk,
                    return_bsa_indices=True,
                    reverse=True,
                    online_topk_method="online",
                )
            )
            print(f"naive top-k latency {topk=} {fwd_latency=} {rev_latency=}")

        # 6.3 test forward/reverse latency with heap
        for topk in [2**i for i in range(LATENCY_LOWER, LATENCY_UPPER)]:
            fwd_latency = latency(
                lambda: qsa(
                    topk,
                    return_bsa_indices=True,
                    reverse=False,
                    online_topk_method="tree",
                )
            )
            rev_latency = latency(
                lambda: qsa(
                    topk,
                    return_bsa_indices=True,
                    reverse=True,
                    online_topk_method="tree",
                )
            )
            print(f"heap top-k latency {topk=} {fwd_latency=} {rev_latency=}")


def check_topk_selection(
    block_idx,  # (BSZ, HEAD, TDST, BSA_K)
    qq,  # (BSZ, HEAD, TDST, D)
    k,  # (BSZ, HEAD, TSRC, D)
    v,  # (BSZ, HEAD, TSRC, D)
    row_sums,  # (BSZ, HEAD, TDST)
    k_block: int,
    sm_scale: float,
):
    r = slice(-1024, None)
    block_idxi = block_idx[0, 0, r]
    qqi = qq[0, 0, r] * sm_scale
    ki = k[0, 0]
    row_sums = row_sums[0, 0, r]

    mask = block_idxi < 987654321  # q, K
    block_idxi = torch.where(mask, block_idxi, 0)
    ki = einx.get_at(
        "... [bk] B d, ... q K -> ... q K B d",
        einx.rearrange("... (bk B) d -> ... bk B d", ki, B=k_block),
        block_idxi // k_block,
    )
    exp_sc = einx.dot("... q d, ... q K B d -> ... q K B", qqi, ki).exp()
    exp_sc = torch.where(mask[:, :, None], exp_sc, 0.0)
    exp_sc = exp_sc.sum(-1)  # (TDST, K_BSA)
    return exp_sc.sum(-1) / row_sums


if __name__ == "__main__":
    test()
