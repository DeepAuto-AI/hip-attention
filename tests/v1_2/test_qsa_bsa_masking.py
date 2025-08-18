import math
import os

# from hip_research.utils.load_checkouts import load_checkouts
from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention
from hip_attn.v1_2.attention_extend import dual_stage_quadratic_hip_attention
from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention
from hip_attn.v1_2.attention_metadata import HiPAttentionArgs, ScanStage
from hip_attn.v1_2.delta.apply_delta import apply_delta
import math
from typing import Any, Tuple

from flash_attn import flash_attn_func
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

def test_qsa() -> None:
    q_block, k_block = 32, 32
    seq = 4096 * 32
    device = 0
    window_size, sink_tokens = 2048, 64

    def qsa(
        q, k, v,
        heap=False,
        return_bsa_indices=False,
        reverse=False,
        K=64,
        mask=None,
    ) -> Tuple[torch.Tensor, ...]:
        bsa_idx, block_sums = None, None
        if return_bsa_indices:
            out, (bsa_idx, block_sums) = query_sparse_attention(
                q=q,
                k=k,
                v=v,
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
                bsa_heap=heap,
                reverse_iter=reverse,
            )
            return out, bsa_idx, block_sums

        out = query_sparse_attention(
            q=q, k=k, v=v,
            mask=mask, k_cache=None, v_cache=None,
            block_table=None, sm_scale=math.sqrt(1 / q.size(-1)),
        )
        return out, None, None

    path = "/home/jeff/python/delta2/checkout/qkvout.pth"
    d = torch.load(path)

    q, k, v, cos, sin = d["q"].cuda(device), d["k"].cuda(device), d["v"].cuda(device), d["cos"].cuda(device), d["sin"].cuda(device)
    q, k, v, cos, sin = q[:, :, :seq], k[:, :, :seq], v[:, :, :seq], cos[:, :seq], sin[:, :seq]
    b, h, s, d = q.size()
    k = k.view(b, -1, 1, s, d).repeat(1, 1, h // k.size(1), 1, 1).reshape(b, h, s, d)
    v = v.view(b, -1, 1, s, d).repeat(1, 1, h // v.size(1), 1, 1).reshape(b, h, s, d)
    print(f"after making q: {q.size()=} {k.size()=} {v.size()=}")
        
    q = q * cos[:, None] + rotate_half(q) * sin[:, None]
    k = k * cos[:, None] + rotate_half(k) * sin[:, None]
    
    # 1. test forward/reverse is equivalent to flash
    s = 2048
    o = flash_attn_func(q[:, :, :s].transpose(1, 2), k[:, :, :s].transpose(1, 2), v[:, :, :s].transpose(1, 2), causal=True)
    o = o.transpose(1, 2)

    
    mask = torch.arange(seq).view(1, seq).repeat(b, 1).cuda(device)
    out, _, _ = qsa(q[:, :, :s], k[:, :, :s], v[:, :, :s], return_bsa_indices=False, mask=mask)
    diff = (o - out).abs()
    assert diff.mean() < 1e-4, f"{diff.mean()=}"
    assert diff.amax() < 1e-1, f"{diff.mean()=}" # due to bfloat15 precision
    print("flash/qsa fwd equivalent ok!")

    out, _, _ = qsa(q[:, :, :s], k[:, :, :s], v[:, :, :s], return_bsa_indices=False, reverse=True, mask=mask)
    diff = (o - out).abs()
    assert diff.mean() < 1e-4, f"{diff.mean()=}"
    assert diff.amax() < 1e-1, f"{diff.mean()=}" # due to bfloat15 precision
    print("flash/qsa rev equivalent ok!")

    mask = mask.view(b, seq // q_block, q_block)[:, :, 0] 
    print(f"{mask.size()=} {q.size()=} {seq=} {q_block=}")
    qq = q.view(b, h, seq // q_block, q_block, d)[:, :, :, 0]

    # 2. test forward/reverse gives same indices scan top-k. One burn in for safety
    _, bsa_idx_fwd, _ = qsa(qq, k, v, return_bsa_indices=True, reverse=False, mask=mask)
    _, bsa_idx_rev, _ = qsa(qq, k, v, return_bsa_indices=True, reverse=True, mask=mask)

    _, bsa_idx_fwd, _ = qsa(qq, k, v, return_bsa_indices=True, reverse=False, mask=mask)
    _, bsa_idx_rev, _ = qsa(qq, k, v, return_bsa_indices=True, reverse=True, mask=mask)
    rand_idx = torch.randperm(bsa_idx_fwd.size(2))[:4]
    eq = bsa_idx_fwd[0, 0, rand_idx].unsqueeze(-1) == bsa_idx_rev[0, 0, rand_idx].unsqueeze(-2)
    eq = eq.sum(-1).sum(-1)
    print(f"linear scan fwd/rev indices match: {eq=}")

    # 3. test forward/reverse gives same indices winner tree top-k. One burn in for safety
    _, bsa_idx_fwd, _ = qsa(qq, k, v, heap=True, return_bsa_indices=True, reverse=False, mask=mask)
    _, bsa_idx_rev, _ = qsa(qq, k, v, heap=True, return_bsa_indices=True, reverse=True, mask=mask)

    _, bsa_idx_fwd, _ = qsa(qq, k, v, heap=True, return_bsa_indices=True, reverse=False, mask=mask)
    _, bsa_idx_rev, _ = qsa(qq, k, v, heap=True, return_bsa_indices=True, reverse=True, mask=mask)
    rand_idx = torch.randperm(bsa_idx_fwd.size(2))[:4]
    eq = bsa_idx_fwd[0, 0, rand_idx].unsqueeze(-1) == bsa_idx_rev[0, 0, rand_idx].unsqueeze(-2)
    eq = eq.sum(-1).sum(-1)
    print(f"winner tree fwd/rev indices match {eq=}")

    # 4. test linear scan and heap give same indices. One burn in for warmup
    _, bsa_idx, _ = qsa(qq, k, v, return_bsa_indices=True, reverse=True, K=16, mask=mask)
    _, bsa_idx_tree, _ = qsa(qq, k, v, heap=True, return_bsa_indices=True, reverse=True, K=16, mask=mask)

    _, bsa_idx, _ = qsa(qq, k, v, return_bsa_indices=True, reverse=True, K=16, mask=mask)
    _, bsa_idx_tree, _ = qsa(qq, k, v, heap=True, return_bsa_indices=True, reverse=True, K=16, mask=mask)

    rand_idx = torch.randperm(bsa_idx.size(2))[:4]
    eq = bsa_idx[0, 0, rand_idx].unsqueeze(-1) == bsa_idx_tree[0, 0, rand_idx].unsqueeze(-2)
    eq = eq.sum(-1).sum(-1)
    print(f"heap and plain returned same indices: {eq=}")
    print("winner tree / linear top-k indices ok! (should be 16 unless a very early block was selected)")
    print(f"selected blocks: {rand_idx=}")

    # 5. test forward/reverse latency with linear scan top-k
    for topk in [2**i for i in range(4, 6)]:
        fwd_latency = latency(lambda: qsa(qq, k, v, return_bsa_indices=True, reverse=False, mask=mask, K=topk))
        rev_latency = latency(lambda: qsa(qq, k, v, return_bsa_indices=True, reverse=True, mask=mask, K=topk))
        print(f"online top-k latency {topk=} {fwd_latency=} {rev_latency=}")

    # 6. test forward/reverse latency with winner tree top-k
    for topk in [2**i for i in range(4, 6)]:
        fwd_latency = latency(lambda: qsa(qq, k, v, heap=True, return_bsa_indices=True, reverse=False, mask=mask, K=topk))
        rev_latency = latency(lambda: qsa(qq, k, v, heap=True, return_bsa_indices=True, reverse=True, mask=mask, K=topk))
        print(f"winner tree top-k latency {topk=} {fwd_latency=} {rev_latency=}")

    # 7. test no-bsa latency
    fwd_latency = latency(lambda: qsa(qq, k, v, return_bsa_indices=False, reverse=False, mask=mask))
    print(f"no bsa indices latency {fwd_latency=}")

    # 8. test flash attention latency
    latency_flash = latency(lambda: flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True))
    print(f"flash attn: {latency_flash:.2f}")


def test_with_hip_bsa() -> None:
    q_block, k_block = 32, 32
    seq = 4096 * 32
    device = 0
    K = 64
    window_size, sink_tokens = 2048, 64

    path = "/home/jeff/python/delta2/checkout/qkvout.pth"
    d = torch.load(path)

    q, k, v, cos, sin = d["q"].cuda(device), d["k"].cuda(device), d["v"].cuda(device), d["cos"].cuda(device), d["sin"].cuda(device)
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

    access_counter = torch.zeros(b, h, seq, dtype=torch.long, device=q.device)
    cache_miss_counter = torch.zeros(b, h, seq, dtype=torch.long, device=q.device)
    seq_lens = torch.arange(1, seq + 1, dtype=torch.long, device=q.device)[None, :]

    args = HiPAttentionArgs(
        block_size_q=q_block,
        block_size_k=k_block,
        position_ids=None,
        sink_token_size=sink_tokens,
        sliding_window_size=window_size,
        logit_softcap=None,
        rope_range=[0, d],
        using_extend=False,
    )

    def qsa(heap=False, reverse=True) -> Tuple[torch.Tensor, ...]:
        out, (bsa_idx, block_sums) = query_sparse_attention(
            q=qp,
            k=kp,
            v=vp,
            mask=mask,
            k_cache=None,
            v_cache=None,
            block_table=None, 
            return_bsa_indices=True,
            sm_scale=math.sqrt(1 / q.size(-1)),
            bsa_top_block_k=K,
            bsa_block_size_k=k_block,
            bsa_mask_sliding_window_size=window_size,
            bsa_mask_sink_token_size=sink_tokens,
            bsa_heap=heap,
            reverse_iter=reverse,
        )

        ks = (bsa_idx < 987654321).sum(dim=-1)
        ks = ks.view(b * h, seq // q_block, 1).repeat(1, 1, q_block).reshape(b * h, seq)
        ks_count = ks.unsqueeze(-1)
        ks_start_end = torch.nn.functional.pad(ks_count, (1, 0), "constant", 0)

        bsa_out = block_sparse_attention(
            q[:, :, :seq].transpose(1, 2) * math.sqrt(1 / q.size(-1)),
            k[:, :, :seq].transpose(1, 2),
            v[:, :, :seq].transpose(1, 2),
            seq_lens,
            bsa_idx.reshape(b * h, bsa_idx.size(2), bsa_idx.size(3)),
            ks,
            ks_count,
            ks_start_end,
            args,
            access_counter,
            cache_miss_counter,
        )
        return out, bsa_out, bsa_idx, block_sums

    o = flash_attn_func(q[:, :, :seq].transpose(1, 2), k[:, :, :seq].transpose(1, 2), v[:, :, :seq].transpose(1, 2), causal=True)
    o = o.transpose(1, 2)
    
    # warmup burn-in. autotune has s dirty init so this is necessary right now
    out, bsa_out, block_idx, _ = qsa(heap=False)
    out, bsa_out, block_idx, _ = qsa(heap=True)

    bsa_out = bsa_out.transpose(1, 2)

    delta = out - bsa_out.view(b, h, seq // q_block, q_block, d)[:, :, :, 0]
    delta = delta.view(b, h, seq // q_block, 1, d).repeat(1, 1, 1, q_block, 1).reshape(b, h, seq, d)
    delta = delta + bsa_out

    delta_cos = torch.nn.functional.cosine_similarity(delta, o, dim=-1)
    bsa_cos = torch.nn.functional.cosine_similarity(bsa_out, o, dim=-1)

    delta_cos_mean = delta_cos.mean().item()
    delta_cos_std = delta_cos.std().item()

    bsa_cos_mean = bsa_cos.mean().item()
    bsa_cos_std = bsa_cos.std().item()

    print(f"delta/flash cos: {delta_cos_mean} +- {delta_cos_std} ")
    print(f"qsa bsa only/flash cos: {bsa_cos_mean} +- {bsa_cos_std}")

    # -------------------------------------------------
    query_seq_dups = int(os.getenv("Q_DUPS", "-1"))
    seq_dups = int(os.getenv("DUPS", "1"))
    if query_seq_dups < 0:
        query_seq_dups = seq_dups
    block_size = int(os.getenv("BLOCK_SIZE", "64"))
    num_samples = int(os.getenv("NUM_SAMPLES", "20"))
    mask_only = int(os.getenv("MASK_ONLY", "0")) == "1"
    k_group_size = int(os.getenv("K_GROUP_SIZE", "1"))

    device = 0

    assert seq_dups > 0

    using_extend = False
    is_decode = False

    # preset = os.getenv("HIP_PRESET", "debug")
    preset = "debug"
    config_stage = {
        "mid": [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=2,
                stage_chunk_size=32,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=2,
                stage_chunk_size=8,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=2,
                stage_k=8192,
                stage_stride=1,
            ),
        ],
        "debug": [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=16,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=2,
                stage_chunk_size=4,
                stage_k=4096,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=1,
                stage_k=2048,
                stage_stride=1,
            ),
        ],
    }[preset]

    config_second_k = {
        "mid": 2048,
        "debug": 2048,
    }[preset]

    config_sa_extend_backend = {
        "mid": "streaming",
        "debug": "streaming",
    }[preset]

    dual_stage_kwargs = dict(
        q=q[:, :, :seq].transpose(1, 2),
        k=k[:, :, :seq].transpose(1, 2),
        v=v[:, :, :seq].transpose(1, 2),
        args=HiPAttentionArgs(
            block_size_k=32,  # BLOCK_CHUNK
            sliding_window_size=window_size,
            sink_token_size=64,
            # position_ids=position_ids,
            using_extend=using_extend,
            need_apply_rope=using_extend,
            rope_cos=None,
            rope_sin=None,
            second_stage_k=config_second_k,
            stages=config_stage,
            block_sparse_block_size_q=block_size,
            model_context_length=131072,
            extend_context_length=131072,
            # scan_early_terminate=1,
            # stage_early_terminate=1,
            scan_extend_backend="relative",
            sa_extend_backend=config_sa_extend_backend,
            stage_early_terminate=k_group_size,
            mask_only=mask_only,
        ),
    )

    # print(f"{dual_stage_kwargs=}")
    hip_out, metadata = dual_stage_quadratic_hip_attention(
        **dual_stage_kwargs, cached_metadata=None
    )
    hip_out = hip_out.transpose(1, 2)

    hip_bsa_cos = torch.nn.functional.cosine_similarity(hip_out, bsa_out, dim=-1).mean()
    print(f"hip/qsa_bsa only cosine similarity: {hip_bsa_cos.item()}")

    # print(f"{hip_out=}\n{bsa_out=}\n{out=}")
    diff = hip_out - bsa_out
    diff = ((diff[0, 0] != 0).sum(dim=-1) != 0).sum()
    print(f"number of differing elements in hip/bsa output: {diff}")

    delta_hip = out - hip_out.view(b, h, seq // q_block, q_block, d)[:, :, :, 0]
    delta_hip = delta_hip.view(b, h, seq // q_block, 1, d).repeat(1, 1, 1, q_block, 1).reshape(b, h, seq, d)
    delta_hip = delta_hip + hip_out

    hip_qsa_delta_cos = torch.nn.functional.cosine_similarity(delta_hip, delta, dim=-1)
    hip_qsa_delta_cos_mean = hip_qsa_delta_cos.mean()
    hip_qsa_delta_cos_std = hip_qsa_delta_cos.std()
    print(f"hip+delta/qsabsa+delta cos: {hip_qsa_delta_cos_mean} +- {hip_qsa_delta_cos_std}")

    hip_delta_cos = torch.nn.functional.cosine_similarity(delta_hip, o, dim=-1)
    hip_cos = torch.nn.functional.cosine_similarity(hip_out, o, dim=-1)

    hip_delta_cos_mean = hip_delta_cos.mean().item()
    hip_delta_cos_std = hip_delta_cos.std().item()

    hip_cos_mean = hip_cos.mean().item()
    hip_cos_std = hip_cos.std().item()

    print(f"hip + delta/flash cos {hip_delta_cos_mean} +- {hip_delta_cos_std}")
    print(f"hip/flash cos: {hip_cos_mean} +- {hip_cos_std}")

    def sllm() -> torch.Tensor:
        sllm_out = block_sparse_attention(
            q[:, :, :seq].transpose(1, 2) * math.sqrt(1 / q.size(-1)),
            k[:, :, :seq].transpose(1, 2),
            v[:, :, :seq].transpose(1, 2),
            seq_lens,
            None, 
            None,
            None,
            None,
            args,
            access_counter,
            cache_miss_counter,
        )
        return sllm_out


    sllm_out = sllm()
    sllm_out = sllm_out.transpose(1, 2)

    delta = out - sllm_out.view(b, h, seq // q_block, q_block, d)[:, :, :, 0]
    delta = delta.view(b, h, seq // q_block, 1, d).repeat(1, 1, 1, q_block, 1).reshape(b, h, seq, d)
    delta = delta + sllm_out

    sllm_delta_cos = torch.nn.functional.cosine_similarity(delta, o, dim=-1)
    sllm_cos = torch.nn.functional.cosine_similarity(sllm_out, o, dim=-1)

    sllm_delta_cos_mean = sllm_delta_cos.mean().item()
    sllm_delta_cos_std = sllm_delta_cos.std().item()

    sllm_cos_mean = sllm_cos.mean().item()
    sllm_cos_std = sllm_cos.std().item()

    print(f"{sllm_delta_cos_mean=} {sllm_delta_cos_std=}")
    print(f"{sllm_cos_mean=} {sllm_cos_std=}")


    latency_sllm = latency(lambda: sllm())
    print(f'sllm: {latency_sllm:.2f} ms took')
    latency_hip = latency(lambda: dual_stage_quadratic_hip_attention(**dual_stage_kwargs, cached_metadata=None))
    print(f'hip: {latency_hip:.2f} ms took')
    print("calling qsa latency")

    latency_qsa = latency(lambda: qsa(return_bsa_indices=False))
    print(f'qsa no bsa indices: {latency_qsa:.2f} ms took')
    latency_qsa = latency(lambda: qsa())
    print(f'qsa no heap: {latency_qsa:.2f} ms took')
    latency_qsa = latency(lambda: qsa(heap=True))
    print(f'qsa heap: {latency_qsa:.2f} ms took')

if __name__ == "__main__":
    test_with_hip_bsa()
    test_qsa()
