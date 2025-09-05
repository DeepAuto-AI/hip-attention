import torch
from typing import Tuple

def collapse_and_topk_batched(
    indices: torch.Tensor,      # shape (..., L), int
    scores: torch.Tensor,       # shape (..., L), float
    k: int,
    reduce: str = "mean",       # "mean" | "sum" | "max"
    fill_index: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collapse duplicates within the last dim per batch, aggregate scores, and return top-k
    per batch without globally sorting the flattened tensor.

    Returns:
        topk_idx:   (..., k) int
        topk_score: (..., k) float
    """
    assert indices.shape == scores.shape and indices.ndim >= 1
    assert reduce in {"mean", "sum", "max"}

    device = scores.device
    dtype  = scores.dtype
    batch_shape = indices.shape[:-1]
    L = indices.shape[-1]

    # Flatten leading dims to B
    B = 1
    for d in batch_shape: B *= d
    idx2 = indices.reshape(B, L).to(torch.int64)
    sc2  = scores.reshape(B, L)

    # Build global ids so (batch, idx) pairs become unique integers
    idx_flat = idx2.reshape(-1)
    sc_flat  = sc2.reshape(-1)
    b_ids    = torch.arange(B, device=device, dtype=torch.int64).repeat_interleave(L)

    min_idx = idx_flat.min()
    domain  = (idx_flat.max() - min_idx + 1).to(torch.int64)  # safe window per batch

    global_ids = (idx_flat - min_idx) + b_ids * domain  # [B*L] int64

    # Find unique (batch, idx), and inverse map for aggregation
    uniq_g, inv = torch.unique(global_ids, sorted=True, return_inverse=True)  # [U], [B*L]
    U = uniq_g.numel()

    # Aggregate scores per unique
    if reduce in ("sum", "mean"):
        agg = torch.zeros(U, device=device, dtype=dtype).index_add_(0, inv, sc_flat)
        if reduce == "mean":
            cnt_u = torch.bincount(inv, minlength=U).to(dtype)
            agg = agg / cnt_u.clamp_min(1)
    else:  # "max"
        base = torch.finfo(dtype).min if sc_flat.is_floating_point() else torch.iinfo(sc_flat.dtype).min
        agg = torch.full((U,), base, device=device, dtype=dtype)
        agg.scatter_reduce_(0, inv, sc_flat, reduce="amax", include_self=False)

    # Recover per-unique batch and original index
    uniq_batch = (uniq_g // domain).to(torch.int64)                 # [U] in [0..B-1]
    uniq_idx   = (uniq_g %  domain + min_idx).to(indices.dtype)     # [U]

    # Because uniq_g is sorted, items are grouped by batch already.
    # Count uniques per batch and compute positions (ranks) within each batch segment
    counts = torch.bincount(uniq_batch, minlength=B)                # [B]
    starts = counts.cumsum(0) - counts                              # [B] start offset per batch
    pos    = torch.arange(U, device=device)                         # [U]
    rank   = pos - starts[uniq_batch]                               # [U], 0..counts[b]-1

    # Dense (B, Mmax) matrices for scores and indices
    Mmax = int(counts.max().item())
    k_eff = min(k, Mmax)
    neg_inf = torch.finfo(dtype).min if sc_flat.is_floating_point() else torch.iinfo(sc_flat.dtype).min

    agg_dense = torch.full((B, Mmax), neg_inf, device=device, dtype=dtype)
    idx_dense = torch.full((B, Mmax), fill_index, device=device, dtype=indices.dtype)

    agg_dense[uniq_batch, rank] = agg
    idx_dense[uniq_batch, rank] = uniq_idx

    # Per-batch top-k (no global sort)
    top_vals, top_pos = torch.topk(agg_dense, k_eff, dim=1, largest=True, sorted=True)
    top_idx = torch.gather(idx_dense, 1, top_pos)

    # Reshape back to original leading dims
    top_idx   = top_idx.reshape(*batch_shape, k_eff)
    top_vals  = top_vals.reshape(*batch_shape, k_eff)
    return top_idx, top_vals



idx = torch.tensor([[7,2,7,9,2,2,5],
                    [3,3,1,1,1,9,9]])
sc  = torch.tensor([[1.,3.,5.,2.,7.,1.,4.],
                    [4.,2.,9.,1.,5.,1.,3.]])

print("idx and scores before")
print(idx, sc)
# Mean reduce, top-2 per batch:
ti, tv = collapse_and_topk_batched(idx, sc, k=2, reduce="max")

print("idx and scores after")
print(ti, tv)
