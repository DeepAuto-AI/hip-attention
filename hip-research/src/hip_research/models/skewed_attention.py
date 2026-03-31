import matplotlib.pyplot as plt
import torch
from torch import Tensor


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    x = (x * cos) + (rotate_half(x) * sin)
    return x


def skewed_mask(
    q: Tensor,
    k: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    sm_scaler: float,
    compress_hid: int = 64,
):
    import einx

    N, H, TDST, HID = q.shape
    N, H_KV, TSRC, HID = k.shape

    t = einx.rearrange("n h t d -> h (n t) d", k).float()
    _, _, proj = torch.linalg.svd(t, full_matrices=False)
    proj = proj.to(q.dtype)

    q = einx.dot(
        "n h t d1, h d1 d2 -> n h t d2",
        q.view(N, H_KV, H // H_KV, TDST, HID).permute(0, 2, 1, 3, 4).flatten(0, 1),
        proj,
        kv=H // H_KV,
    )
    k = einx.dot("n h t d1, h d1 d2 -> n h t d2", k, proj)

    q_colsum = q.permute(1, 0, 2, 3).flatten(1, 2).abs().mean(dim=1, keepdim=False)
    k_colsum = k.permute(1, 0, 2, 3).flatten(1, 2).abs().mean(dim=1, keepdim=False)

    colsum = q_colsum + k_colsum
    colsum = colsum + rotate_half(colsum).abs()
    colsum = colsum[..., : HID // 2]

    _, topk_indices = colsum.topk(dim=-1, k=compress_hid)
    idx_hid_keys = topk_indices.sort(dim=-1).values
    idx_hid_keys = torch.cat([idx_hid_keys, idx_hid_keys + HID // 2], dim=-1)
    idx_hid_keys_q = idx_hid_keys.repeat_interleave(H // H_KV, 0)
    idx_hid_keys_kv = idx_hid_keys

    print(idx_hid_keys_q[0])

    q = q.reshape(N, H, TDST, HID).gather(
        index=idx_hid_keys_q[None, :, None, :].expand(N, H, TDST, -1), dim=-1
    )
    k = k.reshape(N, H_KV, TSRC, HID).gather(
        index=idx_hid_keys_kv[None, :, None, :].expand(N, H_KV, TSRC, -1), dim=-1
    )
    q_cos = (
        rope_cos[None, :, :, :]
        .expand(N, H, TDST, HID)
        .gather(index=idx_hid_keys_q[None, :, None, :].expand(N, H, TDST, -1), dim=-1)
    )
    q_sin = (
        rope_sin[None, :, :, :]
        .expand(N, H, TDST, HID)
        .gather(index=idx_hid_keys_q[None, :, None, :].expand(N, H, TDST, -1), dim=-1)
    )
    k_cos = (
        rope_cos[None, :, :, :]
        .expand(N, H_KV, TSRC, HID)
        .gather(
            index=idx_hid_keys_kv[None, :, None, :].expand(N, H_KV, TSRC, -1), dim=-1
        )
    )
    k_sin = (
        rope_sin[None, :, :, :]
        .expand(N, H_KV, TSRC, HID)
        .gather(
            index=idx_hid_keys_kv[None, :, None, :].expand(N, H_KV, TSRC, -1), dim=-1
        )
    )

    q = q * q_cos + rotate_half(q) * q_sin
    k = k * k_cos + rotate_half(k) * k_sin

    scores = q @ k.repeat_interleave(H // H_KV, dim=1).permute(0, 1, 3, 2)
    scores = scores * sm_scaler

    mask = (
        torch.arange(0, TDST, device=q.device)[:, None]
        >= torch.arange(0, TSRC, device=k.device)[None, :]
    )
    mask = torch.where(mask, 0, -32000.0)
    scores = scores + mask

    probs = torch.softmax(scores, dim=-1).to(q.dtype)

    plt.clf()
    plt.imshow(probs[0, 0].cpu().float().numpy() ** 0.2)
    plt.colorbar()
    plt.savefig("./dummy_skewed.png")

    return q, k, None


def skewed_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    sm_scaler: float,
    layer_idx: int,
):
    N, H, TDST, HID = q.shape
    N, H_KV, TSRC, HID = k.shape
    assert k.shape == v.shape

    q_skewed, k_skewed, est_mask = skewed_mask(q, k, rope_cos, rope_sin, sm_scaler)

    q = apply_rope(q, rope_cos, rope_sin)
    k = apply_rope(k, rope_cos, rope_sin)

    scores = q @ k.repeat_interleave(H // H_KV, dim=1).permute(0, 1, 3, 2)
    scores = scores * sm_scaler

    mask = (
        torch.arange(0, TDST, device=q.device)[:, None]
        >= torch.arange(0, TSRC, device=k.device)[None, :]
    )
    mask = torch.where(mask, 0, -32000.0)
    scores = scores + mask

    probs = torch.softmax(scores, dim=-1).to(q.dtype)

    plt.clf()
    plt.imshow(probs[0, 0].cpu().float().numpy() ** 0.2)
    plt.colorbar()
    plt.savefig("./dummy_not_skewed.png")

    input(">>>")

    context = probs @ v.repeat_interleave(H // H_KV, dim=1)

    return context
