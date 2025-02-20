import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def memory_efficient_llm_ce_cuda(
    HIDDEN,
    stride_hidden_n,
    stride_hidden_hid,
    PROJ,
    stride_proj_kout,
    stride_proj_kin,
    LABEL,
    stride_label_n,
    LOSS,
    stride_loss_n,
    N,
    HID,
    KOUT,
    softcap,
    BLOCK_N: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    BLOCK_KOUT: tl.constexpr,
):
    idx_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = idx_n < N

    target_label = tl.load(
        LABEL + idx_n * stride_label_n,
        mask=mask_n,
    )

    score_max = tl.full((BLOCK_N,), value=float("-inf"), dtype=tl.float64)
    for idx_blabel in range(tl.cdiv(KOUT, BLOCK_KOUT)):
        idx_label = tl.arange(0, BLOCK_KOUT) + idx_blabel * BLOCK_KOUT
        mask_label = idx_label < KOUT

        acc = tl.zeros((BLOCK_N, BLOCK_KOUT), dtype=tl.float32)
        for idx_bhid in range(0, tl.cdiv(HID, BLOCK_HID)):
            idx_hid = tl.arange(0, BLOCK_HID) + BLOCK_HID * idx_bhid
            assert (HID % BLOCK_HID) == 0
            hidden = tl.load(
                HIDDEN
                + idx_n[:, None] * stride_hidden_n
                + idx_hid[None, :] * stride_hidden_hid,
                mask=mask_n[:, None],
                other=0,
            )
            proj = tl.load(
                PROJ
                + idx_label[None, :] * stride_proj_kout
                + idx_hid[:, None] * stride_proj_kin,
                mask=mask_label[None, :],
                other=0,
            )
            acc += tl.dot(
                hidden,
                proj.to(hidden.dtype),
                allow_tf32=True,
            ).to(acc.dtype)
        if softcap is not None:
            acc = tl.extra.cuda.libdevice.tanh(acc / softcap) * softcap
        score_max = tl.maximum(tl.max(acc, axis=1), score_max)

    exp_label_score = tl.zeros((BLOCK_N,), dtype=tl.float64)
    exp_score_sum = tl.zeros((BLOCK_N,), dtype=tl.float64)
    for idx_blabel in range(tl.cdiv(KOUT, BLOCK_KOUT)):
        idx_label = tl.arange(0, BLOCK_KOUT) + idx_blabel * BLOCK_KOUT
        mask_label = idx_label < KOUT

        acc = tl.zeros((BLOCK_N, BLOCK_KOUT), dtype=tl.float32)
        for idx_bhid in range(0, tl.cdiv(HID, BLOCK_HID)):
            idx_hid = tl.arange(0, BLOCK_HID) + BLOCK_HID * idx_bhid
            assert (HID % BLOCK_HID) == 0
            hidden = tl.load(
                HIDDEN
                + idx_n[:, None] * stride_hidden_n
                + idx_hid[None, :] * stride_hidden_hid,
                mask=mask_n[:, None],
                other=0,
            )
            proj = tl.load(
                PROJ
                + idx_label[None, :] * stride_proj_kout
                + idx_hid[:, None] * stride_proj_kin,
                mask=mask_label[None, :],
                other=0,
            )
            acc += tl.dot(
                hidden,
                proj.to(hidden.dtype),
                allow_tf32=True,
            ).to(acc.dtype)
        if softcap is not None:
            acc = tl.extra.cuda.libdevice.tanh(acc / softcap) * softcap
        exp_score = tl.exp(acc - score_max[:, None])
        exp_score_sum += tl.sum(exp_score, axis=1)
        exp_label_score += tl.sum(
            tl.where(
                target_label[:, None] == idx_label[None, :],
                exp_score,
                0,
            ),
            axis=1,
        )

    loss = -tl.log(exp_label_score / exp_score_sum)
    tl.store(LOSS + idx_n * stride_loss_n, mask=mask_n, value=loss)


def memory_efficient_llm_ce(
    hidden_states: Tensor,
    out_proj_weight: Tensor,
    labels: Tensor,
    reduction: str = "mean",
    softcap: float = None,
):
    assert hidden_states.ndim == 2
    assert out_proj_weight.ndim == 2
    assert labels.ndim == 1
    assert labels.dtype in [torch.int32, torch.int64, torch.long]
    assert hidden_states.device == out_proj_weight.device
    assert labels.device == hidden_states.device
    N, HID = hidden_states.shape
    KOUT, KIN = out_proj_weight.shape
    (_N,) = labels.shape
    assert N == _N, f"{N} == {_N}"
    assert HID == KIN

    losses = torch.empty((N,), dtype=torch.float32, device=hidden_states.device)

    BLOCK_N = 64
    BLOCK_HID = 128
    BLOCK_KOUT = 128

    assert (HID % BLOCK_HID) == 0

    grid = (triton.cdiv(N, BLOCK_N),)

    pre_device = torch.get_default_device()
    torch.set_default_device(hidden_states.device)
    memory_efficient_llm_ce_cuda[grid](
        hidden_states,
        *hidden_states.stride(),
        out_proj_weight,
        *out_proj_weight.stride(),
        labels,
        *labels.stride(),
        losses,
        *losses.stride(),
        N,
        HID,
        KOUT,
        softcap,
        BLOCK_N,
        BLOCK_HID,
        BLOCK_KOUT,
        num_warps=16,
    )
    torch.set_default_device(pre_device)

    if reduction == "mean":
        loss = losses.mean()
    elif reduction == "sum":
        loss = losses.sum()
    elif reduction == "none":
        loss = losses
    else:
        raise Exception()

    return loss
