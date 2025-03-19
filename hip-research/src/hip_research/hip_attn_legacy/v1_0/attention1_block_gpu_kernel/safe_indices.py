import torch
import triton
import triton.language as tl
from torch import Tensor


def next_multiple_of(x: int, multiple_by: int = 16):
    return triton.next_power_of_2(max(x, multiple_by))


@triton.jit
def _safe_indices_compute(
    # input tensors
    MASK,
    stride_mask_n,
    stride_mask_tdst,
    stride_mask_k,
    WS,
    stride_ws_n,
    stride_ws_tdst,
    stride_ws_k,
    # output tensors
    INDICES,
    stride_indices_n,
    stride_indices_tdst,
    stride_indices_k,
    N,
    TDST,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ALLOW_COLLISION: tl.constexpr,
    BLOCK_N_TDST: tl.constexpr,
    BLOCK_K: tl.constexpr,
    COLLISION_METHOD: tl.constexpr = "biased",
    # COLLISION_METHOD: tl.constexpr = 'unbiased',
    # COLLISION_METHOD: tl.constexpr = 'unbiased_simple',
    # COLLISION_METHOD: tl.constexpr = 'unbiased_simple_reversed',
):
    if not ALLOW_COLLISION:
        pids = tl.program_id(0) * BLOCK_N_TDST + tl.arange(0, BLOCK_N_TDST)

        idx_n = pids // TDST
        mask_n = idx_n < N

        idx_tdst = pids % TDST
        mask_tdst = idx_tdst < TDST

        mask = mask_n & mask_tdst

        if COLLISION_METHOD == "biased":
            last_col = tl.zeros((BLOCK_N_TDST,), dtype=tl.int64) - 1
            for _idx_k in range(K):
                mask_vec = tl.load(
                    MASK
                    + idx_n * stride_mask_n
                    + idx_tdst * stride_mask_tdst
                    + _idx_k * stride_mask_k,
                    mask=mask,
                    other=0,
                )  # .to(tl.float32)
                ws_vec = tl.load(
                    WS
                    + idx_n * stride_ws_n
                    + idx_tdst * stride_ws_tdst
                    + _idx_k * stride_ws_k,
                    mask=mask,
                    other=0,
                )  # .to(tl.float32)
                indices_float = mask_vec * ws_vec
                col = tl.math.ceil(indices_float / BLOCK_SIZE_K).to(tl.int32)

                # avoid collision
                col = tl.maximum(last_col + 1, col)
                last_col = col

                col = col * BLOCK_SIZE_K

                tl.store(
                    INDICES
                    + idx_n * stride_indices_n
                    + idx_tdst * stride_indices_tdst
                    + _idx_k * stride_indices_k,
                    value=col,
                    mask=mask,
                )
        elif COLLISION_METHOD == "unbiased_simple":
            last_col = tl.zeros((BLOCK_N_TDST,), dtype=tl.int64) - 1
            for _idx_k in range(K // 2, K):
                mask_vec = tl.load(
                    MASK
                    + idx_n * stride_mask_n
                    + idx_tdst * stride_mask_tdst
                    + _idx_k * stride_mask_k,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                ws_vec = tl.load(
                    WS
                    + idx_n * stride_ws_n
                    + idx_tdst * stride_ws_tdst
                    + _idx_k * stride_ws_k,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                indices_float = mask_vec * ws_vec
                col = tl.math.ceil(indices_float / BLOCK_SIZE_K).to(tl.int32)

                # avoid collision
                col = tl.maximum(last_col + 1, col)
                last_col = col

                col = col * BLOCK_SIZE_K

                tl.store(
                    INDICES
                    + idx_n * stride_indices_n
                    + idx_tdst * stride_indices_tdst
                    + _idx_k * stride_indices_k,
                    value=col,
                    mask=mask,
                )

            last_col = tl.zeros((BLOCK_N_TDST,), dtype=tl.int64) + 9999999
            for _idx_k in range(0, K // 2):
                idx_k = K // 2 - _idx_k - 1
                mask_vec = tl.load(
                    MASK
                    + idx_n * stride_mask_n
                    + idx_tdst * stride_mask_tdst
                    + idx_k * stride_mask_k,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                ws_vec = tl.load(
                    WS
                    + idx_n * stride_ws_n
                    + idx_tdst * stride_ws_tdst
                    + idx_k * stride_ws_k,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                indices_float = mask_vec * ws_vec
                col = tl.math.ceil(indices_float / BLOCK_SIZE_K).to(tl.int32)

                # avoid collision
                col = tl.maximum(0, tl.minimum(last_col - 1, col))
                last_col = col

                col = col * BLOCK_SIZE_K

                tl.store(
                    INDICES
                    + idx_n * stride_indices_n
                    + idx_tdst * stride_indices_tdst
                    + idx_k * stride_indices_k,
                    value=col,
                    mask=mask,
                )
        elif COLLISION_METHOD == "unbiased_simple_reversed":
            last_col = tl.zeros((BLOCK_N_TDST,), dtype=tl.int64) - 1
            for _idx_k in range(0, K // 2):
                mask_vec = tl.load(
                    MASK
                    + idx_n * stride_mask_n
                    + idx_tdst * stride_mask_tdst
                    + _idx_k * stride_mask_k,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                ws_vec = tl.load(
                    WS
                    + idx_n * stride_ws_n
                    + idx_tdst * stride_ws_tdst
                    + _idx_k * stride_ws_k,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                indices_float = mask_vec * ws_vec
                col = tl.math.ceil(indices_float / BLOCK_SIZE_K).to(tl.int32)

                # avoid collision
                col = tl.maximum(last_col + 1, col)
                last_col = col

                col = col * BLOCK_SIZE_K

                tl.store(
                    INDICES
                    + idx_n * stride_indices_n
                    + idx_tdst * stride_indices_tdst
                    + _idx_k * stride_indices_k,
                    value=col,
                    mask=mask,
                )

            last_col = tl.zeros((BLOCK_N_TDST,), dtype=tl.int64) + 9999999
            for _idx_k in range(K // 2, K):
                idx_k = K // 2 - _idx_k - 1
                mask_vec = tl.load(
                    MASK
                    + idx_n * stride_mask_n
                    + idx_tdst * stride_mask_tdst
                    + idx_k * stride_mask_k,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                ws_vec = tl.load(
                    WS
                    + idx_n * stride_ws_n
                    + idx_tdst * stride_ws_tdst
                    + idx_k * stride_ws_k,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                indices_float = mask_vec * ws_vec
                col = tl.math.ceil(indices_float / BLOCK_SIZE_K).to(tl.int32)

                # avoid collision
                col = tl.maximum(0, tl.minimum(last_col - 1, col))
                last_col = col

                col = col * BLOCK_SIZE_K

                tl.store(
                    INDICES
                    + idx_n * stride_indices_n
                    + idx_tdst * stride_indices_tdst
                    + idx_k * stride_indices_k,
                    value=col,
                    mask=mask,
                )
        elif COLLISION_METHOD == "unbiased":
            # N^2 extreamly slow
            for _idx_k in range(K):
                mask_vec = tl.load(
                    MASK
                    + idx_n * stride_mask_n
                    + idx_tdst * stride_mask_tdst
                    + _idx_k * stride_mask_k,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                ws_vec = tl.load(
                    WS
                    + idx_n * stride_ws_n
                    + idx_tdst * stride_ws_tdst
                    + _idx_k * stride_ws_k,
                    mask=mask,
                    other=0,
                ).to(tl.float32)
                indices_float = mask_vec * ws_vec
                col = tl.math.ceil(indices_float / BLOCK_SIZE_K).to(tl.int32)

                tl.store(
                    INDICES
                    + idx_n * stride_indices_n
                    + idx_tdst * stride_indices_tdst
                    + _idx_k * stride_indices_k,
                    value=col,
                    mask=mask,
                )

            for idx_iteration in range(K):
                last_col = tl.zeros((BLOCK_N_TDST,), dtype=tl.int64) - 1
                is_collided = tl.zeros((BLOCK_N_TDST,), dtype=tl.int1)
                direction = idx_iteration % 2
                for _idx_k in range(K):
                    if direction:
                        _idx_k = K - _idx_k - 1
                    else:
                        _idx_k = _idx_k

                    col = tl.load(
                        INDICES
                        + idx_n * stride_indices_n
                        + idx_tdst * stride_indices_tdst
                        + _idx_k * stride_indices_k,
                        mask=mask,
                    )

                    # avoid collision
                    new_collided = last_col == col

                    updated_last_col = tl.where(
                        (~new_collided) & is_collided,
                        last_col + (direction * 2 - 1),
                        last_col,
                    )
                    updated_last_col = tl.maximum(0, updated_last_col)

                    is_collided = new_collided
                    last_col = col.to(last_col.dtype)

                    tl.store(
                        INDICES
                        + idx_n * stride_indices_n
                        + idx_tdst * stride_indices_tdst
                        + (_idx_k + (direction * 2 - 1)) * stride_indices_k,
                        value=updated_last_col,
                        mask=(_idx_k > 0) & mask,
                    )

            for _idx_k in range(K):
                col = tl.load(
                    INDICES
                    + idx_n * stride_indices_n
                    + idx_tdst * stride_indices_tdst
                    + _idx_k * stride_indices_k,
                    mask=mask,
                )
                col = col * BLOCK_SIZE_K
                tl.store(
                    INDICES
                    + idx_n * stride_indices_n
                    + idx_tdst * stride_indices_tdst
                    + _idx_k * stride_indices_k,
                    value=col,
                    mask=mask,
                )
        else:
            raise Exception()
    else:
        pids_ntdst = tl.program_id(1) * BLOCK_N_TDST + tl.arange(0, BLOCK_N_TDST)

        idx_n = (pids_ntdst // TDST)[:, None]
        mask_n = idx_n < N

        idx_tdst = (pids_ntdst % TDST)[:, None]
        mask_tdst = idx_tdst < TDST

        _idx_k = (tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K))[None, :]
        mask_k = _idx_k < K

        mask = mask_n & mask_tdst & mask_k

        mask_vec = tl.load(
            MASK
            + idx_n * stride_mask_n
            + idx_tdst * stride_mask_tdst
            + _idx_k * stride_mask_k,
            mask=mask,
            other=0,
        ).to(tl.float32)
        ws_vec = tl.load(
            WS + idx_n * stride_ws_n + idx_tdst * stride_ws_tdst + _idx_k * stride_ws_k,
            mask=mask,
            other=0,
        ).to(tl.float32)

        indices_float = mask_vec * ws_vec
        col = tl.math.ceil(indices_float / BLOCK_SIZE_K).to(tl.int32)
        col = col * BLOCK_SIZE_K

        tl.store(
            INDICES
            + idx_n * stride_indices_n
            + idx_tdst * stride_indices_tdst
            + _idx_k * stride_indices_k,
            value=col,
            mask=mask,
        )


def safe_indices(mask: Tensor, ws, block_size_k, allow_collision=False):
    # mask = mask.sort(dim=-1, descending=False)[0]

    N, TDST, K = mask.shape
    ws = ws.unsqueeze(-1).expand(N, TDST, K)

    indices = torch.empty((N, TDST, K), dtype=torch.int32, device=mask.device)

    BLOCK_N_TDST = 32
    BLOCK_K = 128

    if not allow_collision:
        grid = (triton.cdiv(N * TDST, BLOCK_N_TDST),)
    else:
        grid = (
            triton.cdiv(K, BLOCK_K),
            triton.cdiv(N * TDST, BLOCK_N_TDST),
        )

    assert indices.ndim == 3
    assert mask.ndim == 3
    assert indices.ndim == 3
    orig_device = torch.cuda.current_device()
    torch.cuda.set_device(mask.device)
    _safe_indices_compute[grid](
        mask,
        *mask.stride(),
        ws,
        *ws.stride(),
        indices,
        *indices.stride(),
        N,
        TDST,
        K,
        block_size_k,
        allow_collision,
        BLOCK_N_TDST,
        BLOCK_K,
        num_warps=4 if allow_collision else 1,
    )
    torch.cuda.set_device(orig_device)

    # indices = indices.reshape(N, TDST, K)

    return indices
