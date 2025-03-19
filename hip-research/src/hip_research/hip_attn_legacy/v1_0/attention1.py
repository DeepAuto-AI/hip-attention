"""
- Need to stop expansion when reach #patch
> multiple = 4, #patch:p = 16, k = 64, w = 8192
| w    | z    | z'   | k'   | keep?|
|------|------|------|------|------|
| 64   | 64   | 1    | 16   | True |
| 256  | 64   | 2    | 16   | True |
| 1024 | 64   | 8    | 16   | True |
| 4096 | 64   | 32   | 32   | done |
| 8192 | done | done | done | done |

- When approximator interation stops?
w / T * k >= p

if p and k is constant
w = (p/k)T
approximator is logN, but sparse attention is linear

if p=T/C
w = T^2/(kC) -- log w = 2log T - log kC
approximator is quadratic, but sparse attention is linear

if k=T/C
w = pC
approximator is linear, but sparse attention is quadratic

if p=T/C and k=T/C
w = T
approximator is log N, but sparse attention is quadratic
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import tqdm
from numpy import ndarray
from torch import Tensor


def mask(
    queries: ndarray,
    keys: ndarray,
    w_start: int = 32,
    n_patches: int = 16,
    mask_k: int = 128,
    scale_up: int = 4,
    # w_start: int = 4,
    # n_patches: int = 8,
    # mask_k: int = 16,
    # scale_up: int = 2,
) -> ndarray:
    # NOTE: uncomment for cumsum
    # queries_cs = queries.cumsum(1)
    # keys_cs = keys.cumsum(1)

    dtype = np.float32
    N, T_DST, HID = queries.shape
    _, T_SRC, _ = keys.shape
    assert T_DST <= T_SRC

    # NOTE: width of last query
    w_curr = round(w_start / scale_up)
    t_srcs = (
        np.arange(T_SRC - T_DST + 1, T_SRC + 1, 1, dtype=np.int32)
        .reshape((1, T_DST, 1))
        .repeat(N, axis=0)
    )
    ws = t_srcs.clip(0, w_curr)
    ks = ws.copy()
    mask = np.arange(mask_k, dtype=np.float32).reshape((1, 1, mask_k)) / ks
    t_mask = np.zeros((N, T_DST, mask_k * math.ceil(scale_up)), dtype=np.float32)
    scores = np.zeros((N, T_DST, mask_k * math.ceil(scale_up)), dtype=dtype)

    def to_dense(mask, ks, ws):
        dense = np.zeros((N, T_DST, T_SRC))
        for i in range(N):
            for j in range(T_DST):
                nonzero_k = ks[i, j, 0]
                for k in range(nonzero_k):
                    dense[i, j, int(mask[i, j, k] * ws[i, j, 0])] = 1
        return dense

    while w_curr < T_SRC:
        # scale up, and top k masking
        for i in range(N):
            for j in tqdm.tqdm(range(T_DST)):
                # for each query
                w_old = ws[i, j, 0]
                t_src = t_srcs[i, j, 0]
                w_new = min(round(w_old * scale_up), t_src)
                # print(i, j, w_old, w_new)
                if w_old != w_new:
                    k_old = ks[i, j, 0]
                    k_new = max(n_patches, int(min(mask_k / t_src, 1.0) * w_new))

                    def resize_query(
                        i: int,
                        j: int,
                        mask: ndarray,
                        t_mask: ndarray,
                        k_old: int,
                        w_old: int,
                        w_new: int,
                    ) -> int:
                        num_pixels = 0
                        for k in range(k_old):
                            loc = mask[i, j, k]
                            loc_idx_start = int(loc * w_old)
                            loc_idx_end = loc_idx_start + 1
                            loc_idx_start = int(loc_idx_start / w_old * w_new)
                            loc_idx_end = int(loc_idx_end / w_old * w_new)
                            dup_pixels = loc_idx_end - loc_idx_start
                            for l in range(dup_pixels):
                                t_mask[i, j, num_pixels + l] = (
                                    loc_idx_start + l
                                ) / w_new
                            num_pixels += dup_pixels
                        return num_pixels

                    k_new = min(t_src, max(n_patches, k_new))

                    # mask -> t_mask
                    num_pixels = resize_query(i, j, mask, t_mask, k_old, w_old, w_new)

                    # t_mask -> mask (using scores)
                    if k_new < num_pixels:
                        # need top_k, so compute scores
                        for k in range(num_pixels):
                            vec_q = queries[i, j, :]

                            # NOTE: nearest
                            loc = t_mask[i, j, k]
                            vec_k = keys[i, int(loc * t_src), :]

                            # NOTE: cumsum
                            # loc_start = int(int(loc * w_new) * (t_src / w_new))
                            # loc_end = int((int(loc * w_new) + 1) * (t_src / w_new))
                            # loc_end = max(loc_end, loc_start + 1)
                            # vec_k = keys_cs[i, loc_end - 1, :]# - keys_cs[i, loc_start, :]
                            # if loc_start > 0:
                            #     vec_k -= keys_cs[i, loc_start - 1, :]

                            score = np.dot(vec_q, vec_k)
                            scores[i, j, k] = -score  # NOTE: store negative store

                        topk_indices = np.argpartition(
                            scores[i, j, :num_pixels], kth=k_new
                        )
                        topk_indices = np.sort(topk_indices[:k_new])
                        for k in range(k_new):
                            mask[i, j, k] = t_mask[i, j, topk_indices[k]]
                    else:
                        mask[i, j, :num_pixels] = t_mask[i, j, :num_pixels]

                    ws[i, j, 0] = w_new
                    ks[i, j, 0] = min(k_new, num_pixels)
                # end if w_old != w_new
            # end for j
        # end for i

        # print(t_mask[0, 1000:1016, :5])
        # print(ws[0, 1000:1016])

        # NOTE: debug image output
        x = to_dense(mask, ks, ws)[0]
        x = skimage.measure.block_reduce(x, (4, 4), np.max)
        plt.imshow(x)
        plt.savefig("hello.png", dpi=200)
        input(">>> ")

        w_curr = round(w_curr * scale_up)
    # end while

    # NOTE: for debug image output
    # print mask
    mask = to_dense(mask, ks, ws)[0]
    x = skimage.measure.block_reduce(mask, (4, 4), np.max)
    plt.imshow(x)
    plt.savefig("hello.png", dpi=200)

    # print probabilites
    x = np.matmul(queries[0], keys[0].transpose((-1, -2)))
    x = x + (1 - np.tri(*x.shape)) * (-32000)
    x = np.exp(x - x.max(-1, keepdims=True))
    x = x / x.sum(-1, keepdims=True)
    x = skimage.measure.block_reduce(x, (8, 8), np.max) ** 0.2
    plt.imshow(x)
    plt.savefig("hello_2.png", dpi=200)
    # NOTE: end of debug output

    print(ks)

    return


def sparse_attention(q: ndarray, k: ndarray, v: ndarray, csr_mask: ndarray):
    pass


def attention(q: Tensor, k: Tensor, v: Tensor):
    assert q.ndim == 3
    assert k.ndim == 3
    assert v.ndim == 3
    N, T_DST, HID = q.shape
    _N, T_SRC, _HID = k.shape
    assert k.shape[:-1] == v.shape[:-1]
    assert N == _N
    assert HID == _HID

    q = q.numpy()
    k = k.numpy()
    v = v.numpy()
    csr_scores = mask(q, k)
    # out = sparse_attention(csr_scores, v)

    # return out
