import os

import cv2
import matplotlib.pyplot as plt
import numba
import numpy as np
import torch
from hip_research.utils import setup_seaborn

setup_seaborn(axis_below=True)

TDST = 2048
TSRC = 4096
BQ = 32
BK = 2
MASK_K = 512


@numba.njit
def convert_to_dense(indices, ks, TDST, TSRC, BQ, BK, MASK_K):
    mask = np.zeros((TDST, TSRC))
    for i in range(TDST // BQ):
        kk = ks[0, i]
        for j in range(MASK_K // BK):
            if j < kk:
                t = indices[0, i, j]
                mask[i * BQ : i * BQ + BQ, t : t + BK] = 1
    return mask


def render_plot(cache_path, name, iteration):
    data = torch.load(cache_path, map_location="cpu")
    indices = data["indices"].numpy()
    ks = data["ks"].numpy()

    ws = np.full((TDST,), MASK_K) * (2 ** max(0, iteration - 1))
    tsrcs = np.arange(TSRC - TDST, TSRC)
    tsrcs = tsrcs - (tsrcs % BQ) + BQ
    ws = np.minimum(tsrcs, ws)

    scales = tsrcs / ws

    mask = convert_to_dense(indices, ks, TDST, TSRC, BQ, BK, MASK_K)

    for i in range(TDST):
        scale = scales[i]
        row = mask[i : i + 1, :]
        row_resize = cv2.resize(
            row, None, fx=scale, fy=1.0, interpolation=cv2.INTER_NEAREST
        )
        mask[i : i + 1, :] = row_resize[:, :TSRC]

    root = "./saves/plot_masking_iteration"
    path = os.path.join(root, f"{name}.png")
    os.makedirs(root, exist_ok=True)

    plt.figure(figsize=(4, 3))
    plt.imshow(mask, cmap="summer")
    plt.savefig(path, dpi=400, bbox_inches="tight")

    print("saved", path)


if __name__ == "__main__":
    render_plot("./saves/attention1_block_gpu/checkout_mask_0.pth", "mask_0", 0)
    render_plot("./saves/attention1_block_gpu/checkout_mask_1.pth", "mask_1", 1)
    render_plot("./saves/attention1_block_gpu/checkout_mask_2.pth", "mask_2", 2)
    render_plot("./saves/attention1_block_gpu/checkout_mask_3.pth", "mask_3", 3)
    render_plot("./saves/attention1_block_gpu/checkout_mask_4.pth", "mask_4", 4)
