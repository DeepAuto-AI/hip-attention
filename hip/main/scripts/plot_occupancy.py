from typing import Literal
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
import triton

from hip.models.hip_attention.gen3.attention_extend import(
    load_checkouts, chunk_controllable_sampling_mask_cuda, safe_stride
)

# X: top-k occupancy
# Y: occurance

q, k, v, out, cos, sin = load_checkouts(
    idx=0, 
    window=999, 
    seq_len=131072, 
    dtype=torch.bfloat16, 
    return_cos_sin=True, 
    derope=False,
)

def reshape(x: torch.Tensor):
    return x.unsqueeze(0).permute(0, 2, 1, 3)

q = reshape(q).to(0)
k = reshape(k).to(0)
v = reshape(v).to(0)

q = q[0, -1:, :, :].permute(1, 0, 2)
k = k[0].permute(1, 0, 2).repeat(4, 1, 1)

scores = torch.bmm(q, k.permute(0, 2, 1))[:, 0, :]

def compute(
    chunk_size = 64,
    top_k = 2048
):
    scores_top_indices = scores.topk(k=2048, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.int64)
    mask.scatter_(dim=-1, index=scores_top_indices, value=1)
    stats = mask.view(32, -1, chunk_size).sum(-1).view(-1)

    ys = [0,] * (chunk_size + 1)

    for item in stats.cpu().numpy():
        item = item.item()
        ys[item] += 1

    ys = np.array(ys)
    ys = ys.astype(np.float32) / ys.sum()
    ys = ys * 100
    xs = np.arange(chunk_size + 1).astype(np.float32) / chunk_size * 100
    
    return xs, ys, ys[0]


xs, ys, empty_percent = compute()

print(f'{empty_percent:.2f} % chunks are empty.')

plt.figure(figsize=(4,3))
plt.plot(xs, ys)

plt.grid()
plt.xlabel('Top-k Occupancy (%)')
plt.ylabel('Occurance (%)')

plt.savefig('dummy.png', bbox_inches='tight', pad_inches=0.05)

chunk_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
eps = []
for cs in chunk_sizes:
    _, _, ep = compute(chunk_size=cs)
    eps.append(ep)

plt.figure(figsize=(4,3))
plt.plot(chunk_sizes, eps)

plt.grid()
plt.xlabel('Chunk Size')
plt.ylabel('Empty Chunks (%)')
plt.ylim(0, 100)

plt.savefig('dummy_empty.png', bbox_inches='tight', pad_inches=0.05)