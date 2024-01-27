import os
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from src.models.tree_attention.attention1_block_gpu import tree_attention, attention_matrix, sparse_attention, load_checkouts

def to_dense(
    indices: np.ndarray, 
    ks: np.ndarray, 
    value: np.ndarray,
    N, T_DST, T_SRC, BLOCK_SIZE
):
    # print(indices.shape, ks.shape, value.shape, T_DST, T_SRC)
    out = torch.zeros((N, T_DST, T_SRC), device=indices.device, dtype=value.dtype)
    for idx_n in range(N):
        for idx_bdst in range(indices.shape[1]):
            for idx_k in range(indices.shape[2]):
                if idx_k < ks[idx_n, idx_bdst]:
                    idx_tsrc = indices[idx_n, idx_bdst, idx_k]
                    out[
                        idx_n, 
                        idx_bdst * BLOCK_SIZE: (idx_bdst + 1) * BLOCK_SIZE, 
                        idx_tsrc: idx_tsrc + BLOCK_SIZE
                    ] = value[
                        idx_n,
                        idx_bdst * BLOCK_SIZE: (idx_bdst + 1) * BLOCK_SIZE, 
                        idx_k * BLOCK_SIZE: (idx_k + 1) * BLOCK_SIZE
                    ]
    return out

def imsave(im: Tensor, name: str, gamma: float = 0.2, idx_batch: int = -1):
    im = im[idx_batch].cpu().detach().numpy() ** gamma
    plt.clf()
    plt.title(name)
    plt.imshow(im)
    plt.colorbar()
    os.makedirs('./saves/models/test_tree_block_fwd', exist_ok=True)
    path = f'./saves/models/test_tree_block_fwd/{name}.png'
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.clf()
    print('saved', path)

def main():
    q, k, v, out = load_checkouts(idx=5, window=4, seq_len=4096, dtype=torch.float16)
    
    q = q[:, 2048:, :].contiguous()
    
    q_backup = q.clone()
    k_backup = k.clone()
    v_backup = v.clone()
    
    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape
    BLOCKSIZE = 8
    mask_k = 512
    scale_up = 2
    w_start = mask_k * scale_up
    n_patches = mask_k / scale_up
    
    indices, ks, probs, scores = attention_matrix(
        q, k, 
        w_start,
        n_patches,
        mask_k,
        scale_up,
        BLOCK_SIZE=BLOCKSIZE
    )
    
    print(q.shape, indices.shape, ks.shape, probs.shape, scores.shape)
    print(indices.min(), indices.max())
    
    probs_dense = to_dense(
        indices.cpu(), ks.cpu(), probs.cpu(), 
        N, TDST, TSRC, BLOCKSIZE
    ).to(indices.device)
    
    scores_dense = to_dense(
        indices.cpu(), ks.cpu(), scores.cpu(),
        N, TDST, TSRC, BLOCKSIZE,
    ).to(indices.device)
    
    mask_dense = probs_dense <= 1e-7
    mask_dense = mask_dense.to(probs.dtype) * torch.finfo(probs.dtype).min
    
    scores_truth = torch.bmm(q, k.transpose(-1, -2))
    print(scores_truth.dtype, mask_dense.dtype)
    probs_truth = (scores_truth + mask_dense).softmax(dim=-1)
    
    probs_error_map = (probs_truth - probs_dense).abs()
    probs_error = probs_error_map[:].max()
    print((probs_error_map == probs_error).nonzero())
    
    scores_dense = scores_dense * (probs_dense > 1e-12)
    scores_truth = scores_truth * (probs_dense > 1e-12)
    scores_error_map = (scores_truth - scores_dense).abs()
    scores_error = scores_error_map.max()
    
    imsave(probs, 'probs')
    imsave(probs_truth, 'probs_truth')
    imsave(probs_dense, 'probs_dense')
    imsave(probs_error_map, 'probs_error_map')
    imsave(scores_truth.abs(), 'scores_truth', gamma=1.0)
    imsave(scores_dense.abs(), 'scores_dense', gamma=1.0)
    imsave(scores_error_map, 'scores_error_map', gamma=1.0)
    print('scores_error', scores_error)
    print('probs_error', probs_error)
    
    context_dense = sparse_attention(
        v.contiguous(), indices, ks, probs, BLOCKSIZE
    )
    context_truth = torch.bmm(probs_dense, v)
    
    context_error_map = (context_dense - context_truth).abs()
    # print(context_error_map)
    
    context_error = context_error_map.max()
    context_error_loc = (context_error_map == context_error).nonzero()
    
    imsave(context_error_map, 'context_error_map', gamma=1.0, idx_batch=2)
    print(torch.std_mean(context_dense), torch.std_mean(context_truth))
    print(context_error, context_error_loc)
    
    # print(context_error_map[1, 2, :])
    # print(context_dense[1, 2, :])
    # print(context_truth[1, 2, :])
    
    for i in range(3):
        context_tree, (indices_tree, ks_tree, probs_tree) = tree_attention(
            q, k, v, w_start, n_patches, mask_k, scale_up, BLOCKSIZE
        )
    context_tree_error_map = (context_tree - context_dense).abs()
    context_tree_error = context_tree_error_map.max()
    print((q - q_backup).abs().sum())
    print((k - k_backup).abs().sum())
    print((v - v_backup).abs().sum())
    print('cte', context_tree_error)
    
    probs_dense_tree = to_dense(
        indices_tree.cpu(), ks_tree.cpu(), probs_tree.cpu(), 
        N, TDST, TSRC, BLOCKSIZE
    ).to(indices.device)
    imsave(probs_dense_tree, 'probs_dense_tree')

if __name__ == '__main__':
    for i in range(1):
        print('='*80)
        main()
        print('='*80)