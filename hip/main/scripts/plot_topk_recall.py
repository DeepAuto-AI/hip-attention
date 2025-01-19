import json
import os
from typing import Literal
from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
import triton

from hip.models.hip_attention.gen3.attention_extend import(
    load_checkouts, 
    safe_stride,
)

from hip.models.hip_attention.gen3.attention_extend import dual_stage_quadratic_hip_attention as hip_attention_12
from hip.models.hip_attention.gen3.attention_extend import HiPAttentionArgs as HiPAttentionArgs12

from hip.models.hip_attention.attention2_draft_prefetch import hip_attention as hip_attention_11
from hip.models.hip_attention.attention2_draft_prefetch import HiPAttentionArgs as HiPAttentionArgs11

# X: topk k
# Y: recall
# Legen: method

def compute_oracle(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, top_k: int):
    q = q[0, -1, 0, :]
    k = k[0, :, 0, :]
    scores = (q[None, :] @ k.T)[0]
    return scores.softmax(-1).float().cpu().numpy()

def compute_gen2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, top_k: int, sampling_method: str):
    _, metadata = hip_attention_11(
        q=q[:, -1:, :, :], k=k, v=v, args=HiPAttentionArgs11(
            sample_method=sampling_method,
            block_size_k=8,
            mask_k=top_k,
            sliding_window_size=1024,
            sink_token_size=128,
        )
    )
    indices = metadata.indices.view(-1)
    indices = (indices[:, None] + torch.arange(0, 2, device=q.device)).view(-1)
    return indices.cpu().numpy()

def compute_gen3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, top_k: int):
    _, metadata = hip_attention_12(
        q=q[:, -1:, :, :], k=k, v=v, args=HiPAttentionArgs12(
            second_stage_k=top_k,
            sliding_window_size=1024,
            sink_token_size=128,
        )
    )
    indices = metadata.indices.view(-1)
    indices = (indices[:, None] + torch.arange(0, 8, device=q.device)[None, :]).view(-1)
    return indices.cpu().numpy()

def compute_infllm(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, top_k: int):
    num_repr = 4
    chunk_size = 128
    num_repr = min(chunk_size, num_repr)
    reprs = []
    for i_start in range(0, k.shape[1], chunk_size):
        i_end = i_start + chunk_size
        tq = q[0, i_start: i_end, 0, :]
        tk = k[0, i_start: i_end, 0, :]
        repr_loc = (tq.mean(dim=0, keepdim=True) @ tk.T)[0].topk(k=num_repr).indices
        reprs.append(repr_loc)
    reprs = torch.stack(reprs, dim=0)
    
    curr_q = q[0, -1, 0, :]
    k = k[0, :, 0, :]
    scores = (curr_q[None, :] @ k.T)[0]
    scores = scores.view(-1, chunk_size)
    scores[:(128 // chunk_size)] = -32000
    scores[-(1024 // chunk_size):] = -32000
    indices = scores.gather(dim=1, index=reprs).max(dim=-1).values.topk(k=top_k // chunk_size).indices * chunk_size
    indices = (indices[:, None] + torch.arange(0, chunk_size, device=q.device)[None, :]).view(-1)
    
    return indices.cpu().numpy()

def compute_all(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, top_k: int):
    orcale_indices = compute_oracle(q, k, v, top_k)
    
    # gen2_left_indices = compute_gen2(q, k, v, top_k, 'first')
    gen2_center_indices = compute_gen2(q, k, v, top_k, 'center')
    # gen2_right_indices = compute_gen2(q, k, v, top_k, 'last')
    gen3_indices = compute_gen3(q, k, v, top_k)
    infllm_indices = compute_infllm(q, k, v, top_k)
    
    def recall(est: np.ndarray, oracle: np.ndarray):
        # est = set(est.tolist())
        # oracle = set(oracle.tolist())
        # intersect = set.intersection(est, oracle)
        # return len(intersect) / len(oracle)
        
        x = oracle[est].sum().item()
        x += oracle[:128].sum().item()
        x += oracle[-1024:].sum().item()
        
        return x * 100
    
    # gen2_left_recall = recall(gen2_left_indices, orcale_indices)
    gen2_center_recall = recall(gen2_center_indices, orcale_indices)
    # gen2_right_recall = recall(gen2_right_indices, orcale_indices)
    gen3_recall = recall(gen3_indices, orcale_indices)
    infllm_recall = recall(infllm_indices, orcale_indices)
    
    return {
        # 'gen2_left': gen2_left_recall,
        'HiP ($b_k$=8)': gen2_center_recall,
        # 'gen2_right': gen2_right_recall,
        'InfLLM': infllm_recall,
        'InfiniteHiP (Ours)': gen3_recall,
    }

num_heads = 32
chunk_sizes = [256, 512, 1024, 2048, 4096, 8192]

def run_exp():
    q, k, v, out, cos, sin = load_checkouts(
        idx=0, 
        window=999, 
        seq_len=131072, 
        dtype=torch.bfloat16, 
        return_cos_sin=True, 
        derope=False,
    )

    q_rope, k_rope, _, _, _, _ = load_checkouts(
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
    q_rope = reshape(q_rope).to(0)
    k_rope = reshape(k_rope).to(0)
    
    data = {}

    for target_head in range(num_heads):
        current_q = q[:, :, target_head:target_head+1, :]
        curr_k = k[:, :, target_head // (q.shape[2] // k.shape[2]):target_head // (q.shape[2] // k.shape[2])+1, :]
        curr_v = v[:, :, target_head // (q.shape[2] // k.shape[2]):target_head // (q.shape[2] // k.shape[2])+1, :]
        current_q_rope = q_rope[:, :, target_head:target_head+1, :]
        k_rope = k_rope[:, :, target_head // (q.shape[2] // k.shape[2]):target_head // (q.shape[2] // k.shape[2])+1, :]

        for i, top_k in enumerate(chunk_sizes):
            results = compute_all(current_q, curr_k, curr_v, top_k)
            for key in results:
                if key not in data:
                    data[key] = [0, ] * len(chunk_sizes)
                data[key][i] += results[key]
    
    os.makedirs('./saves/plot_topk_recall', exist_ok=True)
    with open('./saves/plot_topk_recall/data.json', 'w') as f:
        json.dump(data, f)

def render_plot():
    with open('./saves/plot_topk_recall/data.json', 'r') as f:
        data = json.load(f)
    
    plt.figure(figsize=(4, 3))

    for legend in data:
        plt.plot(chunk_sizes, (np.array(data[legend]) / num_heads).tolist(), label=legend)

    plt.grid()
    plt.title('Recall of Attention Probabilities')
    plt.xlabel('Top-k Tokens')
    plt.ylabel('Recall (%)')
    # plt.xscale('log', base=2)
    plt.legend()

    plt.savefig('./saves/plot_topk_recall/plot.pdf', bbox_inches='tight', pad_inches=0.05)
    plt.savefig('./saves/plot_topk_recall/plot.png', bbox_inches='tight', pad_inches=0.05)
    
if __name__ == '__main__':
    # run_exp()
    render_plot()