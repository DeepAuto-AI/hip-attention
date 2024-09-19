import os, torch
import tqdm
from hip.models.hip_attention.attention1_block_gpu import load_checkouts, to_dense
from hip import hip_attention, HiPAttentionArgs

def main():
    seq_len = 1024 * 128
    seq_repeat = 1
    batch_repeat = 1
    
    q, k, v, out, cos, sin = load_checkouts(
        idx=0, 
        window=40, 
        seq_len=seq_len, 
        return_cos_sin=True, 
        dtype=torch.bfloat16
    )
    HEAD = q.shape[0]
    HEAD_KV = k.shape[0]
    
    if seq_repeat > 1 or batch_repeat > 1:
        q = q.repeat(batch_repeat, seq_repeat, 1)
        k = k.repeat(batch_repeat, seq_repeat, 1)
        v = v.repeat(batch_repeat, seq_repeat, 1)
        out = out.repeat(batch_repeat, seq_repeat, 1)
        cos = cos.repeat(seq_repeat, 1)
        sin = sin.repeat(seq_repeat, 1)
    
    # k = k.view(k.shape[0], -1, 2, k.shape[2]).mean(2)
    
    q = q.cuda()
    k = k.cuda()
    
    target_head = 10
    
    score = (
        (q[:, -1:, :] / (q.shape[-1] ** 0.5)) @\
        k[:, 16:-1024, :].repeat_interleave(HEAD // HEAD_KV, 0).permute(0, 2, 1)
    )[target_head, 0].cpu()
    print(score)
    
    mask_k = 512
    token_budget = 512
    _, true_topk = torch.topk(score, k=mask_k)
    true_topk = set(true_topk.tolist())
    
    _, metadata = hip_attention(
        q=q[:, -1:, :].unsqueeze(0).permute(0, 2, 1, 3),
        k=k[:, 16:-1024, :].repeat_interleave(HEAD // HEAD_KV, 0).unsqueeze(0).permute(0, 2, 1, 3),
        v=None,
        mask_only=True,
        args=HiPAttentionArgs(
            mask_k=token_budget,
            block_size_k=1,
            block_stride_k=1,
            # randomize_mask=True
        )
    )
    
    hip_topk_indices = set(metadata.indices[target_head, -1].tolist())
    recall = len(set.intersection(hip_topk_indices, true_topk)) / len(true_topk)
    precision = len(set.intersection(hip_topk_indices, true_topk)) / len(hip_topk_indices)
    print('recall of hip', recall)
    print('precision of hip', precision)
    print('f1 of hip', 2 / (1/recall + 1/precision))
    
    chunk_size = score.shape[0] // mask_k
    max_scores = []
    max_scores_using_hip = []
    for i in tqdm.tqdm(range(0, score.shape[0], chunk_size), dynamic_ncols=True):
        chunk = score[i:i+chunk_size]
        value, idx = torch.max(chunk, dim=0)
        true_topk_hit = 0
        for j in range(chunk_size):
            x = i + j
            if x in true_topk:
                true_topk_hit += 1
        
        hip_topk = 1
        bk = 2
        _, metadata = hip_attention(
            q[:, -1:, :].unsqueeze(0).permute(0, 2, 1, 3), 
            k[:, 16:-1024, :][:, i:i+chunk_size, :].unsqueeze(0).permute(0, 2, 1, 3),
            None,
            args=HiPAttentionArgs(
                mask_k=hip_topk * bk,
                block_size_k=bk,
                block_stride_k=1,
                sink_token_size=0,
                sliding_window_size=0,
            ),
            mask_only=True,
        )
        hip_indices = (metadata.indices[target_head, -1] + bk // 2).tolist()
        
        hip_score = (
            (q[:, -1:, :] / (q.shape[-1] ** 0.5)) @\
            k\
                [:, 16:-1024, :]\
                [:, i:i+chunk_size, :]\
                [:, hip_indices, :]\
                .permute(0, 2, 1)\
                .repeat_interleave(q.shape[0] // k.shape[0], 0)
        )
        hip_score = hip_score[target_head, 0]
        
        # print(hip_indices, hip_score, idx.item(), value.item())
        max_scores.append((value.item(), idx.item(), i, true_topk_hit))
        max_scores_using_hip.append((hip_score[0].item(), hip_indices[0], i, true_topk_hit))
    
    max_scores = list(sorted(max_scores, key=lambda x: x[0], reverse=True))
    max_scores_using_hip = list(sorted(max_scores_using_hip, key=lambda x: x[0], reverse=True))
    
    print(*map(lambda x: x[2], max_scores[:50]))
    print(max(map(lambda x: x[2], max_scores)))
    
    import matplotlib.pyplot as plt
    
    plt.clf()
    plt.plot(list(map(lambda x: x[-1], max_scores)), label='top1 (oracle)')
    plt.plot(list(map(lambda x: x[-1], max_scores_using_hip)), label='hip')
    plt.legend()
    plt.savefig('./dummy_order.png')
    
    plt.clf()
    plt.scatter(
        list(map(lambda x: x[0], max_scores)),
        list(map(lambda x: x[0], max_scores_using_hip))
    )
    plt.xlabel('ground truth score')
    plt.ylabel('hip est. score')
    plt.savefig('./dummy_order_score.png')

if __name__ == '__main__':
    main()