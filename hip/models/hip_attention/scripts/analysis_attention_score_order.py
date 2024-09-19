import os, torch
from hip.models.hip_attention.attention1_block_gpu import load_checkouts, to_dense

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
    
    score = (
        (q[:, -1:, :] / (q.shape[-1] ** 0.5)) @\
        k[:, 16:-1024, :].repeat_interleave(HEAD // HEAD_KV, 0).permute(0, 2, 1)
    )[10, 0].cpu()
    print(score)
    
    k = 512
    _, true_topk = torch.topk(score, k=k)
    true_topk = set(true_topk.tolist())
    
    chunk_size = score.shape[0] // k
    max_scores = []
    max_scores_using_hip = []
    for i in range(0, score.shape[0], chunk_size):
        chunk = score[i:i+chunk_size]
        value, idx = torch.max(chunk, dim=0)
        true_topk_hit = 0
        for j in range(chunk_size):
            x = i + j
            if x in true_topk:
                true_topk_hit += 1
        max_scores.append((value.item(), idx.item(), i, true_topk_hit))
    
    max_scores = list(sorted(max_scores, key=lambda x: x[0], reverse=True))
    
    print(*map(lambda x: x[2], max_scores[:50]))
    print(max(map(lambda x: x[2], max_scores)))
    
    import matplotlib.pyplot as plt
    
    plt.plot(list(map(lambda x: x[-1], max_scores)))
    plt.savefig('./dummy_order.png')

if __name__ == '__main__':
    main()