import os
import torch
from hip import paged_hip_attention_11, HiPAttentionArgs11

state = torch.load(
    'cache/vllm/hip_attn/decode_example.pth', 
    map_location='cuda:0'
)

out = paged_hip_attention_11(
    q=state['q'],
    k_cache=state['k'],
    v_cache=state['v'],
    block_table=state['block_table'],
    cache_seq_lens=state['cache_seq_lens'],
    softmax_scale=state['softmax_scale'],
    args=HiPAttentionArgs11(),
)

truth = state['out']

print(out.shape, out.dtype, truth.shape, truth.dtype)
print(torch.nn.functional.mse_loss(out, truth))