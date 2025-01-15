import torch
from hip.models.hip_attention.gen3.attention_extend import(
    load_checkouts
)

q, k, v, out, cos, sin = load_checkouts(
    idx=0, 
    window=32, 
    seq_len=131072, 
    dtype=torch.bfloat16, 
    return_cos_sin=True, 
    derope=True,
)

print(q.shape)