import torch
import tqdm
from hip import hip_attention_11
from hip.models.hip_attention.attention1_block_gpu import load_checkouts, to_dense
import time

def main():
    refresh_interval = 8
    samples = 500
    batch_size = 32
    sequential_masking = False
    
    q, k, v, out, cos, sin = load_checkouts(
        idx=0, 
        window=40, 
        seq_len=32768, 
        return_cos_sin=True, 
        dtype=torch.bfloat16
    )
    HEAD = q.shape[0]
    HEAD_KV = k.shape[0]
    
    def reshape(x, HEAD):
        N, T, H = x.shape
        x = x.contiguous()\
            .view(N // HEAD, HEAD, T, H)\
            .permute(0, 2, 1, 3)\
            .contiguous()
        assert x.shape == (N // HEAD, T, HEAD, H)
        assert x.is_contiguous()
        return x

    q = reshape(q, HEAD)
    k = reshape(k, HEAD_KV)
    v = reshape(v, HEAD_KV)
    out = reshape(out, HEAD)
    
    q = q[:, -1:, :, :]
    q = q.expand(batch_size, -1, -1, -1)
    k = k.expand(batch_size, -1, -1, -1)
    v = v.expand(batch_size, -1, -1, -1)
    
    sa_stream = torch.cuda.Stream()
    mask_stream = torch.cuda.Stream()
    
    _, meta = hip_attention_11(q, k, v)
    
    t_start = time.time()
    
    for i in tqdm.tqdm(range(samples), dynamic_ncols=True):
        if i == 3:
            t_start = time.time()
        with torch.cuda.stream(mask_stream):
            _, new_meta = hip_attention_11(q, k, v, mask_only=True)
        
        for isa in range(refresh_interval):
            with torch.cuda.stream(sa_stream if not sequential_masking else mask_stream):
                context, _ = hip_attention_11(q, k, v, previous_metadata=meta)
        
        torch.cuda.default_stream().wait_stream(mask_stream)
        torch.cuda.default_stream().wait_stream(sa_stream)
        torch.cuda.synchronize()
    
    t_end = time.time()
    
    elapsed = t_end - t_start
    print(elapsed * 1000)

if __name__ == '__main__':
    main()