import torch
import os

from hip_research.utils.load_checkouts import load_checkouts

from hip_attn.v1_2.query_sparse_attention import query_sparse_attention

def main():
    B, H, H_KV, S, D = 1, 32, 8, 32768, 128
    delta_w = 16
    bsa_block_size_q = 64 // delta_w
    bsa_block_size_k = 64
    bsa_top_block_k = 1024 // bsa_block_size_k
    import math
    scale = math.sqrt(1 / D)
    device = "cuda"
    dtype = torch.bfloat16
    source = "checkout"
    
    if source == "checkout":
        seq_len = int(os.getenv("SEQ_LEN", "131072"))
        query_seq_dups = int(os.getenv("Q_DUPS", "-1"))
        seq_dups = int(os.getenv("DUPS", "1"))
        if query_seq_dups < 0:
            query_seq_dups = seq_dups

        assert seq_dups > 0

        using_extend = True

        q, k, v, out, cos, sin = load_checkouts(
            idx=0,
            window=40,
            seq_len=seq_len,
            return_cos_sin=True,
            derope=using_extend,
            dtype=torch.bfloat16,
        )
        seq_len = seq_len * seq_dups

        q = q.repeat(1, query_seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
        k = (
            k.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
        )
        v = (
            v.repeat(1, seq_dups, 1).permute(1, 0, 2).contiguous().unsqueeze(0)
        )
        if cos is not None:
            cos = cos.repeat(seq_dups, 1)
            sin = sin.repeat(seq_dups, 1)
        
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        
        print(q.shape, k.shape, v.shape, q.dtype, k.dtype, v.dtype)
    else:
        q, k, v = (
            torch.randn(B, H, S, D, device=device, dtype=dtype), 
            torch.randn(B, H_KV, S, D, device=device, dtype=dtype), 
            torch.randn(B, H_KV, S, D, device=device, dtype=dtype)
        )
    mask = torch.arange(0, S, delta_w, device=device)[None, :].repeat(B, 1)
    
    def fwd(return_bsa_indices: bool, debug: bool = False):
        if return_bsa_indices:
            out, (bsa_idx, block_sums) = query_sparse_attention(
                q=q, 
                k=k, 
                v=v, 
                mask=mask, 
                sm_scale=scale, 
                k_cache=None, 
                v_cache=None, 
                block_table=None, 
                return_bsa_indices=True,
                bsa_top_block_k=bsa_top_block_k,
                bsa_block_size_q=bsa_block_size_q,
                bsa_block_size_k=bsa_block_size_k,
            )
            if debug:
                print(bsa_idx)
        else:
            out = query_sparse_attention(
                q=q, 
                k=k, 
                v=v, 
                mask=mask, 
                sm_scale=scale, 
                k_cache=None, 
                v_cache=None, 
                block_table=None, 
                return_bsa_indices=False,
            )
    
    fwd(return_bsa_indices=True, debug=True)
    print(f"[done] return_bsa_indices=True")

    fwd(return_bsa_indices=False, debug=True)
    print(f"[done] return_bsa_indices=False")
    
    def latency(fn, n_sample = 10):
        elapsed = []
        for i in range(n_sample):
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            if i > 3:
                elapsed.append(start.elapsed_time(end))
        return sum(elapsed) / len(elapsed)
    
    latency_return_mask = latency(lambda: fwd(return_bsa_indices=True))
    print(f'with mask: {latency_return_mask:.2f} ms took')
    latency_original = latency(lambda: fwd(return_bsa_indices=False))
    print(f'without mask: {latency_original:.2f} ms took')

if __name__ == "__main__":
    main()