import warnings
import torch
import triton
import triton.language as tl

def naive_flat_csr_softmax(scores: torch.Tensor):
    mask = scores == 0
    scores = scores.masked_fill(mask, torch.finfo(torch.float16).min * 0.5)
    probs = torch.softmax(scores, dim=-1).masked_fill_(mask, 0)
    return probs

def __flat_csr_softmax_py(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    in_values: torch.Tensor,
    out_values: torch.Tensor,
    N: int,
    R: int,
    H: int,
    T_SRC: int,
):
    for n in range(N):
        for ir in range(R):
            crow_start = crow_indices[n, ir]
            crow_end = crow_indices[n, ir+1]
            
            row = in_values[n, crow_start:crow_end]
            
            col_idx = col_indices[n, crow_start:crow_end]
            head_idx = col_idx // T_SRC
            output = torch.zeros_like(row)
            
            for ih in range(H):
                head_mask = (head_idx == ih).float()
                row_per_head = row * head_mask + torch.tensor([torch.finfo(torch.float16).min * 0.5,], device=head_mask.device) * (1-head_mask)
                
                row_max = torch.max(row_per_head)
                row_minus_max = row_per_head - row_max
                numerator = torch.exp(row_minus_max)
                denominator = torch.sum(numerator)
                output = output * (1-head_mask) + head_mask * (numerator / denominator)
                
            out_values[n, crow_start:crow_end] = output

# @triton.autotune(configs=[
#         triton.Config({}, num_warps=1),
#         triton.Config({}, num_warps=2),
#         triton.Config({}, num_warps=4),
#         triton.Config({}, num_warps=8),
#         triton.Config({}, num_warps=16),
#         triton.Config({}, num_warps=32),
#     ],
#     key=['BLOCK_Z','BLOCK_R']
# )
@triton.jit
def __flat_csr_softmax_compute(
    CROW_INDICES,
    stride_crow_n, stride_crow_r,
    COL_INDICES,
    stride_col_n, stride_col_z,
    IN_VALUES,
    stride_in_n, stride_in_z,
    OUT_VALUES,
    stride_out_n, stride_out_z,
    N, R, H, T_SRC,
    BLOCK_Z: tl.constexpr, BLOCK_R:tl.constexpr,
):
    n = tl.program_id(0)
    pid_ir = tl.program_id(1)
    
    for i in range(BLOCK_R):
        ir = pid_ir*BLOCK_R + i
        ir_mask = ir < R
        
        crow_start = tl.load(
            CROW_INDICES\
                + n*stride_crow_n\
                + ir*stride_crow_r,
            mask=ir_mask,
        )
        crow_end = tl.load(
            CROW_INDICES\
                + n*stride_crow_n\
                + (ir+1)*stride_crow_r,
            mask=ir_mask,
        )
        
        row_mask = (tl.arange(0, BLOCK_Z) + crow_start) < crow_end
        row = tl.load(
            IN_VALUES\
                + n*stride_in_n\
                + (tl.arange(0, BLOCK_Z) + crow_start)*stride_in_z,
            mask=row_mask & ir_mask,
            other=-float('inf')
        )
        
        col_idx = tl.load(
            COL_INDICES\
                + n*stride_col_n\
                + (tl.arange(0, BLOCK_Z) + crow_start)*stride_col_z,
            mask=row_mask & ir_mask,
            other=0,
        )
        head_idx = col_idx // T_SRC
        
        output = tl.zeros_like(row)
        for ih in range(H):
            head_mask = head_idx == ih
            row_per_head = tl.where(head_mask, row, -float('inf'))
            
            row_max = tl.max(row_per_head)
            row_minus_max = row_per_head - row_max
            numerator = tl.exp(row_minus_max)
            denominator = tl.sum(numerator)
            softmax_result = numerator / denominator
            
            output += tl.where(head_mask, softmax_result, 0)
        
        tl.store(
            OUT_VALUES\
                + n*stride_out_n\
                + (tl.arange(0, BLOCK_Z) + crow_start)*stride_out_z,
            output,
            mask=row_mask & ir_mask
        )

def flat_csr_softmax(scores: torch.Tensor, H:int, T_SRC:int, max_z_per_row:int=None):
    assert scores.is_sparse_csr
    crow_indices = scores.crow_indices()
    col_indices = scores.col_indices()
    in_values = scores.values()
    out_values = in_values.clone()
    
    if max_z_per_row is None:
        max_z_per_row = (crow_indices[:,1:] - crow_indices[:,:-1]).max().item()
    
    N, R_1 = crow_indices.shape
    R = R_1 - 1
    N, Z = col_indices.shape
    
    # __flatten_csr_softmax_py(
    #     crow_indices, col_indices, in_values, out_values,
    #     N, R, H, T_SRC
    # )
    
    BLOCK_Z = triton.next_power_of_2(max_z_per_row)
    num_warps = 4
    if BLOCK_Z >= 2048:
        num_warps = 8
    if BLOCK_Z >= 4096:
        num_warps = 16
    BLOCK_R = 1
    if R >= 8192:
        BLOCK_R = triton.cdiv(R, 4096)
    grid = (N, triton.cdiv(R, BLOCK_R))
    # print(grid)
    __flat_csr_softmax_compute[grid](
        crow_indices,
        crow_indices.stride(0), crow_indices.stride(1),
        col_indices,
        col_indices.stride(0), col_indices.stride(1),
        in_values,
        in_values.stride(0), in_values.stride(1),
        out_values,
        out_values.stride(0), out_values.stride(1),
        N, R, H, T_SRC,
        BLOCK_Z, BLOCK_R,
        num_warps=num_warps,
    )
    
    return torch.sparse_csr_tensor(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=out_values,
        size=scores.shape,
    )

def test_main():
    from .....utils import seed
    from .....utils.bench import bench
    from .causal_resize_m_to_t import resize_from_m_to_t_csr
    from .causal_topk_masking import causal_topk_masking
    from .flat_csr_masked_bmm import flat_csr_masked_bmm
    from .flat_csr_to_dense import flat_csr_to_dense

    seed()
    
    N = 1
    H = 2
    T = 300
    T_DST = 300
    T_M = 4
    K = 2
    HID = 64
    
    N = 1
    H = 12
    T = 4096
    T_DST = 4096
    T_M = 128
    K = 64
    HID = 64
    
    FP_MIN = torch.finfo(torch.float16).min * 0.5
    device = 0
    
    estimated_scores = torch.randn((N, H, T_DST, T_M), device=device)
    estimated_probs = torch.softmax(estimated_scores, dim=-1)
    causal_attention_mask = ((torch.arange(T, device=device).view(1, T) > torch.arange(T, device=device).view(T, 1)) * FP_MIN).view(1, 1, T, T)
    causal_attention_mask = causal_attention_mask[:, :, -T_DST:, :]
    attention_mask = causal_attention_mask[:,:,-1:,:]
    dst_attention_mask = causal_attention_mask[:,:,:,:1]
    
    compressed_mask = causal_topk_masking(
        estimated_probs, 
        k=K, 
        attention_mask=attention_mask, 
        dst_attention_mask=dst_attention_mask, 
        causal_attention_mask=causal_attention_mask
    )
    
    csr_mask = resize_from_m_to_t_csr(
        compressed_mask, 0, K,
        target_width=causal_attention_mask.shape[-1]
    )
    
    query_layer = torch.randn((N, H, T_DST, HID), device=device)
    key_layer = torch.randn((N, H, T, HID), device=device)
    csr_score = flat_csr_masked_bmm(
        query_layer, 
        key_layer, 
        csr_mask, 
        None
    )
    # csr_score_dense = csr_score.to_dense().view(N, T_DST, H, T).transpose(1, 2).reshape(N, H, T_DST, T)
    csr_score_dense = flat_csr_to_dense(csr_score, T, H)
    
    def bench_naive():
        with torch.no_grad():
            return naive_flat_csr_softmax(
                csr_score_dense
            )
    
    def bench_sparse():
        with torch.no_grad():
            return flat_csr_softmax(csr_score, H, T)
    
    # probs_sparse = bench_sparse().to_dense().view(N, T_DST, H, T).transpose(1, 2).reshape(N, H, T_DST, T)
    probs_sparse = flat_csr_to_dense(bench_sparse(), T, H)
    probs = bench_naive()
    
    # print(score[0,0])
    # print(score_sparse[0,0,9,:])
    
    max_error = (probs - probs_sparse).abs().max()
    print(max_error)
    if max_error > 1e-1:
        warnings.warn('max error exceed threshold')
        for i in range(N):
            for j in range(H):
                for k in range(T_DST):
                    for m in range(T):
                        err = (probs[i,j,k,m] - probs_sparse[i,j,k,m]).abs().item()
                        if err > 1e-1:
                            print(i,j,k,m,err,probs[i,j,k,m],probs_sparse[i,j,k,m])
                            return
    
    bench('sparse_softmax', bench_sparse, 0.5, 3, 'ms')
    bench('naive_softmax', bench_naive, 0.5, 3, 'ms')

if __name__ == '__main__':
    test_main()