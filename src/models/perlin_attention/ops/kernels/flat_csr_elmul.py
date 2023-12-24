import torch
import warnings
import triton
import triton.language as tl

def __flat_csr_elmul_py(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    in_values: torch.Tensor,
    out_values: torch.Tensor,
    other: torch.Tensor,
    N, H, T_DST, T_SRC, R, Z
):
    for n in range(N):
        for ir in range(R):
            crow_start = crow_indices[n, ir]
            crow_end = crow_indices[n, ir+1]
            
            idx_ht = col_indices[n, crow_start:crow_end]
            values = in_values[n, crow_start:crow_end]
            
            idx_heads = idx_ht // T_SRC
            idx_cols = idx_ht % T_SRC
            
            for i in range(crow_end-crow_start):
                idx_head = idx_heads[i]
                idx_col = idx_cols[i]
                out_values[n, crow_start+i] = values[i] * other[n, idx_head, ir, idx_col]

# DBG
# @triton.autotune(configs=[
#         triton.Config({}, num_warps=1),
#         triton.Config({}, num_warps=2),
#         triton.Config({}, num_warps=4),
#         triton.Config({}, num_warps=8),
#         triton.Config({}, num_warps=16),
#         triton.Config({}, num_warps=32),
#     ],
#     key=[],
#     # key=['MAX_ROW_Z']
# )
@triton.jit
def __flat_csr_elmul_compute(
    CROW_INDICES,
    stride_crow_n, stride_crow_r,
    COL_INDICES,
    stride_col_n, stride_col_z,
    IN_VALUES,
    stride_in_n, stride_in_z,
    OUT_VALUES,
    stride_out_n, stride_out_z,
    OTHER,
    stride_other_n, stride_other_h, stride_other_tdst, stride_other_tsrc,
    N, H, T_DST, T_SRC, R, Z,
    MAX_ROW_Z: tl.constexpr, BLOCK_R: tl.constexpr,
):
    n = tl.program_id(0)
    ir = tl.program_id(1)
    ir = ir * BLOCK_R + tl.arange(0, BLOCK_R)
    ir_mask = ir < R
    
    crow_start = tl.load(
        CROW_INDICES\
            + n*stride_crow_n\
            + ir*stride_crow_r,
        mask=ir_mask
    )
    crow_end = tl.load(
        CROW_INDICES\
            + n*stride_crow_n\
            + (ir+1)*stride_crow_r,
        mask=ir_mask
    )
    
    idx_ht = tl.load(
        COL_INDICES\
            + n*stride_col_n\
            + (tl.arange(0, MAX_ROW_Z)[None,:] + crow_start[:, None])*stride_col_z,
        mask = (tl.arange(0, MAX_ROW_Z)[None, :] < (crow_end[:, None] - crow_start[:, None])) and ir_mask[:, None]
    )
    
    idx_heads = idx_ht // T_SRC
    idx_cols = idx_ht % T_SRC
    
    in_values = tl.load(
        IN_VALUES\
            + n*stride_in_n\
            + (tl.arange(0, MAX_ROW_Z)[None,:] + crow_start[:, None])*stride_in_z,
        mask = (tl.arange(0, MAX_ROW_Z)[None, :] < (crow_end[:, None] - crow_start[:, None])) and ir_mask[:, None]
    )
    other_values = tl.load(
        OTHER\
            + n*stride_other_n\
            + idx_heads*stride_other_h\
            + ir[:, None]*stride_other_tdst\
            + idx_cols*stride_other_tsrc,
        mask=(tl.arange(0, MAX_ROW_Z)[None, :] < (crow_end[:, None] - crow_start[:, None])) and ir_mask[:, None]
    )
    
    out_values = in_values * other_values
    
    tl.store(
        OUT_VALUES\
            + n*stride_out_n\
            + (tl.arange(0, MAX_ROW_Z)[None, :] + crow_start[:, None])*stride_out_z,
        out_values,
        mask=(tl.arange(0, MAX_ROW_Z)[None, :] < (crow_end[:, None] - crow_start[:, None])) and ir_mask[:, None]
    )

def flat_csr_elmul(probs: torch.Tensor, dense: torch.Tensor, max_z_per_row:int=None):
    assert probs.is_sparse_csr
    N, T_DST, H_T = probs.shape
    _N, H, _T_DST, T = dense.shape
    assert T_DST == _T_DST
    assert N == _N
    assert H_T == H*T
    
    crow_indices = probs.crow_indices()
    col_indices = probs.col_indices()
    _N, R_1 = crow_indices.shape
    R = R_1 - 1
    assert N == _N
    _N, Z = col_indices.shape
    assert N == _N
    in_values = probs.values()
    out_values = in_values.clone()
    
    if max_z_per_row is None:
        max_z_per_row = (crow_indices[:,1:] - crow_indices[:,:-1]).max().item()
    
    # __flatten_csr_elmul_py(
    #     crow_indices, col_indices, in_values, out_values, dense,
    #     N, H, T_DST, T, R, Z
    # )
    
    MAX_ROW_Z = triton.next_power_of_2(max_z_per_row)
    BLOCK_R = 1
    if R >= 4096:
        BLOCK_R = triton.next_power_of_2(triton.cdiv(R, 2048))
    GRID_R = triton.cdiv(R, BLOCK_R)
    grid = (N, GRID_R)
    __flat_csr_elmul_compute[grid](
        crow_indices,
        crow_indices.stride(0), crow_indices.stride(1),
        col_indices,
        col_indices.stride(0), col_indices.stride(1),
        in_values,
        in_values.stride(0), in_values.stride(1),
        out_values,
        out_values.stride(0), out_values.stride(1),
        dense,
        dense.stride(0), dense.stride(1), dense.stride(2), dense.stride(3),
        N, H, T_DST, T, R, Z,
        MAX_ROW_Z, BLOCK_R,
    )
    
    return torch.sparse_csr_tensor(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=out_values,
        size=probs.shape
    )

def naive_flat_csr_elmul(probs, dense):
    return probs * dense

def test_main():
    from .....utils import seed
    from .....utils.bench import bench
    from .causal_resize_m_to_t import resize_from_m_to_t_csr
    from .causal_topk_masking import causal_topk_masking
    from .flat_csr_masked_bmm import flat_csr_masked_bmm
    from .flat_csr_to_dense import flat_csr_to_dense
    from .flat_csr_softmax import flat_csr_softmax

    seed()
    
    # N = 1
    # H = 2
    # T = 4
    # T_DST = 4
    # T_M = 4
    # K = 2
    # HID = 64
    
    N = 1
    H = 12
    T = 4096*4
    T_DST = 4096*4
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
    
    csr_probs = flat_csr_softmax(
        csr_score, H, T
    )
    if T <= 4096:
        csr_probs_dense = flat_csr_to_dense(csr_probs, T, H)
    dense_scaler = torch.randn((N, H, T, 1), device=device).expand(N, H, T_DST, T)
    
    def bench_naive():
        with torch.no_grad():
            return naive_flat_csr_elmul(
                csr_probs_dense, dense_scaler
            )
    
    def bench_sparse():
        with torch.no_grad():
            return flat_csr_elmul(csr_probs, dense_scaler)
    
    if T <= 4096:
        scaled_sparse = flat_csr_to_dense(bench_sparse(), T, H)
        scaled = bench_naive()
    idx_batch = 0
    idx_head = 0
    # print(csr_probs_dense[idx_batch, idx_head])
    # print(dense_scaler[idx_batch, idx_head])
    # print(scaled[idx_batch, idx_head])
    # print(scaled_sparse[idx_batch, idx_head])
    
    if T <= 4096:
        max_error = (scaled - scaled_sparse).abs().max()
        print(max_error)
        if max_error > 1e-1:
            warnings.warn('max error exceed threshold')
            for i in range(N):
                for j in range(H):
                    for k in range(T_DST):
                        for m in range(T):
                            err = (scaled[i,j,k,m] - scaled_sparse[i,j,k,m]).abs().item()
                            if err > 1e-1:
                                print(i,j,k,m,err,scaled[i,j,k,m],scaled_sparse[i,j,k,m])
                                return
    
    bench('sparse_elmul', bench_sparse, 0.5, 3, 'ms')
    if T <= 4096:
        bench('naive_elmul', bench_naive, 0.5, 3, 'ms')

if __name__ == '__main__':
    test_main()