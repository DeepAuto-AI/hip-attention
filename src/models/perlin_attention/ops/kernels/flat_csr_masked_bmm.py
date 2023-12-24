import math
import warnings
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

def __flat_csr_masked_bmm_py(
    crow_indices,
    col_indices,
    a,
    b,
    out_values,
    N, R, T_SRC,
):
    for n in range(N): # prange
        for ir in range(R): # prange
            crow_start = crow_indices[n, ir]
            crow_end = crow_indices[n, ir+1]
            for ic in range(crow_start, crow_end):
                index_row = ir
                _index_col = col_indices[n, ic]
                index_col = _index_col % T_SRC
                index_head = _index_col // T_SRC
                a_vec = a[n, index_head, index_row]
                b_vec = b[n, index_head, index_col]
                out_values[n, ic] = torch.dot(a_vec, b_vec)

# @triton.autotune(configs=[
#         triton.Config({}, num_warps=1),
#         triton.Config({}, num_warps=2),
#         triton.Config({}, num_warps=4),
#         triton.Config({}, num_warps=8),
#         triton.Config({}, num_warps=16),
#         triton.Config({}, num_warps=32),
#     ],
#     key=['BLOCK_ROW','BLOCK_COL','BLOCK_HID']
# )
@triton.jit
def __flat_csr_masked_bmm_compute(
    CROW_INDICES,
    stride_crow_n, stride_crow_r1,
    COL_INDICES,
    stride_col_n, stride_col_z,
    A,
    stride_a_n, stride_a_h, stride_a_t, stride_a_d,
    B,
    stride_b_n, stride_b_h, stride_b_t, stride_b_d,
    OUT_VALUES,
    stride_out_n, stride_out_z,
    N, R, T_SRC, HID,
    GRID_ROW, GRID_COL,
    BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr, BLOCK_HID: tl.constexpr,
):
    n = tl.program_id(0)
    pid_ir = tl.program_id(1)
    pid_icol = tl.program_id(2)
    
    for _ir in range(BLOCK_ROW):
        # ir = pid_ir * BLOCK_ROW + _ir
        ir = _ir * GRID_ROW + pid_ir
        ir_mask = ir < R
        
        crow_start = tl.load(
            CROW_INDICES\
                + n*stride_crow_n\
                + ir*stride_crow_r1,
            mask=ir_mask
        )
        crow_end = tl.load(
            CROW_INDICES\
                + n*stride_crow_n\
                + (ir+1)*stride_crow_r1,
            mask=ir_mask
        )
        
        index_row = ir
        
        for ic in range(BLOCK_COL):
            # index_col = index_cols[i]
            # index_head = index_heads[i]
            icol = (ic + pid_icol * BLOCK_COL + crow_start)
            # icol = (ic * GRID_COL + pid_icol + crow_start)
            _index_col = tl.load(
                COL_INDICES\
                    + n*stride_col_n\
                    + icol*stride_col_z,
                mask=(icol<crow_end) & ir_mask,
            )
            index_col = _index_col % T_SRC
            index_head = _index_col // T_SRC
            
            accumulator = 0.0
            for ih in range(0, tl.cdiv(HID, BLOCK_HID)):
                index_hids = tl.arange(0, BLOCK_HID) + ih*BLOCK_HID
                index_hids_mask = index_hids < HID
                
                a_vec = tl.load(
                    A\
                        + n*stride_a_n\
                        + index_head*stride_a_h\
                        + index_row*stride_a_t\
                        + index_hids*stride_a_d,
                    mask = index_hids_mask & ir_mask,
                    other = 0
                )
                b_vec = tl.load(
                    B\
                        + n*stride_b_n\
                        + index_head*stride_b_h\
                        + index_col*stride_b_t\
                        + index_hids*stride_b_d,
                    mask = index_hids_mask & ir_mask,
                    other = 0
                )
                t = tl.sum(a_vec * b_vec)
                accumulator += t
            
            tl.store(
                OUT_VALUES\
                    + n*stride_out_n\
                    + icol*stride_out_z,
                accumulator,
                mask=(icol < crow_end) & ir_mask
            )
    
    # accumulator = accumulator.to(OUT_VALUES.dtype)
    
    # tl.store(
    #     OUT_VALUES\
    #         + n*stride_out_n\
    #         + ics*stride_out_z,
    #     accumulator,
    #     mask=ics_mask
    # )

def flat_csr_masked_bmm(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, max_z_per_row: int=None):
    assert mask.is_sparse_csr
    
    assert a.ndim == b.ndim
    assert a.ndim == 4
    N, H, T_DST, HID = a.shape
    assert b.shape[:2] == (N, H)
    _, _, T_SRC, HID = b.shape
    assert mask.shape == (N, T_DST, H*T_SRC)
    
    crow_indices = mask.crow_indices()
    col_indices = mask.col_indices()
    out_values = mask.values().clone()
    
    assert crow_indices.shape[0] == N
    _, R_1 = crow_indices.shape
    R = R_1 - 1
    _, Z = col_indices.shape
    assert out_values.shape == (N, Z)
    
    # __flatten_csr_masked_bmm_py(
    #     crow_indices, col_indices, a, b, out_values,
    #     N, R, T_SRC
    # )
    
    if max_z_per_row is None:
        max_z_per_row = mask.crow_indices()
        max_z_per_row = (max_z_per_row[:,1:] - max_z_per_row[:,:-1]).max().item()
    
    #TODO improve following heuristics
    n_warps = 1
    BLOCK_ROW = 8
    BLOCK_COL = 8
    BLOCK_HID = 64
    grid = (N, triton.cdiv(R, BLOCK_ROW), triton.cdiv(max_z_per_row, BLOCK_COL))
    # print(grid)
    __flat_csr_masked_bmm_compute[grid](
        crow_indices,
        crow_indices.stride(0), crow_indices.stride(1),
        col_indices,
        col_indices.stride(0), col_indices.stride(1),
        a,
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        b,
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        out_values,
        out_values.stride(0), out_values.stride(1),
        N, R, T_SRC, HID,
        grid[1], grid[2],
        BLOCK_ROW, BLOCK_COL, BLOCK_HID,
        num_warps=n_warps,
    )
    
    return torch.sparse_csr_tensor(
        crow_indices=crow_indices,
        col_indices=col_indices,
        values=out_values,
        size=mask.shape,
    )
                

def naive_flat_csr_masked_bmm(a, b, mask):
    # a: N*H, T_DST, HID
    # b: N*H, T_SRC, HID
    # mask: N*H, T_DST, T_SRC
    
    score = torch.matmul(a, b.transpose(-1, -2))
    score = score * mask
    return score

def test_main():
    from .....utils import seed
    from .....utils.bench import bench
    from .causal_resize_m_to_t import resize_from_m_to_t_csr
    from .causal_topk_masking import causal_topk_masking
    from .flat_csr_to_dense import flat_csr_to_dense

    seed()
    
    # N = 1
    # H = 1
    # T = 30
    # T_DST = 30
    # T_M = 2
    # K = 1
    # HID = 4
    
    N = 1
    H = 12
    T = 4096
    T_DST = 4096
    T_M = 128
    K = 64
    HID = 64
    
    FP_MIN = torch.finfo(torch.float16).min * 0.5
    device = 0
    
    # estimated_scores = torch.randn((N, H, T_DST, T_M), device=device)
    # def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    #     delta = (res[0] / shape[0], res[1] / shape[1])
    #     d = (shape[0] // res[0], shape[1] // res[1])
        
    #     grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
    #     angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
    #     gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
        
    #     tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    #     dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
        
    #     n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    #     n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    #     n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    #     n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    #     t = fade(grid[:shape[0], :shape[1]])
    #     return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
    # estimated_scores = F.interpolate(rand_perlin_2d((128, 128), (16, 16)).view(1, 1, 128, 128), (T_DST, T_M)).expand(N, H, T_DST, T_M).contiguous().to(device)
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
    # csr_mask_dense = torch.clamp_max(csr_mask.to_dense(), 1).view(N, T_DST, H, T).transpose(1, 2).reshape(N, H, T_DST, T)
    csr_mask_dense = torch.clamp_max(flat_csr_to_dense(csr_mask, T, H), 1)
    
    # print(csr_mask.crow_indices())
    # print(csr_mask.col_indices())
    # print(csr_mask.values())
    # return
    
    query_layer = torch.randn((N, H, T_DST, HID), device=device)
    key_layer = torch.randn((N, H, T, HID), device=device)
    
    def bench_naive():
        with torch.no_grad():
            return naive_flat_csr_masked_bmm(
                query_layer, 
                key_layer, 
                csr_mask_dense,
            )
    
    def bench_sparse():
        with torch.no_grad():
            return flat_csr_masked_bmm(query_layer, key_layer, csr_mask, None)
    
    score = bench_naive()
    # score_sparse = bench_sparse().to_dense().view(N, T_DST, H, T).transpose(1, 2).reshape(N, H, T_DST, T)
    score_sparse = flat_csr_to_dense(bench_sparse(), T, H)
    mask_dense = flat_csr_to_dense(csr_mask, T, H) #torch.clamp_max(csr_mask.to_dense(), 333).view(N, T_DST, H, T).transpose(1, 2).reshape(N, H, T_DST, T)
    
    idx_batch = 0
    idx_head = 0
    # print('score', score[idx_batch,idx_head])
    # print('score_sparse', score_sparse[idx_batch,idx_head])
    # print(mask_dense[idx_batch,idx_head])
    # print(0,0,8,4, score_sparse[0,0,8,4])
    # return
    
    max_error = (score - score_sparse).abs().max()
    print(max_error)
    if max_error > 1e-1:
        warnings.warn('max error exceed threshold')
        for i in range(N):
            for j in range(H):
                for k in range(T_DST):
                    for m in range(T):
                        err = (score[i,j,k,m] - score_sparse[i,j,k,m]).abs().item()
                        if err > 1e-1:
                            print(i,j,k,m,err,score[i,j,k,m],score_sparse[i,j,k,m])
                            return
    
    bench('sparse_bmm', bench_sparse, 0.5, 3)
    bench('naive_bmm', bench_naive, 0.5, 3)

if __name__ == '__main__':
    test_main()