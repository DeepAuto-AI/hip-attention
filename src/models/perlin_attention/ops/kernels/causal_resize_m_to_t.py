from logging import warning
import warnings
import torch, math
import os, tqdm, gc
import torch.nn.functional as F
import time
import triton
import triton.language as tl

def scan_col_py(x, original_width, target_width, max_col_z):
    N, A, B = x.shape
    assert target_width.shape == (A,)
    ncols = torch.zeros((N, A), dtype=torch.long, device=x.device)
    col_indices = torch.zeros((N, A, max_col_z), device=x.device)
    for n in range(N): #prange
        for a in range(A): #prange
            last_index = 0
            for b in range(B):
                if x[n, a, b] != 0:
                    n_pixel = 0
                    v = b % original_width #x[n, a, b]
                    scale = target_width[a] / original_width
                    v_start = torch.round(v * scale)
                    v_end = torch.round((v+1) * scale)
                    n_pixel = v_end - v_start
                    n_pixel = int(n_pixel.item())
                    for i in range(n_pixel):
                        col_indices[n, a, last_index+i] = (v_start + i) + (b // original_width * target_width[-1])
                    last_index += n_pixel
            ncols[n, a] = last_index
    return ncols, col_indices

@triton.jit
def __scan_col_compute_old(
    X,
    stride_xn, stride_xa, stride_xb,
    N, A, B: tl.constexpr, BLOCK_A: tl.constexpr,
    SCALE,
    stride_scale,
    NCOLS,
    stride_ncolsn, stride_ncolsa,
    COL_INDICES,
    stride_coln, stride_cola, stride_colz,
    MAX_Z: tl.constexpr,
    MAX_INTERP: tl.constexpr,
    ORIGINAL_WIDTH: tl.constexpr,
    TARGET_WIDTH_MAX: tl.constexpr,
    GRID_N, GRID_A,
):
    # for n in range(N): #prange
    #     for a in range(A): #prange
    #         last_index = 0
    #         for b in range(B):
    #             if x[n, a, b] != 0:
    #                 n_pixel = 0
    #                 v = b #x[n, a, b]
    #                 scale = scales[a]
    #                 v_start = torch.round(v * scale)
    #                 v_end = torch.round((v+1) * scale)
    #                 n_pixel = v_end - v_start
    #                 n_pixel = int(n_pixel.item())
    #                 for i in range(n_pixel):
    #                     col_indices[n, a, last_index+i] = (v_start + i)
    #                 last_index += n_pixel
    #         ncols[n, a] = last_index
    n = tl.program_id(0)
    pid_a = tl.program_id(1)
    
    for ia in range(BLOCK_A):
        # a = pid_a * BLOCK_A + ia
        a = ia * GRID_A + pid_a
        # mask_a = (a < A) & (a < 19)
        mask_a = a < A
        
        scales_a = tl.load(
            SCALE\
                + a*stride_scale, 
            mask=mask_a, 
            other=0
        )
        
        last_index = int(0)
        for _b in range(B):
            b = _b % ORIGINAL_WIDTH
            x_mask = tl.load(
                X \
                    + n*stride_xn \
                    + a*stride_xa \
                    + _b*stride_xb, 
                mask=mask_a, 
                other=0
            ).to(tl.int32)
            v_start = tl.math.round(b*scales_a)
            v_end = tl.math.round((b+1)*scales_a)
            n_pixel = (v_end-v_start).to(tl.int32) * x_mask
            tl.store(
                COL_INDICES \
                    + n*stride_coln \
                    + a*stride_cola \
                    + (tl.arange(0, MAX_INTERP) + last_index.to(tl.int64)) * stride_colz,
                tl.arange(0, MAX_INTERP) + v_start + tl.math.floor(tl.math.floor(_b / ORIGINAL_WIDTH) * TARGET_WIDTH_MAX),
                mask=(tl.arange(0, MAX_INTERP) < n_pixel) & mask_a,
            )
            last_index += n_pixel
        
        tl.store(NCOLS + n*stride_ncolsn + a*stride_ncolsa, last_index, mask=mask_a)

@triton.autotune(configs=[
        triton.Config({'BLOCK_A': 4}, num_warps=1),
        triton.Config({'BLOCK_A': 16}, num_warps=2),
        triton.Config({'BLOCK_A': 32}, num_warps=4),
        triton.Config({'BLOCK_A': 64}, num_warps=8),
        triton.Config({'BLOCK_A': 128}, num_warps=16),
        triton.Config({'BLOCK_A': 256}, num_warps=32),
        triton.Config({'BLOCK_A': 8}, num_warps=1),
        triton.Config({'BLOCK_A': 16}, num_warps=2),
        triton.Config({'BLOCK_A': 32}, num_warps=4),
        triton.Config({'BLOCK_A': 64}, num_warps=8),
        triton.Config({'BLOCK_A': 128}, num_warps=16),
        triton.Config({'BLOCK_A': 256}, num_warps=32),
        triton.Config({'BLOCK_A': 16}, num_warps=1),
        triton.Config({'BLOCK_A': 32}, num_warps=2),
        triton.Config({'BLOCK_A': 64}, num_warps=4),
        triton.Config({'BLOCK_A': 128}, num_warps=8),
        triton.Config({'BLOCK_A': 256}, num_warps=16),
        triton.Config({'BLOCK_A': 512}, num_warps=32),
    ],
    key=['A', 'MAX_INTERP'],
)
@triton.jit
def __scan_col_compute(
    X,
    stride_xn, stride_xa, stride_xb,
    N, A, B: tl.constexpr, #BLOCK_A: tl.constexpr,
    SCALE,
    stride_scale,
    NCOLS,
    stride_ncolsn, stride_ncolsa,
    COL_INDICES,
    stride_coln, stride_cola, stride_colz,
    MAX_Z,
    # N_PIXELS,
    # stride_np_n, stride_np_a, stride_np_b,
    MAX_INTERP: tl.constexpr, ORIGINAL_WIDTH: tl.constexpr, TARGET_WIDTH_MAX: tl.constexpr, 
    BLOCK_A: tl.constexpr,
):
    # BLOCK_A = meta['BLOCK_A']
    
    n = tl.program_id(0)
    pid_a = tl.program_id(1)
    
    # for ia in range(BLOCK_A):
    index_as = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)
    # index_as = pid_a + tl.arange(0, BLOCK_A) * GRID_A
    mask_as = index_as < A
    
    scales_as = tl.load(
        SCALE\
            + index_as*stride_scale, 
        mask=mask_as, 
        other=0
    )
    
    last_index = tl.zeros((BLOCK_A,), dtype=tl.int32)
    for _b in range(B):
        b = _b % ORIGINAL_WIDTH
        x_mask = tl.load(
            X \
                + n*stride_xn \
                + index_as*stride_xa \
                + _b*stride_xb, 
            mask=mask_as, 
            other=0
        ).to(tl.int32)
        v_start = tl.math.round(b*scales_as)
        v_end = tl.math.round((b+1)*scales_as)
        n_pixel = (v_end-v_start).to(tl.int32) * x_mask
        tl.store(
            COL_INDICES \
                + n*stride_coln \
                + index_as[:, None]*stride_cola \
                + (tl.arange(0, MAX_INTERP)[None,:] + last_index[:,None]) * stride_colz,
            tl.arange(0, MAX_INTERP)[None,:] + v_start[:, None] + tl.math.floor(tl.math.floor(_b / ORIGINAL_WIDTH) * TARGET_WIDTH_MAX),
            mask=(tl.arange(0, MAX_INTERP)[None,:] < n_pixel[:,None]) & mask_as[:, None],
        )
        # tl.store(
        #     N_PIXELS\
        #         + n*stride_np_n\
        #         + index_as*stride_np_a\
        #         + _b*stride_np_b,
        #     # v_start,
        #     # v_end,
        #     n_pixel,
        #     # b*scales_as,
        #     mask=mask_as,
        # )
        # print("np", n_pixel)
        # print("n", n)
        # print("a", index_as)
        last_index += n_pixel
    
    tl.store(NCOLS + n*stride_ncolsn + index_as*stride_ncolsa, last_index, mask=mask_as)

# @triton.autotune(configs=[
#         triton.Config({}, num_warps=1),
#         triton.Config({}, num_warps=2),
#         triton.Config({}, num_warps=4),
#         triton.Config({}, num_warps=8),
#         triton.Config({}, num_warps=16),
#         triton.Config({}, num_warps=32),
#     ],
#     key=['BLOCK_N']
# )
@triton.jit
def __triton_round_compute(
    X,
    stride_x_n,
    N,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    grid_n = tl.num_programs(0)
    
    n = tl.arange(0, BLOCK_N) * grid_n + pid_n
    n_mask = n < N
    
    xs = tl.load(
        X + n*stride_x_n,
        mask=n_mask
    )
    
    ys = tl.math.round(xs)
    
    tl.store(
        X + n*stride_x_n,
        ys,
        mask=n_mask
    )

def triton_round(x: torch.Tensor, inline=False):
    x_shape = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    
    if inline:
        y = x
        assert False, "has bug"
    else:
        y = x.clone().view(-1)
    
    N = y.shape[0]
    BLOCK_N = 1024
    num_warps = BLOCK_N // 32
    
    grid = (triton.cdiv(N, BLOCK_N), )
    __triton_round_compute[grid](
        y,
        y.stride(0),
        N,
        BLOCK_N,
        num_warps=num_warps,
    )
    
    return y.view(x_shape).contiguous()

def __scal_col_2_py(
    pixel_indices,
    v_starts,
    col_indices,
    N, M, H, T_M, TARGET_WIDTH_MAX,
):
    for n in range(N):
        for m in range(M):
            idx_tdst = m // (H*T_M)
            idx_h = (m % (H*T_M)) // T_M
            idx_tm = m % T_M
            
            v_start = v_starts[idx_tdst, idx_tm]
            
            col_start = pixel_indices[n, m - 1] if m > 0 else 0
            col_end = pixel_indices[n, m]
            col_len = (col_end - col_start).item()
            
            if col_len > 0:
                range_start = v_start + (idx_h * TARGET_WIDTH_MAX)
                range_end = range_start + col_len
                col_indices[n, col_start:col_end] = torch.arange(range_start, range_end, device=col_indices.device)

@triton.autotune(configs=[
        # triton.Config({}, num_warps=1),
        # triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['BLOCK_M', 'BLOCK_M', 'MAX_INTERP']
)
@triton.jit
def __scan_col_2_compute(
    PIXEL_INDICES,
    stride_pixel_n, stride_pixel_m,
    V_STARTS,
    stride_vs_tdst, stride_vs_tm,
    COL_INDICES,
    stride_col_n, stride_col_z,
    N, M, H, T_M, TARGET_WIDTH_MAX,
    BLOCK_N:tl.constexpr, GROUP_M:tl.constexpr, BLOCK_M:tl.constexpr, MAX_INTERP:tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    grid_m = tl.program_id(1)
    
    for _n in range(BLOCK_N):
        # for _m in range(BLOCK_M):
        #     n = pid_n*BLOCK_N + _n
        #     m = pid_m*BLOCK_M + _m
            
        #     idx_tdst = m // (H*T_M)
        #     idx_h = (m % (H*T_M)) // T_M
        #     idx_tm = m % T_M
            
        #     v_start = tl.load(
        #         V_STARTS\
        #             + idx_tdst * stride_vs_tdst\
        #             + idx_tm * stride_vs_tm,
        #     )
            
        #     col_start = tl.load(
        #         PIXEL_INDICES\
        #             + n * stride_pixel_n\
        #             + (m - 1) * stride_pixel_m,
        #         mask=(m - 1) >= 0 and m < M,
        #     )
            
        #     col_end = tl.load(
        #         PIXEL_INDICES\
        #             + n * stride_pixel_n\
        #             + m * stride_pixel_m,
        #         mask=m >= 0 and m < M,
        #     )
            
        #     col_len = col_end - col_start
            
        #     range_start = v_start + (idx_h * TARGET_WIDTH_MAX)
        #     tl.store(
        #         COL_INDICES\
        #             + n * stride_col_n\
        #             + (tl.arange(0, MAX_INTERP) + col_start) * stride_col_z,
        #         tl.arange(0, MAX_INTERP) + range_start,
        #         mask=tl.arange(0, MAX_INTERP) < col_len
        #     )

        for _m in range(0, GROUP_M):
            n = pid_n*BLOCK_N + _n
            ms = pid_m*BLOCK_M*GROUP_M + _m*BLOCK_M + tl.arange(0, BLOCK_M)
            # ms = _m*BLOCK_M*GROUP_M + tl.arange(0, BLOCK_M)*grid_m + pid_m
            ms_mask = ms < M
            
            idx_tdst = ms // (H*T_M)
            idx_h = (ms % (H*T_M)) // T_M
            idx_tm = ms % T_M
            
            v_start = tl.load(
                V_STARTS\
                    + idx_tdst * stride_vs_tdst\
                    + idx_tm * stride_vs_tm,
                mask = ms_mask
            )
            
            col_start = tl.load(
                PIXEL_INDICES\
                    + n * stride_pixel_n\
                    + (ms - 1) * stride_pixel_m,
                mask=(((ms - 1) >= 0) and (ms < M)) and ms_mask,
            )
            
            col_end = tl.load(
                PIXEL_INDICES\
                    + n * stride_pixel_n\
                    + ms * stride_pixel_m,
                mask=((ms >= 0) and (ms < M)) and ms_mask,
            )
            
            col_len = col_end - col_start
            
            range_start = v_start + (idx_h * TARGET_WIDTH_MAX)
            tl.store(
                COL_INDICES\
                    + n * stride_col_n\
                    + (tl.arange(0, MAX_INTERP)[None, :] + col_start[:, None]) * stride_col_z,
                tl.arange(0, MAX_INTERP)[None, :] + range_start[:, None],
                mask=(tl.arange(0, MAX_INTERP)[None, :] < col_len[:, None]) and (ms_mask[:, None])
            )

@triton.autotune(configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['BLOCK_N_ZERO', 'BLOCK_ROW', 'MAX_INTERP']
)
@triton.jit
def __scan_col_3_compute(
    NON_ZERO_ROWS,
    stride_nzr_n, stride_nzr_d,
    PIXEL_INDICES,
    stride_pixel_n, stride_pixel_m,
    V_STARTS,
    stride_vs_tdst, stride_vs_tm,
    COL_INDICES,
    stride_col_n, stride_col_z,
    N, M, H, T_M, 
    TARGET_WIDTH_MAX: tl.constexpr, MAX_INTERP: tl.constexpr, 
    NZR_N, NZR_D, BLOCK_N_ZERO: tl.constexpr, 
    NCOL_PER_ROW, BLOCK_ROW: tl.constexpr,
):
    pid_nzr = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    for _i_nzr in range(BLOCK_N_ZERO):
        i_nzr = pid_nzr * BLOCK_N_ZERO + _i_nzr
        mask_nzr = i_nzr < NZR_N
        
        i_batch = tl.load(
            NON_ZERO_ROWS +\
                i_nzr * stride_nzr_n +\
                0 * stride_nzr_d,
            mask=mask_nzr
        )
        i_row = tl.load(
            NON_ZERO_ROWS +\
                i_nzr * stride_nzr_n +\
                1 * stride_nzr_d,
            mask=mask_nzr
        )
        
        n = i_batch
        ms = pid_col * BLOCK_ROW + tl.arange(0, BLOCK_ROW) + i_row * NCOL_PER_ROW
        ms_mask = (pid_col * BLOCK_ROW + tl.arange(0, BLOCK_ROW)) < NCOL_PER_ROW
        
        idx_tdst = ms // (H*T_M)
        idx_h = (ms % (H*T_M)) // T_M
        idx_tm = ms % T_M
        
        v_start = tl.load(
            V_STARTS\
                + idx_tdst * stride_vs_tdst\
                + idx_tm * stride_vs_tm,
            mask = ms_mask
        )
        
        col_start = tl.load(
            PIXEL_INDICES\
                + n * stride_pixel_n\
                + (ms - 1) * stride_pixel_m,
            mask=(((ms - 1) >= 0) and (ms < M)) and ms_mask,
        )
        
        col_end = tl.load(
            PIXEL_INDICES\
                + n * stride_pixel_n\
                + ms * stride_pixel_m,
            mask=((ms >= 0) and (ms < M)) and ms_mask,
        )
        
        col_len = col_end - col_start
        
        range_start = v_start + (idx_h * TARGET_WIDTH_MAX)
        tl.store(
            COL_INDICES\
                + n * stride_col_n\
                + (tl.arange(0, MAX_INTERP)[None, :] + col_start[:, None]) * stride_col_z,
            tl.arange(0, MAX_INTERP)[None, :] + range_start[:, None],
            mask=(tl.arange(0, MAX_INTERP)[None, :] < col_len[:, None]) and (ms_mask[:, None])
        )

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
#     # key=['TARGET_WIDTH_MAX', 'MAX_INTER_PADDED', 'BLOCK_N_ZERO']
# )
@triton.jit
def __scan_col_4_compute(
    NON_ZERO_PIXELS,
    stride_nzp_n, stride_nzp_d,
    PIXEL_INDICES,
    stride_pixel_n, stride_pixel_m,
    V_STARTS,
    stride_vs_tdst, stride_vs_tm,
    V_ENDS,
    stride_ve_tdst, stride_ve_tm,
    COL_INDICES,
    stride_col_n, stride_col_z,
    N, M, H, T_M, 
    TARGET_WIDTH_MAX, MAX_INTER_PADDED: tl.constexpr, MAX_INTERP,
    NZR_N, NZR_D, BLOCK_N_ZERO: tl.constexpr,
):
    pid_nzp = tl.program_id(0)
    
    i_nzp_n = pid_nzp * BLOCK_N_ZERO + tl.arange(0, BLOCK_N_ZERO)
    mask_i_nzp = i_nzp_n < NZR_N
    is_batch = tl.load(
        NON_ZERO_PIXELS +\
            i_nzp_n * stride_nzp_n+\
            0 * stride_nzp_d,
        mask = mask_i_nzp
    )
    is_col = tl.load(
        NON_ZERO_PIXELS +\
            i_nzp_n * stride_nzp_n+\
            1 * stride_nzp_d,
        mask = mask_i_nzp
    )
    
    idx_tdst = is_col // (H*T_M)
    idx_h = (is_col % (H*T_M)) // T_M
    idx_tm = is_col % T_M
    
    v_start = tl.load(
        V_STARTS\
            + idx_tdst * stride_vs_tdst\
            + idx_tm * stride_vs_tm,
        mask = mask_i_nzp
    )
    
    v_end = tl.load(
        V_ENDS\
            + idx_tdst * stride_ve_tdst\
            + idx_tm * stride_ve_tm,
        mask = mask_i_nzp
    )
    
    col_start = tl.load(
        PIXEL_INDICES\
            + is_batch * stride_pixel_n\
            + (is_col - 1) * stride_pixel_m,
        mask=(((is_col - 1) >= 0) and (is_col < M)) and mask_i_nzp,
        other=0,
    )
    
    col_end = tl.load(
        PIXEL_INDICES\
            + is_batch * stride_pixel_n\
            + is_col * stride_pixel_m,
        mask=((is_col >= 0) and (is_col < M)) and mask_i_nzp,
    )
    
    col_len = col_end - col_start
    
    range_start = v_start + (idx_h * TARGET_WIDTH_MAX)
    range_end = v_end + (idx_h * TARGET_WIDTH_MAX)
    tl.store(
        COL_INDICES\
            + is_batch[:, None] * stride_col_n\
            + (tl.arange(0, MAX_INTER_PADDED)[None, :] + col_start[:, None]) * stride_col_z,
        # 77,
        # (tl.arange(0, MAX_INTER_PADDED)[None, :] * ((range_end[:, None] - range_start[:, None]) / col_len[:, None])).to(tl.int32) + range_start[:, None],
        range_end[:, None] - (tl.arange(0, MAX_INTER_PADDED)[None, :] * ((range_end[:, None] - range_start[:, None]) / col_len[:, None])).to(tl.int32) - 1,
        # mask=((tl.arange(0, MAX_INTER_PADDED)[None, :] < col_len[:, None])) and (mask_i_nzp[:, None])
        mask=((tl.arange(0, MAX_INTER_PADDED)[None, :] < col_len[:, None]) and (tl.arange(0, MAX_INTER_PADDED)[None, :] < MAX_INTERP)) and (mask_i_nzp[:, None])
    )
    
    # for _i_nzr in range(BLOCK_N_ZERO):
    #     i_nzr = pid_nzr * BLOCK_N_ZERO + _i_nzr
    #     mask_nzr = i_nzr < NZR_N
        
    #     i_batch = tl.load(
    #         NON_ZERO_ROWS +\
    #             i_nzr * stride_nzr_n +\
    #             0 * stride_nzr_d,
    #         mask=mask_nzr
    #     )
    #     i_row = tl.load(
    #         NON_ZERO_ROWS +\
    #             i_nzr * stride_nzr_n +\
    #             1 * stride_nzr_d,
    #         mask=mask_nzr
    #     )
        
    #     n = i_batch
    #     ms = pid_col * BLOCK_ROW + tl.arange(0, BLOCK_ROW) + i_row * NCOL_PER_ROW
    #     ms_mask = (pid_col * BLOCK_ROW + tl.arange(0, BLOCK_ROW)) < NCOL_PER_ROW
        
    #     idx_tdst = ms // (H*T_M)
    #     idx_h = (ms % (H*T_M)) // T_M
    #     idx_tm = ms % T_M
        
    #     v_start = tl.load(
    #         V_STARTS\
    #             + idx_tdst * stride_vs_tdst\
    #             + idx_tm * stride_vs_tm,
    #         mask = ms_mask
    #     )
        
    #     col_start = tl.load(
    #         PIXEL_INDICES\
    #             + n * stride_pixel_n\
    #             + (ms - 1) * stride_pixel_m,
    #         mask=(((ms - 1) >= 0) and (ms < M)) and ms_mask,
    #     )
        
    #     col_end = tl.load(
    #         PIXEL_INDICES\
    #             + n * stride_pixel_n\
    #             + ms * stride_pixel_m,
    #         mask=((ms >= 0) and (ms < M)) and ms_mask,
    #     )
        
    #     col_len = col_end - col_start
        
    #     range_start = v_start + (idx_h * TARGET_WIDTH_MAX)
    #     tl.store(
    #         COL_INDICES\
    #             + n * stride_col_n\
    #             + (tl.arange(0, MAX_INTERP)[None, :] + col_start[:, None]) * stride_col_z,
    #         tl.arange(0, MAX_INTERP)[None, :] + range_start[:, None],
    #         mask=(tl.arange(0, MAX_INTERP)[None, :] < col_len[:, None]) and (ms_mask[:, None])
    #     )

def scan_col(
    x: torch.Tensor, 
    original_width: int, 
    target_width_max: int, 
    target_width: torch.Tensor, 
    max_col_z: int, 
    max_k: int, 
    oversampled: float = None
):
    N, T_DST, H_T = x.shape # N, T_DST, H*T_M
    assert target_width.shape == (T_DST,)
    scales = target_width / original_width
    
    # METHOD = 2 if target_width_max <= 2048 else 1
    METHOD = 1
    
    # for high sparsity
    if METHOD == 1:
        with get_bench().region("scan_col.setup"):
            T_M = original_width
            H = H_T // T_M
            b = torch.arange(0, T_M, device=x.device).view(1, T_M)
            with get_bench().region("scan_col.triton_round"):
                v_starts = triton_round(b*scales.view(T_DST, 1))
                v_ends = triton_round((b + 1)*scales.view(T_DST, 1))
            with get_bench().region("scan_col.npixels"):
                n_pixels = ((v_ends - v_starts).view(1, T_DST, 1, T_M).to(torch.int32) * x.view(N, T_DST, H, T_M).to(torch.int32))
                # NOTE we set upper bound of pixel duplication
                torch.clamp_max(n_pixels, max_k, out=n_pixels)
                # print(n_pixels.view(N, T_DST, -1)[0])
            
            with get_bench().region("scan_col.cumsum"):
                # TODO fusing kernel
                pixel_indices = n_pixels.view(N, -1).cumsum(-1) # N, M
            M = pixel_indices.shape[-1]
            
            Z = pixel_indices.view(N, T_DST, -1)[:, :, -1].max().item()
            crow_indices = torch.zeros((N, T_DST+1), dtype=torch.long, device=x.device)
            col_indices = torch.zeros((N, Z), dtype=torch.long, device=x.device)
            values = torch.ones_like(col_indices, dtype=x.dtype)
            
            crow_indices[:, 1:] = pixel_indices.view(N, T_DST, -1)[:,:,-1]
        
        # with get_bench().region("scan_col.compute"):
        #     BLOCK_N = 1
        #     GROUP_M = 256
        #     BLOCK_M = triton.next_power_of_2(triton.cdiv(triton.cdiv(M, GROUP_M), 4096))
        #     # BLOCK_M = 64
        #     num_warps = min(32, BLOCK_M // 32)
        #     MAX_INTERP = triton.next_power_of_2(triton.cdiv(target_width_max, original_width))
        #     grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M*GROUP_M))
        #     # print(grid, num_warps, GROUP_M, BLOCK_M, MAX_INTERP)
        #     __scan_col_2_compute[grid](
        #         pixel_indices,
        #         pixel_indices.stride(0),pixel_indices.stride(1),
        #         v_starts,
        #         v_starts.stride(0), v_starts.stride(1),
        #         col_indices,
        #         col_indices.stride(0), col_indices.stride(1),
        #         N, M, H, T_M, target_width_max,
        #         BLOCK_N, GROUP_M, BLOCK_M, MAX_INTERP,
        #         # num_warps=num_warps,
        #     )
        
        # skiping non zero rows
        # with get_bench().region("scan_col.compute"):
        #     non_zero_rows = (crow_indices[:, 1:] - crow_indices[:, :-1]).nonzero()
            
        #     NZR_N, NZR_D = non_zero_rows.shape # (idx_batch, idx_row)
        #     BLOCK_N_ZERO = 1
        #     NCOLS_PER_ROW = pixel_indices.numel() // N // T_DST
        #     BLOCK_ROW = 16
        #     MAX_INTERP = triton.next_power_of_2(triton.cdiv(target_width_max, original_width))
        #     grid = (triton.cdiv(NZR_N, BLOCK_N_ZERO), triton.cdiv(NCOLS_PER_ROW, BLOCK_ROW))
        #     # print(grid)
        #     __scan_col_3_compute[grid](
        #         non_zero_rows,
        #         non_zero_rows.stride(0), non_zero_rows.stride(1),
        #         pixel_indices,
        #         pixel_indices.stride(0), pixel_indices.stride(1),
        #         v_starts,
        #         v_starts.stride(0), v_starts.stride(1),
        #         col_indices,
        #         col_indices.stride(0), col_indices.stride(1),
        #         N, M, H, T_M, 
        #         target_width_max, MAX_INTERP, 
        #         NZR_N, NZR_D, BLOCK_N_ZERO, 
        #         NCOLS_PER_ROW, BLOCK_ROW
        #     )
        
        # skiping non zero entries
        with get_bench().region("scan_col.compute"):
            # crow_indices[:, 1:] = pixel_indices.view(N, T_DST, -1)[:,:,-1]
            non_zero_pixels = (n_pixels.view(N, -1)[:, :]).nonzero()
            
            # print('nzp', non_zero_pixels)
            
            NZP_N, NZP_D = non_zero_pixels.shape # (idx_batch, idx_row)
            BLOCK_N_ZERO = 128
            # MAX_INTERP = triton.next_power_of_2(triton.cdiv(target_width_max, original_width))
            MAX_INTERP = min(triton.cdiv(target_width_max, original_width), max_k)
            MAX_INTERP_PADDED = triton.next_power_of_2(MAX_INTERP)
            grid = (triton.cdiv(NZP_N, BLOCK_N_ZERO),)
            # print(grid)
            __scan_col_4_compute[grid](
                non_zero_pixels,
                non_zero_pixels.stride(0), non_zero_pixels.stride(1),
                pixel_indices,
                pixel_indices.stride(0), pixel_indices.stride(1),
                v_starts,
                v_starts.stride(0), v_starts.stride(1),
                v_ends,
                v_ends.stride(0), v_ends.stride(1),
                col_indices,
                col_indices.stride(0), col_indices.stride(1),
                N, M, H, T_M, 
                target_width_max, MAX_INTERP_PADDED, MAX_INTERP, 
                NZP_N, NZP_D, BLOCK_N_ZERO, 
            )
            
            # print('pi', n_pixels.view(N, -1)[:, :-1])
            # print('pi', pixel_indices)
            # print(crow_indices)
            # print(col_indices)
            # exit()
        
        return torch.sparse_csr_tensor(
            crow_indices=crow_indices,
            col_indices=col_indices,
            values=values,
            size=(N, T_DST, H*target_width_max)
        )
    elif METHOD == 2:
        # for low sparsity. somehow allow quadratic thread allocation
        
        # npixels_debug = torch.zeros_like(x, dtype=torch.float32)
        ncols = torch.zeros((N, T_DST), dtype=torch.long, device=x.device)
        col_indices = torch.zeros((N, T_DST, max_col_z), device=x.device, dtype=torch.long)
        
        # truth = scan_col_py(x, original_width=original_width, target_width=target_width, max_col_z=max_col_z)
        # return truth
        
        # BLOCK_A = 1
        # if T_DST < 256:
        #     BLOCK_A = 1
        # elif T_DST < 1024:
        #     BLOCK_A = 2
        # elif T_DST < 2048:
        #     BLOCK_A = 4
        # elif T_DST < 4096:
        #     BLOCK_A = 8
        # elif T_DST < 8192:
        #     BLOCK_A = 16
        # elif T_DST < 16384:
        #     BLOCK_A = 16
        # else:
        #     BLOCK_A = 32
        MAX_INTERP = triton.next_power_of_2(int(math.ceil(target_width_max / original_width)))
        grid = lambda META: (N, triton.cdiv(T_DST, META['BLOCK_A']),)
        __scan_col_compute[grid](
            x, 
            x.stride(0), x.stride(1), x.stride(2),
            N, T_DST, H_T, #BLOCK_A,
            scales, 
            scales.stride(0),
            ncols, 
            ncols.stride(0), ncols.stride(1), 
            col_indices, 
            col_indices.stride(0), col_indices.stride(1), col_indices.stride(2), 
            max_col_z, 
            # npixels_debug,
            # npixels_debug.stride(0), npixels_debug.stride(1), npixels_debug.stride(2),
            MAX_INTERP, original_width, target_width_max, 
        )
        
        # print(ncols[0].cumsum(0))
        # print('np', n_pixels[0, 7].view(-1))
        # print('vs', v_starts[7])
        # print('ve', v_ends[7])
        # print('bs', (b*scales.view(T_DST, 1))[7])
        # print('dg', npixels_debug[0, 7])
        return ncols, col_indices
        
    raise Exception()

def compact_cols_py(ncols_cs, col_indices, out_col_indices):
    N, A, _ = col_indices.shape
    for n in range(N):
        for a in range(A):
            out_col_indices[n, ncols_cs[n, a]:ncols_cs[n, a+1]] = col_indices[n, a, :ncols_cs[n, a+1]-ncols_cs[n, a]]

@triton.autotune(configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=['BLOCK_A','MAX_NCOLS']
)
@triton.jit
def __compact_cols_compute(
    NCOLS_CS,
    stride_ncols_cs_n, stride_ncols_cs_a,
    N, A,
    COL_INDICES,
    stride_col_indices_n, stride_col_indices_a, stride_col_indices_mz,
    OUT_COL_INDICES,
    stride_out_col_indices_n, stride_out_col_indices_z,
    MAX_NCOLS:tl.constexpr,
    BLOCK_A:tl.constexpr,
):
    n = tl.program_id(0)
    pid_a = tl.program_id(1)
    
    # a = pid_a * BLOCK_A + tl.arange(BLOCK_A)
    
    #out_col_indices[n, ncols_cs[n, a]:ncols_cs[n, a+1]] = col_indices[n, a, :ncols_cs[n, a+1]-ncols_cs[n, a]]
    for ia in range(BLOCK_A):
        a = pid_a * BLOCK_A + ia
        mask_a = a < A
        
        cs_start = tl.load(NCOLS_CS+n*stride_ncols_cs_n+a*stride_ncols_cs_a, mask=mask_a)
        cs_end = tl.load(NCOLS_CS+n*stride_ncols_cs_n+(a + 1)*stride_ncols_cs_a, mask=mask_a)
        cs_len = cs_end - cs_start
        col_indices = tl.load(
            COL_INDICES \
                + n*stride_col_indices_n\
                + a*stride_col_indices_a\
                + tl.arange(0, MAX_NCOLS),
            mask = (tl.arange(0, MAX_NCOLS) < cs_len) & mask_a
        )
        tl.store(
            OUT_COL_INDICES\
                + n*stride_out_col_indices_n\
                + (tl.arange(0, MAX_NCOLS) + cs_start)*stride_out_col_indices_z,
            col_indices,
            mask = (tl.arange(0, MAX_NCOLS) < cs_len) & mask_a
        )

def compact_cols(ncols, col_indices: torch.Tensor):
    N, A = ncols.shape
    N, A, MZ = col_indices.shape
    ncols_cs = F.pad(ncols.view(1, 1, N, A), pad=(1, 0), mode='constant', value=0).view(N, A+1).cumsum(-1)
    z_per_batch = ncols_cs[:,-1].max()
    # print(ncols_cs[:, -1], z_per_batch)
    if not torch.all(z_per_batch == ncols_cs[:, -1]):
        warnings.warn(f"all batch should have same number of elements {z_per_batch}=={ncols_cs[:, -1]}")
    out_col_indices = torch.zeros((N, z_per_batch), dtype=torch.long, device=ncols.device).fill_(-1) # type: torch.Tensor
    
    # print()
    # BLOCK_A = 16 if A > 512 else 1
    num_warps = 4
    BLOCK_A = 4
    grid = (N, triton.cdiv(A,BLOCK_A))
    # print(triton.next_power_of_2(max(1, int(torch.max(ncols).item()))))
    __compact_cols_compute[grid](
        ncols_cs,
        ncols_cs.stride(0), ncols_cs.stride(1),
        N, A,
        col_indices,
        col_indices.stride(0), col_indices.stride(1), col_indices.stride(2),
        out_col_indices,
        out_col_indices.stride(0), out_col_indices.stride(1),
        triton.next_power_of_2(max(1, int(torch.max(ncols).item()))),
        BLOCK_A,
        # num_warps=num_warps,
    )
    
    return ncols_cs, out_col_indices

from .....utils import get_bench
from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

def resize_from_m_to_t_csr(
    x, 
    masked_fill_value, 
    k, 
    target_width=None, 
    training=False, 
    need_assert=False, 
    is_causal=True, 
    max_col_z = None, 
    benchmarking = False,
    oversampled = None,
):
    if benchmarking:
        timer = lambda name: get_bench().region(name)
    else:
        timer = lambda name: nullcontext()
    
    with timer("resize_from_m_to_t_csr"):
        with timer("resize_from_m_to_t_csr.setup"):
            _x = x
            _target_width = target_width
            _k = k
            assert not training
            assert masked_fill_value == 0
            N, H, T_DST, T_M = x.shape
            if target_width is not None:
                T_SRC = target_width
            else:
                T_SRC = T_DST
            
            x = x.transpose(1, 2).reshape(N, T_DST, H*T_M)
            
            if max_col_z is None:
                if is_causal:
                    # max_col_z = H*(k*(max(1, math.ceil(T_SRC / T_M)) + 1))
                    # t = k + math.ceil(T_SRC / T_M)
                    # max_col_z = H*t
                    max_col_z = H*(k + math.ceil(T_SRC / T_M))
                else:
                    # max_col_z = H*(k*(max(1, math.ceil(T_SRC / T_M)) + 1))
                    max_col_z = H*(k + math.ceil(T_SRC / T_M))
            assert isinstance(max_col_z, (int, float))
            
            if is_causal:
                target_width = torch.arange(1, T_SRC+1, device=x.device)[-T_DST:]
            else:
                #TODO confirm correctness
                target_width = torch.full((T_SRC,), T_SRC, device=x.device)[-T_DST:]
        
        with timer("resize_from_m_to_t_csr.scan_col"):
            ret = scan_col(
                x,
                original_width=T_M, 
                target_width_max=T_SRC, 
                target_width=target_width, 
                max_col_z=max_col_z,
                max_k=k,
                oversampled=oversampled,
            )
            if isinstance(ret, torch.Tensor):
                assert ret.is_sparse_csr
                return ret

            ncols, _col_indices = ret
        
        if need_assert and False:
            assert ncols.max() < max_col_z, f"{max_col_z}, {H}, {k}, {ncols.max()}"
        else:
            ncols_max = ncols.max().item()
            if ncols_max >= max_col_z:
                print(f'ncols({ncols_max}) out of range ({max_col_z}), please fix this')
                del target_width, x, ncols, _col_indices
                return resize_from_m_to_t_csr(
                    _x, 
                    masked_fill_value, 
                    _k, 
                    target_width=_target_width, 
                    training=training, 
                    need_assert=need_assert, 
                    is_causal=is_causal, 
                    max_col_z=ncols_max + 1
                )
        # print(ncols, _col_indices)
        # print(ncols.shape, _col_indices.shape)
        
        with timer("resize_from_m_to_t_csr.compact_cols"):
            crows_indices, col_indices = compact_cols(ncols, _col_indices)
            # print(crows_indices, col_indices, _col_indices)
        
        with timer("resize_from_m_to_t_csr.csr"):
            # print(crows_indices[0])
            
            return torch.sparse_csr_tensor(
                crow_indices=crows_indices,
                col_indices=col_indices,
                values=torch.ones(col_indices.shape, device=col_indices.device),
                size=(N, T_DST, H*T_SRC),
            )

def test_config(
    IS_CAUSAL, N, H, T, T_DST, T_M, K, K_OS, only_bench=False
):
    from .....utils import seed
    from .....utils.bench import bench
    from .causal_topk_masking import causal_topk_masking
    from .flat_csr_to_dense import flat_csr_to_dense
    from .resize_m_to_t import resize_from_m_to_t

    seed()
    
    FP_MIN = torch.finfo(torch.float16).min * 0.5
    device = 0
    
    def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        
        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
        
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
        
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
    estimated_scores = F.interpolate(rand_perlin_2d((128, 128), (16, 16)).view(1, 1, 128, 128), (T_DST, T_M)).expand(N, H, T_DST, T_M).contiguous().to(device)
    
    # estimated_scores = torch.randn((N, H, T_DST, T_M), device=device)
    
    estimated_probs = torch.softmax(estimated_scores, dim=-1)
    causal_attention_mask = ((torch.arange(T, device=device).view(1, T) > torch.arange(T, device=device).view(T, 1)) * FP_MIN).view(1, 1, T, T)
    causal_attention_mask = causal_attention_mask[:, :, -T_DST:, :]
    attention_mask = causal_attention_mask[:,:,-1:,:]
    dst_attention_mask = causal_attention_mask[:,:,:,:1]
    if not IS_CAUSAL:
        mask = attention_mask
    else:
        mask = causal_attention_mask
    
    compressed_mask = causal_topk_masking(
        estimated_probs, 
        k=K * K_OS, 
        attention_mask=attention_mask, 
        dst_attention_mask=dst_attention_mask, 
        causal_attention_mask=causal_attention_mask,
        is_causal=IS_CAUSAL
    )
    
    if not only_bench:
        t = resize_from_m_to_t_csr(
            compressed_mask, 0, K,
            target_width=mask.shape[-1], 
            is_causal=IS_CAUSAL,
            oversampled=K_OS,
        )
        if t is not None:
            print(t.shape)
            # resized_mask_csr = t.to_dense().view(N, T_DST, H, T).transpose(1, 2).reshape(N, H, T_DST, T)
            resized_mask_csr = flat_csr_to_dense(t, T, H)
        
        # print(resized_mask_csr)
        # return
        
        resized_mask = resize_from_m_to_t(
            compressed_mask, 0, 
            attention_mask=mask,
            target_width=mask.shape[-1],
            is_causal=IS_CAUSAL,
            k=K,
            oversampled=K_OS,
        )
        
        os.makedirs('./saves/tests/ops/causal_resize_m_to_t', exist_ok=True)
        torch.save({
            'tm': compressed_mask,
            'csr': resized_mask_csr,
            'dense': resized_mask,
        }, './saves/tests/ops/causal_resize_m_to_t/state.pth')
        print('saved sample ./saves/tests/ops/causal_resize_m_to_t/state.pth')
    
    # print(resized_mask)
    return
    
    # for i in range(H):
    #     print('-=-')
    #     print('old')
    #     print(resized_mask[0,i])
    #     print('new')
    #     print(resized_mask_csr[0,i])
    
    # return
    
    def bench_naive_convert():
        resized_mask = resize_from_m_to_t(
            compressed_mask, 0, 
            attention_mask=mask,
            target_width=mask.shape[-1],
            is_causal=IS_CAUSAL,
            k=K,
            oversampled=K_OS,
        )
        resized_mask = resized_mask.transpose(1, 2).reshape(N, T_DST, H*T)
        for i in range(N):
            strided = resized_mask[i:i+1]
            strided.to_sparse_csr()
        # return resized_mask.to_sparse_csr()
    
    def bench_csr_convert():
        return resize_from_m_to_t_csr(
            compressed_mask, 0, K,
            target_width=mask.shape[-1],
            is_causal=IS_CAUSAL,
            benchmarking=True,
            oversampled=K_OS,
        )
    
    bench('csr_convert (trace)', bench_csr_convert, t_warmup=0.5, t_sample=3, tracetree=True)
    bench('csr_convert', bench_csr_convert, t_warmup=0.5, t_sample=3)
    if not only_bench:
        bench('naive_convert', bench_naive_convert, t_warmup=0.5, t_sample=3)

def test_main():
    IS_CAUSAL = True
    
    N = 1
    H = 1
    T = 32
    T_DST = 32
    T_M = 2
    K = 4
    K_OS = 1.0
    
    N = 1
    H = 1
    T = 64
    T_DST = 64
    T_M = 8
    K = 16
    K_OS = 1.0
    
    # N = 1
    # H = 12
    # T = 4096
    # T = 1024*32
    # T_DST = T
    # T_M = 128
    # K = 32
    
    test_config(
        IS_CAUSAL, N, H, T, T_DST, T_M, K, K_OS, only_bench=T > 4096
    )
    
    # for t in [2048, 4096, 8192]:
    #     test_config(
    #         IS_CAUSAL, N, H, t, t, T_M, K
    #     )

if __name__ == '__main__':
    test_main()