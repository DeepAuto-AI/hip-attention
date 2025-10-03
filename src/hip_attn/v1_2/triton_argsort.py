import numpy as np
import triton
import triton.language as tl
import triton.language.core as core
from triton.language.standard import _log2


@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)

    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape)
    right_idx = tl.reshape(right_idx, x.shape)

    # actual compare-and-swap
    if x.dtype.is_floating():
        if core.constexpr(x.dtype.primitive_bitwidth) == 16:
            dtype_int = core.int16
        elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
            dtype_int = core.int32
        elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
            dtype_int = core.int64
        else:
            raise ValueError("Unsupported dtype")
        ileft = left.to(dtype_int, bitcast=True)
        iright = right.to(dtype_int, bitcast=True)
        ix = x.to(dtype_int, bitcast=True)
    else:
        ileft = left
        iright = right
        ix = x

    cond = (left > right) ^ flip
    ret = ix ^ tl.where(cond != 0, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond != 0, left_idx ^ right_idx, tl.zeros_like(ids))

    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
    x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip denotes whether to re-arrange sub-sequences of elements in ascending or
    # descending order.
    # if flip = 00000000... then all elements will be re-arranged ascendingly at this stage
    # if flip = 00110011... then all the elements will be re-arranged alternatingly (with
    # a stride of 2) at this stage
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(
            tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape
        )
    else:
        flip = order
    # perform `stage` rounds of `compare-and-swap`
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(x, ids, descending: tl.constexpr = False, stages: tl.constexpr = None):
    # iteratively run bitonic merge-sort steps
    if stages is None:
        n_dims: tl.constexpr = _log2(x.shape[1])
    else:
        n_dims: tl.constexpr = stages
    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids


def test():

    @triton.jit
    def sort_kernel(
        # Pointers to matrices
        x_ptr,
        o_ptr,
        id_ptr,
        stride_m,
        stride_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        m_offset = pid_m * stride_m * BLOCK_M
        k_off = tl.arange(0, BLOCK_N)

        x_ptrs = (
            x_ptr
            + m_offset
            + (tl.arange(0, BLOCK_M)[:, None] * stride_m + k_off[None, :])
        )

        # shape: [BLOCK_M, BLOCK_N]
        x = tl.load(x_ptrs)
        ids = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_M, BLOCK_N))

        o, ids = argsort(x, ids, False)

        o_ptrs = (
            o_ptr
            + m_offset
            + (tl.arange(0, BLOCK_M)[:, None] * stride_m + k_off[None, :])
        )
        id_ptrs = (
            id_ptr
            + m_offset
            + (tl.arange(0, BLOCK_M)[:, None] * stride_m + k_off[None, :])
        )
        tl.store(o_ptrs, o)
        tl.store(id_ptrs, ids)

    import torch

    x = np.random.randn(8192, 1024)
    b = x

    x = torch.tensor(
        x,
        dtype=torch.float16,
        device="cuda",
    )
    o = torch.empty_like(x)
    # ids = torch.empty(x.shape, dtype=torch.int, device='cuda')
    ids = torch.empty(x.shape, dtype=torch.int64, device="cuda")

    BLOCK_M = 1
    BLOCK_N = 1024

    grid = (
        triton.cdiv(x.shape[0], BLOCK_M),
        triton.cdiv(x.shape[1], BLOCK_N),
    )

    k = sort_kernel[grid](x, o, ids, x.stride(0), x.stride(1), BLOCK_M, BLOCK_N)

    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()
    k = sort_kernel[grid](x, o, ids, x.stride(0), x.stride(1), BLOCK_M, BLOCK_N)
    end_time.record()
    torch.cuda.synchronize()
    print("Elapsed time: ", start_time.elapsed_time(end_time))

    # path = os.path.join(os.path.dirname(__file__), 'ttgir.mlir')
    # with open(path, 'w') as f:
    #     f.write(k.asm['ttgir'])
    #
    # path = os.path.join(os.path.dirname(__file__), 'ttir.mlir')
    # with open(path, 'w') as f:
    #     f.write(k.asm['ttir'])

    print(k.asm.keys())

    print("result: ")
    print(o)

    print("ids: ")
    print(ids)

    # ref_o, ref_ids = torch.sort(x, 1, True)
    ref_o, ref_ids = torch.sort(x, 1, False)
    print("ref: ")
    print(ref_o)
    print(ref_ids)
    print(ref_ids.dtype)


if __name__ == "__main__":
    test()
