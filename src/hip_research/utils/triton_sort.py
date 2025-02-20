# sort

from triton.language import core, math, zeros_like
from triton.runtime.jit import jit


@jit
def _indicator(n_dims: core.constexpr, idx: core.constexpr, pos: core.constexpr):
    core.static_assert(idx < n_dims)
    core.static_assert((pos == 0) or (pos == 1))
    y = core.arange(0, 2)
    if pos == 0:
        y = 1 - y

    for n in core.static_range(0, n_dims):
        if n != n_dims - 1 - idx:
            y = core.expand_dims(y, n)
    return y


@jit
def _take_slice(
    x,
    n_dims: core.constexpr,
    idx: core.constexpr,
    pos: core.constexpr,
    keep_dim: core.constexpr = True,
):
    y = sum(x * _indicator(n_dims, idx, pos), n_dims - 1 - idx)
    if keep_dim:
        y = core.expand_dims(y, n_dims - 1 - idx)

    return y


@jit
def _compare_and_swap(x, desc_mask, n_dims: core.constexpr, idx: core.constexpr):
    l = _take_slice(x, n_dims, idx, 0)
    r = _take_slice(x, n_dims, idx, 1)

    x_int = x
    l_int = l
    r_int = r
    if x.dtype.is_floating():
        if core.constexpr(x.dtype.primitive_bitwidth) == 16:
            dtype_int = core.int16
        elif core.constexpr(x.dtype.primitive_bitwidth) == 32:
            dtype_int = core.int32
        elif core.constexpr(x.dtype.primitive_bitwidth) == 64:
            dtype_int = core.int64
        else:
            raise ValueError("Unsupported dtype")
        x_int = x.to(dtype_int, bitcast=True)
        l_int = l.to(dtype_int, bitcast=True)
        r_int = r.to(dtype_int, bitcast=True)
    desc_mask = desc_mask.to(x_int.dtype)
    zero = zeros_like(x_int)
    y = x_int ^ core.where((l > r) ^ desc_mask, l_int ^ r_int, zero)
    y = y.to(x.dtype, bitcast=True)
    return y


@jit
def _bitonic_merge(
    x, n_dims: core.constexpr, active_dims: core.constexpr, order_type: core.constexpr
):
    """
    order_type 0 == ascending
    order_type 1 == descending
    order_type 2 == alternating
    """
    core.static_assert(active_dims <= n_dims)

    if order_type == 2:
        desc_mask = _indicator(n_dims, active_dims, 1)
    else:
        desc_mask = order_type

    for i in core.static_range(active_dims):
        x = _compare_and_swap(x, desc_mask, n_dims, active_dims - 1 - i)

    return x


def _log2(i: core.constexpr):
    log2 = 0
    n = i.value
    while n > 1:
        n >>= 1
        log2 += 1
    return core.constexpr(log2)


def _is_power_of_two(i: core.constexpr):
    n = i.value
    return core.constexpr((n & (n - 1)) == 0 and n != 0)


def _unwrap_if_constexpr(o):
    return o.value if isinstance(o, core.constexpr) else o


def _get_sort_dim(dim, shape):
    dim = _unwrap_if_constexpr(dim)
    shape = _unwrap_if_constexpr(shape)
    if dim is None:
        dim = len(shape) - 1
    assert dim == len(shape) - 1, "Currently only support sorting on the last dimension"
    return core.constexpr(dim)


@jit
def sort(x, dim=None, descending: core.constexpr = 0):
    core.static_assert(_is_power_of_two(x.shape[_get_sort_dim(dim, x.shape)]))
    core.static_assert(_is_power_of_two(x.numel))
    # reshape the tensor to have all dimensions be 2.
    # TODO: We shouldn't have to change the dimensions not sorted.
    y = core.reshape(x, [2] * _log2(x.numel))
    for i in core.static_range(1, _log2(x.shape[_get_sort_dim(dim, x.shape)]) + 1):
        y = _bitonic_merge(
            y,
            _log2(x.numel),
            i,
            (descending if (i == _log2(x.shape[_get_sort_dim(dim, x.shape)])) else 2),
        )

    x = core.reshape(y, x.shape)
    return x
