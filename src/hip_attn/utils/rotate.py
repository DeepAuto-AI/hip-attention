import triton
import triton.language as tl


@triton.jit
def push_right(
    x: tl.tensor,
    new_right: tl.tensor,
):
    T: tl.constexpr = x.shape[0]
    D: tl.constexpr = x.shape[1]
    offset = new_right.shape[1]
    x = x.reshape(T, 2, D // 2)
    x = x.trans(0, 2, 1)
    a, b = x.split()
    cond: tl.constexpr = D // 2 == offset
    if cond:
        new_b = tl.join(b, new_right)
        return a, new_b
    else:
        mid, new_b = push_right_1(b, new_right)
        left, new_a = push_right_1(a, mid)
        x = tl.join(new_a, new_b)
        return left, x


@triton.jit
def push_right_1(
    x: tl.tensor,
    new_right: tl.tensor,
):
    T: tl.constexpr = x.shape[0]
    D: tl.constexpr = x.shape[1]
    offset = new_right.shape[1]
    x = x.reshape(T, 2, D // 2)
    x = x.trans(0, 2, 1)
    a, b = x.split()
    cond: tl.constexpr = D // 2 == offset
    if cond:
        new_b = tl.join(b, new_right)
        return a, new_b
    else:
        mid, new_b = push_right_2(b, new_right)
        left, new_a = push_right_2(a, mid)
        x = tl.join(new_a, new_b)
        return left, x


@triton.jit
def push_right_2(
    x: tl.tensor,
    new_right: tl.tensor,
):
    T: tl.constexpr = x.shape[0]
    D: tl.constexpr = x.shape[1]
    offset = new_right.shape[1]
    x = x.reshape(T, 2, D // 2)
    x = x.trans(0, 2, 1)
    a, b = x.split()
    cond: tl.constexpr = D // 2 == offset
    if cond:
        new_b = tl.join(b, new_right)
        return a, new_b
    else:
        mid, new_b = push_right_3(b, new_right)
        left, new_a = push_right_3(a, mid)
        x = tl.join(new_a, new_b)
        return left, x


@triton.jit
def push_right_3(
    x: tl.tensor,
    new_right: tl.tensor,
):
    T: tl.constexpr = x.shape[0]
    D: tl.constexpr = x.shape[1]
    offset = new_right.shape[1]
    x = x.reshape(T, 2, D // 2)
    x = x.trans(0, 2, 1)
    a, b = x.split()
    cond: tl.constexpr = D // 2 == offset
    tl.static_assert(cond)
    new_b = tl.join(b, new_right)
    return a, new_b


@triton.jit
def push_left(
    new_left: tl.tensor,
    x: tl.tensor,
):
    T: tl.constexpr = x.shape[0]
    D: tl.constexpr = x.shape[1]
    offset = new_left.shape[1]
    x = x.reshape(T, 2, D // 2)
    x = x.trans(0, 2, 1)
    a, b = x.split()
    cond: tl.constexpr = D // 2 == offset
    if cond:
        new_a = tl.join(new_left, a)
        return new_a, b
    else:
        new_a, mid = push_left_1(new_left, a)
        new_b, right = push_left_1(mid, b)
        x = tl.join(new_a, new_b)
        return x, right


@triton.jit
def push_left_1(
    new_left: tl.tensor,
    x: tl.tensor,
):
    T: tl.constexpr = x.shape[0]
    D: tl.constexpr = x.shape[1]
    offset = new_left.shape[1]
    x = x.reshape(T, 2, D // 2)
    x = x.trans(0, 2, 1)
    a, b = x.split()
    cond: tl.constexpr = D // 2 == offset
    if cond:
        new_a = tl.join(new_left, a)
        return new_a, b
    else:
        new_a, mid = push_left_2(new_left, a)
        new_b, right = push_left_2(mid, b)
        x = tl.join(new_a, new_b)
        return x, right


@triton.jit
def push_left_2(
    new_left: tl.tensor,
    x: tl.tensor,
):
    T: tl.constexpr = x.shape[0]
    D: tl.constexpr = x.shape[1]
    offset = new_left.shape[1]
    x = x.reshape(T, 2, D // 2)
    x = x.trans(0, 2, 1)
    a, b = x.split()
    cond: tl.constexpr = D // 2 == offset
    if cond:
        new_a = tl.join(new_left, a)
        return new_a, b
    else:
        new_a, mid = push_left_3(new_left, a)
        new_b, right = push_left_3(mid, b)
        x = tl.join(new_a, new_b)
        return x, right


@triton.jit
def push_left_3(
    new_left: tl.tensor,
    x: tl.tensor,
):
    T: tl.constexpr = x.shape[0]
    D: tl.constexpr = x.shape[1]
    offset = new_left.shape[1]
    x = x.reshape(T, 2, D // 2)
    x = x.trans(0, 2, 1)
    a, b = x.split()
    cond: tl.constexpr = D // 2 == offset
    tl.static_assert(cond)
    new_a = tl.join(new_left, a)
    return new_a, b


@triton.jit
def rotate_left(x: tl.tensor, offset: tl.constexpr):
    return push_right(x, tl.zeros([x.shape[0], offset], dtype=x.dtype))[1]


@triton.jit
def rotate_right(x: tl.tensor, offset: tl.constexpr):
    return push_left(tl.zeros([x.shape[0], offset], dtype=x.dtype), x)[0]
