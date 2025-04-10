import triton
import triton.language as tl


@triton.jit
def split_half(x: tl.tensor, T: tl.constexpr, HID: tl.constexpr):
    x = x.reshape(T, 2, HID // 2)
    x = x.trans(0, 2, 1)
    return x.split()


@triton.jit
def merge_half(left: tl.tensor, right: tl.tensor, T: tl.constexpr, HID: tl.constexpr):
    assert left.shape == right.shape
    x = tl.join(left, right)
    x = x.trans(0, 2, 1)
    x = x.reshape(T, HID)
    return x


@triton.jit
def de_rope(
    vec: tl.tensor, cos: tl.tensor, sin: tl.tensor, T: tl.constexpr, HID: tl.constexpr
):
    c0, ch = split_half(cos, T, HID)
    s0, sh = split_half(sin, T, HID)
    vr0, vrh = split_half(vec, T, HID)

    out0 = (vrh * s0 + vr0 * ch) / (c0 * ch + sh * s0 + 1e-20)
    outh = (out0 * c0 - vr0) / (s0 + 1e-20)
    out = merge_half(out0, outh, T, HID)
    return out


@triton.jit
def rotate_half(vec: tl.tensor, T: tl.constexpr, HID: tl.constexpr):
    left, right = split_half(vec, T, HID)
    out0 = -right
    outh = left
    return merge_half(out0, outh, T, HID)


@triton.jit
def apply_rope(
    vec: tl.tensor, cos: tl.tensor, sin: tl.tensor, T: tl.constexpr, HID: tl.constexpr
):
    vec = vec * cos + rotate_half(vec, T, HID) * sin
    return vec


@triton.jit
def adjust_rope(
    tokens: tl.tensor,
    old_t: tl.tensor,
    new_t: tl.tensor,
    mask_t: tl.tensor,
    idx_hid: tl.tensor,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    T: tl.constexpr,
    HID: tl.constexpr,
    HID_DIM,
    NEED_APPLY_ROPE: tl.constexpr,
    rope_range_begin: tl.constexpr = 0,
    rope_range_end: tl.constexpr = None,
    rope_is_neox_style: tl.constexpr = True,
):
    if rope_range_end is None:
        rope_range_end = HID_DIM
    tl.static_assert(rope_is_neox_style, "interleaved style not supported")
    tl.device_assert(
        tl.min(idx_hid) == rope_range_begin, "tl.min(idx_hid) != rope_range_begin"
    )
    tl.device_assert(
        tl.max(idx_hid) == rope_range_end - 1, "tl.max(idx_hid) != rope_range_end - 1"
    )

    if not NEED_APPLY_ROPE:
        mask_t = mask_t & (old_t != 0)

        cos_old = tl.load(
            COS
            + old_t[:, None].to(tl.int64) * stride_cos_t
            + idx_hid[None, :] * stride_cos_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0,
        )
        sin_old = tl.load(
            SIN
            + old_t[:, None].to(tl.int64) * stride_sin_t
            + idx_hid[None, :] * stride_sin_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0,
        )

        cos_new = tl.load(
            COS
            + new_t[:, None].to(tl.int64) * stride_cos_t
            + idx_hid[None, :] * stride_cos_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0,
        )
        sin_new = tl.load(
            SIN
            + new_t[:, None].to(tl.int64) * stride_sin_t
            + idx_hid[None, :] * stride_sin_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0,
        )

        tokens_adjusted = de_rope(
            tokens.to(tl.float32),
            cos_old.to(tl.float32),
            sin_old.to(tl.float32),
            T,
            HID,
        )
        tokens_adjusted = apply_rope(
            tokens_adjusted.to(tl.float32),
            cos_new.to(tl.float32),
            sin_new.to(tl.float32),
            T,
            HID,
        )

        tokens = tl.where(mask_t[:, None], tokens_adjusted.to(tokens.dtype), tokens)

        return tokens
    else:
        cos_new = tl.load(
            COS
            + new_t[:, None].to(tl.int64) * stride_cos_t
            + idx_hid[None, :] * stride_cos_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0.0,
        )
        sin_new = tl.load(
            SIN
            + new_t[:, None].to(tl.int64) * stride_sin_t
            + idx_hid[None, :] * stride_sin_hid,
            mask=tl.ravel(mask_t)[:, None],
            other=0.0,
        )

        tokens = apply_rope(
            tokens.to(tl.float32),
            cos_new.to(tl.float32),
            sin_new.to(tl.float32),
            T,
            HID,
            rope_range_begin,
            rope_range_end,
        ).to(tokens.dtype)

        # tokens = tl.where(mask_t[:, None], tokens_adjusted.to(tokens.dtype), tokens)

        return tokens
