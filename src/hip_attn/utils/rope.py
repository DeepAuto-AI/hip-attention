import triton
import triton.language as tl

from hip_attn.utils.rotate import rotate_left, rotate_right


@triton.jit
def de_rope(
    vec: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    T: tl.constexpr,
    HID: tl.constexpr,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
):
    ROPE_DIM = rope_range_end - rope_range_begin
    idx_hid = tl.arange(0, HID)

    c0, ch = cos, rotate_left(cos, ROPE_DIM // 2)
    s0, sh = sin, rotate_left(sin, ROPE_DIM // 2)
    vr0, vrh = vec, rotate_left(vec, ROPE_DIM // 2)

    out0 = (vrh * s0 + vr0 * ch) / (c0 * ch + sh * s0 + 1e-20)
    outh = (out0 * c0 - vr0) / (s0 + 1e-20)

    outh = rotate_right(outh, ROPE_DIM // 2)
    out = tl.where(
        (rope_range_begin <= idx_hid) & (idx_hid < rope_range_end),
        tl.where(
            idx_hid < rope_range_begin + ROPE_DIM // 2,
            out0,
            outh,
        ),
        vec,
    )
    return out


@triton.jit
def rotate_half(
    vec: tl.tensor,
    T: tl.constexpr,
    HID: tl.constexpr,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
):
    idx_hid = tl.arange(0, HID)
    idx_rope_range = idx_hid - rope_range_begin
    ROPE_DIM = rope_range_end - rope_range_begin

    vec *= ((idx_rope_range + ROPE_DIM // 2 < ROPE_DIM) * (-2) + 1).to(vec.dtype)

    return vec


@triton.jit
def apply_rope(
    vec: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
    T: tl.constexpr,
    HID: tl.constexpr,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
):
    idx_hid = tl.arange(0, HID)
    vec_rot = rotate_half(vec, T, HID, rope_range_begin, rope_range_end)
    vec = tl.where(
        (rope_range_begin <= idx_hid) & (idx_hid < rope_range_end),
        vec * cos + vec_rot * sin,
        vec,
    )
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
    NEED_APPLY_ROPE: tl.constexpr,
    rope_range_begin: tl.constexpr = 0,
    rope_range_end: tl.constexpr = None,
):
    if rope_range_end is None:
        rope_range_end: tl.constexpr = HID

    idx_rope_range = idx_hid - rope_range_begin
    rope_mask = (rope_range_begin <= idx_hid) & (idx_hid < rope_range_end)

    if not NEED_APPLY_ROPE:
        mask_t = mask_t & (old_t != 0)

        cos_old = tl.load(
            COS
            + old_t[:, None].to(tl.int64) * stride_cos_t
            + idx_rope_range[None, :] * stride_cos_hid,
            mask=tl.ravel(mask_t)[:, None] & rope_mask[None, :],
            other=0,
        )
        sin_old = tl.load(
            SIN
            + old_t[:, None].to(tl.int64) * stride_sin_t
            + idx_rope_range[None, :] * stride_sin_hid,
            mask=tl.ravel(mask_t)[:, None] & rope_mask[None, :],
            other=0,
        )

        cos_new = tl.load(
            COS
            + new_t[:, None].to(tl.int64) * stride_cos_t
            + idx_rope_range[None, :] * stride_cos_hid,
            mask=tl.ravel(mask_t)[:, None] & rope_mask[None, :],
            other=0,
        )
        sin_new = tl.load(
            SIN
            + new_t[:, None].to(tl.int64) * stride_sin_t
            + idx_rope_range[None, :] * stride_sin_hid,
            mask=tl.ravel(mask_t)[:, None] & rope_mask[None, :],
            other=0,
        )

        tokens_adjusted = de_rope(
            tokens.to(tl.float32),
            cos_old.to(tl.float32),
            sin_old.to(tl.float32),
            T,
            HID,
            rope_range_begin,
            rope_range_end,
        )
        tokens_adjusted = apply_rope(
            tokens_adjusted.to(tl.float32),
            cos_new.to(tl.float32),
            sin_new.to(tl.float32),
            T,
            HID,
            rope_range_begin,
            rope_range_end,
        )

        tokens = tl.where(mask_t[:, None], tokens_adjusted.to(tokens.dtype), tokens)

        return tokens
    else:
        cos_new = tl.load(
            COS
            + new_t[:, None].to(tl.int64) * stride_cos_t
            + idx_rope_range[None, :] * stride_cos_hid,
            mask=tl.ravel(mask_t)[:, None] & rope_mask[None, :],
            other=0.0,
        )
        sin_new = tl.load(
            SIN
            + new_t[:, None].to(tl.int64) * stride_sin_t
            + idx_rope_range[None, :] * stride_sin_hid,
            mask=tl.ravel(mask_t)[:, None] & rope_mask[None, :],
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
