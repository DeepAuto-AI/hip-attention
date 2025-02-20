"""
Streaming-LLM: Triton Implementation
"""

import math

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function


@triton.jit
def load_rotary_embedded_vector(
    QK,
    stride_qk_n,
    stride_qk_t,
    stride_qk_hid,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    idx_n,
    idx_t_qk,
    idx_t_rope,
    HID,
    BLOCK_HID,
):
    idx_hid = tl.arange(0, BLOCK_HID).to(tl.int64)
    mask_hid = idx_hid < HID

    idx_hid_rot = ((idx_hid + HID // 2) % HID).to(tl.int64)
    mask_hid_rot = mask_hid

    vec = tl.load(
        QK
        + idx_n.to(tl.int64) * stride_qk_n
        + idx_t_qk.to(tl.int64) * stride_qk_t
        + idx_hid.to(tl.int64) * stride_qk_hid,
        mask=mask_hid,
        other=0,
    )

    vec_rot = tl.load(
        QK
        + idx_n.to(tl.int64) * stride_qk_n
        + idx_t_qk.to(tl.int64) * stride_qk_t
        + idx_hid_rot.to(tl.int64) * stride_qk_hid,
        mask=mask_hid_rot,
        other=0,
    )
    vec_rot = tl.where(idx_hid < HID // 2, -vec_rot, vec_rot)

    cos = tl.load(
        COS
        + idx_t_rope.to(tl.int64) * stride_cos_t
        + idx_hid.to(tl.int64) * stride_cos_hid,
        mask=mask_hid,
        other=0,
    )
    sin = tl.load(
        SIN
        + idx_t_rope.to(tl.int64) * stride_sin_t
        + idx_hid.to(tl.int64) * stride_sin_hid,
        mask=mask_hid,
        other=0,
    )

    vec_rope = ((vec.to(tl.float32) * cos) + (vec_rot.to(tl.float32) * sin)).to(
        vec.dtype
    )

    return vec_rope, vec, vec_rot, cos, sin


@triton.jit
def grad_rotary_embedded_vector(
    grad_vec_rope,
    vec_origin,
    vec_rot,
    cos,
    sin,
    HID,
    BLOCK_HID,
):
    grad_vec_origin = grad_vec_rope * cos
    idx_vec_origin_hid = tl.arange(0, BLOCK_HID)

    grad_vec_rot = grad_vec_rope * sin
    grad_vec_rot = tl.where(idx_vec_origin_hid < HID // 2, -grad_vec_rot, grad_vec_rot)
    idx_vec_rot_hid = (idx_vec_origin_hid + HID // 2) % HID

    return grad_vec_origin, idx_vec_origin_hid, grad_vec_rot, idx_vec_rot_hid


@triton.jit
def _attention_scores_compute(
    # input tensors
    Q,
    stride_q_n,
    stride_q_tdst,
    stride_q_hid,
    K,
    stride_k_n,
    stride_k_tsrc,
    stride_k_hid,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    # output tensors
    INDICES,
    stride_indices_d,
    stride_indices_z,
    VALUES,
    stride_values_z,
    # input variables
    N,
    TDST,
    TSRC,
    HID,
    NUM_SINK,
    WINDOW_SIZE,
    # kernel constants
    BLOCK_HID: tl.constexpr,
):
    idx_n = tl.program_id(0).to(tl.int64)
    idx_tdst = tl.program_id(1).to(tl.int64)
    idx_k = tl.program_id(2).to(tl.int64)

    tdst = idx_tdst + TSRC - TDST

    if idx_k < NUM_SINK:
        idx_tsrc = idx_k
    else:
        window_offset = idx_k - NUM_SINK
        t_tsrc = tdst - WINDOW_SIZE + 1 + window_offset
        idx_tsrc = tl.maximum(idx_k, t_tsrc)

    # load key
    key, _, _, _, _ = load_rotary_embedded_vector(
        K,
        stride_k_n,
        stride_k_tsrc,
        stride_k_hid,
        COS,
        stride_cos_t,
        stride_cos_hid,
        SIN,
        stride_sin_t,
        stride_sin_hid,
        idx_n,
        idx_tsrc,
        idx_k,
        HID,
        BLOCK_HID,
    )

    # load query
    query, _, _, _, _ = load_rotary_embedded_vector(
        Q,
        stride_q_n,
        stride_q_tdst,
        stride_q_hid,
        COS,
        stride_cos_t,
        stride_cos_hid,
        SIN,
        stride_sin_t,
        stride_sin_hid,
        idx_n,
        idx_tdst,
        tl.minimum(tdst, WINDOW_SIZE + NUM_SINK - 1),
        HID,
        BLOCK_HID,
    )

    # calc dot product.
    score = tl.sum(query.to(tl.float32) * key.to(tl.float32))
    score = score * (1 / tl.sqrt(HID.to(tl.float32)))
    # score = tl.extra.cuda.libdevice.tanh(score / 50.0) * 50.0 # gemma2
    score = tl.where(idx_tsrc <= tdst, score, float("-inf"))

    # output
    idx_z = (
        idx_n.to(tl.int64) * TDST * (WINDOW_SIZE + NUM_SINK)
        + idx_tdst.to(tl.int64) * (WINDOW_SIZE + NUM_SINK)
        + idx_k.to(tl.int64)
    )
    tl.store(VALUES + idx_z.to(tl.int64) * stride_values_z, value=score)
    zero = tl.zeros((1,), dtype=tl.int64)
    one = zero + 1
    tl.store(
        INDICES + zero * stride_indices_d + idx_z.to(tl.int64) * stride_indices_z,
        value=idx_n,
    )
    tl.store(
        INDICES + one * stride_indices_d + idx_z.to(tl.int64) * stride_indices_z,
        value=idx_tdst,
    )
    tl.store(
        INDICES + (one * 2) * stride_indices_d + idx_z.to(tl.int64) * stride_indices_z,
        value=idx_tsrc,
    )


@triton.jit
def _attention_score_backward_compute(
    # input tensors
    GRAD_VALUES,
    stride_grad_values_z,
    Q,
    stride_q_n,
    stride_q_tdst,
    stride_q_hid,
    K,
    stride_k_n,
    stride_k_tsrc,
    stride_k_hid,
    INDICES,
    stride_indices_d,
    stride_indices_z,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    # output tensors
    GRAD_Q,
    stride_grad_q_n,
    stride_grad_q_tdst,
    stride_grad_q_hid,
    GRAD_K,
    stride_grad_k_n,
    stride_grad_k_tsrc,
    stride_grad_k_hid,
    # input variables
    N,
    TDST,
    TSRC,
    HID,
    NNZ,
    NUM_SINK,
    WINDOW_SIZE,
    # block constant
    BLOCK_HID: tl.constexpr,
):
    idx_z = tl.program_id(0)

    idx_n = tl.load(INDICES + 0 * stride_indices_d + idx_z * stride_indices_z).to(
        tl.int64
    )
    idx_tdst = tl.load(INDICES + 1 * stride_indices_d + idx_z * stride_indices_z).to(
        tl.int64
    )
    idx_tsrc = tl.load(INDICES + 2 * stride_indices_d + idx_z * stride_indices_z).to(
        tl.int64
    )
    tdst = idx_tdst + TSRC - TDST

    idx_k = idx_z % (NUM_SINK + WINDOW_SIZE)

    # load key
    key, key_origin, key_rot, cos_k, sin_k = load_rotary_embedded_vector(
        K,
        stride_k_n,
        stride_k_tsrc,
        stride_k_hid,
        COS,
        stride_cos_t,
        stride_cos_hid,
        SIN,
        stride_sin_t,
        stride_sin_hid,
        idx_n,
        idx_tsrc,
        idx_k,
        HID,
        BLOCK_HID,
    )

    # load query
    query, query_origin, query_rot, cos_q, sin_q = load_rotary_embedded_vector(
        Q,
        stride_q_n,
        stride_q_tdst,
        stride_q_hid,
        COS,
        stride_cos_t,
        stride_cos_hid,
        SIN,
        stride_sin_t,
        stride_sin_hid,
        idx_n,
        idx_tdst,
        tl.minimum(tdst, WINDOW_SIZE + NUM_SINK - 1),
        HID,
        BLOCK_HID,
    )

    # load value grad
    grad_score = tl.load(
        GRAD_VALUES + idx_z * stride_grad_values_z,
    )

    grad_score = tl.where(idx_tsrc <= tdst, grad_score, 0)
    grad_score = grad_score * (1 / tl.sqrt(HID.to(tl.float32)))

    grad_key = grad_score * query
    grad_query = grad_score * key

    grad_key_origin, idx_key_origin_hid, grad_key_rot, idx_key_rot_hid = (
        grad_rotary_embedded_vector(
            grad_key, key_origin, key_rot, cos_k, sin_k, HID, BLOCK_HID
        )
    )
    grad_query_origin, idx_query_origin_hid, grad_query_rot, idx_query_rot_hid = (
        grad_rotary_embedded_vector(
            grad_query, query_origin, query_rot, cos_q, sin_q, HID, BLOCK_HID
        )
    )

    mask_hid = tl.arange(0, BLOCK_HID) < HID

    tl.atomic_add(
        GRAD_K
        + idx_n * stride_grad_k_n
        + idx_tsrc * stride_grad_k_tsrc
        + idx_key_origin_hid * stride_grad_k_hid,
        mask=mask_hid,
        val=grad_key_origin,
    )
    tl.atomic_add(
        GRAD_K
        + idx_n * stride_grad_k_n
        + idx_tsrc * stride_grad_k_tsrc
        + idx_key_rot_hid * stride_grad_k_hid,
        mask=mask_hid,
        val=grad_key_rot,
    )

    tl.atomic_add(
        GRAD_Q
        + idx_n * stride_grad_q_n
        + idx_tdst * stride_grad_q_tdst
        + idx_query_origin_hid * stride_grad_q_hid,
        mask=mask_hid,
        val=grad_query_origin,
    )
    tl.atomic_add(
        GRAD_Q
        + idx_n * stride_grad_q_n
        + idx_tdst * stride_grad_q_tdst
        + idx_query_rot_hid * stride_grad_q_hid,
        mask=mask_hid,
        val=grad_query_rot,
    )


class AttentionScoreFunc(Function):
    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        cos: Tensor,
        sin: Tensor,
        num_sink: int,
        window_size: int,
    ):
        q = q.contiguous()
        k = k.contiguous()

        assert q.ndim == 3
        assert k.ndim == 3
        assert cos.ndim == 2, cos.shape
        assert sin.ndim == 2, sin.shape
        N, TDST, HID = q.shape
        _, TSRC, _ = k.shape
        assert k.shape == (N, TSRC, HID)
        assert cos.shape[-1] == HID
        assert sin.shape[-1] == HID

        device = q.device
        if q.requires_grad or k.requires_grad:
            dtype = torch.float32  # q.dtype
        else:
            dtype = q.dtype

        nnz = N * TDST * (num_sink + window_size)
        indices = torch.zeros((3, nnz), dtype=torch.int64, device=device)
        values = torch.zeros((nnz,), dtype=dtype, device=device)

        BLOCK_HID = triton.next_power_of_2(HID)

        grid = (N, TDST, num_sink + window_size)

        _device = torch.cuda.current_device()
        torch.cuda.set_device(q.device)
        try:
            _attention_scores_compute[grid](
                q,
                *q.stride(),
                k,
                *k.stride(),
                cos,
                *cos.stride(),
                sin,
                *sin.stride(),
                indices,
                *indices.stride(),
                values,
                *values.stride(),
                N,
                TDST,
                TSRC,
                HID,
                num_sink,
                window_size,
                BLOCK_HID,
                num_warps=1,
                num_stages=1,
            )
        except RuntimeError as ex:
            print(
                N,
                TDST,
                TSRC,
                HID,
                BLOCK_HID,
                num_sink,
                window_size,
                _device,
                "\n",
                q.shape,
                q.dtype,
                q.is_contiguous(),
                q.device,
                "\n",
                k.shape,
                k.dtype,
                k.is_contiguous(),
                k.device,
                "\n",
                cos.shape,
                cos.dtype,
                cos.is_contiguous(),
                cos.device,
                "\n",
                sin.shape,
                sin.dtype,
                sin.is_contiguous(),
                sin.device,
                "\n",
                indices.shape,
                indices.dtype,
                indices.is_contiguous(),
                indices.device,
                "\n",
                values.shape,
                values.dtype,
                values.is_contiguous(),
                values.device,
                "\n",
            )
            raise Exception() from ex
        torch.cuda.set_device(_device)

        ctx.save_for_backward(q, k, cos, sin, indices)
        ctx.num_sink = num_sink
        ctx.window_size = window_size

        return indices, values

    @staticmethod
    def backward(ctx, grad_indices: Tensor, grad_values: Tensor):
        q, k, cos, sin, indices = ctx.saved_tensors
        num_sink = ctx.num_sink
        window_size = ctx.window_size

        N, TDST, HID = q.shape
        _, TSRC, _ = k.shape
        _, NNZ = indices.shape

        assert q.ndim == 3
        assert k.ndim == 3
        assert cos.ndim == 2
        assert sin.ndim == 2
        assert indices.ndim == 2
        assert grad_values.ndim == 1

        grad_q = torch.zeros_like(q, dtype=torch.float32)
        grad_k = torch.zeros_like(k, dtype=torch.float32)

        BLOCK_HID = triton.next_power_of_2(HID)

        grid = (NNZ,)

        _device = torch.cuda.current_device()
        torch.cuda.set_device(q.device)
        _attention_score_backward_compute[grid](
            grad_values,
            *grad_values.stride(),
            q,
            *q.stride(),
            k,
            *k.stride(),
            indices,
            *indices.stride(),
            cos,
            *cos.stride(),
            sin,
            *sin.stride(),
            grad_q,
            *grad_q.stride(),
            grad_k,
            *grad_k.stride(),
            N,
            TDST,
            TSRC,
            HID,
            NNZ,
            num_sink,
            window_size,
            BLOCK_HID,
            num_warps=1,
            num_stages=1,
        )
        torch.cuda.set_device(_device)

        return (
            grad_q,  # q: Tensor
            grad_k,  # k: Tensor
            None,  # cos: Tensor
            None,  # sin: Tensor
            None,  # num_sink: int
            None,  # window_size: int
        )


def attention_scores(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    num_sink: int = 4,
    window_size: int = 512,
):
    N, TDST, HID = q.shape
    _, TSRC, _ = k.shape

    window_size = min(window_size, TSRC - num_sink)

    indices, values = AttentionScoreFunc.apply(
        q,
        k,
        cos,
        sin,
        num_sink,
        window_size,
    )

    values = values.view(-1, num_sink + window_size).softmax(-1).view(-1).contiguous()

    probs = torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=(N, TDST, TSRC),
        requires_grad=q.requires_grad,
        dtype=values.dtype,
        device=values.device,
        check_invariants=False,
    )

    return probs


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if position_ids is None:
        N = q.shape[0]
        TDST = q.shape[1]
        position_ids = torch.arange(0, TDST, device=q.device)[None, :].expand(N, TDST)

    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos  # [seq_len, dim]
    sin = sin  # [seq_len, dim]
    assert cos.ndim == 2
    cos = cos[position_ids]  # [bs, seq_len, dim]
    sin = sin[position_ids]  # [bs, seq_len, dim]
    assert position_ids.ndim == 2
    assert cos.ndim == 3

    q_embed = (q * cos) + (rotate_half(q) * sin)

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


@triton.jit
def _sparse_attention_compute(
    # input matrix
    INDICES,
    stride_indices_d,
    stride_indices_z,
    VALUES,
    stride_values_z,
    V,
    stride_v_n,
    stride_v_tsrc,
    stride_v_hid,
    # output matrix
    CONTEXT,
    stride_context_n,
    stride_context_tdst,
    stride_context_hid,
    # input variables
    N,
    TDST,
    TSRC,
    HID,
    BK,
    NUM_SINK,
    WINDOW_SIZE,
    # block constant
    BLOCK_HID: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    zero = tl.zeros((1,), dtype=tl.int64)
    one = zero + 1
    two = zero + 2

    idx_n = tl.program_id(0).to(tl.int64)
    idx_tdst = tl.program_id(1).to(tl.int64)
    # idx_bk = tl.program_id(2).to(tl.int64)

    idx_hid = tl.arange(0, BLOCK_HID).to(tl.int64)
    mask_hid = idx_hid < HID

    acc = tl.zeros((BLOCK_HID,), dtype=tl.float32)

    for idx_bk in range(BK):
        CACHE_SIZE = NUM_SINK + WINDOW_SIZE
        idx_k = idx_bk.to(tl.int64) * BLOCK_K + tl.arange(0, BLOCK_K).to(tl.int64)
        mask_k = idx_k < CACHE_SIZE

        idx_z = idx_n * TDST * CACHE_SIZE + idx_tdst * CACHE_SIZE + idx_k
        mask_z = mask_k

        idx_tsrc = tl.load(
            INDICES + two * stride_indices_d + idx_z * stride_indices_z,
            mask=mask_z,
            other=0,
        )
        mask_tsrc = mask_z

        score = tl.load(
            VALUES + idx_z * stride_values_z,
            mask=mask_z,
            other=0,
        )

        value = tl.load(
            V
            + idx_n * stride_v_n
            + idx_tsrc[:, None] * stride_v_tsrc
            + idx_hid[None, :] * stride_v_hid,
            mask=mask_tsrc[:, None] & mask_hid[None, :],
            other=0,
        )

        context = tl.sum(score[:, None] * value, axis=0)
        acc += context.to(tl.float32)

    tl.store(
        CONTEXT
        + idx_n * stride_context_n
        + idx_tdst * stride_context_tdst
        + idx_hid * stride_context_hid,
        mask=mask_hid,
        value=acc,
    )


def sparse_attention(
    probs: Tensor,
    v: Tensor,
    num_sink: int,
    window_size: int,
):
    N, TDST, TSRC = probs.shape
    _, _, HID = v.shape

    window_size = min(window_size, TSRC - num_sink)

    values = probs._values()
    indices = probs._indices()

    context = torch.zeros((N, TDST, HID), dtype=v.dtype, device=v.device)

    BLOCK_HID = triton.next_power_of_2(HID)
    BLOCK_K = 128

    grid = (N, TDST)

    assert indices.ndim == 2
    assert values.ndim == 1
    assert v.ndim == 3
    assert context.ndim == 3
    _device = torch.cuda.current_device()
    torch.cuda.set_device(v.device)
    _sparse_attention_compute[grid](
        indices,
        *indices.stride(),
        values,
        *values.stride(),
        v,
        *v.stride(),
        context,
        *context.stride(),
        N,
        TDST,
        TSRC,
        HID,
        triton.cdiv(num_sink + window_size, BLOCK_K),
        num_sink,
        window_size,
        BLOCK_HID,
        BLOCK_K,
    )
    torch.cuda.set_device(_device)

    return context


def sink_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cos: Tensor,
    sin: Tensor,
    num_sink: int = 4,
    window_size: int = 512,
    BENCHMARK: bool = False,
):
    chunk_tdst = 4096
    if q.shape[1] > chunk_tdst:
        contexts = torch.empty_like(q)
        for i_start_tdst in range(0, q.shape[1], chunk_tdst):
            i_end_tdst = min(i_start_tdst + chunk_tdst, q.shape[1])
            t = sink_attention(
                q=q[:, i_start_tdst:i_end_tdst],
                k=k[:, :i_end_tdst],
                v=v[:, :i_end_tdst],
                cos=cos,
                sin=sin,
                num_sink=num_sink,
                window_size=window_size,
                BENCHMARK=BENCHMARK,
            )
            contexts.index_copy_(
                index=torch.arange(i_start_tdst, i_end_tdst, device=t.device),
                source=t,
                dim=1,
            )
        return contexts

    if BENCHMARK:
        event_scores_start = torch.cuda.Event(enable_timing=True)
        event_scores_end = torch.cuda.Event(enable_timing=True)
        event_bmm_start = torch.cuda.Event(enable_timing=True)
        event_bmm_end = torch.cuda.Event(enable_timing=True)
        event_scores_start.record()

    _dtype = v.dtype

    # COO format
    probs = attention_scores(
        q,
        k,
        cos,
        sin,
        num_sink=num_sink,
        window_size=window_size,
    )

    if BENCHMARK:
        event_scores_end.record()
        event_bmm_start.record()

    try:
        if q.requires_grad or k.requires_grad or v.requires_grad:
            if v.dtype in [torch.bfloat16, torch.float16]:
                v = v.to(torch.float32)
            context = torch.bmm(probs, v)
        else:
            context = sparse_attention(probs, v, num_sink, window_size)
    except torch.cuda.OutOfMemoryError as ex:
        print(probs.shape, v.shape)
        raise Exception() from ex

    if context.dtype != _dtype:
        context = context.to(_dtype)

    if BENCHMARK:
        event_bmm_end.record()

        torch.cuda.synchronize()
        elapsed_scores = event_scores_start.elapsed_time(event_scores_end)
        elapsed_bmm = event_bmm_start.elapsed_time(event_bmm_end)

        print(elapsed_scores, elapsed_bmm)

    return context


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_one(x, cos, sin, position_ids=None, unsqueeze_dim=1):
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def sink_attention_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cos: Tensor,
    sin: Tensor,
    num_sink: int = 4,
    window_size: int = 512,
):
    scale = 1 / math.sqrt(q.size(-1))
    outs = []
    for i in range(q.size(1)):
        _q = q[:, i : i + 1]
        _k = k[:, : i + 1]
        _v = v[:, : i + 1]

        if i + 1 > num_sink + window_size:
            k_sinks = _k[:, :num_sink]
            v_sinks = _v[:, :num_sink]

            k_window = _k[:, -window_size:]
            v_window = _v[:, -window_size:]

            _k = torch.cat((k_sinks, k_window), dim=1)
            _v = torch.cat((v_sinks, v_window), dim=1)

        _cos, _sin = cos[: _v.size(1)].unsqueeze(0), sin[: _v.size(1)].unsqueeze(0)

        _q = apply_rotary_pos_emb_one(_q, _cos[:, -1:], _sin[:, -1:])
        _k = apply_rotary_pos_emb_one(_k, _cos, _sin)

        atten = torch.einsum("bqd,bkd->bqk", _q, _k) * scale
        atten = atten.softmax(dim=-1)
        out = torch.bmm(atten, _v)
        outs += [out]

    return torch.cat(outs, dim=1)


def test_against_reference():
    eps = 1.0
    q = torch.nn.Parameter(torch.randn((N, T, HID), device=0) * eps)
    k = torch.nn.Parameter(torch.randn((N, T, HID), device=0) * eps)
    v = torch.nn.Parameter(torch.randn((N, T, HID), device=0) * eps)
    cos = torch.randn((T, HID), device=q.device)
    sin = torch.randn((T, HID), device=q.device)

    x = sink_attention(q, k, v, cos, sin, num_sink=NSINK, window_size=WIND)
    x.mean().backward()

    kernel_q_grad = q.grad
    kernel_k_grad = k.grad
    kernel_v_grad = v.grad

    q.grad = None
    k.grad = None
    v.grad = None

    x_reference = sink_attention_reference(
        q, k, v, cos, sin, num_sink=NSINK, window_size=WIND
    )

    diff = (x - x_reference).abs().amax()
    print(f"max difference between reference forward: {diff}")

    x_reference.mean().backward()

    for name, kern, reference in zip(
        ("q", "k", "v"), (kernel_q_grad, kernel_k_grad, kernel_v_grad), (q, k, v)
    ):
        diff = (kern - reference.grad).abs().amax()
        print(f"max difference between reference grad for {name}: {diff}")

    # sanity check reference against pytorch. big sink and window makes it
    # full attention.
    sanity_check = False
    if sanity_check:
        x_sdpa = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=1 / math.sqrt(q.size(-1))
        )
        x_reference = sink_attention_reference(
            q, k, v, cos, sin, num_sink=128, window_size=128
        )

        diff = (x_sdpa - x_reference).abs().amax()
        print(f"{diff=}")


if __name__ == "__main__":
    N = 32
    T = 1024
    HID = 128
    NSINK = 4
    WIND = 512
    QSIZE = T

    # test_against_reference()

    dtype = torch.float32
    std = 0.2
    q = torch.nn.Parameter(
        (torch.randn((N, T, HID), device=0, dtype=dtype) * std)[
            :, :QSIZE, :
        ].contiguous()
    )
    k = torch.nn.Parameter(torch.randn((N, T, HID), device=0, dtype=dtype) * std)
    v = torch.nn.Parameter(torch.randn((N, T, HID), device=0, dtype=dtype) * std)
    cos = torch.randn((T, HID), device=q.device, dtype=dtype)
    sin = cos.clone()

    for istep in range(10):
        with torch.no_grad():
            sink_attention(
                q.detach(),
                k.detach(),
                v.detach(),
                cos,
                sin,
                num_sink=NSINK,
                window_size=WIND,
                BENCHMARK=True,
            )

    for istep in range(10000):
        x = sink_attention(q, k, v, cos, sin, num_sink=NSINK, window_size=WIND)

        # look up diagonal
        loss = (x - v[:, : q.shape[1], :]).square().mean() * 1000
        (loss * (2**6 if dtype == torch.float16 else 1)).backward()

        lr = 1e-3

        if q.grad is not None:
            q.data += -q.grad * lr
            q.grad = None
        if k.grad is not None:
            k.data += -k.grad * lr
            k.grad = None
        if v.grad is not None:
            v.data += -v.grad * lr * 0.1
            v.grad = None

        if (istep % 5) == 0:
            print("loss", loss.item())
