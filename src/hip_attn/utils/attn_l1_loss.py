import torch
import torch.autograd
import torch.utils.checkpoint
import triton
import triton.language as tl


@triton.jit
def compute_attn_lp_loss_kernel(
    q,
    q_stride_n,
    q_stride_h,
    q_stride_t,
    q_stride_hdim,
    k,
    k_stride_n,
    k_stride_h,
    k_stride_t,
    k_stride_hdim,
    p: float,
    H: int,
    TDST: int,
    TSRC: int,
    HDIM: int,
    HDIM_MAX: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    attend_lengths,
    attend_lengths_stride_n,
    attend_lengths_stride_t,
    l,
    l_stride_n,
    l_stride_h,
    l_stride_t,
    m,
    m_stride_n,
    m_stride_h,
    m_stride_t,
    output,
    output_stride_n,
    output_stride_h,
    output_stride_t,
):
    batch_idx = tl.program_id(1)
    n_idx = batch_idx // H
    h_idx = batch_idx % H
    q_begin = tl.program_id(0) * Q_BLOCK_SIZE
    q_idx = tl.arange(0, Q_BLOCK_SIZE)
    kv_idx = tl.arange(0, KV_BLOCK_SIZE)
    d_idx = tl.arange(0, HDIM_MAX)

    q_chunk = tl.load(
        q
        + n_idx * q_stride_n
        + h_idx * q_stride_h
        + (q_begin + q_idx)[:, None] * q_stride_t
        + d_idx[None, :] * q_stride_hdim,
        mask=((q_begin + q_idx < TDST)[:, None] & (d_idx < HDIM)[None, :]),
        other=0,
    )  # [q_blk, hd]

    attend_lengths_chunk = None
    if attend_lengths is not None:
        attend_lengths_chunk = tl.load(
            attend_lengths
            + n_idx * attend_lengths_stride_n
            + (q_begin + q_idx) * attend_lengths_stride_t,
            mask=(q_begin + q_idx < TDST),
            other=0,
        )  # [q_blk]

    for kv_begin in range(0, TSRC, KV_BLOCK_SIZE):
        k_chunk = tl.load(
            k
            + n_idx * k_stride_n
            + h_idx * k_stride_h
            + (kv_begin + kv_idx)[None, :] * k_stride_t
            + d_idx[:, None] * k_stride_hdim,
            mask=((kv_begin + kv_idx < TSRC)[None, :] & (d_idx < HDIM)[:, None]),
            other=0,
        )  # [hd, kv_blk]
        output_chunk = tl.load(
            output
            + n_idx * output_stride_n
            + h_idx * output_stride_h
            + (q_begin + q_idx)[:, None] * output_stride_t,
            mask=((q_begin + q_idx < TDST)[:, None]),
            other=0,
        )  # [q_blk, 1]
        l_chunk = tl.load(
            l
            + n_idx * l_stride_n
            + h_idx * l_stride_h
            + (q_begin + q_idx)[:, None] * l_stride_t,
            mask=((q_begin + q_idx < TDST)[:, None]),
            other=0,
        )  # [q_blk, 1]
        m_chunk = tl.load(
            m
            + n_idx * m_stride_n
            + h_idx * m_stride_h
            + (q_begin + q_idx)[:, None] * m_stride_t,
            mask=((q_begin + q_idx < TDST)[:, None]),
            other=-1e9,
        )  # [q_blk, 1]
        attn_scores = tl.dot(q_chunk.to(tl.float16), k_chunk.to(tl.float16)).to(
            tl.float32
        )  # [q_blk, kv_blk]
        if IS_CAUSAL:
            attn_scores = tl.where(
                (kv_begin + kv_idx)[None, :] > (q_begin + q_idx)[:, None],
                -1e9,
                attn_scores,
            )
        if attend_lengths is not None:
            attn_scores = tl.where(
                (kv_begin + kv_idx)[None, :] >= attend_lengths_chunk[:, None],
                -1e9,
                attn_scores,
            )
        m_tilde = tl.max(attn_scores, axis=1)[:, None]  # [q_blk, 1]
        P_tilde = tl.exp(attn_scores - m_tilde)  # [q_blk, kv_blk]
        l_tilde = tl.sum(P_tilde, axis=1)[:, None]  # [q_blk, 1]
        m_new = tl.maximum(m_chunk, m_tilde)  # [q_blk, 1]
        l_new = (
            tl.exp(m_chunk - m_new) * l_chunk + tl.exp(m_tilde - m_new) * l_tilde
        )  # [q_blk, 1]

        loss_new = tl.exp(tl.log(l_new) * -p) * (
            tl.exp(p * (tl.log(l_chunk) + m_chunk - m_new)) * output_chunk
            + tl.exp(p * (m_tilde - m_new))
            * tl.sum(tl.exp((attn_scores - m_tilde) * p), axis=1)[:, None]
        )  # [q_blk, 1]
        tl.store(
            output
            + n_idx * output_stride_n
            + h_idx * output_stride_h
            + (q_begin + q_idx)[:, None] * output_stride_t,
            loss_new,
            mask=(q_begin + q_idx < TDST)[:, None],
        )
        tl.store(
            m
            + n_idx * m_stride_n
            + h_idx * m_stride_h
            + (q_begin + q_idx)[:, None] * m_stride_t,
            m_new,
            mask=(q_begin + q_idx < TDST)[:, None],
        )
        tl.store(
            l
            + n_idx * l_stride_n
            + h_idx * l_stride_h
            + (q_begin + q_idx)[:, None] * l_stride_t,
            l_new,
            mask=(q_begin + q_idx < TDST)[:, None],
        )


@triton.jit
def compute_attn_lp_loss_kernel_backward(
    q,
    q_stride_n,
    q_stride_h,
    q_stride_t,
    q_stride_hdim,
    k,
    k_stride_n,
    k_stride_h,
    k_stride_t,
    k_stride_hdim,
    output,
    output_stride_n,
    output_stride_h,
    output_stride_t,
    grad_output,
    grad_output_stride_n,
    grad_output_stride_h,
    grad_output_stride_t,
    p: float,
    H: int,
    TDST: int,
    TSRC: int,
    HDIM: int,
    HDIM_MAX: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    attend_lengths,
    attend_lengths_stride_n,
    attend_lengths_stride_t,
    l,
    l_stride_n,
    l_stride_h,
    l_stride_t,
    m,
    m_stride_n,
    m_stride_h,
    m_stride_t,
    grad_q,
    grad_q_stride_n,
    grad_q_stride_h,
    grad_q_stride_t,
    grad_q_stride_hdim,
    grad_k,
    grad_k_stride_n,
    grad_k_stride_h,
    grad_k_stride_t,
    grad_k_stride_hdim,
):
    batch_idx = tl.program_id(1)
    n_idx = batch_idx // H
    h_idx = batch_idx % H
    q_begin = tl.program_id(0) * Q_BLOCK_SIZE
    q_idx = tl.arange(0, Q_BLOCK_SIZE)
    kv_idx = tl.arange(0, KV_BLOCK_SIZE)
    d_idx = tl.arange(0, HDIM_MAX)

    q_chunk = tl.load(
        q
        + n_idx * q_stride_n
        + h_idx * q_stride_h
        + (q_begin + q_idx)[None, :] * q_stride_t
        + d_idx[:, None] * q_stride_hdim,
        mask=((q_begin + q_idx < TDST)[None, :] & (d_idx < HDIM)[:, None]),
        other=0,
    ).to(
        tl.float32
    )  # [hd, q_blk]
    output_chunk = tl.load(
        output
        + n_idx * output_stride_n
        + h_idx * output_stride_h
        + (q_begin + q_idx)[None, :] * output_stride_t,
        mask=((q_begin + q_idx < TDST)[None, :]),
        other=0,
    ).to(
        tl.float32
    )  # [1, q_blk]
    grad_output_chunk = tl.load(
        grad_output
        + n_idx * grad_output_stride_n
        + h_idx * grad_output_stride_h
        + (q_begin + q_idx)[None, :] * grad_output_stride_t,
        mask=((q_begin + q_idx < TDST)[None, :]),
        other=0,
    ).to(
        tl.float32
    )  # [1, q_blk]
    l_chunk = tl.load(
        l
        + n_idx * l_stride_n
        + h_idx * l_stride_h
        + (q_begin + q_idx)[None, :] * l_stride_t,
        mask=((q_begin + q_idx < TDST)[None, :]),
        other=0,
    ).to(
        tl.float32
    )  # [1, q_blk]
    m_chunk = tl.load(
        m
        + n_idx * m_stride_n
        + h_idx * m_stride_h
        + (q_begin + q_idx)[None, :] * m_stride_t,
        mask=((q_begin + q_idx < TDST)[None, :]),
        other=-1e9,
    ).to(
        tl.float32
    )  # [1, q_blk]

    attend_lengths_chunk = None
    if attend_lengths is not None:
        attend_lengths_chunk = tl.load(
            attend_lengths
            + n_idx * attend_lengths_stride_n
            + (q_begin + q_idx) * attend_lengths_stride_t,
            mask=(q_begin + q_idx < TDST),
            other=0,
        )

    for kv_begin in range(0, TSRC, KV_BLOCK_SIZE):
        k_chunk = tl.load(
            k
            + n_idx * k_stride_n
            + h_idx * k_stride_h
            + (kv_begin + kv_idx)[:, None] * k_stride_t
            + d_idx[None, :] * k_stride_hdim,
            mask=((kv_begin + kv_idx < TSRC)[:, None] & (d_idx < HDIM)[None, :]),
            other=0,
        ).to(
            tl.float32
        )  # [kv_blk, hd]

        attn_scores = tl.dot(k_chunk, q_chunk).to(tl.float32)  # [kv_blk, q_blk]
        logP = attn_scores - m_chunk - tl.log(l_chunk)  # [kv_blk, q_blk]
        grad_P = grad_output_chunk * p * tl.exp(logP * (p - 1))  # [kv_blk, q_blk]

        D = grad_output_chunk * p * output_chunk  # [1, q_blk]
        grad_S = tl.exp(logP) * (grad_P - D)  # [kv_blk, q_blk]
        if IS_CAUSAL:
            grad_S = tl.where(
                (kv_begin + kv_idx)[:, None] > (q_begin + q_idx)[None, :], 0.0, grad_S
            )
        if attend_lengths is not None:
            grad_S = tl.where(
                (kv_begin + kv_idx)[:, None] >= attend_lengths_chunk[None, :],
                0.0,
                grad_S,
            )

        grad_q_new = tl.dot(tl.trans(grad_S), k_chunk).to(tl.float32)  # [q_blk, hd]
        tl.atomic_add(
            grad_q
            + n_idx * grad_q_stride_n
            + h_idx * grad_q_stride_h
            + (q_begin + q_idx)[:, None] * grad_q_stride_t
            + d_idx[None, :] * grad_q_stride_hdim,
            grad_q_new,
            mask=((q_begin + q_idx < TDST)[:, None] & (d_idx < HDIM)[None, :]),
        )
        grad_k_chunk = tl.dot(q_chunk, tl.trans(grad_S)).to(tl.float32)  # [hd, kv_blk]
        tl.atomic_add(
            grad_k
            + n_idx * grad_k_stride_n
            + h_idx * grad_k_stride_h
            + (kv_begin + kv_idx)[None, :] * grad_k_stride_t
            + d_idx[:, None] * grad_k_stride_hdim,
            grad_k_chunk,
            mask=((kv_begin + kv_idx < TSRC)[None, :] & (d_idx < HDIM)[:, None]),
        )


class AttnLpLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,  # noqa
        q,
        k,
        N,
        H,
        TDST,
        TSRC,
        HDIM,
        p,
        is_causal,
        attend_lengths,
        KV_BLOCK_SIZE,
        Q_BLOCK_SIZE,
    ):
        assert q.ndim == 4
        assert k.ndim == 4
        assert attend_lengths.ndim == 2 if attend_lengths is not None else True
        l = torch.full((N, H, TDST), 0.0, device=q.device)  # [bsz, num_heads, q_len]
        m = torch.full((N, H, TDST), -1e9, device=q.device)  # [bsz, num_heads, q_len]
        result = torch.zeros((N, H, TDST), device=q.device)

        orig_device = torch.cuda.current_device()
        torch.cuda.set_device(q.device)
        compute_attn_lp_loss_kernel[(triton.cdiv(TDST, Q_BLOCK_SIZE), N * H)](
            q,
            *q.stride(),
            k,
            *k.stride(),
            p,
            H,
            TDST,
            TSRC,
            HDIM,
            triton.next_power_of_2(HDIM),
            KV_BLOCK_SIZE,
            Q_BLOCK_SIZE,
            is_causal,
            attend_lengths,
            *(attend_lengths.stride() if attend_lengths is not None else (None, None)),
            l,
            *l.stride(),
            m,
            *m.stride(),
            result,
            *result.stride(),
        )
        torch.cuda.set_device(orig_device)

        if attend_lengths is not None:
            ctx.save_for_backward(q, k, l, m, result, attend_lengths)
        else:
            ctx.save_for_backward(q, k, l, m, result)
        ctx.has_attend_lengths = attend_lengths is not None
        ctx.N, ctx.H, ctx.TDST, ctx.TSRC, ctx.HDIM = N, H, TDST, TSRC, HDIM
        ctx.p, ctx.is_causal = p, is_causal
        ctx.KV_BLOCK_SIZE, ctx.Q_BLOCK_SIZE = KV_BLOCK_SIZE, Q_BLOCK_SIZE

        result = result ** (1 / p)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # noqa
        if ctx.has_attend_lengths:
            q, k, l, m, result, attend_lengths = ctx.saved_tensors
        else:
            q, k, l, m, result = ctx.saved_tensors
            attend_lengths = None
        N, H, TDST, TSRC, HDIM = ctx.N, ctx.H, ctx.TDST, ctx.TSRC, ctx.HDIM
        p, is_causal = ctx.p, ctx.is_causal
        KV_BLOCK_SIZE, Q_BLOCK_SIZE = ctx.KV_BLOCK_SIZE, ctx.Q_BLOCK_SIZE

        grad_output *= (1 / p) * result ** (1 / p - 1)

        grad_q = torch.full((N, H, TDST, HDIM), 0.0, device=q.device)
        grad_k = torch.full((N, H, TSRC, HDIM), 0.0, device=q.device)

        orig_device = torch.cuda.current_device()
        torch.cuda.set_device(q.device)
        compute_attn_lp_loss_kernel_backward[(triton.cdiv(TDST, Q_BLOCK_SIZE), N * H)](
            q,
            *q.stride(),
            k,
            *k.stride(),
            result,
            *result.stride(),
            grad_output,
            *grad_output.stride(),
            p,
            H,
            TDST,
            TSRC,
            HDIM,
            triton.next_power_of_2(HDIM),
            KV_BLOCK_SIZE,
            Q_BLOCK_SIZE,
            is_causal,
            attend_lengths,
            *(attend_lengths.stride() if attend_lengths is not None else (None, None)),
            l,
            *l.stride(),
            m,
            *m.stride(),
            grad_q,
            *grad_q.stride(),
            grad_k,
            *grad_k.stride(),
        )
        torch.cuda.set_device(orig_device)

        return (
            grad_q,
            grad_k,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def compute_attn_lp_loss_triton(
    q,
    k,
    p,
    is_causal=True,
    attend_lengths=None,
    do_average=True,
    KV_BLOCK_SIZE=64,
    Q_BLOCK_SIZE=64,
):
    assert q.ndim == 4 and k.ndim == 4
    N, H, TDST, TSRC, HDIM = q.shape[0], q.shape[1], q.shape[2], k.shape[2], q.shape[3]
    result = AttnLpLoss.apply(
        q,
        k,
        N,
        H,
        TDST,
        TSRC,
        HDIM,
        p,
        is_causal,
        attend_lengths,
        KV_BLOCK_SIZE,
        Q_BLOCK_SIZE,
    )
    if do_average:
        result = result.mean(dim=-1)  # [bsz, num_heads]
    return result


def process_block(
    q_chunk,
    k_chunk,
    m_chunk,
    l_chunk,
    loss_chunk,
    p: float,
    is_causal: bool,
    offset: int,
):
    attn_scores = torch.einsum("bhqd,bhkd->bhqk", q_chunk, k_chunk)
    if is_causal:
        invalid_mask = torch.triu(
            torch.ones(attn_scores.shape[-2:], device=q_chunk.device, dtype=torch.bool),
            offset + 1,
        )
        attn_scores = attn_scores.masked_fill(invalid_mask, -1e9)
    m_tilde = attn_scores.amax(dim=-1, keepdim=True)  # [*, q_blk, 1]
    P_tilde = torch.exp(attn_scores - m_tilde)  # [*, q_blk, kv_blk]
    l_tilde = P_tilde.sum(dim=-1, keepdim=True)  # [*, q_blk, 1]
    m_new = torch.maximum(m_chunk, m_tilde)  # [*, q_blk, 1]
    l_new = (
        torch.exp(m_chunk - m_new) * l_chunk + torch.exp(m_tilde - m_new) * l_tilde
    )  # [*, q_blk, 1]

    loss_new = torch.exp(torch.log(l_new) * -p) * (
        torch.exp(p * (torch.log(l_chunk) + m_chunk - m_new)) * loss_chunk
        + torch.exp(p * (m_tilde - m_new))
        * torch.exp((attn_scores - m_tilde) * p).sum(dim=-1, keepdim=True)
    )
    return loss_new, m_new, l_new


def compute_attn_lp_loss(
    q,
    k,
    p,
    is_causal=True,
    do_average=True,
    KV_BLOCK_SIZE=4096,
    Q_BLOCK_SIZE=4096,
    use_checkpoint=True,
):
    assert q.ndim == 4 and k.ndim == 4
    N, H, TDST, TSRC, HDIM = q.shape[0], q.shape[1], q.shape[2], k.shape[2], q.shape[3]
    # q: shape [bsz, num_heads, q_len, head_dim]
    # k: shape [bsz, num_heads, kv_len, head_dim]
    m = torch.full((N, H, TDST), -1e9, device=q.device)  # [bsz, num_heads, q_len]
    l = torch.full((N, H, TDST), 0.0, device=q.device)  # [bsz, num_heads, q_len]
    attn_sparsity_loss = torch.zeros((N, H, TDST), device=q.device)

    q_chunks = list(torch.split(q, Q_BLOCK_SIZE, dim=2))
    m_chunks = list(torch.split(m.unsqueeze(-1), Q_BLOCK_SIZE, dim=2))
    l_chunks = list(torch.split(l.unsqueeze(-1), Q_BLOCK_SIZE, dim=2))
    loss_chunks = list(
        torch.split(attn_sparsity_loss.unsqueeze(-1), Q_BLOCK_SIZE, dim=2)
    )

    for i, k_chunk in enumerate(torch.split(k, KV_BLOCK_SIZE, dim=2)):
        for j in range(len(q_chunks)):
            offset = j * Q_BLOCK_SIZE - i * KV_BLOCK_SIZE
            if use_checkpoint:
                loss_new, m_new, l_new = torch.utils.checkpoint.checkpoint(
                    process_block,
                    q_chunks[j],
                    k_chunk,
                    m_chunks[j],
                    l_chunks[j],
                    loss_chunks[j],
                    p,
                    is_causal,
                    offset,
                    use_reentrant=False,
                )
            else:
                loss_new, m_new, l_new = process_block(
                    q_chunks[j],
                    k_chunk,
                    m_chunks[j],
                    l_chunks[j],
                    loss_chunks[j],
                    p,
                    is_causal,
                    offset,
                )
            loss_chunks[j] = loss_new
            m_chunks[j] = m_new
            l_chunks[j] = l_new

    attn_sparsity_loss = torch.cat(loss_chunks, dim=2).squeeze(-1) ** (1 / p)
    if do_average:
        attn_sparsity_loss = attn_sparsity_loss.mean(dim=-1)  # [bsz, num_heads]
    return attn_sparsity_loss


def compute_attn_lp_loss_orig(
    q, k, p, is_causal=True, attend_lengths=None, do_average=True
):
    assert q.ndim == 4 and k.ndim == 4
    attn_scores = torch.einsum(
        "bhqd,bhkd->bhqk", q, k
    )  # shape [bsz, num_heads, q_len, kv_len]
    assert not (is_causal and attend_lengths is not None)
    if is_causal:
        invalid_mask = torch.triu(
            torch.ones(attn_scores.shape[-2:], device=q.device, dtype=torch.bool), 1
        )
        attn_scores = attn_scores.masked_fill(invalid_mask, -1e9)
    if attend_lengths is not None:  # shape [bsz, q_len]
        attn_scores[
            (
                torch.arange(k.shape[-2], device=q.device)
                >= attend_lengths[:, None, :, None]
            ).expand_as(attn_scores)
        ] = -1e9
    attn_log_probs = torch.nn.functional.log_softmax(attn_scores, dim=-1)
    attn_sparsity_loss = torch.exp(attn_log_probs * p).sum(dim=-1)
    attn_sparsity_loss = attn_sparsity_loss ** (1 / p)
    if do_average:
        attn_sparsity_loss = attn_sparsity_loss.mean(dim=-1)  # [bsz, num_heads]
    return attn_sparsity_loss
