import math

import torch

try:
    from flash_attn import flash_attn_func as flash_attn_func_cuda
except ImportError:
    flash_attn_func_cuda = None

from .flash_attn_triton_for_hyper import flash_attn_func


def indexing(x, indices, chunk_size=-1):
    """
    inputs:
        - x: 4d-tensor with shape [b, h, n, d]
        - indices: 3d-tensor with shape [b, h, s] where each entry should be in [0, n-1]
    output:
        - out: 4d-tensor with shape [b, h, s, d] where out[i,j] = x[i,j][indices[i,j],:]

    A naive implementation:
        out = torch.zeros(b, h, s, d)
        for i in range(b):
            for j in range(h):
                out[i,j] = x[i,j][idx[i,j],:]
        return out
    """
    if chunk_size < 0 or (chunk_size > 0 and x.shape[-2] % chunk_size == 0):
        return x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
    else:
        x = x.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        new_n = math.ceil(x.shape[2] / chunk_size) * chunk_size
        if new_n <= 0 or new_n - x.shape[2] <= 0:
            raise ValueError("new_n <= 0 or new_n - x.shape[2] <= 0")
        #    import pdb
        #    pdb.set_trace()
        return torch.nn.functional.pad(
            x, (0, 0, 0, new_n - x.shape[2]), mode="constant", value=0.0
        )


def add_self_attentions(attn1, lse1, attn2, lse2):
    """
    inputs:
        - attn1, attn2: 4d-tensors with shape [b, h, n, d]
        - lse1, lse2: 4d-tensors of log-sum-exp with shape [b, h, n, 1]
    output:
        - attn
        = (attn1 * exp(lse1) + attn2 * exp(lse2)) / (exp(lse1) + exp(lse2))
        = (attn1 + attn2 * exp(lse2 - lse1)) / (1 + exp(lse2-lse1))
        = attn1 * c + attn2 * (1-c), where c=1/(1 + exp(lse2-lse1)),
        - lse
        = log(exp(lse1) + exp(lse2))
        = log(exp(lse1) * (1 + exp(lse2 - lse1)))
        = lse1 + log(1 + exp(lse2 - lse1)) = lse1 - log(c)
    """
    c = (1 / (1 + (lse2 - lse1).exp())).to(dtype=attn1.dtype)
    attn = c * attn1 + (1 - c) * attn2
    lse = lse1 - (c + torch.finfo(lse1.dtype).eps).log()
    return attn, lse


def exact_attention(query, key, value, softmax_scale, causal=False, bias=None):
    if query.dtype not in [torch.bfloat16, torch.float16]:
        qk = query @ key.transpose(-1, -2) * softmax_scale
        if causal:
            qk += (
                (
                    torch.ones(query.shape[2], key.shape[2], device=query.device)
                    * torch.finfo(query.dtype).min
                )
                .triu(1)
                .reshape(1, 1, query.shape[2], key.shape[2])
            )
        out = qk.softmax(dim=-1) @ value
        lse = torch.logsumexp(qk, dim=-1, keepdim=True)
        return out, lse

    out, lse = flash_attn_func(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        bias,
        causal,
        softmax_scale,
    )
    out = out.transpose(1, 2)

    lse = lse.detach()
    if lse.shape[2] != out.shape[2]:
        lse = lse[:, :, : out.shape[2]]
    lse = lse.unsqueeze(-1)
    return out, lse


def exact_attention_cuda(query, key, value, softmax_scale, causal, bias=None):
    if flash_attn_func_cuda is None:
        raise ImportError(
            "Please install flash_attn (pip install flash-attn --no-build-isolation)"
        )
    out, lse, _ = flash_attn_func_cuda(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=True,
    )
    out = out.transpose(1, 2)
    lse = lse.unsqueeze(-1)
    return out, lse
