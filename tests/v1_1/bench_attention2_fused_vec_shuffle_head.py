import time

import torch
import tqdm
from hip_research.utils.load_checkouts import load_checkouts
from torch import Tensor

from hip_attn.v1_1.attention2_draft_causal_batch_gpu_fused_vec import (
    block_sparse_attention,
    hip_masking,
)


def test_random_shuffle(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    indices: Tensor,
    ks: Tensor,
    ks_count: Tensor,
    ks_start_end: Tensor,
    mask_k,
    warm_up,
    samples,
    sliding_window_size,
):
    N, T, H, D = q.shape

    # permutation = torch.randperm(H, device=q.device)

    # def permute_seq(x: torch.Tensor):
    #     return x[:, :, permutation, :]
    # q = permute_seq(q)
    # k = permute_seq(k)
    # v = permute_seq(v)
    # assert q.shape == (N, T, H, D)

    # if indices is not None:
    #     B, _, _ = indices.shape
    #     indices = indices[permutation.repeat(B // H)].fill_(0)
    #     ks = ks[permutation.repeat(B // H)]
    #     ks_count = ks_count[permutation.repeat(B // H)]
    #     ks_start_end = ks_start_end[permutation.repeat(B // H)]

    with torch.no_grad():
        for i in range(warm_up + samples):
            if i == warm_up:
                torch.cuda.synchronize()
                t = time.time()
            block_sparse_attention(
                q=q,
                k=k,
                v=v,
                indices=indices,
                ks=ks,
                ks_count=ks_count,
                ks_start_end=ks_start_end,
                block_size_q=32,
                block_size_k=8,
                mask_k=mask_k,
                sliding_window_size=sliding_window_size,
            )
    torch.cuda.synchronize()
    elapsed = ((time.time() - t) / samples) * 1000
    # return permutation.cpu(), elapsed
    return None, elapsed


def main():
    q, k, v, out, cos, sin = load_checkouts(
        idx=0, window=32, seq_len=32768, dtype=torch.float16, return_cos_sin=True
    )
    q = q.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    k = k.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    v = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    q_quant = q.to(torch.float8_e5m2).view(torch.uint8)
    k_quant = k.to(torch.float8_e5m2).view(torch.uint8)

    print(q.shape, k.shape, v.shape)

    # q = q[:1, :1, :1, :1].expand_as(q)
    # k = k[:1, :1, :1, :1].expand_as(k)
    # v = v[:1, :1, :1, :1].expand_as(v)
    # q_quant = q_quant[:1, :1, :1, :1].expand_as(q)
    # k_quant = k_quant[:1, :1, :1, :1].expand_as(k)

    num_warmups = 3
    num_samples = 50
    num_tests = 10

    num_warmups = 1
    num_samples = 1
    num_tests = 1

    mask_k = 512
    sliding_window_size = 512

    for i in range(num_warmups + num_samples):
        if i == num_warmups:
            torch.cuda.synchronize()
            t = time.time()
        if mask_k > 0:
            (indices, ks, ks_count, ks_start_end, key_access_log, key_access_count) = (
                hip_masking(
                    q=q_quant,
                    k=k_quant,
                    mask_k=mask_k,
                    block_size_q=32,
                    block_stride_q=2,
                    block_size_k=8,
                    block_stride_k=4,
                    sliding_window_size=sliding_window_size,
                    sink_token_size=32,
                )
            )
        else:
            indices = ks = ks_count = ks_start_end = None
    torch.cuda.synchronize()
    elapsed = (time.time() - t) * 1000 / num_samples
    print("masking took", elapsed)

    samples = []
    for i in tqdm.tqdm(range(num_tests), dynamic_ncols=True):
        samples.append(
            test_random_shuffle(
                q,
                k,
                v,
                indices,
                ks,
                ks_count,
                ks_start_end,
                mask_k,
                num_warmups,
                num_samples,
                sliding_window_size,
            )
        )

    samples = list(sorted(samples, key=lambda x: x[1]))
    print(samples[0])
    print(samples[-1])

    from flash_attn import flash_attn_func

    for i in range(num_warmups + num_samples):
        if i == num_warmups:
            torch.cuda.synchronize()
            t = time.time()
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    elapsed_flash = (time.time() - t) * 1000 / num_samples

    from mamba_ssm import Mamba2

    mamba = (
        Mamba2(
            d_model=4096,
            d_state=128,
            d_conv=4,
            expand=2,
            headdim=64,
            ngroups=8,
            D_has_hdim=False,
            rmsnorm=True,
            norm_before_gate=False,
            bias=False,
            conv_bias=True,
            chunk_size=128,
        )
        .to("cuda")
        .eval()
        .half()
    )
    x = v.reshape(v.shape[0], v.shape[1], -1)
    for i in range(num_warmups + num_samples):
        if i == num_warmups:
            torch.cuda.synchronize()
            t = time.time()
        with torch.no_grad():
            mamba(x)
    torch.cuda.synchronize()
    elapsed_mamba = (time.time() - t) * 1000 / num_samples

    print(f"hip masking      : {elapsed:.4f} ms")
    print(f"sparse attention : {samples[0][1]:.4f} ms")
    print(f"total (hip)      : {elapsed + samples[0][1]:.4f} ms")
    print(f"total (flash)    : {elapsed_flash:.4f} ms")
    print(f"total (mamba2)   : {elapsed_mamba:.4f} ms")


if __name__ == "__main__":
    main()
