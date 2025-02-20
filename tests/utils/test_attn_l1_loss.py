import time
import unittest

import torch
import torch.autograd
import torch.utils.checkpoint

from hip_attn.utils.attn_l1_loss import (
    compute_attn_lp_loss_orig,
    compute_attn_lp_loss_triton,
)


class TestAttnL1Loss(unittest.TestCase):

    def test_attn_l1_loss(self):
        # torch.autograd.set_detect_anomaly(True)
        N = 1
        H = 40
        TDST, TSRC = 1024, 1024
        HDIM = 64
        KV_BLOCK_SIZE, Q_BLOCK_SIZE = 64, 64
        p = 0.5
        q = torch.randn(N, H, TDST, HDIM, device="cuda", requires_grad=True)
        k = torch.randn(N, H, TSRC, HDIM, device="cuda", requires_grad=True)
        do_backward = True
        do_compare = True
        do_average = False
        is_causal = False
        use_attend_lengths = False
        attend_lengths = None
        if use_attend_lengths:
            attend_lengths = torch.randperm(TDST, device="cuda").expand(N, TDST)
        noise = (
            torch.randn(N, H, device="cuda")
            if do_average
            else torch.randn(N, H, TDST, device="cuda")
        )

        for _ in range(3):
            torch.cuda.synchronize()

            if do_compare:
                torch.cuda.reset_peak_memory_stats()
                time_begin = time.time()
                l1_loss_orig = compute_attn_lp_loss_orig(
                    q,
                    k,
                    p,
                    is_causal=is_causal,
                    do_average=do_average,
                    attend_lengths=attend_lengths,
                )
                # KV_BLOCK_SIZE=KV_BLOCK_SIZE,
                # Q_BLOCK_SIZE=Q_BLOCK_SIZE)
                if do_backward:
                    (l1_loss_orig * noise).sum().backward()
                    grad_q_orig = q.grad.detach()
                    q.grad = None
                    grad_k_orig = k.grad.detach()
                    k.grad = None
                l1_loss_orig = l1_loss_orig.detach()
                torch.cuda.synchronize()
                print("orig time:", time.time() - time_begin)
                print(torch.cuda.max_memory_allocated() / 1024**2, "MB")

            torch.cuda.reset_peak_memory_stats()
            time_begin = time.time()
            l1_loss = compute_attn_lp_loss_triton(
                q,
                k,
                p,
                is_causal=is_causal,
                do_average=do_average,
                attend_lengths=attend_lengths,
                KV_BLOCK_SIZE=KV_BLOCK_SIZE,
                Q_BLOCK_SIZE=Q_BLOCK_SIZE,
            )
            if do_backward:
                (l1_loss * noise).sum().backward()
                grad_q = q.grad.detach()
                q.grad = None
                grad_k = k.grad.detach()
                k.grad = None
            l1_loss = l1_loss.detach()
            torch.cuda.synchronize()
            print("triton time:", time.time() - time_begin)
            print(torch.cuda.max_memory_allocated() / 1024**2, "MB")

            if do_compare:
                print("loss")
                compare(l1_loss_orig, l1_loss)

                if do_backward:
                    print("grad_q")
                    compare(grad_q_orig, grad_q)
                    print("grad_k")
                    compare(grad_k_orig, grad_k)


def compare(orig, new):
    stderr = (new - orig).abs().mean().item()
    stdcontext = torch.std_mean(orig)[0].item()

    print(
        f"err = {stderr:.6f} ({stderr / stdcontext:.4f} sigma), out_std = {stdcontext:.6f}"
    )
