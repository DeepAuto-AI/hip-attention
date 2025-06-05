import torch
import triton
import triton.language as tl
from ..utils import capture

@triton.jit
def _fused_apply_delta(
    
):
    pass

@capture
def apply_delta(
    context_dense: torch.Tensor,
    context_sparse_for_diff: torch.Tensor,
    context_sparse: torch.Tensor,
    idx: torch.Tensor,
    num_last_dense: int,
    args_w: int,
    args_smooth: bool,
):
    if False:
        _fused_apply_delta(
            
        )
    else:
        context_dense_concat = context_dense
        context_dense, last_context_dense = (
            context_dense[:, :-num_last_dense],
            context_dense[:, -num_last_dense:],
        )

        # context_sparse_for_diff_norm = context_sparse_for_diff.float().square().sum(dim=-1, keepdim=True).sqrt()
        # context_dense_norm = context_dense.float().square().sum(dim=-1, keepdim=True).sqrt()
        # scale = context_dense_norm / context_sparse_for_diff_norm

        # take difference
        context_diff = context_dense - context_sparse_for_diff# * scale

        context_diff = context_diff.repeat_interleave(
            args_w, dim=1
        )

        if args_smooth:
            # (exp) linear interpolate diff
            context_diff_shift = torch.roll(context_diff, -args_w, 1)
            context_diff_shift[:, -args_w:] = context_diff[:, -1:]

            offset = torch.arange(0, context_diff.shape[1], device=context_diff.device)
            offset = (offset % args_w).float() / args_w
            context_diff = context_diff + (context_diff_shift - context_diff) * offset[None, :, None, None]

        # context_sparse_norm = context_sparse.float().square().sum(dim=-1, keepdim=True).sqrt()
        # scale = context_dense_norm.repeat_interleave(delta_attention_args_w, dim=1) / context_sparse_norm

        # context = context_sparse * scale + context_diff
        context = context_sparse + context_diff
        context = torch.cat([context, last_context_dense], dim=1).to(context_sparse.dtype)
        context[:, idx] = context_dense_concat
        
        # if get_local_rank() == 0:
        #     print(
        #         'hit', layer_id, 
        #         context_diff.shape, 
        #         context_sparse.shape, 
        #         context_diff.abs().mean().item(), 
        #         context_sparse.abs().mean().item()
        #     )
    
    return context