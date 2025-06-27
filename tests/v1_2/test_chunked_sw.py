import torch
import triton
from sglang.srt.layers.attention.flashattention_backend import (
    make_local_attention_virtual_batches,
)

from hip_attn.v1_2.attention_extend_bsa import HiPAttentionArgs, block_sparse_attention

device = torch.device("cuda:0")
dtype = torch.bfloat16
seq_len = 256 * 1024
head = 40
head_kv = 8
tp_size = 8
hid = 5120

q = torch.zeros((1, seq_len, head // tp_size, hid // head), device=device, dtype=dtype)
k = torch.zeros(
    (1, seq_len, head_kv // tp_size, hid // head), device=device, dtype=dtype
)
v = k.clone()
position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)[None, :]


def sample():
    BSZ, TDST, HEAD, HID = q.shape

    args = HiPAttentionArgs(
        k_cache=None,
        v_cache=None,
        offload_cache=None,
        block_table=None,
        cache_seq_lens=None,
        position_ids=position_ids,
        block_size_k=64,
        sliding_window_size=1024,
        sink_token_size=128,
        using_extend=False,
        need_apply_rope=False,
        rope_cos=None,
        rope_sin=None,
        rope_range=None,
        rope_is_neox_style=True,
        logit_softcap=None,
        second_stage_k=1024,
        stages=None,
        model_context_length=seq_len,
        extend_context_length=seq_len,
        block_sparse_block_size_q=64,
        scan_extend_backend="relative",
        sa_extend_backend="streaming",
        online_update_cache=False,
        require_cache_statistics=False,
        disable_flashdecode=True,
        q_mask=None,
        sliding_window_indices=None,
        layer_id=4,
        v_hidden_dim=v.shape[-1],
        using_chunked_sliding_window=True,
    )
    if args.rope_range is None:
        args.rope_range = (0, HID)
    args.block_size_q = args.block_sparse_block_size_q
    args.block_size_k = 128
    args.second_stage_k = 0
    args.sink_token_size = 0
    args.sliding_window_size = 8192
    args.sliding_window_indices = None

    BDST = triton.cdiv(TDST, args.block_size_q)
    BH = BSZ * HEAD

    indices = torch.zeros((BH, BDST, 0), dtype=torch.int64, device=q.device)
    ks = torch.zeros((BH, BDST), dtype=torch.int64, device=q.device)
    ks_count = ks.unsqueeze(-1)
    ks_start_end = torch.zeros((BH, BDST, 2), dtype=torch.int64, device=q.device)

    context = block_sparse_attention(
        q=q,
        k=k,
        v=v,
        seq_lens=args.position_ids + 1,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        access_counter=None,
        cache_miss_counter=None,
        EXTEND_BACKEND=args.sa_extend_backend,
        model_context_length=args.model_context_length,
        extend_context_length=args.extend_context_length,
        offload_update_cache=False,
        args=args,
    )
    context = context.to(q.dtype)
    # metadata = None


samples = []
for i in range(10):
    start_event = torch.cuda.Event(True)
    end_event = torch.cuda.Event(True)
    start_event.record()
    sample()
    end_event.record()
    end_event.synchronize()
    latency = start_event.elapsed_time(end_event)

    print(latency)

    if i > 2:
        samples.append(latency)

print(sum(samples) / len(samples))
