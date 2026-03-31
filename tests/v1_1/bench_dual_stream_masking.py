import argparse
import time

import torch
import tqdm
from hip_research.utils.load_checkouts import load_checkouts

from hip_attn import hip_attention_11


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh_interval", default=8, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--sequential_masking", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    print(args)

    samples = 500
    refresh_interval = args.refresh_interval
    batch_size = args.batch_size
    sequential_masking = args.sequential_masking

    q, k, v, out, cos, sin = load_checkouts(
        idx=0, window=40, seq_len=32768, return_cos_sin=True, dtype=torch.bfloat16
    )
    HEAD = q.shape[0]
    HEAD_KV = k.shape[0]

    def reshape(x, HEAD):
        N, T, H = x.shape
        x = x.contiguous().view(N // HEAD, HEAD, T, H).permute(0, 2, 1, 3).contiguous()
        assert x.shape == (N // HEAD, T, HEAD, H)
        assert x.is_contiguous()
        return x

    q = reshape(q, HEAD)
    k = reshape(k, HEAD_KV)
    v = reshape(v, HEAD_KV)
    out = reshape(out, HEAD)

    q = q[:, -1:, :, :]
    q = q.expand(batch_size, -1, -1, -1)
    k = k.expand(batch_size, -1, -1, -1)
    v = v.expand(batch_size, -1, -1, -1)

    torch.cuda.synchronize()

    sa_stream = torch.cuda.Stream()
    mask_stream = sa_stream if sequential_masking else torch.cuda.Stream()

    print(sa_stream.stream_id, mask_stream.stream_id)

    _, meta = hip_attention_11(q, k, v)

    sa_graph = torch.cuda.CUDAGraph()
    context, _ = hip_attention_11(q, k, v, previous_metadata=meta)
    with torch.cuda.graph(sa_graph, stream=sa_stream):
        context, _ = hip_attention_11(q, k, v, previous_metadata=meta)
    mask_graph = torch.cuda.CUDAGraph()
    _, new_meta = hip_attention_11(q, k, v, mask_only=True)
    with torch.cuda.graph(mask_graph, stream=mask_stream):
        _, new_meta = hip_attention_11(q, k, v, mask_only=True)

    t_start = time.time()

    torch.cuda.synchronize()

    for i in tqdm.tqdm(range(samples), dynamic_ncols=True):
        if i == 3:
            t_start = time.time()
        with torch.cuda.stream(mask_stream):
            mask_graph.replay()
            # _ , new_meta = hip_attention_11(q, k, v, mask_only=True)
        with torch.cuda.stream(sa_stream):
            for _ in range(refresh_interval):
                sa_graph.replay()
                # context, _ = hip_attention_11(q, k, v, previous_metadata=meta)
        sa_stream.wait_stream(mask_stream)
    sa_stream.synchronize()

    t_end = time.time()

    elapsed = t_end - t_start
    print(elapsed * 1000)


if __name__ == "__main__":
    main()
