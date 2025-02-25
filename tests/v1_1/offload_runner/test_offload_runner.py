import argparse
import unittest

from hip_attn.v1_1.attention2_draft_prefetch import HiPAttentionArgs
from hip_attn.v1_1.offload_runner.offload_runner import Runner

TEMPLATE = """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi, can you describe about following document? Here is document,

```
{document}
```

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


class TestOffloadRunner(unittest.TestCase):

    def test_offload_runner(self):
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument("--method", default="hip", type=str)
            parser.add_argument("--cache_backend", default="cuda", type=str)
            parser.add_argument("--input", default="./samples/32k.md", type=str)
            parser.add_argument("--model", default="llama3.1_8b", type=str)
            parser.add_argument("--batch_size", default=16, type=int)
            parser.add_argument("--kv_share", default=1, type=int)
            parser.add_argument("--max_tokens", default=64, type=int)
            parser.add_argument("--k", default=512, type=int)
            parser.add_argument("--sw", default=256, type=int)
            parser.add_argument("--cache_size", default=8192, type=int)
            parser.add_argument(
                "--offload-cache", action=argparse.BooleanOptionalAction
            )
            parser.add_argument("--block_size_k", default=2, type=int)
            parser.add_argument("--refresh_interval", default=8, type=int)
            parser.add_argument(
                "--simulate-hit-ratio", action=argparse.BooleanOptionalAction
            )
            parser.add_argument("--simulated_mask_hit_ratio", default=0.8, type=float)
            parser.add_argument("--simulated_sa_hit_ratio", default=0.99, type=float)
            parser.add_argument(
                "--offload_cache_method", default="single_level", type=str
            )
            parser.add_argument("--sample_method", default="last", type=str)

            args = parser.parse_args()

            print(args)

            with open(args.input, "r") as f:
                document = f.read()

            sample_input = TEMPLATE.format(document=document)
            results = Runner(
                {
                    "llama3.1_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "llama2_7b": "meta-llama/Llama-2-7b-chat-hf",
                    "llama2_13b": "meta-llama/Llama-2-13b-chat-hf",
                }[args.model],
                method=args.method,
                cache_backend=args.cache_backend,
                kv_share=args.kv_share,
                using_offload_cache=args.offload_cache,
                cache_size=args.cache_size,
                refresh_interval=args.refresh_interval,
                prefix_query=True,
                hip_args=HiPAttentionArgs(
                    mask_k=args.k,
                    block_size_k=args.block_size_k,
                    block_stride_k=1,
                    sliding_window_size=args.sw,
                    sample_method=args.sample_method,
                    offload_cache_method=args.offload_cache_method,
                ),
                offload_cache_method=args.offload_cache_method,
            ).generate(
                sample_input,
                item_repeat=args.batch_size,
                max_tokens=args.max_tokens,
                simulate_hit_ratio=args.simulate_hit_ratio,
                simulated_mask_hit_ratio=args.simulated_mask_hit_ratio,
                simulated_sa_hit_ratio=args.simulated_sa_hit_ratio,
            )
            print("-" * 20, "example", "-" * 20)
            print(results[-1])
            print("-" * 50)
            for result in results:
                result = result.replace("\n", "\\n")
                print(f"{result[:100]} [...] {len(result)}")
        finally:
            import torch.distributed

            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
