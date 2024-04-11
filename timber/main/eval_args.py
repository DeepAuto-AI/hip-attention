import argparse
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class ArgsType:
    model: Literal['llama32k', 'llama16b', 'qwen'] = 'llama32k'
    job: Literal['ppl', 'mmlu', 'mmmu', 'stream', 'bench_single_layer'] = 'ppl'
    method: Literal['none', 'timber'] = 'timber'
    stride: int = -1
    lora_r: int = 32
    checkpoint: Optional[str] = None
    count: int = 100
    block_size_q: int = 32
    block_size_k: int = 2
    batch_size: int = 1
    k: int = 512
    dense_queries: int = 0
    dense_layers: int = 3

def eval_args(
    default_model = 'llama32k',
    default_job = 'ppl',
) -> ArgsType:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=default_model)
    parser.add_argument('--job', type=str, default=default_job)
    parser.add_argument('--method', type=str, default='none')
    parser.add_argument('--stride', type=int, default=-1)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--block_size_q', type=int, default=32)
    parser.add_argument('--block_size_k', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--dense_layers', type=int, default=3)
    parser.add_argument('--dense_queries', type=int, default=0)
    parser.add_argument('--name', type=str, default='dev')
    parser.add_argument('--disable_prompt', default=False, action='store_true')
    parser.add_argument('--no_sample', default=False, action='store_true')
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--no_quantize', default=False, action='store_true')
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--rope_method', type=str, default='none')
    parser.add_argument('--disable_flash', default=False, action='store_true')
    parser.add_argument('--disable_sparq', default=False, action='store_true')
    parser.add_argument('--disable_sliding_window', default=False, action='store_true')
    parser.add_argument('--sampling_method', default='random', type=str)
    parser.add_argument('--overwrite', default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    return args
