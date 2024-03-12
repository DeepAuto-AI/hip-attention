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
    dense_queries: int = 2048
    ensemble: bool = False
    ensemble_model_setting : str = "random_pruning"
    ensemble_method : str = "final_attn"
    ensemble_method_final : str = "none"
    ensemble_per_layer_n : int = 1
    ensemble_per_attn_iter_n : int = 5
    ensemble_model_n : int = 5


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
    parser.add_argument('--dense_queries', type=int, default=2048)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=512)

    parser.add_argument('--sampling-method', type=str, default='random') # NOTE JIN added

    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--ensemble-model-setting', type=str, default='random_pruning')
    parser.add_argument('--ensemble-method', type=str, default='final_attn')
    parser.add_argument('--ensemble-method-final', type=str, default='all_agree')
    parser.add_argument('--ensemble-per-layer-n', type=int, default=1)
    parser.add_argument('--ensemble-per-attn-iter-n', type=int, default=5)
    parser.add_argument('--ensemble-model-n', type=int, default=5)
    parser.add_argument('--ensemble-particular-layer', type=int, default=None)

    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()
    print(args)
    return args