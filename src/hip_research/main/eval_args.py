import argparse
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ArgsType:
    model: str = "llama3.1_8b"
    job: str = "ppl"
    method: str = "fa2"
    stride: int = -1
    lora_r: int = 32
    checkpoint: Optional[str] = None

    count: int = -1
    batch_size: int = 1
    name: str = "dev"

    k: int = 512
    block_size_q: int = 64
    block_stride_q: int = 64
    block_size_k: int = 2
    block_stride_k: int = 1
    dense_queries: int = -1
    dense_layers: int = 3

    disable_prompt: bool = False
    no_sample: bool = False

    input: Optional[str] = None
    output: Optional[str] = None

    no_quantize: bool = False

    max_tokens: int = 512

    rope_method: str = "none"

    disable_flash: bool = False
    disable_sparq: bool = False
    disable_sliding_window: bool = False

    sampling_method: str = "center"

    overwrite: bool = False

    dataset: str = "wikitext"

    endpoint: str = "http://localhost:30000/"

    seed: int = 42


def eval_args(
    default_model="llama3.1_8b",
    default_job="ppl",
) -> ArgsType:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--job", type=str, default=default_job)
    parser.add_argument("--method", type=str, default=ArgsType.method)
    parser.add_argument("--stride", type=int, default=ArgsType.stride)
    parser.add_argument("--lora_r", type=int, default=ArgsType.lora_r)
    parser.add_argument("--checkpoint", type=str, default=ArgsType.checkpoint)
    parser.add_argument("--count", type=int, default=ArgsType.count)
    parser.add_argument("--batch_size", type=int, default=ArgsType.batch_size)
    parser.add_argument("--name", type=str, default=ArgsType.name)
    parser.add_argument("--block_size_q", type=int, default=ArgsType.block_size_q)
    parser.add_argument("--block_stride_q", type=int, default=ArgsType.block_stride_q)
    parser.add_argument("--block_size_k", type=int, default=ArgsType.block_size_k)
    parser.add_argument("--block_stride_k", type=int, default=ArgsType.block_stride_k)
    parser.add_argument("--k", type=int, default=ArgsType.k)
    parser.add_argument("--dense_layers", type=int, default=ArgsType.dense_layers)
    parser.add_argument("--dense_queries", type=int, default=ArgsType.dense_queries)
    parser.add_argument(
        "--disable_prompt", default=ArgsType.disable_prompt, action="store_true"
    )
    parser.add_argument("--no_sample", default=ArgsType.no_sample, action="store_true")
    parser.add_argument("--input", type=str, default=ArgsType.input)
    parser.add_argument("--output", type=str, default=ArgsType.output)
    parser.add_argument(
        "--no_quantize", default=ArgsType.no_quantize, action="store_true"
    )
    parser.add_argument("--max_tokens", type=int, default=ArgsType.max_tokens)
    parser.add_argument("--rope_method", type=str, default=ArgsType.rope_method)
    parser.add_argument(
        "--disable_flash", default=ArgsType.disable_flash, action="store_true"
    )
    parser.add_argument(
        "--disable_sparq", default=ArgsType.disable_sparq, action="store_true"
    )
    parser.add_argument(
        "--disable_sliding_window",
        default=ArgsType.disable_sliding_window,
        action="store_true",
    )
    parser.add_argument("--sampling_method", default=ArgsType.sampling_method, type=str)
    parser.add_argument("--overwrite", default=ArgsType.overwrite, action="store_true")
    parser.add_argument("--dataset", default=ArgsType.dataset, type=str)
    parser.add_argument("--endpoint", default=ArgsType.endpoint, type=str)

    # h2o
    parser.add_argument("--h2o-shift-q-pos", action="store_true")
    parser.add_argument("--h2o-streaming", action="store_true")
    parser.add_argument("--h2o-reduce-for-gqa", type=str, default="average")

    parser.add_argument("--seed", type=int, default=ArgsType.seed)
    args = parser.parse_args()
    print(args)
    return args
