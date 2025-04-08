import argparse
import itertools
import json
import math
import time
import traceback
from dataclasses import dataclass
from typing import List

import pandas as pd
import requests
import tqdm
import transformers
from hip_research.utils.seed import seed


@dataclass
class RequestStatistics:
    decode_throughput: int
    prefill_throughput: int
    num_returned: int


def stream_chat_completion(
    endpoint: str, messages, num_prefill: int, num_decode: int, num_concurrent: int
):
    url = f"{endpoint}/flush_cache"
    requests.post(url)

    url = f"{endpoint}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer sk-dummy",
        "Content-Type": "application/json",
    }
    # Note the 'stream': True parameter
    data = {
        "model": "anything",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": num_decode,
        "min_tokens": num_decode,
        "stream": True,
        "n": num_concurrent,
    }

    is_decode = False
    timestamp_start = time.time()
    timestamp_first_token = None
    timestamp_end = None
    num_returned = 0

    try:
        with requests.post(url, headers=headers, json=data, stream=True) as resp:
            # Make sure we got a successful response
            resp.raise_for_status()
            for chunk in resp.iter_lines(decode_unicode=True):
                if chunk:
                    # The stream sends lines prefixed with 'data: '
                    # We only want actual JSON lines, not empty or other SSE fields
                    if chunk.startswith("data: "):
                        if not is_decode:
                            timestamp_first_token = time.time()
                            is_decode = True
                        data_str = chunk[len("data: ") :].strip()
                        # The termination line is: data: [DONE]
                        if data_str == "[DONE]":
                            break

                        # Otherwise, parse the JSON for the token(s)
                        try:
                            parsed_data = json.loads(data_str)
                            # print(data_str)
                            # Each chunk generally has a 'choices' list with 'delta' content
                            if "choices" in parsed_data:
                                for choice in parsed_data["choices"]:
                                    # The content token is in choice["delta"]["content"], if present
                                    if (
                                        "delta" in choice
                                        and "content" in choice["delta"]
                                    ):
                                        num_returned += 1
                        except json.JSONDecodeError:
                            # If there's an error, ignore or handle as needed
                            continue
    except Exception as ex:
        traceback.print_exc()

        return RequestStatistics(
            prefill_throughput=float("nan"),
            decode_throughput=float("nan"),
            num_returned=0,
        )

    timestamp_end = time.time()

    return RequestStatistics(
        prefill_throughput=num_prefill / (timestamp_first_token - timestamp_start),
        decode_throughput=(num_decode * num_concurrent)
        / (timestamp_end - timestamp_first_token),
        num_returned=num_returned,
    )


def get_random_passkey(tokenizer: transformers.LlamaTokenizer, seq_len: int):
    header = "There is a passkey hidden inside a lot of irrelevant text. Find the passkey and memorize it. I will quiz you about the the passkey."
    passkey = "HERE IS THE PASSKEY! The passkey is $000310$. $000310$ is the passkey. **the passkey is $000310$** LOOK BEHIND FOR PASSKEY"
    footer = "In previous text, you have seen the passkey. You had to remember that passkey. What was the passkey? Just answer the secret keyword without any verbal text."
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    len_filler = tokenizer(filler, return_tensors="pt").input_ids.shape[-1]
    num_filler = math.ceil(seq_len * 1024 / len_filler * 0.95)

    text = (
        header
        + filler * (num_filler // 2)
        + passkey
        + filler * (num_filler // 2)
        + footer
    )

    return [
        {
            "role": "system",
            "content": "You are a knowledge retrieval. You have to find a given passkey from random text, and answer only passkey numbers.",
        },
        {"role": "user", "content": text},
    ]


def get_random_example(tokenizer, dataset: str, seq_len: int):
    if dataset == "longbook":
        return get_random_passkey(tokenizer, seq_len)
    elif dataset == "passkey":
        return get_random_passkey(tokenizer, seq_len)
    raise Exception(dataset)


def benchmark(
    endpoint: str,
    model_name: str,
    datasets: List[str],
    seq_lens: List[int],
    decode_lens: List[int],
    num_concurrents: List[int],
):
    data = []

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    for (
        dataset,
        seq_len,
        decode_len,
        num_concurrent,
    ) in tqdm.tqdm(
        list(itertools.product(datasets, seq_lens, decode_lens, num_concurrents)),
        dynamic_ncols=True,
        leave=False,
    ):
        example = get_random_example(tokenizer, dataset, seq_len)
        # run warmup
        stream_chat_completion(
            endpoint, example, seq_len * 1024, decode_len, num_concurrent
        )
        # sample
        result = stream_chat_completion(
            endpoint, example, seq_len * 1024, decode_len, num_concurrent
        )
        data.append(
            {
                "dataset": dataset,
                "prompt_len": seq_len,
                "decode_len": decode_len,
                "num_batch": num_concurrent,
                "throughput_prefill": result.prefill_throughput,
                "throughput_decode": result.decode_throughput,
            }
        )
        tqdm.tqdm.write(
            f"{dataset=}, {seq_len=}, {decode_len=}, {num_concurrent=}, {result=}"
        )

    for line in data:
        print(",".join(map(lambda x: str(x), line.values())))


# Example usage:
if __name__ == "__main__":
    seed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--dataset", nargs="+", type=str)
    parser.add_argument("--prompt", nargs="+", type=int)
    parser.add_argument("--decode", nargs="+", type=int)
    parser.add_argument("--batch", nargs="+", type=int)

    args = parser.parse_args()

    benchmark(
        args.endpoint, args.model, args.dataset, args.prompt, args.decode, args.batch
    )
