import json
import os
import subprocess

import torch

n_gpus = torch.cuda.device_count()
model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
output_file = "cache/sglang/result.json"
chunked_prefill_size = int(os.getenv("CHUNK_SIZE", "8192"))


def reset_result():
    os.makedirs("cache/sglang", exist_ok=True)
    if os.path.exists(output_file):
        os.remove(output_file)


reset_result()


def run_sample(seq_len: int, batch_size: int, envs: dict):
    reset_result()
    envs["SRT_MAX_BATCH"] = str(batch_size)

    seq_len = seq_len * 1024
    current_chunked_prefill_size = min(seq_len, chunked_prefill_size)

    subprocess.call(
        (
            f"python -m sglang.bench_latency --model-path {model} --context-length {seq_len + 1024} "
            f"--batch {batch_size} --input-len {seq_len} --output-len {256} --tp-size {n_gpus} "
            f"--mem-fraction-static 0.7 --enable-p2p-check --chunked-prefill-size {current_chunked_prefill_size} "
            f"--max-prefill-tokens {current_chunked_prefill_size} --result-filename {output_file}"
        ).split(),
        env=envs,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError:
                return None, None
            prefill = data["prefill_throughput"]
            decode = data["p90_decode_throughput"]
            return prefill, decode
    else:
        return None, None


seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512]
batch_sizes = [1, 2, 4, 8, 16, 32]

# seq_lens = [256]
# batch_size = [1, 8, 32]

parent_envs = os.environ.copy()
parent_envs.update(
    {
        "POPULATION_FILE": "none",
        "HIP_DISABLE_AUTOTUNE": "1",
        "SRT_MAX_BATCH": "32",
    }
)
test_envs = {
    "HIP G3E": {
        "SRT_ATTENTION_BACKEND": "HIP_ATTN",
        "HIP_EXTEND": "1",
        "EXTEND_LEN": "600",
    },
    "HIP G2": {
        "SRT_ATTENTION_BACKEND": "HIP_ATTN",
        "HIP_EXTEND": "0",
        "HIP_K": "1024",
        "EXTEND_LEN": "600",
    },
    "SRT": {
        "SRT_ATTENTION_BACKEND": "SRT",
        "EXTEND_LEN": "600",
    },
}

results = {
    "settings": {
        "seq_lens": seq_lens,
        "batch_sizes": batch_sizes,
    }
}
for test_exp_name, test_exp_update in test_envs.items():
    print("=" * 80)
    envs = parent_envs.copy()
    envs.update(test_exp_update)

    print(f"test {test_exp_name}")

    data_prefill = []
    data_decode = []
    for batch_size in batch_sizes:
        row_prefill = []
        row_decode = []
        for seq_len in seq_lens:
            prefill_throughput, decode_throughput = run_sample(
                seq_len=seq_len,
                batch_size=batch_size,
                envs=envs,
            )
            row_prefill.append(prefill_throughput)
            row_decode.append(decode_throughput)
            print(
                f"\033[92m>>> [exp={test_exp_name}, batch_size={batch_size}, seq_len={seq_len}] "
                f"prefill={(prefill_throughput if prefill_throughput is not None else 0.0):.2f}, "
                f"decode={(decode_throughput if decode_throughput is not None else 0.0):.2f}\033[0m"
            )
        data_prefill.append(row_prefill)
        data_decode.append(row_decode)
    results[test_exp_name] = {
        "prefill": data_prefill,
        "decode": data_decode,
    }

os.makedirs("saves/bench_latency_sglang", exist_ok=True)

json_path = "saves/bench_latency_sglang/result.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

csv_path = "saves/bench_latency_sglang/result.csv"
lines = []
for exp_name in test_envs.keys():
    lines.append(
        ",".join(
            [exp_name, "prefill", f"{chunked_prefill_size}"]
            + list(map(lambda x: str(x), results[exp_name]["prefill"][0]))
        ).replace("None", "")
    )
    for i, bsz in enumerate(batch_sizes):
        lines.append(
            ",".join(
                [exp_name, "decode", f"{bsz}"]
                + list(map(lambda x: str(x), results[exp_name]["decode"][i]))
            ).replace("None", "")
        )

csv = "\n".join(lines) + "\n"
with open(csv_path, "w") as f:
    f.write(csv)
print("=" * 80)
print(csv, end="")
