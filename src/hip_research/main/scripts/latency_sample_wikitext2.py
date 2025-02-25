import json
import os
import subprocess
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class Config:
    method: Literal["none", "flash", "hip", "streaming"]
    hidden_size: int
    head_size: int
    seq_len: int
    k: int = 512


CONFIGS = [
    Config("none", hidden_size=128, head_size=32, seq_len=4 * 1024),
    Config("none", hidden_size=128, head_size=32, seq_len=32 * 1024),
    Config("flash", hidden_size=128, head_size=32, seq_len=4 * 1024),
    Config("flash", hidden_size=128, head_size=32, seq_len=32 * 1024),
    Config("streaming", hidden_size=128, head_size=32, seq_len=4 * 1024, k=512),
    Config("streaming", hidden_size=128, head_size=32, seq_len=32 * 1024, k=512),
    Config("streaming", hidden_size=128, head_size=32, seq_len=4 * 1024, k=1024),
    Config("streaming", hidden_size=128, head_size=32, seq_len=32 * 1024, k=1024),
    Config("hip", hidden_size=128, head_size=32, seq_len=4 * 1024, k=512),
    Config("hip", hidden_size=128, head_size=32, seq_len=32 * 1024, k=512),
    Config("hip", hidden_size=128, head_size=32, seq_len=4 * 1024, k=1024),
    Config("hip", hidden_size=128, head_size=32, seq_len=32 * 1024, k=1024),
    Config("none", hidden_size=128, head_size=40, seq_len=4 * 1024),
    Config("none", hidden_size=128, head_size=40, seq_len=32 * 1024),
    Config("flash", hidden_size=128, head_size=40, seq_len=4 * 1024),
    Config("flash", hidden_size=128, head_size=40, seq_len=32 * 1024),
    Config("streaming", hidden_size=128, head_size=40, seq_len=4 * 1024, k=512),
    Config("streaming", hidden_size=128, head_size=40, seq_len=32 * 1024, k=512),
    Config("streaming", hidden_size=128, head_size=40, seq_len=4 * 1024, k=1024),
    Config("streaming", hidden_size=128, head_size=40, seq_len=32 * 1024, k=1024),
    Config("hip", hidden_size=128, head_size=40, seq_len=4 * 1024, k=512),
    Config("hip", hidden_size=128, head_size=40, seq_len=32 * 1024, k=512),
    Config("hip", hidden_size=128, head_size=40, seq_len=4 * 1024, k=1024),
    Config("hip", hidden_size=128, head_size=40, seq_len=32 * 1024, k=1024),
]


def sample():
    cache_path = "./cache/attention1_block_gpu/result.json"
    query_size = 1
    num_samples = 1000
    block_size_q = 32
    block_size_k = 2
    refresh_interval = 8

    results = []

    for config in CONFIGS:
        batch_size = {4 * 1024: 64, 32 * 1024: 32}[config.seq_len]
        if config.method in ["flash", "none"]:
            cmd = [
                "python",
                "hip/models/hip_attention/attention1_block_gpu.py",
                "--method",
                config.method,
                "--query_size",
                str(query_size),
                "--dups",
                str(config.seq_len // 1024),
                "--batch_size",
                str(batch_size),
                "--samples",
                str(num_samples),
                "--head_size",
                str(config.head_size),
            ]
        elif config.method == "hip":
            cmd = [
                "python",
                "hip/models/hip_attention/attention1_block_gpu.py",
                "--method",
                "hip",
                "--block_size_q",
                str(block_size_q),
                "--block_size_k",
                str(block_size_k),
                "--k",
                str(config.k),
                "--query_size",
                str(query_size),
                "--dups",
                str(config.seq_len // 1024),
                "--batch_size",
                str(batch_size),
                "--samples",
                str(num_samples),
                "--head_size",
                str(config.head_size),
                "--refresh_interval",
                str(refresh_interval),
            ]
        elif config.method == "streaming":
            cmd = [
                "python",
                "hip/models/hip_attention/attention1_block_gpu.py",
                "--method",
                "streaming",
                "--k",
                str(config.k),
                "--query_size",
                str(query_size),
                "--dups",
                str(config.seq_len // 1024),
                "--batch_size",
                str(batch_size),
                "--samples",
                str(num_samples),
                "--head_size",
                str(config.head_size),
            ]
        else:
            raise Exception()
        print(" ".join(cmd))
        subprocess.call(cmd)

        with open(cache_path, "r") as f:
            latency = json.load(f)["mean"]
        os.remove(cache_path)
        results.append(
            {
                "config": asdict(config),
                "latency": latency,
                "latency_per_batch": f"{latency / batch_size * 1000:.2f}",
            }
        )

    root = "saves/latency_sample_wikitext2"
    os.makedirs(root, exist_ok=True)
    json_path = os.path.join(root, "result.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", json_path)


def plot():
    root = "saves/latency_sample_wikitext2"
    os.makedirs(root, exist_ok=True)
    json_path = os.path.join(root, "result.json")
    with open(json_path, "r") as f:
        results = json.load(f)

    def print_table(data):
        print("-" * 20)
        for i, line in enumerate(data):
            print(f'{line["latency_per_batch"]:>10}', end="")
            if (i % 2) == 0:
                speedup = f'{float(data[0]["latency_per_batch"]) / float(line["latency_per_batch"]):.2f}'
                print(f"{speedup:>10}", end="")
            elif (i % 2) == 1:
                speedup = f'{float(data[1]["latency_per_batch"]) / float(line["latency_per_batch"]):.2f}'
                print(f"{speedup:>10}", end="")
                print()
        print("-" * 20)

    print_table(results[: len(results) // 2])
    print_table(results[len(results) // 2 :])


if __name__ == "__main__":
    sample()
    plot()
