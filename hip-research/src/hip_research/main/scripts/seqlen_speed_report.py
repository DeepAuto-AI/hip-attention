import json
import os
import subprocess

import seaborn as sns
from hip_research.utils import setup_seaborn
from matplotlib import pyplot as plt

setup_seaborn(legend_fontsize=6)

dups = range(1, 17)


def samples(query_size=1, step_size=1):
    block_size = 32
    block_size_ks = [2, 4]
    k = 1024
    batch_sizes = {
        1: 256,
        3: 192,
        4: 128,
        5: 96,
        6: 80,
        7: 80,
        8: 64,
        9: 64,
        10: 64,
        11: 64,
        12: 48,
        13: 48,
        14: 40,
        16: 32,
        24: 16,
        32: 8,
        48: 4,
    }
    num_samples = 200
    cache_path = "./cache/attention1_block_gpu/result.json"
    refresh_interval = 8

    batch_size = max(1, max(list(batch_sizes.values())) // query_size)
    results = {}
    for dup in dups:
        dup *= step_size
        if dup in batch_sizes:
            batch_size = max(1, batch_sizes[dup] // query_size)

        latency_hips = []
        for block_size_k in block_size_ks:
            cmd = [
                "python",
                "hip/models/hip_attention/attention1_block_gpu.py",
                "--method",
                "hip",
                "--block_size_q",
                str(block_size),
                "--block_size_k",
                str(block_size_k),
                "--k",
                str(k),
                "--query_size",
                str(query_size),
                "--dups",
                str(dup),
                "--batch_size",
                str(batch_size),
                "--samples",
                str(num_samples),
                "--refresh_interval",
                str(refresh_interval),
            ]
            print(" ".join(cmd))
            subprocess.call(cmd)
            with open(cache_path, "r") as f:
                latency_hip = json.load(f)["mean"]
            os.remove(cache_path)
            latency_hips.append(latency_hip)

        subprocess.call(
            [
                "python",
                "hip/models/hip_attention/attention1_block_gpu.py",
                "--method",
                "none",
                "--block_size_q",
                str(block_size),
                "--block_size_k",
                str(block_size),
                "--k",
                str(k),
                "--query_size",
                str(query_size),
                "--dups",
                str(dup),
                "--batch_size",
                str(batch_size),
                "--samples",
                str(num_samples),
            ]
        )
        with open(cache_path, "r") as f:
            latency_base = json.load(f)["mean"]
        os.remove(cache_path)

        subprocess.call(
            [
                "python",
                "hip/models/hip_attention/attention1_block_gpu.py",
                "--method",
                "flash",
                "--block_size_q",
                str(block_size),
                "--block_size_k",
                str(block_size),
                "--k",
                str(k),
                "--query_size",
                str(query_size),
                "--dups",
                str(dup),
                "--batch_size",
                str(batch_size),
                "--samples",
                str(num_samples),
            ]
        )
        with open(cache_path, "r") as f:
            latency_flash = json.load(f)["mean"]
        os.remove(cache_path)

        seq_len = dup * 1024
        results[f"s{seq_len}"] = {
            "block_size_ks": block_size_ks,
            "latency_hips": latency_hips,
            "latency_flash": latency_flash,
            "latency_base": latency_base,
            "batch_size": batch_size,
            "query_size": query_size,
            "seq_len": seq_len,
            "dups": dup,
            "k": k,
            "speedup": latency_base / latency_hip,
            "speedup_flash": latency_base / latency_flash,
        }

    os.makedirs("./saves/seqlen_speed_report", exist_ok=True)
    path = f"./saves/seqlen_speed_report/result_q{query_size}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("dumped", path)


LINEWIDTH = 1.5


def plot(query_size=1, step_size=1):
    path = f"./saves/seqlen_speed_report/result_q{query_size}.json"
    with open(path, "r") as f:
        data = json.load(f)

    block_size_ks = data[list(data.keys())[0]]["block_size_ks"]

    xss = []
    ys_hips = []
    ys_speedups = []

    for iks, block_size_k in enumerate(block_size_ks):
        xs = []
        ys_flash = []
        ys_base = []
        ys_hip = []
        ys_speedup = []
        ys_speedup_flash = []
        for dup in dups:
            dup = dup * step_size
            entry = data[f"s{dup * 1024}"]
            xs.append(entry["seq_len"] / 1024)
            ys_base.append(entry["latency_base"] / entry["batch_size"] * 1000)
            ys_flash.append(entry["latency_flash"] / entry["batch_size"] * 1000)
            ys_hip.append(entry["latency_hips"][iks] / entry["batch_size"] * 1000)
            ys_speedup.append(entry["latency_base"] / entry["latency_hips"][iks])
            ys_speedup_flash.append(entry["latency_base"] / entry["latency_flash"])
        xss.append(xs)
        ys_hips.append(ys_hip)
        ys_speedups.append(ys_speedup)

    figsize = (2.5, 2.0)

    plt.figure(figsize=figsize)

    sns.lineplot(x=xs, y=ys_base, label="Torch", linewidth=LINEWIDTH)
    sns.lineplot(x=xs, y=ys_flash, label="FlashAttenion2", linewidth=LINEWIDTH)
    for iks, block_size_k in enumerate(block_size_ks):
        sns.lineplot(
            x=xs,
            y=ys_hips[iks],
            label=f"HiP ($b_k$={block_size_k})",
            linewidth=LINEWIDTH,
        )
    plt.legend()
    if query_size == 1:
        plt.title("Decoding Latency ($k$=1024, $b_q$=32)")
    else:
        plt.title("Prompt Latency ($k$=1024, $b_q$=32)")
    plt.xlabel("Seq. Length (k) ↑")
    plt.ylabel("Latency (us) ↓")
    plt.xlim(0, 17 * step_size)

    fig_path = f"./saves/seqlen_speed_report/plot_seqlen_latency_q{query_size}"
    plt.savefig(fig_path + ".png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.savefig(fig_path + ".pdf", dpi=200, bbox_inches="tight", pad_inches=0)
    print(f"saved {fig_path}.png")

    plt.figure(figsize=figsize)

    if query_size == 1:
        plt.title("Decoding Speedup ($k$=1024, $b_q$=32)")
    else:
        plt.title("Prompt Speedup ($k$=1024, $b_q$=32)")
    sns.lineplot(
        x=xs,
        y=[
            1.0,
        ]
        * len(xs),
        label="Torch",
        linewidth=LINEWIDTH,
    )
    sns.lineplot(x=xs, y=ys_speedup_flash, label="FlashAttention2", linewidth=LINEWIDTH)
    for iks, block_size_k in enumerate(block_size_ks):
        sns.lineplot(
            x=xs,
            y=ys_speedups[iks],
            label=f"HiP ($b_k$={block_size_k})",
            linewidth=LINEWIDTH,
        )
    plt.xlabel("Seq. Length (k) ↑")
    plt.ylabel("Speedup ↑")
    plt.xlim(0, 17 * step_size)
    plt.legend()

    fig_path = f"./saves/seqlen_speed_report/plot_seqlen_speedup_q{query_size}"
    plt.savefig(fig_path + ".png", dpi=200, bbox_inches="tight", pad_inches=0)
    plt.savefig(fig_path + ".pdf", dpi=200, bbox_inches="tight", pad_inches=0)
    print(f"saved {fig_path}.png")


def main():
    samples(query_size=1, step_size=4)
    plot(query_size=1, step_size=4)

    samples(query_size=1024, step_size=4)
    plot(query_size=1024, step_size=4)


if __name__ == "__main__":
    main()
