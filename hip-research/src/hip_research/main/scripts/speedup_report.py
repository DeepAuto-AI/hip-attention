import itertools
import json
import os
import subprocess

import matplotlib.pyplot as plt
import pypareto
import seaborn as sns
import tqdm
from hip_research.utils import setup_seaborn

setup_seaborn()

os.environ["PYTHONPATH"] = "./"

# query_sizes = [1, 2, 4, 8, 16]
query_sizes = [
    1,
]
block_size_qs = [8, 16, 32]
block_size_ks = [1, 2, 4]
ks = [256, 512, 1024]


def samples():
    num_samples = 200
    batch_size = 256
    results = {}
    cache_path = "./cache/attention1_block_gpu/result.json"
    refresh_interval = 8

    for query_size in tqdm.tqdm(query_sizes, dynamic_ncols=True, desc="none"):
        subprocess.call(
            [
                "python",
                "hip/models/hip_attention/attention1_block_gpu.py",
                "--method",
                "none",
                "--query_size",
                str(query_size),
                "--dups",
                "4",
                "--batch_size",
                str(batch_size),
                "--samples",
                str(num_samples),
            ]
        )
        with open(cache_path, "r") as f:
            latency = json.load(f)["mean"]
        os.remove(cache_path)
        results[f"none_q{query_size}"] = {
            "query_size": query_size,
            "latency": latency,
            "method": "none",
        }

    for query_size, block_size_q, block_size_k, k in tqdm.tqdm(
        list(itertools.product(query_sizes, block_size_qs, block_size_ks, ks)),
        dynamic_ncols=True,
        desc="eval",
    ):
        subprocess.call(
            [
                "python",
                "hip/models/hip_attention/attention1_block_gpu.py",
                "--method",
                "hip",
                "--block_size_q",
                str(block_size_q),
                "--block_size_k",
                str(block_size_k),
                "--k",
                str(k),
                "--query_size",
                str(query_size),
                "--dups",
                "4",
                "--batch_size",
                str(batch_size),
                "--samples",
                str(num_samples),
                "--refresh_interval",
                str(refresh_interval),
            ]
        )
        with open(cache_path, "r") as f:
            latency = json.load(f)["mean"]
        os.remove(cache_path)
        results[f"hip_q{query_size}_bq{block_size_q}_bk{block_size_k}_k{k}"] = {
            "query_size": query_size,
            "k": k,
            "block_size_q": block_size_q,
            "block_size_k": block_size_k,
            "latency": latency,
            "method": "hip",
        }

    os.makedirs("./saves/speedup_report", exist_ok=True)
    path = "./saves/speedup_report/result.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", path)


def plot():
    path = "./saves/speedup_report/result.json"
    with open(path, "r") as f:
        data = json.load(f)

    plt.figure(figsize=(5, 4))

    for query_size in query_sizes:
        xs = []
        ys = []
        for block_size_q, block_size_k, k in itertools.product(
            block_size_qs, block_size_ks, ks
        ):
            hip_entry = data[
                f"hip_q{query_size}_bq{block_size_q}_bk{block_size_k}_k{k}"
            ]
            base_entry = data[f"none_q{query_size}"]
            num_blocks = k / block_size_k
            speedup = base_entry["latency"] / hip_entry["latency"]
            xs.append(num_blocks)
            ys.append(speedup)
        sns.scatterplot(x=xs, y=ys, label=f"Query Length: {query_size}")

    plt.title("Decode Speedup (4k) / Num. Blocks")
    plt.xlabel("Num. Blocks")
    plt.ylabel("Speedup")
    plt.axhline(1.0, color="#555", linestyle="--", linewidth=1)
    # plt.yscale('log', base=2)
    # plt.xscale('log', base=2)

    plt.savefig(
        "./saves/speedup_report/plot_speedup_report.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.savefig(
        "./saves/speedup_report/plot_speedup_report.pdf",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )
    print("saved", "./saves/speedup_report/plot_speedup_report.png")


def by_value(a, b):
    if isinstance(a, (tuple, list)):
        return pypareto.Domination.EQUAL

    if a > b:
        return pypareto.Domination.GREATER
    elif a < b:
        return pypareto.Domination.LESS
    else:
        return pypareto.Domination.EQUAL


LINEWIDTH = 1.5


def plot_ppl(query_size=1):
    path = "./saves/speedup_report/result.json"
    with open(path, "r") as f:
        data_latency = json.load(f)

    path = "./saves/ppl_report/report.json"
    with open(path, "r") as f:
        data_ppl = json.load(f)

    xs = []
    ys = []
    entries = []
    for entry_ppl in data_ppl.values():
        ys.append(entry_ppl["ppl"])
        k = entry_ppl["k"]
        block_size_q = entry_ppl["block_size_q"]
        block_size_k = entry_ppl["block_size_k"]
        latency_hip = data_latency[
            f"hip_q{query_size}_bq{block_size_q}_bk{block_size_k}_k{k}"
        ]["latency"]
        latency_base = data_latency[f"none_q{query_size}"]["latency"]
        entries.append(
            {
                "k": k,
                "block_size_q": block_size_q,
                "block_size_k": block_size_k,
            }
        )
        xs.append(latency_base / latency_hip)

    data = list(sorted(zip(xs, ys, entries), key=lambda x: x[0]))
    xs = [d[0] for d in data]
    ys = [d[1] for d in data]
    entries = [d[2] for d in data]

    pts = list(zip(xs, ys, map(lambda x: (x,), range(len(data_ppl)))))
    chain = pypareto.Comparison(
        by_value,
        pypareto.MaxMinList(
            pypareto.MaxMin.MAX,
            pypareto.MaxMin.MIN,
            pypareto.MaxMin.MIN,
        ),
    ).as_chain()
    pts = chain.split_by_pareto(pts)[0]
    xs_front = [pt[0] for pt in pts]
    ys_front = [pt[1] for pt in pts]
    idxs_front = [pt[2][0] for pt in pts]

    plt.figure(figsize=(2.5, 2.0))

    if query_size == 1:
        plt.title(f"PPL. / Decoding Speedup (4k)")
    else:
        plt.title(f"PPL. / Decoding Speedup (4k, #Q:{query_size})")
    plt.ylabel("PPL. (w/o train) ↓")
    plt.xlabel("Decoding Speedup ↑")
    sns.scatterplot(x=xs, y=ys)
    sns.lineplot(x=xs_front, y=ys_front, linewidth=LINEWIDTH)
    last_x = 0
    for idx in range(len(idxs_front)):
        if abs(xs_front[idx] - last_x) > 0.8:
            x = plt.annotate(
                f'$k$:{entries[idxs_front[idx]]["k"]}\n$b_q$:{entries[idxs_front[idx]]["block_size_q"]}, $b_k$:{entries[idxs_front[idx]]["block_size_k"]}',
                (xs_front[idx] - 0.5, ys_front[idx] + 0.00),
                horizontalalignment="center",
                verticalalignment="bottom",
                fontsize=6.0,
                color="0.1",
            )
            x.set_alpha(0.4)
            last_x = xs_front[idx]

    # baseline_ppl = 5.59
    baseline_ppl = 4.682
    plt.axhline(
        baseline_ppl, color="#555", linestyle="--", linewidth=LINEWIDTH, zorder=1000
    )
    plt.axvline(1.0, color="#555", linestyle="--", linewidth=LINEWIDTH, zorder=1000)
    # plt.yscale('log', base=2)
    # plt.xscale('log', base=2)

    plt.savefig(
        f"./saves/speedup_report/plot_speedup_report_ppl_q{query_size}.png",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.savefig(
        f"./saves/speedup_report/plot_speedup_report_ppl_q{query_size}.pdf",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0,
    )
    print("saved", f"./saves/speedup_report/plot_speedup_report_ppl_q{query_size}.png")


def main():
    samples()
    plot()
    for q in query_sizes:
        plot_ppl(q)


if __name__ == "__main__":
    main()
