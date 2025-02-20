import itertools
import json
import math
import os
import subprocess
import sys

import seaborn as sns
import tqdm

sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import pypareto

os.environ["PYTHONPATH"] = "./"

block_size_qs = [8, 16, 32]
block_size_ks = [1, 2, 4]
ks = [256, 512, 1024]


def samples():
    results = {}
    for block_size_q, block_size_k, k in tqdm.tqdm(
        list(itertools.product(block_size_qs, block_size_ks, ks)),
        desc="exam",
        dynamic_ncols=True,
    ):
        print(f"ppl measure bq{block_size_q}, bk{block_size_k}, k{k}")
        cache_path = (
            f"./cache/llama_eval/dev/ppl_hip_llama13b_s4096_dl4_k{k}_ckptFalse.json"
        )
        subprocess.call(
            [
                "python",
                "hip/main/model_eval.py",
                "--model",
                "llama13b",
                "--method",
                "hip",
                "--stride",
                "4096",
                "--block_size_q",
                str(block_size_q),
                "--block_size_k",
                str(block_size_k),
                "--dense_layers",
                str(4),
                "--k",
                str(k),
            ]
        )
        with open(cache_path, "r") as f:
            ppl = json.load(f)["ppl"]
        os.remove(cache_path)
        print(f"ppl measured {ppl} (bq{block_size_q}, bk{block_size_k}, k{k})")
        results[f"bq{block_size_q}_bk{block_size_k}_k{k}"] = {
            "block_size_q": block_size_q,
            "block_size_k": block_size_k,
            "k": k,
            "num_blocks": math.ceil(k / block_size_k),
            "ppl": ppl,
        }

    os.makedirs("./saves/ppl_report", exist_ok=True)
    with open("./saves/ppl_report/report.json", "w") as f:
        json.dump(results, f, indent=2)


def by_value(a, b):
    if isinstance(a, (tuple, list)):
        return pypareto.Domination.EQUAL

    if a > b:
        return pypareto.Domination.GREATER
    elif a < b:
        return pypareto.Domination.LESS
    else:
        return pypareto.Domination.EQUAL


def plots():
    # llama32k
    # baseline_ppl = 5.59
    baseline_ppl = 4.682

    with open("./saves/ppl_report/report.json", "r") as f:
        data = json.load(f)

    entries = list(data.values())
    xs = []  # num blocks
    ys = []  # ppl

    for entry in entries:
        xs.append(entry["num_blocks"])
        ys.append(entry["ppl"])

    pts = list(zip(xs, ys, map(lambda x: (x,), range(len(xs)))))
    chain = pypareto.Comparison(
        by_value,
        pypareto.MaxMinList(
            pypareto.MaxMin.MIN, pypareto.MaxMin.MIN, pypareto.MaxMin.MIN
        ),
    ).as_chain()
    pts = chain.split_by_pareto(pts)[0]
    xs_front = [pt[0] for pt in pts]
    ys_front = [pt[1] for pt in pts]
    idxs_front = [pt[2][0] for pt in pts]

    plt.figure(figsize=(5, 4))

    sns.lineplot(x=xs_front, y=ys_front)
    for idx in range(len(idxs_front)):
        plt.annotate(
            f'k:{entries[idxs_front[idx]]["k"]}, bq:{entries[idxs_front[idx]]["block_size_q"]}, bk:{entries[idxs_front[idx]]["block_size_k"]}',
            (xs_front[idx], ys_front[idx]),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=9,
        )

    plt.axhline(baseline_ppl, color="#555", linestyle="--", linewidth=1)
    sns.scatterplot(x=xs, y=ys)

    plt.title("Perplexity / Num. Blocks")
    plt.xlabel("Num. Blocks")
    plt.ylabel("PPL. (w/o train)")
    # plt.yscale('log', base=2)

    plt.savefig("./saves/ppl_report/plot_ppl_report.png", dpi=200, bbox_inches="tight")
    plt.savefig("./saves/ppl_report/plot_ppl_report.pdf", dpi=200, bbox_inches="tight")
    print("saved", "./saves/ppl_report/plot_ppl_report.png")


def main():
    samples()
    plots()


if __name__ == "__main__":
    main()
