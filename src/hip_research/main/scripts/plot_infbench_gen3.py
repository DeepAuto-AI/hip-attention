import os

import matplotlib.pyplot as plt
import numpy as np


def render(
    data, name, ylabel, fn=None, ylims=(None, None), show_legend=True, fa2_limit=None
):
    from matplotlib import font_manager

    font_path = "NotoSans-Medium.ttf"  # Your font path goes here
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = prop.get_name()

    import seaborn as sns

    label_fontsize = 10
    legend_fontsize = 9
    axes_label_fontsize = 9
    font_weight = 500
    axes_label_weight = 600
    axis_below = True
    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette=[
            "#ff8370",
            "#00b1b0",
            "#fec84d",
            "#e42256",
            "#34586e",
            "#45BDC6",
            "#7AAAF7",
            "#CDCDFF",
        ],
        font="Noto Sans",
        font_scale=1.0,
        color_codes=True,
        rc={
            "axes.titlesize": str(label_fontsize),
            "font.weight": font_weight,
            "axes.labelweight": axes_label_weight,
            "axes.titleweight": "600",
            "legend.fontsize": str(legend_fontsize),
            "axes.grid.which": "both",
            "ytick.labelsize": str(axes_label_fontsize),
            "xtick.labelsize": str(axes_label_fontsize),
            "axes.labelsize": str(label_fontsize),
            "ytick.major.pad": "1.0",
            "xtick.major.pad": "1.0",
            "axes.axisbelow": axis_below,
        },
    )

    plt.figure(figsize=(3, 1.65))

    for legend in data:
        if legend == "T":
            continue
        xs = data["T"]
        ys = (np.array(data[legend]) * 100).tolist()
        plt.plot(xs, ys, label=legend, linewidth=2, marker=".", linestyle=":")
    if fa2_limit is not None:
        plt.axhline(fa2_limit * 100, color="red")

    plt.grid(True)
    plt.xlabel("Context Length (k)")
    plt.ylabel(f"{ylabel}")
    if show_legend:
        plt.legend()
    plt.ylim(*ylims)

    if fn is not None:
        fn(data)

    os.makedirs("./saves/plot_infbench_gen3", exist_ok=True)
    plt.savefig(
        f"./saves/plot_infbench_gen3/{name}.pdf", bbox_inches="tight", pad_inches=0.025
    )
    plt.savefig(
        f"./saves/plot_infbench_gen3/{name}.png", bbox_inches="tight", pad_inches=0.025
    )
    print(f"./saves/plot_infbench_gen3/{name}.png")


data_hip = {
    "T": [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384],
    "HiP": [
        0.520,
        0.528,
        0.528,
        0.507,
        0.476,
        0.393,
        0.402,
        0.358,
        0.354,
        0.367,
        0.349,
        0.354,
    ],
    "FA2": [
        0.515,
        0.546,
        0.629,
        0.629,
        0.563,
        0.590,
        0.568,
        0.550,
        0.520,
        0.515,
        0.550,
        float("nan"),
    ],
    "Ours": [
        0.568,
        0.611,
        0.633,
        0.611,
        0.611,
        0.638,
        0.646,
        0.638,
        0.629,
        0.633,
        0.633,
        0.638,
    ],
}

data_mc_gemma = {
    "T": [4, 8, 16, 32, 64, 128, 192, 256],
    "Gemma2": [0.3755, 0.4585, 0.5546, 0.5983, 0.6157, 0.7162, 0.7380, 0.7249],
    "Exaone3": [0.3057, 0.3100, 0.3275, 0.3843, 0.3712, 0.3843, 0.3886, 0.3930],
    "Exaone3.5": [0.3930, 0.4541, 0.4891, 0.5066, 0.5633, 0.6026, 0.5939, 0.5983],
}

data_qa_gemma = {
    "T": [4, 8, 16, 32, 64, 128, 192, 256],
    "Gemma2": [0.1699, 0.2300, 0.2742, 0.3651, 0.4299, 0.4623, 0.4623, 0.4470],
    "Exaone3": [0.2312, 0.2757, 0.3003, 0.3077, 0.3485, 0.3283, 0.3189, 0.3341],
    "Exaone3.5": [0.1998, 0.2461, 0.3002, 0.3538, 0.4197, 0.4616, 0.4728, 0.4739],
}


def data_hip_post(data):
    plt.legend(loc="lower right")
    plt.xlim(None, 384)
    for legend in data:
        if legend == "T":
            continue
        xs = data["T"][4:]
        ys = (np.array(data[legend]) * 100).tolist()[4:]
        t = plt.scatter(
            xs, ys, s=200, marker="*", edgecolor="#333", linewidth=0, zorder=0
        )


def data_mc_gemma_post(data):
    plt.xlabel(None)
    for legend in data:
        if legend == "T":
            continue
        skip = 0
        if legend == "Gemma2":
            skip = 2
        elif legend == "Exaone3.5":
            skip = 4
        elif legend == "Exaone3":
            skip = 1
        xs = data["T"][skip:]
        ys = (np.array(data[legend]) * 100).tolist()[skip:]
        t = plt.scatter(
            xs, ys, s=200, marker="*", edgecolor="#333", linewidth=0, zorder=0
        )


def data_qa_gemma_post(data):
    for legend in data:
        if legend == "T":
            continue
        skip = 0
        if legend == "Gemma2":
            skip = 2
        elif legend == "Exaone3.5":
            skip = 4
        elif legend == "Exaone3":
            skip = 1
        xs = data["T"][skip:]
        ys = (np.array(data[legend]) * 100).tolist()[skip:]
        t = plt.scatter(
            xs, ys, s=200, marker="*", edgecolor="#333", linewidth=0, zorder=0
        )


render(
    data_hip, "plot_hip_fa2_gen3_enmc", "En.MC Acc. (%)", data_hip_post, ylims=(30, 70)
)
render(
    data_mc_gemma,
    "plot_gemma_mc",
    "En.MC Acc. (%)",
    data_mc_gemma_post,
    show_legend=False,
)
render(data_qa_gemma, "plot_gemma_qa", "En.QA Recall. (%)", data_qa_gemma_post)
