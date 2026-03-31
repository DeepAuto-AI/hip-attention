import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hip_research.utils import setup_seaborn
from matplotlib.patches import Patch

setup_seaborn(axis_below=True)


def proc_copy_paste(t: str, scale: float = 1):
    return np.array(list(map(lambda x: float(x) * scale, t.split())))


Ts = np.array([8, 16, 32, 64, 128])

LINEAR = "Linear"
FLASH_ATTN = "Flash Attention (Kernel)"
MASKING = "HiP Masking"
SA = "Sparse Attention"
SA_AFTER_MASK = "Sparse Attention (w/ Masking)"

latency_data = {
    "index": Ts,
    LINEAR: proc_copy_paste("21.361	21.361	21.361	21.361	21.361"),
    FLASH_ATTN: proc_copy_paste("35.952	71.294	141.753	282.196	563.900"),
    MASKING: proc_copy_paste("24.590	26.830	35.520	39.760	48.900"),
    SA: proc_copy_paste("11.740	13.280	13.730	13.900	15.370"),
}

CLASS_FLASH_ATTN = "Flash Attention"
CLASS_HIP_CACHED = "HiP Cached"
CLASS_HIP_MASKING = "HiP Mask Refresh"

plot_data = {"seq_len": [], "latency": [], "class": [], "method": []}


def add_data(latency, class_name, method_name):
    plot_data["seq_len"] += Ts.tolist()
    plot_data["latency"] += latency.tolist()
    plot_data["class"] += [class_name] * len(Ts)
    plot_data["method"] += [method_name] * len(Ts)


METHOD_LINEAR = "Linear"
METHOD_MASKING = "Masking"
METHOD_FA = "Flash Attention"
METHOD_SA = "Sparse Attention"

add_data(latency_data[LINEAR], CLASS_FLASH_ATTN, METHOD_LINEAR)
add_data(latency_data[LINEAR], CLASS_HIP_MASKING, METHOD_LINEAR)
add_data(latency_data[LINEAR], CLASS_HIP_CACHED, METHOD_LINEAR)
add_data(latency_data[FLASH_ATTN], CLASS_FLASH_ATTN, METHOD_FA)
add_data(latency_data[MASKING], CLASS_HIP_MASKING, METHOD_MASKING)
add_data(latency_data[SA], CLASS_HIP_CACHED, METHOD_SA)
add_data(latency_data[SA], CLASS_HIP_MASKING, METHOD_SA)

import pandas as pd

plot_data = pd.DataFrame(plot_data)


def stack_catplot(x, y, cat, stack, data, palette=sns.color_palette("Set2")):
    plt.figure(figsize=(4, 2))
    ax = plt.gca()
    # pivot the data based on categories and stacks
    df = data.pivot_table(
        values=y, index=[cat, x], columns=stack, dropna=False, aggfunc="sum"
    ).fillna(
        0
    )  # type: pd.DataFrame
    ncat = data[cat].nunique()
    nx = data[x].nunique()
    nstack = data[stack].nunique()
    range_x = np.arange(nx)
    width = 0.9 / ncat  # width of each bar

    for i, c in enumerate(data[cat].unique()):
        # iterate over categories, i.e., Conditions
        # calculate the location of each bar
        loc_x = (0.5 + i - ncat / 2) * width + range_x
        bottom = 0
        for j, s in enumerate(data[stack].unique()):
            # iterate over stacks, i.e., Hosts
            # obtain the height of each stack of a bar
            height = df.loc[c][s].values
            # plot the bar, you can customize the color yourself
            ax.bar(
                x=loc_x,
                height=height,
                bottom=bottom,
                width=width,
                color=palette[j % len(palette)],
                zorder=1,
            )
            # change the bottom attribute to achieve a stacked barplot
            bottom += height
    # make xlabel
    ax.set_xticks(range_x)
    ax.set_xticklabels(data[x].unique(), rotation=0)
    ax.set_ylabel("Latency ($\\mu$s)")
    ax.set_ylim(0, 120)
    ax.set_xlabel("$T$ (k)")

    for i, text in enumerate(
        ["Flash Attention", "HiP (Mask Refresh)", "HiP (Mask Cached)"]
    ):
        ax.text(
            len(Ts) - 1 - width * 1 + 2 * width * (i / 2) + 0.04,
            5,
            text,
            fontsize=7,
            fontweight=600,
            rotation=90,
            ha="center",
            va="bottom",
        )

    for i in range(3):
        ax.annotate(
            f"{latency_data[LINEAR][2+i]+latency_data[FLASH_ATTN][2+i]:.1f} μs",
            xy=(2 + i - width, 120),
            xytext=(2 + i - width, 112),
            fontsize=7,
            fontweight=600,
            va="center",
            ha="center",
            color="#ad4f1c",
            # arrowprops=dict(
            #     arrowstyle='<|-',
            #     edgecolor='#ad4f1c',
            #     facecolor='#ad4f1c'
            # )
        )

    plt.title("Latency Breakdown in Single Layer Decoding")
    # make legend
    plt.legend(
        [Patch(facecolor=palette[i % len(palette)]) for i in range(nstack)],
        [f"{s}" for s in data[stack].unique()],
    )
    plt.grid(True)


stack_catplot(x="seq_len", y="latency", cat="class", stack="method", data=plot_data)

# Show
working_directory = "./saves/plot_latency_breakdown"
os.makedirs(working_directory, exist_ok=True)
plt.savefig(
    os.path.join(working_directory, "plot_latency_breakdown.png"),
    bbox_inches="tight",
    pad_inches=0,
)
plt.savefig(
    os.path.join(working_directory, "plot_latency_breakdown.pdf"),
    bbox_inches="tight",
    pad_inches=0,
)
plt.savefig(
    os.path.join(working_directory, "plot_latency_breakdown.svg"),
    bbox_inches="tight",
    pad_inches=0,
)
