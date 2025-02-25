import os

import matplotlib
from matplotlib import font_manager
from matplotlib import pyplot as plt

font_path = "NotoSans-Medium.ttf"  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = prop.get_name()

import seaborn as sns

COLORS = [
    "#ff8370",
    "#00b1b0",
    "#fec84d",
    "#e42256",
    "#34586e",
    "#45BDC6",
    "#7AAAF7",
    "#CDCDFF",
]

label_fontsize = 10
legend_fontsize = 9
axes_label_fontsize = 9
font_weight = 500
axes_label_weight = 600
axis_below = True
sns.set_theme(
    context="paper",
    style="whitegrid",
    palette=COLORS,
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

CMAP = {
    "SRT": COLORS[1],
    "SRT (Estimated)": COLORS[1],
    "Ours": COLORS[0],
    "Ours (Estimated)": COLORS[0],
    "Ours Offload Fast": COLORS[2],
    "Ours Offload Flash": COLORS[3],
}


def render(data, name, show_xlabel=True):
    plt.figure(figsize=(2.5, 1.3))
    lines = []
    for legend in data:
        if legend == "T":
            continue
        xs = data["T"]
        ys = data[legend]
        is_est = "Est" in legend
        (line,) = plt.plot(
            xs,
            ys,
            label=legend,
            linestyle=":" if is_est else "-",
            color=CMAP[legend],
            linewidth=2,
            marker=None if is_est else "o",
        )
        if not is_est:
            lines.append(line)
    plt.grid(True)
    # plt.legend()
    # plt.title(name)
    plt.ylabel(f"{name} (tok/s)")
    plt.ylim(0, None)
    if show_xlabel:
        plt.xlabel("Context Length (k)")

    os.makedirs("saves/plot_sglang_decoding_gen3", exist_ok=True)
    filename = f"saves/plot_sglang_decoding_gen3/plot_sglang_decoding_{name}"
    plt.savefig(filename + ".png", bbox_inches="tight", pad_inches=0.025)
    plt.savefig(filename + ".pdf", bbox_inches="tight", pad_inches=0.025)
    print(filename + ".png")
    return lines


data_4090 = {
    "T": [64, 96, 128, 192, 256, 384, 512, 768, 1024],
    "SRT": [
        88.8,
        74.3,
        63.2,
        49.4,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "SRT (Estimated)": [88.8, 73.8, 63.2, 49.0, 40.1, 29.3, 23.1, 16.3, 12.5],
    "Ours": [
        113.3,
        112.5,
        112.1,
        110.6,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "Ours (Estimated)": [113.3, 112.5, 112.1, 110.6, 109.6, 107.3, 105.0, 100.8, 97.0],
    "Ours Offload Fast": [64.5, 59.6, 55.9, 51.1, 46.7, 39.9, 31.8, 21.6, 17.3],
    "Ours Offload Flash": [66.0, 62.7, 60.3, 58.2, 56.6, 53.5, 49.5, 44.0, 40.1],
}

data_l40s = {
    "T": [64, 128, 256, 512, 1024, 2048, 3072],
    "SRT": [
        69.6,
        48.6,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "SRT (Estimated)": [69.6, 48.6, 30.4, 17.3, 9.3, 4.9, 3.3],
    "Ours": [
        98.7,
        97.6,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    ],
    "Ours (Estimated)": [98.7, 97.6, 95.7, 92.0, 85.4, 74.7, 66.4],
    "Ours Offload Fast": [55.3, 43.5, 37.6, 34.1, 24.2, 10.5, 7.6],
    "Ours Offload Flash": [56.6, 52.0, 49.4, 43.7, 35.2, 28.0, 23.8],
}

render(data_4090, "RTX4090", show_xlabel=True)
lines = render(data_l40s, "L40s")

figlegend = plt.figure(figsize=(3, 2))
figlegend.legend(handles=lines, loc="center", ncols=4)
filename = f"saves/plot_sglang_decoding_gen3/plot_sglang_decoding_legend"
figlegend.savefig(filename + ".png", bbox_inches="tight", pad_inches=0.025)
figlegend.savefig(filename + ".pdf", bbox_inches="tight", pad_inches=0.025)
print(filename + ".png")
