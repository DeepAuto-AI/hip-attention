import numpy as np
import torch
from hip_research.utils.load_checkouts import load_checkouts

# X: top-k occupancy
# Y: occurance

q, k, v, out, cos, sin = load_checkouts(
    idx=0,
    window=999,
    seq_len=131072,
    dtype=torch.bfloat16,
    return_cos_sin=True,
    derope=False,
)


def reshape(x: torch.Tensor):
    return x.unsqueeze(0).permute(0, 2, 1, 3)


q = reshape(q).to(0)
k = reshape(k).to(0)
v = reshape(v).to(0)

q = q[0, -1:, :, :].permute(1, 0, 2)
k = k[0].permute(1, 0, 2).repeat(4, 1, 1)

scores = torch.bmm(q, k.permute(0, 2, 1))[:, 0, :]


def compute(chunk_size=64, top_k=2048):
    scores_top_indices = scores.topk(k=2048, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.int64)
    mask.scatter_(dim=-1, index=scores_top_indices, value=1)
    stats = mask.view(32, -1, chunk_size).sum(-1).view(-1)

    ys = [
        0,
    ] * (chunk_size + 1)

    for item in stats.cpu().numpy():
        item = item.item()
        ys[item] += 1

    ys = np.array(ys)
    ys = ys.astype(np.float32) / ys.sum() * 100
    # ys = ys * 100
    xs = (np.arange(chunk_size + 1)).astype(np.float32) / chunk_size * 100

    return xs, ys, ys[0]  # / ys.sum() * 100


chunk_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

xss, yss = [], []
eps = []
for cs in chunk_sizes:
    xs, ys, empty_percent = compute(chunk_size=cs)
    print(f"{empty_percent:.2f} % chunks are empty.")
    eps.append(empty_percent)
    xss.append(xs)
    yss.append(ys)

import matplotlib.pyplot as plt
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
axis_below = False
sns.set_theme(
    context="paper",
    style="whitegrid",
    palette=[
        "#4286F4",
        "#EA4335",
        "#FBBC04",
        "#33A853",
        "#FF6D01",
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

scale = 0.4
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(scale * 4, scale * 4 * 1.68))
fig.subplots_adjust(hspace=0.05)  # adjust space between Axes

for i, (cs, xs, ys) in list(enumerate(list(zip(chunk_sizes, xss, yss))[3:])):
    lb = f"{cs}" if cs > 8 else f"$l_c$={cs}"
    counts, bins = np.histogram(xs, 8, weights=ys)
    ax2.bar(
        (bins[:-1] + bins[1:]) / 2,
        counts,
        width=bins[1:] - bins[:-1],
        label=lb,
        color=f"C{i}",
    )

for i, (cs, xs, ys) in list(enumerate(list(zip(chunk_sizes, xss, yss))[3:]))[::-1]:
    lb = f"{cs}" if cs > 8 else f"$l_c$={cs}"
    counts, bins = np.histogram(xs, 8, weights=ys)
    ax1.bar(
        (bins[:-1] + bins[1:]) / 2,
        counts,
        width=bins[1:] - bins[:-1],
        label=lb,
        color=f"C{i}",
    )
    counts, bins = counts[:1], bins[:2]
    ax2.bar(
        (bins[:-1] + bins[1:]) / 2, counts, width=bins[1:] - bins[:-1], color=f"C{i}"
    )

ax1.set_xlim(-1, 101)
ax1.set_ylim(90, 100)
ax1.set_ylabel("Occurence (%)" + " " * 30, labelpad=-2.0)

ax2.set_xlabel("Top-k Occupancy (%)  ")
ax2.set_yscale("log")
ax2.set_ylim(0.01, 20.0)
ax2.set_yticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1.0", "10.0"])
ax2.set_yticks([], [], minor=True)
ax2.legend(loc="upper right", bbox_to_anchor=(1, 2.06))

d = 0.5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-1, -d), (1, d)],
    markersize=12,
    linestyle="none",
    color="k",
    mec="k",
    mew=1,
    clip_on=False,
)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

plt.savefig("mot1.pdf", bbox_inches="tight", pad_inches=0.05)

plt.figure(figsize=(scale * 4, scale * 4 * 1.68))
plt.plot(chunk_sizes, eps, lw=3, marker="o")

plt.grid("--")
plt.xlabel("Chunk Size ($l_c$)")
plt.ylabel("Empty Chunks (%)", labelpad=-2.0)
plt.ylim(0, 100)

plt.savefig("mot2.pdf", bbox_inches="tight", pad_inches=0.05)
