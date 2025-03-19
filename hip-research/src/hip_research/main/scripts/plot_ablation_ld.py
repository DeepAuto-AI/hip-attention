import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hip_research.utils import setup_seaborn

setup_seaborn(axis_below=True)

data = [
    5.841,
    5.699,
    5.665,
    5.824,
    5.669,
    5.666,
    5.826,
    5.686,
    5.667,
    5.785,
    5.616,
    5.590,
    5.753,
    5.578,
    5.542,
    5.751,
    5.577,
    5.540,
    5.717,
    5.532,
    5.482,
    5.692,
    5.488,
    5.409,
    5.671,
    5.459,
    5.373,
    5.635,
    5.402,
    5.312,
    5.600,
    5.355,
    5.257,
    5.597,
    5.351,
    5.254,
]

data = np.array(data)
data = data.reshape([-1, 3])
data = data.T

xs = [
    0,
    1,
    2,
    3,
    4,
    5,
    10,
    15,
    20,
    25,
    30,
    32,
]

plt.figure(figsize=(3.2, 2.0))
for i, label in enumerate(["$T$=4k", "$T$=8k", "$T$=12k"]):
    sns.lineplot(x=xs, y=data[i], label=label, marker="o", zorder=10)

plt.annotate("$l_d=0$\nFull HiP", (0.5, np.min(data)), fontsize=7, zorder=20)
plt.annotate("$l_d=3$\nDefault", (3.5, np.min(data) + 0.1), fontsize=7, zorder=20)
plt.annotate(
    "$l_d=32$\nFull Dense", (31.5, np.min(data)), fontsize=7, ha="right", zorder=20
)

plt.axvline(0, linestyle="--", color="#888", zorder=1)
plt.axvline(3, linestyle="--", color="#89F", zorder=1)
plt.axvline(32, linestyle="--", color="#F88", zorder=1)

plt.xlabel("# of Dense Layers")
plt.ylabel("PPL.↓")
plt.title("Perplexity on Wikitext2 / Number of Dense Layers ($l_d$)")

root = "./saves/plot_ablation_ld/"
os.makedirs(root, exist_ok=True)
plt.savefig(
    os.path.join(root, "plot_ablation_ld.png"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.savefig(
    os.path.join(root, "plot_ablation_ld.pdf"), bbox_inches="tight", pad_inches=0.1
)
print("saved", os.path.join(root, "plot_ablation_ld.pdf"))
print("saved", os.path.join(root, "plot_ablation_ld.png"))
