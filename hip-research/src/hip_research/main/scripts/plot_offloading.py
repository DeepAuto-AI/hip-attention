import math
import os

import matplotlib.pyplot as plt
from hip_research.utils import setup_seaborn

setup_seaborn(
    label_fontsize=11,
    legend_fontsize=8,
    axes_label_fontsize=8,
    axis_below=True,
)

HIP_CUDA = "HiP w/o Offload"
HIP_UVM = "HiP UVM w/o Cache"
HIP_CACHE = "HiP UVM w/ Vector Map"
HIP_HASHMAP = "HiP UVM w/ Hash Map"
FA_CUDA = "Flash Attn w/o Offload"
FA_UVM = "Flash Attn UVM"


def proc_copy_paste(t: str, scale: float = 1):
    return list(map(lambda x: float(x.replace(",", "")) * scale, t.split()))


Ts = [8, 16, 32, 64]

LINEWIDTH = {
    FA_CUDA: 5,
    HIP_CUDA: 3,
}

DEAD_REASON = {
    FA_CUDA: "OOM",
    HIP_CUDA: "OOM",
}

gpu_memory_data = {
    FA_CUDA: proc_copy_paste("4001	3918	NaN	NaN"),
    HIP_CUDA: proc_copy_paste("4001	3,918.00	NaN	NaN"),
    FA_UVM: proc_copy_paste("-1000	-1000	-1000	-1000"),
    HIP_UVM: proc_copy_paste("-1000	-1000	-1000	-1000"),
    HIP_CACHE: proc_copy_paste("4510	4600	3598	2283"),
    HIP_HASHMAP: proc_copy_paste("4554	4554	3416	2104"),
}

cpu_memory_data = {
    HIP_CACHE: proc_copy_paste("16,004.00	31,344.00	48,780.00	41,425.50"),
    HIP_HASHMAP: proc_copy_paste("15,556.00	31,344.00	48,780.00	41,257.50"),
}

decode_throughput_data = {
    FA_CUDA: proc_copy_paste("183.4	92.0	NaN	NaN"),
    HIP_CUDA: proc_copy_paste("187.8	94.2	NaN	NaN"),
    FA_UVM: proc_copy_paste("13.4	6.8	3.3	1.9"),
    HIP_UVM: proc_copy_paste("27.3	26.3	24.8	22.9"),
    HIP_CACHE: proc_copy_paste("174.2	154.1	125.1	95.5"),
    HIP_HASHMAP: proc_copy_paste("32.5	25.0	20.5	10.2"),
}

MARKERS = {
    HIP_CUDA: ",",
    FA_CUDA: ",",
    HIP_UVM: "^",
    FA_UVM: "^",
    HIP_CACHE: "*",
    HIP_HASHMAP: "*",
}

root = "./saves/plot_offloading"
os.makedirs(root, exist_ok=True)

fig, ax1 = plt.subplots(figsize=(4, 3))


def render_data(plot_data, ax=None, linestyle="-"):
    def render_line(xs, ys, label, method=None):
        # line_ax = sb.lineplot(
        #     x=xs,
        #     y=ys,
        #     label=label,
        #     ax=ax,
        #     legend=False,
        #     linewidth=3.0,
        #     linestyle=linestyle,
        #     markers=True,
        #     markersize=10,
        # )
        line_ax = (ax if ax is not None else plt).plot(
            xs,
            ys,
            label=label,
            linewidth=LINEWIDTH.get(label, 3),
            linestyle=linestyle,
            marker=MARKERS[label],
            markersize=10,
        )
        if any(map(math.isnan, ys)):
            for last_okay, y in enumerate(ys):
                if math.isnan(y):
                    last_okay -= 1
                    break
            last_okay_x = xs[last_okay]
            last_okay_y = ys[last_okay]
            base_color = line_ax[-1].get_color()
            line_color = tuple(map(lambda x: x * 0.66, base_color))
            font_color = tuple(map(lambda x: x * 0.66, base_color))
            font_color = "darkgray"
            (ax if ax is not None else plt).annotate(
                DEAD_REASON[method],
                xy=(last_okay_x, last_okay_y),
                xytext=(last_okay_x + 2, last_okay_y),
                fontsize=10,
                va="center",
                fontweight=800,
                linespacing=0.9,
                color=font_color,
                zorder=100,
            )
            (ax if ax is not None else plt).plot(
                [last_okay_x],
                [last_okay_y],
                marker="x",
                color=line_color,
                markersize=10,
                zorder=100,
            )

    for label, data in plot_data.items():
        if isinstance(data, dict):
            for inner_label, inner_data in data.items():
                render_line(Ts, inner_data, f"{label} {inner_label}", method=label)
        else:
            render_line(Ts, data, label, method=label)


ax2 = ax1.twinx()
render_data(decode_throughput_data, ax=ax2)
render_data(gpu_memory_data, ax=ax1, linestyle=":")
# render_data(cpu_memory_data, ax=ax1, linestyle=':')

ax2.legend()
ax2.set_title("Decode Throughput and GPU KV Memory", fontsize=13, pad=12)
ax2.set_ylabel("Decode Throughput (tok/s) ↑", labelpad=5)
ax1.set_xlabel("$T$ (k)")

ax1.set_ylim(-200, 5000)
ax1.set_ylabel("GPU KV Memory (MB)", labelpad=5)
ax2.grid(False)

plt.savefig(
    os.path.join(root, "plot_offloading.png"), bbox_inches="tight", pad_inches=0
)
plt.savefig(
    os.path.join(root, "plot_offloading.pdf"), bbox_inches="tight", pad_inches=0
)
