import math
import os

import matplotlib.pyplot as plt
from hip_research.utils import setup_seaborn


def proc_copy_paste(t: str, scale: float = 1):
    return list(map(lambda x: float(x) * scale, t.split()))


FLASH_ATTN = "Flash Attn"
HIP_ATTN = "HiP Attn"
BIGBIRD = "BigBird"
SLLM = "StreamingLLM"
H2O = "H2O"
HYPER_ATTN = "Hyper Attn"

Ts = [8, 16, 32, 64, 128]

FIGSIZE = (3.0, 2.5)

LINESTYLE = {f"{HIP_ATTN} Heal": "--"}

LINEWIDTH = {
    f"{HIP_ATTN} Heal": 2,
    FLASH_ATTN: 3,
}


DEAD_REASON = {
    HYPER_ATTN: "Crashed\nafter this point",
    SLLM: "OOM\nafter this point",
}

prefill_data = {
    FLASH_ATTN: proc_copy_paste("3.82	13.49	53.47	213.46	861.40"),
    HIP_ATTN: proc_copy_paste("3.22	7.67	18.14	42.20	95.69"),
    BIGBIRD: proc_copy_paste("1.41	2.87	6.30	14.47	30.97"),
    SLLM: proc_copy_paste("178.8	359.5	721.2	NaN	NaN"),
    H2O: proc_copy_paste("36.6	142.6	569.5	2563	12576"),
    HYPER_ATTN: proc_copy_paste("15.65	36.40	80.52	NaN	NaN"),
}

decode_data = {
    FLASH_ATTN: proc_copy_paste("0.0360	0.0713	0.1418	0.2822	0.5639", scale=1000),
    HIP_ATTN: proc_copy_paste("0.0149	0.0159	0.0168	0.0180	0.0188", scale=1000),
    BIGBIRD: proc_copy_paste("0.0131	0.0135	0.0137	0.0138	0.0138", scale=1000),
    SLLM: proc_copy_paste("0.0134	0.0134	0.0134	NaN	NaN", scale=1000),
    H2O: proc_copy_paste(
        "0.03913788	0.03913788	0.03913788	0.03913788	0.03913788", scale=1000
    ),
    # HYPER_ATTN: proc_copy_paste('NaN	NaN	NaN	NaN	NaN', scale=1000),
}

pg19_data = {
    FLASH_ATTN: proc_copy_paste("8.3581	8.1028	7.9056	7.7571	7.6737"),
    HIP_ATTN: {
        "": proc_copy_paste("8.3937	8.2025	8.0977	8.0554	8.1850"),
        "Heal": proc_copy_paste("8.3895	8.1822	8.0623	8.0088	8.1277"),
    },
    BIGBIRD: proc_copy_paste("9.0230	9.0442	9.1189	9.2325	9.4282"),
    SLLM: proc_copy_paste("9.2065	9.2128	9.2402	NaN	NaN"),
    HYPER_ATTN: {
        "$l_d$=3": proc_copy_paste("63.9827	73.6145	67.735	NaN	NaN"),
        "$l_d$=25": proc_copy_paste("10.4919	10.5695	10.4721	NaN	NaN"),
    },
}

wt2_data = {
    FLASH_ATTN: proc_copy_paste("5.6080	5.4728	5.3986	5.3470	5.3467"),
    HIP_ATTN: proc_copy_paste("5.6179	5.5037	5.4529	5.4305	5.5889"),
    H2O: {
        "": proc_copy_paste("5.9855	6.0205	6.0843	6.1506	6.2805"),
        # 'stream': proc_copy_paste('6.5778	6.5412	6.5701	6.6008	6.7079'),
    },
}

setup_seaborn(
    label_fontsize=11,
    legend_fontsize=8,
    axes_label_fontsize=8,
    axis_below=True,
)

import seaborn as sb

working_directory = "./saves/plot_ppl/"
os.makedirs(working_directory, exist_ok=True)


def render(
    plot_data: dict,
    figsize=FIGSIZE,
    reset_figure=True,
    ax=None,
    text_offset_scale_y=0.9,
):
    if reset_figure:
        plt.figure(figsize=figsize)

    def render_line(xs, ys, label, method=None):
        line_ax = sb.lineplot(
            x=xs,
            y=ys,
            label=label,
            ax=ax,
            legend=False,
            linewidth=LINEWIDTH.get(label, 3),
            linestyle=LINESTYLE.get(label, "-"),
        )
        if any(map(math.isnan, ys)):
            for last_okay, y in enumerate(ys):
                if math.isnan(y):
                    last_okay -= 1
                    break
            last_okay_x = xs[last_okay]
            last_okay_y = ys[last_okay]
            base_color = line_ax.lines[-1].get_color()
            line_color = tuple(map(lambda x: x * 0.66, base_color))
            font_color = tuple(map(lambda x: x * 0.66, base_color))
            font_color = "darkgray"
            (ax if ax is not None else plt).annotate(
                DEAD_REASON[method],
                xy=(last_okay_x, last_okay_y),
                xytext=(last_okay_x + 3, last_okay_y),
                fontsize=6,
                va="center",
                fontweight=800,
                linespacing=0.9,
                color=font_color,
                # arrowprops=dict(
                #     # width=2,
                #     # headwidth=4,
                #     # headlength=4,
                #     shrinkA=0,
                #     shrinkB=0,
                #     relpos=(0.0,0.5),
                #     arrowstyle = '-|>',
                #     linestyle='-',
                #     linewidth=0.5,
                #     edgecolor=line_color,
                #     facecolor=line_color,
                #     # connectionstyle = 'arc3',
                #     # facecolor=line_color,
                # )
            )
            (ax if ax is not None else plt).plot(
                [last_okay_x],
                [last_okay_y],
                marker="x",
                color=line_color,
            )

    for label, data in plot_data.items():
        if isinstance(data, dict):
            for inner_label, inner_data in data.items():
                render_line(Ts, inner_data, f"{label} {inner_label}", method=label)
        else:
            render_line(Ts, data, label, method=label)
    if reset_figure:
        plt.grid(True)
    return


render(prefill_data, text_offset_scale_y=0.7)
plt.title("Prefill Latency", fontsize=13)
plt.ylim(0, 300)
plt.legend(loc="upper right")
plt.xlabel("$T$ (k)")
plt.ylabel("Latency (ms)")
plt.savefig(
    os.path.join(working_directory, "plot_ppl_prefill.png"),
    bbox_inches="tight",
    pad_inches=0,
)
plt.savefig(
    os.path.join(working_directory, "plot_ppl_prefill.pdf"),
    bbox_inches="tight",
    pad_inches=0,
)

render(decode_data, text_offset_scale_y=0.7)
plt.title("Decode Latency", fontsize=13)
plt.ylim(10, 42)
plt.legend(loc="upper right")
plt.xlabel("$T$ (k)")
plt.ylabel("Latency ($\\mu$s)")
plt.savefig(
    os.path.join(working_directory, "plot_ppl_decode.png"),
    bbox_inches="tight",
    pad_inches=0,
)
plt.savefig(
    os.path.join(working_directory, "plot_ppl_decode.pdf"),
    bbox_inches="tight",
    pad_inches=0,
)


def split_plot(plot_data, range1, range2, legend_anchor, text_offset_scale_y=0.9):
    plt.clf()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=FIGSIZE, gridspec_kw={"height_ratios": [1, 2]}
    )
    fig.subplots_adjust(hspace=0.05)

    render(
        plot_data, ax=ax1, reset_figure=False, text_offset_scale_y=text_offset_scale_y
    )
    render(
        plot_data, ax=ax2, reset_figure=False, text_offset_scale_y=text_offset_scale_y
    )

    ax1.set_ylim(*range2)
    ax2.set_ylim(*range1)
    ax2.legend(bbox_to_anchor=legend_anchor)

    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    # ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)
    # ax2.xaxis.tick_bottom()

    d = 0.5
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=5,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    return fig, ax1, ax2


# fig, ax1, ax2 = split_plot(decode_data, (0, 50), (175, 225), (0.42, 0.72), text_offset_scale_y=1.5)

# ax1.set_title('Decode Latency', fontsize=13)
# ax2.set_xlabel('$T$ (k)', fontsize=11, fontweight=800)
# ax2.set_ylabel('Latency ($\mu$s)')
# ax2.yaxis.set_label_coords(-0.1, 0.75)
# fig.savefig(os.path.join(working_directory, 'plot_ppl_decode.png'), bbox_inches='tight', pad_inches=0)
# fig.savefig(os.path.join(working_directory, 'plot_ppl_decode.pdf'), bbox_inches='tight', pad_inches=0)

fig, ax1, ax2 = split_plot(
    pg19_data, (7, 11.5), (60, 80), (0.37, 0.42), text_offset_scale_y=0.96
)
ax1.set_title("PG19 Perplexity", fontsize=13)
ax2.set_xlabel("$T$ (k)", fontsize=11, fontweight=800)
ax2.set_ylabel("Perplexity")
ax2.yaxis.set_label_coords(-0.1, 0.75)
fig.savefig(
    os.path.join(working_directory, "plot_ppl_pg19.png"),
    bbox_inches="tight",
    pad_inches=0,
)
fig.savefig(
    os.path.join(working_directory, "plot_ppl_pg19.pdf"),
    bbox_inches="tight",
    pad_inches=0,
)

render(wt2_data)
plt.title("Wikitext2 Perplexity", fontsize=13)
plt.ylim(5, 7)
plt.legend()
plt.xlabel("$T$ (k)")
plt.ylabel("Perplexity")
plt.savefig(
    os.path.join(working_directory, "plot_ppl_wt2.png"),
    bbox_inches="tight",
    pad_inches=0,
)
plt.savefig(
    os.path.join(working_directory, "plot_ppl_wt2.pdf"),
    bbox_inches="tight",
    pad_inches=0,
)
