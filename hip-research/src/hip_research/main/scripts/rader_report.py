import os

import matplotlib.pyplot as plt
import numpy as np
from hip_research.utils import setup_seaborn

DATA_LONGBENCH = {
    "row_headers": ["Qwen1.5 7B", "S.LLM k512", "S.LLM k1024", "HiP k512", "HiP k1024"],
    "col_headers": [
        "NarrativeQA",
        "Qasper",
        "HotpotQA",
        "2WikiMQA",
        "GovReport",
        "MultiNews",
        "Average",
    ],
    "data": [
        [20.94, 40.25, 47.92, 34.14, 30.24, 24.69, 33.03],
        [13.37, 17.79, 23.46, 22.09, 20.18, 20.49, 19.56],
        [12.82, 21.18, 27.07, 21.93, 22.29, 23.08, 21.40],
        [16.50, 36.29, 37.89, 29.88, 27.92, 24.25, 28.79],
        [17.61, 39.15, 43.73, 31.44, 29.20, 24.75, 30.98],
    ],
    "theta_offset": 0.5,
    "ymin": 30.0,
}

DATA_MMLU = {
    "row_headers": [
        "LLaMA2 7B",
        "S.LLM k512",
        "S.LLM k1024",
        "HiP k512",
        "HiP k1024",
    ],
    "col_headers": ["Humanities", "STEM", "Social Sciences", "Other", "Average"],
    "data": [
        [47.24, 34.81, 48.51, 44.59, 42.76],
        [45.74, 35.17, 45.76, 42.16, 41.40],
        [47.00, 34.37, 48.09, 44.61, 42.47],
        [45.99, 34.61, 46.85, 44.87, 42.12],
        [47.25, 35.03, 48.55, 45.61, 43.08],
    ],
    "theta_offset": 0.5,
    "ymin": 90.0,
}

DATA_LMEVAL = {
    # 'row_headers': ['LLaMA2 7B', 'S.LLM k128', 'S.LLM k256', 'S.LLM k512', 'HiP k128 PP', 'HiP k256 PP', 'HiP k512 PP'],
    "row_headers": ["LLaMA2 7B", "S.LLM k256", "S.LLM k512", "HiP k256", "HiP k512"],
    "col_headers": ["ARC", "H.S.", "MMLU", "T.QA", "Win.", "GSM8k", "Avg."],
    "data": [
        [43.60, 75.37, 42.80, 32.41, 71.43, 5.38, 45.17],
        # [],
        [43.69, 74.70, 41.11, 32.41, 71.43, 2.65, 44.33],
        [43.60, 75.12, 41.76, 32.41, 71.43, 4.47, 44.80],
        # [],
        [43.60, 75.13, 42.21, 32.41, 71.35, 4.85, 44.93],
        [43.60, 75.37, 42.56, 32.41, 71.35, 5.69, 45.16],
    ],
    "theta_offset": 0.5,
    "ymin": 80.0,
}

DATA_LMMSEVAL = {
    "row_headers": [
        "LLaVAv1.6 13B",
        "S.LLM k512",
        "S.LLM k1024",
        "HiP k512",
        "HiP k1024",
    ],
    "col_headers": [
        "MME Cog.",
        "MME Percep.",
        "MMMU",
        "DocVQA",
        "GQA",
        "TextVQA",
        "Average",
    ],
    "data": [
        [324.29, 1575.21, 0.359, 0.7738, 0.6536, 0.6692, 100.0],
        [242.14, 1093.55, 0.313, 0.2475, 0.5171, 0.2676, 63.73],
        [326.43, 1331.79, 0.323, 0.4195, 0.6000, 0.4636, 81.75],
        [306.43, 1541.75, 0.353, 0.7220, 0.6454, 0.6429, 96.47],
        [325.36, 1561.47, 0.351, 0.7601, 0.6492, 0.6638, 99.00],
    ],
    "theta_offset": 0.5,
    "ymin": 20.0,
}

DATA_LONGBENCH_LLAMA2_7B = {
    "row_headers": ["LLaMA2 7B", "S.LLM k512", "S.LLM k1024", "HiP k512", "HiP k1024"],
    "col_headers": [
        "NarrativeQA",
        "Qasper",
        "HotpotQA",
        "2WikiMQA",
        "GovReport",
        "MultiNews",
        "Average",
    ],
    "data": [
        [4.35, 12.37, 14.70, 12.33, 30.75, 27.21, 16.95],
        [4.49, 5.66, 7.90, 8.87, 20.24, 20.61, 11.30],
        [4.54, 6.65, 7.54, 8.24, 22.21, 23.49, 12.11],
        [4.15, 9.19, 11.94, 11.59, 23.56, 25.86, 14.38],
        [3.38, 11.05, 11.79, 12.15, 26.88, 26.05, 15.22],
    ],
    "theta_offset": 0.5,
    "ymin": 30.0,
}
DATA_LONGBENCH_QWEN_7B = DATA_LONGBENCH
DATA_LONGBENCH_LLAMA2_13B = {
    "row_headers": ["LLaMA2 13B", "S.LLM k512", "S.LLM k1024", "HiP k512", "HiP k1024"],
    "col_headers": [
        "NarrativeQA",
        "Qasper",
        "HotpotQA",
        "2WikiMQA",
        "GovReport",
        "MultiNews",
        "Average",
    ],
    "data": [
        [13.45, 11.39, 18.13, 21.21, 28.84, 26.38, 19.90],
        [7.16, 10.58, 8.18, 12.76, 17.00, 18.16, 12.31],
        [8.00, 10.46, 10.74, 17.54, 18.92, 20.68, 14.39],
        [10.72, 10.21, 13.64, 18.56, 23.99, 25.33, 17.08],
        [11.51, 11.98, 16.37, 20.76, 25.85, 25.85, 18.72],
    ],
    "theta_offset": 0.5,
    "ymin": 30.0,
}
DATA_LONGBENCH_QWEN_14B = {
    "row_headers": [
        "Qwen1.5 14B",
        "S.LLM k512",
        "S.LLM k1024",
        "HiP k512",
        "HiP k1024",
    ],
    "col_headers": [
        "NarrativeQA",
        "Qasper",
        "HotpotQA",
        "2WikiMQA",
        "GovReport",
        "MultiNews",
        "Average",
    ],
    "data": [
        [22.85, 39.83, 53.82, 41.35, 29.84, 24.41, 35.35],
        [12.56, 20.85, 32.92, 22.22, 20.94, 19.99, 21.58],
        [14.63, 24.36, 34.02, 27.21, 23.25, 22.36, 24.31],
        [17.50, 38.35, 40.64, 35.64, 28.10, 23.97, 30.70],
        [20.12, 38.19, 48.17, 41.35, 28.61, 24.29, 33.46],
    ],
    "theta_offset": 0.5,
    "ymin": 30.0,
}


setup_seaborn(
    legend_fontsize=5,
    axis_below=True,
)


def loop_list(lst):
    return [*lst, lst[0]]


def render_rader(chart, name, title, root="saves/raders/"):
    os.makedirs(root, exist_ok=True)

    data = np.array(chart["data"])
    data = np.clip(data / data[:1, :], 0, 100.0) * 100

    categories = loop_list(chart["col_headers"])
    categories[0] = ""
    grades = []
    for grade in data:
        grades.append(loop_list(grade.tolist()))

    label_loc = np.linspace(0, 2 * np.pi, len(grades[0]))

    plt.figure(figsize=(2.4, 2.8))
    ax = plt.subplot(polar=True)
    ax.set_title(title, pad=15)
    ax.set_theta_offset(chart["theta_offset"])

    ax.tick_params(axis="x", which="major", pad=-5)
    plt.xticks(label_loc, labels=categories)
    for idx, (grade, label) in enumerate(zip(grades, chart["row_headers"])):
        if idx == 0:
            ax.plot(
                label_loc,
                grade,
                label=label,
                color="black",
                alpha=0.33,
                linestyle=":",
                zorder=1000,
                linewidth=2.0,
            )
        else:
            linestyle = "-" if "hip" in label.lower() else "-."
            ax.plot(label_loc, grade, label=label, linewidth=2.0, linestyle=linestyle)

    labels = []
    for label, angle in zip(
        ax.get_xticklabels()[1:],
        np.degrees(
            [
                (i + 1) / len(chart["col_headers"]) * 3.14 * 2 + chart["theta_offset"]
                for i in range(len(chart["col_headers"]))
            ]
        ),
    ):
        x, y = label.get_position()
        lab = ax.text(
            x,
            y,
            label.get_text(),
            transform=label.get_transform(),
            ha=label.get_ha(),
            va=label.get_va(),
            fontsize=6,
        )
        angle = (angle % 360) + 90
        if angle >= 90 and angle <= 270:
            angle += 180
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([])

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncols=3)
    ax.set_ylim([chart["ymin"], np.max(data) * 1.0])

    plt.savefig(os.path.join(root, name + ".png"), dpi=200, pad_inches=0)
    plt.savefig(os.path.join(root, name + ".pdf"), dpi=200, pad_inches=0)
    print("saved", os.path.join(root, name + ".png"))


if __name__ == "__main__":
    render_rader(DATA_MMLU, "plot_rader_mmlu", "MMLU")
    render_rader(DATA_LONGBENCH, "plot_rader_longbench", "LongBench")
    render_rader(DATA_LMEVAL, "plot_rader_lmeval", "LM-Eval")
    render_rader(DATA_LMMSEVAL, "plot_rader_lmmseval", "LMMs-Eval")

    render_rader(
        DATA_LONGBENCH_LLAMA2_7B,
        "plot_rader_longbench_llama7b",
        "LLaMA2 7B (LongBench)",
    )
    render_rader(
        DATA_LONGBENCH_LLAMA2_13B,
        "plot_rader_longbench_llama13b",
        "LLaMA2 13B (LongBench)",
    )
    render_rader(
        DATA_LONGBENCH_QWEN_7B, "plot_rader_longbench_qwen7b", "Qwen1.5 7B (LongBench)"
    )
    render_rader(
        DATA_LONGBENCH_QWEN_14B,
        "plot_rader_longbench_qwen14b",
        "Qwen1.5 14B (LongBench)",
    )
