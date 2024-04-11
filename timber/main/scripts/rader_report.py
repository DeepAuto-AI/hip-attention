import os

DATA_LONGBENCH = {
    'row_headers': ['LLaMA2 7B', 'StreamingLLM k3500', 'HiP k1024 PP', 'HiP k2048 PP'],
    'col_headers': ['NarrativeQA', 'Qasper', 'HotpotQA', '2WikiMQA', 'GovReport', 'MultiNews', 'Average'],
    'data': [
        [18.7, 19.2, 25.4, 32.8, 27.3, 25.8, 24.9],
        [11.6, 16.9, 21.6, 28.2, 23.9, 25.0, 21.2],
        [15.7, 20.3, 27.4, 31.4, 26.0, 25.9, 24.4],
        [19.1, 22.0, 27.3, 29.7, 26.3, 25.6, 25.0],
    ],
    'theta_offset': 1.0,
}

DATA_LMEVAL = {
    'row_headers': ['LLaMA2 7B', 'StreamingLLM k128', 'StreamingLLM k256', 'StreamingLLM k512', 'HiP k128 PP', 'HiP k256 PP', 'HiP k512 PP'],
    'col_headers': ['ARC', 'Hellaswag', 'MMLU', 'TruthfulQA', 'Winogrande', 'GSM8k'],
    'data': [
        [43.60, 75.37, 42.80, 32.41, 71.43, 5.38, 45.17],
        [],
        [43.69, 74.70, 41.11, 32.41, 71.43, 2.65, 44.33],
        [],
        [],
        [43.60, 75.13, 42.21, 32.41, 71.35, 4.85, 44.93],
        [43.60, 75.37, 42.56, 32.41, 71.35, 5.69, 45.16],
    ]
}

DATA_LMMSEVAL = {
    'row_headers': ['LLaVA2 13B', 'StremaingLLM k128', 'StremaingLLM k256', 'StremaingLLM k512', 'HiP k128 PP', 'HiP k256 PP', 'HiP k512 PP'],
    'col_headers': [''],
    'data': []
}

import numpy as np
import matplotlib.pyplot as plt
from timber.utils import setup_seaborn

setup_seaborn()

def loop_list(lst):
    return [*lst, lst[0]]

def render_rader(chart, name, title, root='saves/raders/'):
    os.makedirs(root, exist_ok=True)
    
    data = np.array(chart['data'])
    data = np.clip(data / data[:1, :], 0, 100.0) * 100
    
    categories = loop_list(chart['col_headers'])
    categories[0] = ''
    grades = []
    for grade in data:
        grades.append(loop_list(grade.tolist()))
    
    label_loc = np.linspace(0, 2*np.pi, len(grades[0]))
    
    plt.figure(figsize=(2.5, 2.5))
    ax = plt.subplot(polar=True)
    ax.set_title(title)
    ax.set_theta_offset(chart['theta_offset'])
    ax.tick_params(axis='x', which='major', pad=-5)
    plt.xticks(label_loc, labels=categories)
    for idx, (grade, label) in enumerate(zip(grades, chart['row_headers'])):
        if idx == 0:
            ax.plot(label_loc, grade, label=label, color='black', alpha=0.33, linestyle=':', zorder=1000, linewidth=2.0)
        else:
            ax.plot(label_loc, grade, label=label, linewidth=2.0)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncols=2)
    ax.set_ylim([50, np.max(data) * 1.0])
    
    plt.savefig(os.path.join(root, name + '.png'), dpi=200, bbox_inches="tight", pad_inches=0)
    plt.savefig(os.path.join(root, name + '.pdf'), dpi=200, bbox_inches="tight", pad_inches=0)
    print('saved', os.path.join(root, name + '.png'))

if __name__ == '__main__':
    render_rader(DATA_LONGBENCH, 'plot_rader_longbench', 'LM-Eval')