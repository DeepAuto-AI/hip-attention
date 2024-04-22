import os

DATA_LONGBENCH = {
    'row_headers': [
        'Qwen1.5 7B', 
        'StreamingLLM k512', 
        'StreamingLLM k512', 
        'HiP k512 PP', 
        'HiP k1024 PP'
    ],
    'col_headers': ['N.QA', 'Qasper', 'H.QA', '2W.MQA', 'G.R.', 'M.N.', 'Avg.'],
    'data': [
        [20.94, 40.25, 47.92, 34.14, 30.24, 24.69, 33.03],
        [13.37, 17.79, 23.46, 22.09, 20.18, 20.49, 19.56],
        [12.82, 21.18, 27.07, 21.93, 22.29, 23.08, 21.40],
        [16.50, 36.29, 37.89, 29.88, 27.92, 24.25, 28.79],
        [17.61, 39.15, 43.73, 31.44, 29.20, 24.75, 30.98],
    ],
    'theta_offset': 0.5,
    'ymin': 30.0,
}

DATA_MMLU = {
    'row_headers': [
        'LLaMA2 7B', 
        'StreamingLLM k512', 
        'StreamingLLM k1024', 
        'HiP k512 PP', 
        'HiP k1024 PP',
    ],
    'col_headers': ['Humani-\nties', 'STEM', 'Social\nSciences', 'Other', 'Avg.'],
    'data': [
        [47.24, 34.81, 48.51, 44.59, 42.76],
        [45.74, 35.17, 45.76, 42.16, 41.40],
        [47.00, 34.37, 48.09, 44.61, 42.47],
        [45.99, 34.61, 46.85, 44.87, 42.12],
        [47.25, 35.03, 48.55, 45.61, 43.08],
    ],
    'theta_offset': 0.5,
    'ymin': 90.0,
}

DATA_LMEVAL = {
    # 'row_headers': ['LLaMA2 7B', 'StreamingLLM k128', 'StreamingLLM k256', 'StreamingLLM k512', 'HiP k128 PP', 'HiP k256 PP', 'HiP k512 PP'],
    'row_headers': [
        'LLaMA2 7B', 
        'StreamingLLM k256', 
        'StreamingLLM k512', 
        'HiP k256 PP', 
        'HiP k512 PP'
    ],
    'col_headers': ['ARC', 'H.S.', 'MMLU', 'T.QA', 'Win.', 'GSM8k', 'Avg.'],
    'data': [
        [43.60, 75.37, 42.80, 32.41, 71.43, 5.38, 45.17],
        # [],
        [43.69, 74.70, 41.11, 32.41, 71.43, 2.65, 44.33],
        [43.60, 75.12, 41.76, 32.41, 71.43, 4.47, 44.80],
        # [],
        [43.60, 75.13, 42.21, 32.41, 71.35, 4.85, 44.93],
        [43.60, 75.37, 42.56, 32.41, 71.35, 5.69, 45.16],
    ],
    'theta_offset': 0.5,
    'ymin': 80.0,
}

DATA_LMMSEVAL = {
    'row_headers': [
        'LLaVAv1.6 13B',
        'StremaingLLM k512',
        'StremaingLLM k1024',
        'HiP k512 PP',
        'HiP k1024 PP',
    ],
    'col_headers': ['MME\ncog.', 'MME\npercep.', 'MMMU', 'D.VQA', 'GQA', 'T.VQA', 'Average'],
    'data': [
        [324.29, 1575.21, 0.359, 0.7738, 0.6536, 0.6692, 100.0],
        [242.14, 1093.55, 0.313, 0.2475, 0.5171, 0.2676, 63.73],
        [326.43, 1331.79, 0.323, 0.4195, 0.6000, 0.4636, 81.75],
        [291.43, 1518.11, 0.353, 0.6921, 0.6465, 0.6251, 94.39],
        [330.71, 1546.57, 0.358, 0.7478, 0.6506, 0.6568, 99.04],
    ],
    'theta_offset': 0.5,
    'ymin': 20.0,
}

import numpy as np
import matplotlib.pyplot as plt
from timber.utils import setup_seaborn

setup_seaborn(
    legend_fontsize=6,
    axis_below=False
)

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
    ax.tick_params(axis='x', which='major', pad=-25)
    plt.xticks(label_loc, labels=categories)
    for idx, (grade, label) in enumerate(zip(grades, chart['row_headers'])):
        if idx == 0:
            ax.plot(label_loc, grade, label=label, color='black', alpha=0.33, linestyle=':', zorder=1000, linewidth=2.0)
        else:
            linestyle = '-' if 'hip' in label.lower() else '-.'
            ax.plot(label_loc, grade, label=label, linewidth=2.0, linestyle=linestyle)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncols=2)
    ax.set_ylim([chart['ymin'], np.max(data) * 1.0])
    
    plt.savefig(os.path.join(root, name + '.png'), dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.savefig(os.path.join(root, name + '.pdf'), dpi=200, bbox_inches="tight", pad_inches=0.01)
    print('saved', os.path.join(root, name + '.png'))

if __name__ == '__main__':
    render_rader(DATA_MMLU, 'plot_rader_mmlu', 'MMLU')
    render_rader(DATA_LONGBENCH, 'plot_rader_longbench', 'LongBench')
    render_rader(DATA_LMEVAL, 'plot_rader_lmeval', 'LM-Eval')
    render_rader(DATA_LMMSEVAL, 'plot_rader_lmmseval', 'LMMs-Eval')