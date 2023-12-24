import json
import math
import matplotlib.pyplot as plt
import os

import numpy as np
plt.style.use('seaborn-bright')
import matplotlib

matplotlib.rcParams['font.family'] = 'Noto Sans, DejaVu Sans'

METHOD_NAMES = {
    'none': 'Vanilla',
    'perlin': 'Ours',
    'performer': 'Performer',
    'reformer': 'Reformer',
    'scatterbrain': 'ScatterBrain',
    'sinkhorn': 'Sinkhorn',
    'synthesizer': 'Synthesizer',
}
COLORS = {
    'none': 'green',
    'perlin': 'pink',
    'performer': 'blue',
    'reformer': 'purple',
    'scatterbrain': 'gray',
    'sinkhorn': 'orange',
    'synthesizer': 'yellow',
}
EDGECOLORS = {
    'none': '#040',
    'performer': '#005',
    'reformer': '#303',
    'scatterbrain': 'black',
    'sinkhorn': 'red',
    'synthesizer': 'orange',
}
MARKERS = {
    'none': 'o',
    'perlin': '*'
}
MARKER_SIZE = {
    'perlin': 40,
    'none': 40,
    'default': 20
}

def load_metrics(path):
    with open(path, 'r') as f:
        data = json.load(f)
    for k, v in data.items():
        v['method'] = k.split(',')[0]
    return data

def load_benchmark(path):
    with open(path, 'r') as f:
        return json.load(f)

metrics = load_metrics('./plots/main/opt_albation.json')
benchmarks = load_benchmark('./plots/main/benchmark_opt_ablation/data.json')
methods = ['perlin', 'none', 'reformer', 'performer']

PERLIN_MARKERS = []
PERLIN_COLORS = []
PERLIN_ZORDERS = []
SUBMARKERS = {
    'k:32': 'D',
    'k:64': 's',
    'k:128': 'X',
}
SUBCOLORS = {
    'w:64': ('#306', 100000),
    'w:128': ('#f03', 10000),
    'w:256': ('#603', 1000),
    'w:384': ('#30f', 100),
}

def render_plot(ax, metric, benchmark, benchmark_metric, x_label):
    plot_data = []
    for method in methods:
        if method == 'perlin':
            xs = []
            ys = []
            for k, v in metric.items():
                if v['method'] == method:
                    y = v['metric']
                    x = benchmark[k][benchmark_metric]
                    xs.append(x)
                    ys.append(y)
                    
                    for _k, _v in SUBMARKERS.items():
                        if _k in k:
                            sub_marker = _v
                            break
                    for _k, _v in SUBCOLORS.items():
                        if _k in k:
                            sub_color, zorder = _v
                            break
                    PERLIN_COLORS.append(sub_color)
                    PERLIN_MARKERS.append(sub_marker)
                    PERLIN_ZORDERS.append(zorder)
                    
                    ax.scatter(
                        x, 
                        y, 
                        label=METHOD_NAMES[method], 
                        edgecolor=sub_color, 
                        lw=1.5,
                        color=COLORS[method],
                        marker=sub_marker, 
                        s=MARKER_SIZE.get(method, MARKER_SIZE['default']) * 1.5,
                        zorder=zorder,
                    )
            plot_data.append([xs, ys])
        else:
            xs = []
            ys = []
            ss = []
            for k, v in metric.items():
                if v['method'] == method:
                    y = v['metric']
                    s = {
                        'reformer': {
                            2: 0.5,
                            4: 1.0,
                            8: 1.5,
                            16: 2.0
                        },
                        'performer': {
                            8: 0.5,
                            4: 1.0,
                            2: 1.5,
                            1: 2.0,
                        },
                        'none': {
                            'none': 0.66,
                        }
                    }[method][v[{'reformer': 'n_hash', 'performer': 'nbf', 'none': 'method'}[method]]]
                    ss.append(s * MARKER_SIZE.get(method, MARKER_SIZE['default']))
                    x = benchmark[k][benchmark_metric]
                    xs.append(x)
                    ys.append(y)
            # print(xs, ys, method)
            ax.scatter(
                xs, 
                ys, 
                s=ss, 
                marker=MARKERS.get(method, 'o'), 
                color=COLORS.get(method, 'gray'),
                edgecolor=EDGECOLORS.get(method, 'black'),
                lw=1.0,
                label=METHOD_NAMES[method],
                zorder=100000 if method == 'none' else 0,
            )
            plot_data.append([xs, ys])
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel('PPL.↓')
    return plot_data

root = './plots/main/figure_baseline_opt'
os.makedirs(root, exist_ok=True)

nrows = 1
ncols = 2
fig, axs = plt.subplots(nrows, ncols)
fig.set_figwidth(3.2*ncols+0.5)
fig.set_figheight(2.25)
# fig.suptitle('Comparison of Trade-off Between Computational Cost and Accuracy on Wikitext2', fontsize=13, fontweight=500)

ax = axs[-2]
ax.set_title(f'Memory', fontsize=11, fontweight=500)
plot_data_mem = render_plot(ax, metrics, benchmarks, 'mem', 'MB↓')

ax = axs[-1]
ax.set_title(f'Latency', fontsize=11, fontweight=500)
plot_data_latency = render_plot(ax, metrics, benchmarks, 'latency', 'ms↓')

# ax = axs[0]
# plot_data = [plot_data_mem, plot_data_latency]
# ax.set_title(f'Overall Efficiency', fontsize=11, fontweight=500)
# for imethod, method in enumerate(methods):
#     data = [
#         i[imethod]
#         for i in plot_data 
#     ]
#     data = np.array(data)
#     data[1,0,:] = data[1,0,:]*50
#     data = data.mean(0)
#     # print(method, data)
#     xs = data[0, :]
#     ys = data[1, :]
#     ax.scatter(
#         xs, 
#         ys, 
#         s=MARKER_SIZE.get(method, MARKER_SIZE['default']), 
#         marker=MARKERS.get(method, 'o'), 
#         color=COLORS.get(method, 'gray'),
#         label=METHOD_NAMES[method]
#     )
#     ax.grid(True)
#     ax.set_xlabel('50*Lat.+Mem.', fontweight=500)
#     ax.set_ylabel('PPL. (Lower is better)', fontweight=500)

fig.subplots_adjust(top=0.77, bottom=0.0, wspace=0.25, right=0.80, left=0)
handles, labels = ax.get_legend_handles_labels()
label_clip = sum([1 if x == 'Ours' else 0 for x in labels])
labels = labels[label_clip:]
handles = handles[label_clip:]
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
our_labels = []
our_handles = []
for k in [32, 64, 128]:
    our_labels.append(f'Ours ($k$={k})')
    markers = []
    for c in SUBCOLORS.values():
        markers.append(Line2D(
            [0], [0],
            color='w',
            markeredgecolor='w',
            markerfacecolor='#f99',
            marker=SUBMARKERS[f'k:{k}'],
            markersize=8,
        ))
        break
    our_handles.append(tuple(markers))
for w in [64, 128, 256, 384]:
    our_labels.append(f'Ours ($K$={w})')
    markers = []
    for m in SUBMARKERS.values():
        markers.append(Line2D(
            [0], [0],
            color='w',
            markeredgecolor=SUBCOLORS[f'w:{w}'][0],
            markerfacecolor='w',
            marker='o',
            markersize=8,
        ))
        break
    our_handles.append(tuple(markers))
handles = our_handles + handles
labels = our_labels + labels
import itertools
# ncols = math.ceil(len(labels)/2)
ncols = 1
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
fig.legend(
    flip(handles, ncols), flip(labels, ncols), 
    loc='center right', ncol=ncols, handler_map={tuple: HandlerTuple(ndivide=None)},
    fontsize=8.5,
)

plt.savefig(os.path.join(root, 'plot_baseline_opt.png'), bbox_inches='tight')
plt.savefig(os.path.join(root, 'plot_baseline_opt.pdf'), bbox_inches='tight')
print(os.path.join(root, 'plot_baseline_opt.png'))