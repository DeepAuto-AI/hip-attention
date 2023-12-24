import json
import math
import matplotlib.pyplot as plt
import os
import matplotlib

import numpy as np
plt.style.use('seaborn-bright')

matplotlib.rcParams['font.family'] = 'Noto Sans, DejaVu Sans'
# plt.rcParams['text.usetex'] = True

COLORS = {
    'none': 'green',
    'perlin': 'pink',
    'performer': 'blue',
    'reformer': 'purple',
    'scatterbrain': '#2EB',
    'sinkhorn': 'orange',
    'synthesizer': 'lime',
    'cosformer': 'skyblue',
}
EDGECOLORS = {
    'none': '#040',
    'performer': '#005',
    'reformer': '#303',
    'scatterbrain': '#197',
    'sinkhorn': '#f55',
    'synthesizer': '#5b5',
    'cosformer': 'blue',
}
METHOD_NAMES = {
    'none': 'Vanilla',
    'perlin': 'Ours',
    'performer': 'Performer',
    'reformer': 'Reformer',
    'scatterbrain': 'ScatterBrain',
    'sinkhorn': 'Sinkhorn',
    'synthesizer': 'Synthesizer',
    'cosformer': 'Cosformer',
}
MARKERS = {
    'none': 'x',
    'perlin': '*'
}
MARKER_SIZE = {
    'perlin': 40,
    'none': 40,
    'default': 20
}
NAMES = {
    'cola': 'CoLA',
    'mrpc': 'MRPC',
    'mnli': 'MNLI',
}

def import_jn_format(path):
    """
    return {
        "{method}[,nbf:{nbf}][,k:{k}][,w:{w}]" : {
            "method": method,
            "metric": metric,
        }
    }
    "metric/f1" has priority
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    ret = {}
    for k, v in data.items():
        method = k.split()[0]
        if 'metric' in v:
            metric = float(v['metric']) * 100
        elif 'metric/f1' in v:
            metric = float(v['metric/f1']) * 100
        elif 'metric/accuracy' in v:
            metric = float(v['metric/accuracy']) * 100
        else:
            raise Exception(v)
        
        if method == 'perlin':
            nbf = v['nbf']
            k = v['k']
            w = v['w']
            ret[f'perlin,nbf:{nbf},k:{k},w:{w}'] = {
                'metric': metric,
                'nbf': nbf,
                'k': k,
                'method': 'perlin'
            }
        elif method == 'performer':
            nbf = v['nbf']
            ret[f'performer,nbf:{nbf}'] = {
                'metric': metric,
                'nbf': nbf,
                'method': 'performer'
            }
        elif method == 'cosformer':
            ret['cosformer'] = {
                'metric': metric,
                'method': 'cosformer',
            }
        else:
            k = v['k']
            ret[f'{method},k:{k}'] = {
                'metric': metric,
                'k': k,
                'method': method
            }
    return ret

def import_benchmark(path):
    """
    return {
        "{method}[,nbf:{nbf}][,k:{k}][,w:{w}]" : {
            "method": method,
            "latency": latency,
            "mem": mem,
        }
    }
    """
    with open(path, 'r') as f:
        return json.load(f)

data_cola = import_jn_format('./plots/main/bert_cola_ablation.json')
data_mrpc = import_jn_format('./plots/main/bert_mrpc_ablation.json')
data_mnli = import_jn_format('./plots/main/bert_mnli_ablation.json')
metrics = {
    'mrpc': data_mrpc,
    'cola': data_cola,
    'mnli': data_mnli,
}
benchmarks = import_benchmark('./plots/main/benchmark_bert_ablation/data.json')

methods = [set(map(lambda x: x['method'], v.values())) for v in metrics.values()]
methods = list(set.intersection(*methods))
methods.pop(methods.index('perlin'))
methods.insert(0, 'perlin')

root = './plots/main/figure_baseline_glue'
os.makedirs(root, exist_ok=True)

PERLIN_MARKERS = []
PERLIN_COLORS = []
PERLIN_ZORDERS = []
SUBMARKERS = {
    'k:7': 'D',
    'k:13': 's',
    'k:25': 'X',
}
SUBCOLORS = {
    'w:32': ('#f03', 10000),
    'w:64': ('#603', 1000),
    'w:128': ('#30f', 100),
}

ss_methods = {}

def render_fig(ax, data, benchmark, benchmark_metric='latency', x_label='ms', y_label='Acc.'):
    plot_data = []
    for method in methods:
        if method == 'perlin':
            xs = []
            ys = []
            for k, v in data.items():
                if v['method'] == method:
                    y = v['metric']
                    assert k in benchmark, k
                    x = benchmark[k][benchmark_metric]
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
                    xs.append(x)
                    ys.append(y)
            plot_data.append([xs, ys])
        else:
            xs = []
            ys = []
            ss = []
            for k, v in data.items():
                if v['method'] == method:
                    print(k, v)
                    s_default = MARKER_SIZE.get(method, MARKER_SIZE['default'])
                    y = v['metric']
                    assert k in benchmark, k
                    x = benchmark[k][benchmark_metric]
                    s = {
                        'none': {
                            'none': 1.0
                        },
                        'performer': {
                            '8': 0.5,
                            '4': 1.0,
                            '2': 1.5,
                            '1': 2.0,
                        },
                        'reformer': {
                            '7': 0.5,
                            '13': 1.0,
                            '25': 2.0
                        },
                        'scatterbrain': {
                            '7': 0.5,
                            '13': 1.0,
                            '25': 2.0
                        },
                        'sinkhorn': {
                            '7': 0.5,
                            '13': 1.0,
                            '25': 2.0
                        },
                        'synthesizer': {
                            '7': 0.5,
                            '13': 1.0,
                            '25': 2.0
                        },
                        'cosformer': {
                            'cosformer': 1.0,
                        },
                    }[method][v[{
                        'none': 'method',
                        'performer': 'nbf',
                        'reformer': 'k',
                        'scatterbrain': 'k',
                        'sinkhorn': 'k',
                        'synthesizer': 'k',
                        'cosformer': 'method',
                    }[method]]]
                    ss.append(s * s_default)
                    xs.append(x)
                    ys.append(y)
            
            ss_methods[method] = ss
            
            ax.scatter(
                xs, ys, 
                label=METHOD_NAMES[method], 
                color=COLORS[method], 
                edgecolor=EDGECOLORS[method],
                lw=1.0,
                marker=MARKERS.get(method, 'o'), 
                s=ss,
                zorder=100000 if method == 'none' else 0,
            )
            plot_data.append([xs, ys])
    ax.grid(True)
    ax.set_ylabel(y_label, fontweight=500)
    ax.set_xlabel(x_label, fontweight=500)
    return plot_data

ncols = len(metrics.keys()) + 1
nrows = 2

fig, axs = plt.subplots(nrows, ncols)
fig.set_figwidth(3.5*ncols)
fig.set_figheight(2.5*nrows)
# fig.set_facecolor('black')
# fig.suptitle('Comparison of Trade-off Between Computational Cost and Accuracy', fontsize=14, fontweight=500)

all_plot_data = []
for isubset, subset in enumerate(metrics.keys()):
    ylabel = {
        'mrpc': 'F1↑',
        'cola': "Matthew's Corr.↑",
        'mnli': 'Acc.↑'
    }[subset]
    ax = axs[0, isubset + 1]
    ax.set_title(f'Memory ({NAMES[subset]})', fontsize=12, fontweight=500)
    plot_data_memory = render_fig(ax, metrics[subset], benchmarks, 'mem', 'MB↓', ylabel)
    
    ax = axs[1, isubset + 1]
    ax.set_title(f'Latency ({NAMES[subset]})', fontsize=12, fontweight=500)
    plot_data_latency = render_fig(ax, metrics[subset], benchmarks, 'latency', 'ms↓', ylabel)
    all_plot_data.append([plot_data_memory, plot_data_latency])

fig.subplots_adjust(right=0.885, hspace=0.50, wspace=0.30)
handles, labels = ax.get_legend_handles_labels()
label_clip = sum([1 if x == 'Ours' else 0 for x in labels])
labels = labels[label_clip:]
handles = handles[label_clip:]
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
our_labels = []
our_handles = []
for k in [7, 13, 25]:
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
for w in [32, 64, 128]:
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
ncols = math.ceil(len(labels)/12)
def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])
fig.legend(
    flip(handles, ncols), flip(labels, ncols), 
    loc='center right', ncol=ncols, handler_map={tuple: HandlerTuple(ndivide=None)}
)
# fig.tight_layout()

# plt.savefig(os.path.join(root, 'plot_baseline_glue.png'), bbox_inches='tight')
# plt.savefig(os.path.join(root, 'plot_baseline_glue.pdf'), bbox_inches='tight')
# print(os.path.join(root, 'plot_baseline_glue.png'))

# plt.clf()
# nrows = 2
# ncols = 1
# fig, axs = plt.subplots(nrows, ncols)
# fig.set_figwidth(3.5*ncols)
# fig.set_figheight(2.7*nrows+1)
# plt.figure(figsize=(4,4))
def render_merged(ax, axis, xlabel, title):
    for imethod, method in enumerate(methods):
        data = [
            [
                j[imethod]
                for j in i # for memory and latency
            ]
            for i in all_plot_data # for each subset
        ]
        # print(imethod, method, data)
        try:
            data = np.array(data)
        except ValueError:
            print(method, data)
        # print(data.shape)
        # scale latency x axis
        # data[:,1,0,:] = data[:,1,0,:] * 10
        data = data[:,axis:axis+1,:,:]
        # weights = [{
        #     'mnli': 433000,
        #     'cola': 10657,
        #     'mrpc': 5801,
        # }[k] for k in metrics.keys()]
        weights = [{ # size of test set
            'mnli': 9800,
            'cola': 1050,
            'mrpc': 1725,
        }[k] for k in metrics.keys()]
        for i in range(len(weights)):
            w = weights[i] / sum(weights)
            data[i] = data[i] * w
        data = data.sum(0).mean(0)
        xs = data[0]
        ys = data[1]
        # print(data)
        if method == 'perlin':
            for i, (x, y) in enumerate(zip(xs, ys)):
                ax.scatter(
                    x, y, 
                    s=MARKER_SIZE.get(method, MARKER_SIZE['default']) * 2, 
                    label=METHOD_NAMES[method], 
                    edgecolor=PERLIN_COLORS[i], 
                    marker=PERLIN_MARKERS[i],
                    zorder=PERLIN_ZORDERS[i],
                    lw=1.5,
                    color=COLORS[method],
                )
        else:
            ax.scatter(
                xs, 
                ys, 
                # s=MARKER_SIZE.get(method, MARKER_SIZE['default']), 
                s = ss_methods[method],
                edgecolor=EDGECOLORS[method],
                lw=1.0,
                label=METHOD_NAMES[method], 
                color=COLORS[method], 
                marker=MARKERS.get(method, 'o'),
                zorder=100000 if method == 'none' else 0,
            )

    # plt.legend()
    ax.grid(True)
    ax.set_title(title, fontsize=13, fontweight=500)
    ax.set_xlabel(xlabel, fontweight=500)
    ax.set_ylabel('Average Metric↑', fontweight=500)

render_merged(axs[0, 0], 0, 'MB↓', 'Memory (Avg.)')
render_merged(axs[1, 0], 1, 'ms↓', 'Latency (Avg.)')
# fig.subplots_adjust(hspace=0.36, top=0.90)
# fig.suptitle("Averaged Among Subsets", fontsize=15, fontweight=500)

plt.savefig(os.path.join(root, 'plot_baseline_glue_all.png'), bbox_inches='tight')
plt.savefig(os.path.join(root, 'plot_baseline_glue_all.pdf'), bbox_inches='tight')
print(os.path.join(root, 'plot_baseline_glue_all.png'))