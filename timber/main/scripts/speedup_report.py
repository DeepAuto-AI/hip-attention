import json
import math
import sys, subprocess, os, itertools, tqdm
import seaborn as sns
sns.set_style('whitegrid')
import pypareto
import matplotlib.pyplot as plt

os.environ['PYTHONPATH'] = './'

block_sizes = [1, 2, 4, 8, 16]
query_sizes = [1, 2, 4, 8, 16]
ks = [128, 256, 512, 1024]

def samples():
    num_samples = 200
    batch_size = 256
    results = {}
    cache_path = './cache/attention1_block_gpu/result.json'
    
    for query_size in tqdm.tqdm(query_sizes, dynamic_ncols=True, desc='none'):
        subprocess.call([
            'python', 'timber/models/timber_attention/attention1_block_gpu.py',
            '--method', 'none',
            '--query_size', str(query_size),
            '--dups', '4',
            '--batch_size', str(batch_size),
            '--samples', str(num_samples),
        ])
        with open(cache_path, 'r') as f:
            latency = json.load(f)['mean']
        os.remove(cache_path)
        results[f'none_q{query_size}'] = {
            'query_size': query_size,
            'latency': latency,
            'method': 'none'
        }
    
    for query_size, block_size, k in tqdm.tqdm(
        list(itertools.product(query_sizes, block_sizes, ks)),
        dynamic_ncols=True, desc='eval',
    ):
        subprocess.call([
            'python', 'timber/models/timber_attention/attention1_block_gpu.py',
            '--method', 'timber',
            '--block_size_q', str(block_size),
            '--block_size_k', str(block_size),
            '--k', str(k),
            '--query_size', str(query_size),
            '--dups', '4',
            '--batch_size', str(batch_size),
            '--samples', str(num_samples),
        ])
        with open(cache_path, 'r') as f:
            latency = json.load(f)['mean']
        os.remove(cache_path)
        results[f'timber_q{query_size}_b{block_size}_k{k}'] = {
            'query_size': query_size,
            'k': k,
            'block_size': block_size,
            'block_size_k': block_size,
            'latency': latency,
            'method': 'timber',
        }
    
    os.makedirs('./saves/speedup_report', exist_ok=True)
    path = './saves/speedup_report/result.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('saved', path)

def plot():
    path = './saves/speedup_report/result.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    plt.figure(figsize=(5, 4))
    
    for query_size in query_sizes:
        xs = []
        ys = []
        for block_size, k in itertools.product(block_sizes, ks):
            timber_entry = data[f'timber_q{query_size}_b{block_size}_k{k}']
            base_entry = data[f'none_q{query_size}']
            num_blocks = k / block_size
            speedup = base_entry['latency'] / timber_entry['latency']
            xs.append(num_blocks)
            ys.append(speedup)
        sns.scatterplot(x=xs, y=ys, label=f'Query Length: {query_size}')
    
    plt.title('Single Query Speedup / Num. Blocks')
    plt.xlabel('Num. Blocks')
    plt.ylabel('Speedup')
    plt.axhline(1.0, color='#555', linestyle='--', linewidth=1)
    # plt.yscale('log', base=2)
    # plt.xscale('log', base=2)
    
    plt.savefig('./saves/speedup_report/plot_speedup_report.png', dpi=200, bbox_inches='tight')
    plt.savefig('./saves/speedup_report/plot_speedup_report.pdf', dpi=200, bbox_inches='tight')
    print('saved', './saves/speedup_report/plot_speedup_report.png')

def by_value(a, b):
    if isinstance(a, (tuple, list)):
        return pypareto.Domination.EQUAL

    if a > b:
        return pypareto.Domination.GREATER
    elif a < b:
        return pypareto.Domination.LESS
    else:
        return pypareto.Domination.EQUAL

def plot_ppl(query_size=1):
    path = './saves/speedup_report/result.json'
    with open(path, 'r') as f:
        data_latency = json.load(f)
    
    path = './saves/ppl_report/report.json'
    with open(path, 'r') as f:
        data_ppl = json.load(f)
    
    xs = []
    ys = []
    entries = []
    for entry_ppl in data_ppl.values():
        ys.append(entry_ppl['ppl'])
        k = entry_ppl["k"]
        block_size = entry_ppl["block_size"]
        latency_timber = data_latency[f'timber_q{query_size}_b{block_size}_k{k}']['latency']
        latency_base = data_latency[f'none_q{query_size}']['latency']
        entries.append({
            'k': k,
            'block_size': block_size,
        })
        xs.append(latency_base / latency_timber)
    
    pts = list(zip(xs, ys, map(lambda x: (x,), range(len(data_ppl)))))
    chain = pypareto.Comparison(by_value, pypareto.MaxMinList(pypareto.MaxMin.MAX, pypareto.MaxMin.MIN, pypareto.MaxMin.MIN)).as_chain()
    pts = chain.split_by_pareto(pts)[0]
    xs_front = [pt[0] for pt in pts]
    ys_front = [pt[1] for pt in pts]
    idxs_front = [pt[2][0] for pt in pts]
    
    plt.figure(figsize=(5, 4))
    
    plt.title(f'Perplexity / Decoding Speedup (#Query: {query_size})')
    plt.ylabel('PPL. (w/o train)')
    plt.xlabel('Decoding Speedup')
    sns.scatterplot(x=xs, y=ys)
    sns.lineplot(x=xs_front, y=ys_front)
    for idx in range(len(idxs_front)):
        plt.annotate(
            f'k:{entries[idxs_front[idx]]["k"]}, b:{entries[idxs_front[idx]]["block_size"]}', 
            (xs_front[idx], ys_front[idx] + 0.2),
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=9,
        )
    
    # baseline_ppl = 5.59
    baseline_ppl = 4.682
    plt.axhline(baseline_ppl, color='#555', linestyle='--', linewidth=1)
    plt.axvline(1.0, color='#555', linestyle='--', linewidth=1)
    plt.yscale('log', base=2)
    # plt.xscale('log', base=2)
    
    plt.savefig(f'./saves/speedup_report/plot_speedup_report_ppl_q{query_size}.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'./saves/speedup_report/plot_speedup_report_ppl_q{query_size}.pdf', dpi=200, bbox_inches='tight')
    print('saved', f'./saves/speedup_report/plot_speedup_report_ppl_q{query_size}.png')

def main():
    samples()
    plot()
    for q in query_sizes: plot_ppl(q)

if __name__ == '__main__':
    main()