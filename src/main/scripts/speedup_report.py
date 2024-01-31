import json
import math
import sys, subprocess, os, itertools, tqdm
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

os.environ['PYTHONPATH'] = './'

block_sizes = [1, 2, 4, 8, 16]
query_sizes = [1, 2, 4, 8, 16]
ks = [128, 256, 512, 1024]

def samples():
    num_samples = 200
    batch_size = 128
    results = {}
    
    for query_size, block_size, k in tqdm.tqdm(
        list(itertools.product(query_sizes, block_sizes, ks)),
        dynamic_ncols=True, desc='eval',
    ):
        subprocess.call([
            'python', 'src/models/tree_attention/attention1_block_gpu.py',
            '--method', 'tree',
            '--block_size', str(block_size),
            '--k', str(k),
            '--query_size', str(query_size),
            '--dups', '4',
            '--batch_size', str(batch_size),
            '--samples', str(num_samples),
        ])
        with open('./cache/attention1_block_gpu/result.json', 'r') as f:
            latency = json.load(f)['mean']
        results[f'tree_q{query_size}_b{block_size}_k{k}'] = {
            'query_size': query_size,
            'k': k,
            'block_size': block_size,
            'latency': latency,
            'method': 'tree',
        }
    
    for query_size in tqdm.tqdm(query_sizes, dynamic_ncols=True, desc='none'):
        subprocess.call([
            'python', 'src/models/tree_attention/attention1_block_gpu.py',
            '--method', 'none',
            '--query_size', str(query_size),
            '--dups', '4',
            '--batch_size', str(batch_size),
            '--samples', str(num_samples),
        ])
        with open('./cache/attention1_block_gpu/result.json', 'r') as f:
            latency = json.load(f)['mean']
        results[f'none_q{query_size}'] = {
            'query_size': query_size,
            'latency': latency,
            'method': 'none'
        }
    
    os.makedirs('./cache/speedup_report', exist_ok=True)
    path = './cache/speedup_report/result.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('saved', path)

def plot():
    path = './cache/speedup_report/result.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    plt.clf()
    
    for query_size in query_sizes:
        xs = []
        ys = []
        for block_size, k in itertools.product(block_sizes, ks):
            tree_entry = data[f'tree_q{query_size}_b{block_size}_k{k}']
            base_entry = data[f'none_q{query_size}']
            num_blocks = k / block_size
            speedup = base_entry['latency'] / tree_entry['latency']
            xs.append(num_blocks)
            ys.append(speedup)
        sns.scatterplot(x=xs, y=ys, label=f'Query Length: {query_size}')
    
    plt.title('Speedup / Num. Blocks')
    plt.xlabel('Num. Blocks')
    plt.ylabel('Speedup')
    plt.axhline(1.0, color='lightgray', linestyle='--', linewidth=1)
    
    plt.savefig('./cache/speedup_report/plot_speedup_report.png', dpi=200, bbox_inches='tight')
    plt.savefig('./cache/speedup_report/plot_speedup_report.pdf', dpi=200, bbox_inches='tight')
    print('saved', './cache/speedup_report/plot_speedup_report.png')

def plot_ppl():
    path = './cache/speedup_report/result.json'
    with open(path, 'r') as f:
        data_latency = json.load(f)
    
    path = './cache/ppl_report/report.json'
    with open(path, 'r') as f:
        data_ppl = json.load(f)

def main():
    samples()
    plot()

if __name__ == '__main__':
    main()