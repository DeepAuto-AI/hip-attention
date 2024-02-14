import subprocess
import os, json

from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

dups = range(1, 17)

def samples():
    block_size = 16
    block_size_ks = [1, 2, 4, 8, 16]
    query_size = 1
    k = 512
    batch_sizes = {
        1: 256,
        4: 256,
        5: 192,
        6: 160,
        7: 160,
        8: 128,
        9: 96,
        10: 96,
        11: 64,
        12: 64,
        13: 64,
        14: 64,
        16: 64,
    }
    num_samples = 200
    
    batch_size = max(list(batch_sizes.values()))
    results = {}
    for dup in dups:
        if dup in batch_sizes:
            batch_size = batch_sizes[dup]
        
        latency_timbers = []
        for block_size_k in block_size_ks:
            cmd = [
                'python', 'timber/models/timber_attention/attention1_block_gpu.py',
                '--method', 'timber',
                '--block_size_q', str(block_size),
                '--block_size_k', str(block_size_k),
                '--k', str(k),
                '--query_size', str(query_size),
                '--dups', str(dup),
                '--batch_size', str(batch_size),
                '--samples', str(num_samples),
            ]
            print(' '.join(cmd))
            subprocess.call(cmd)
            with open('./cache/attention1_block_gpu/result.json', 'r') as f:
                latency_timber = json.load(f)['mean']
            latency_timbers.append(latency_timber)
        
        subprocess.call([
            'python', 'timber/models/timber_attention/attention1_block_gpu.py',
            '--method', 'none',
            '--block_size_q', str(block_size),
            '--block_size_k', str(block_size),
            '--k', str(k),
            '--query_size', str(query_size),
            '--dups', str(dup),
            '--batch_size', str(batch_size),
            '--samples', str(num_samples),
        ])
        with open('./cache/attention1_block_gpu/result.json', 'r') as f:
            latency_base = json.load(f)['mean']
        
        seq_len = dup * 1024
        results[f's{seq_len}'] = {
            'block_size_ks': block_size_ks,
            'latency_timbers': latency_timbers,
            'latency_base': latency_base,
            'batch_size': batch_size,
            'query_size': query_size,
            'seq_len': seq_len,
            'dups': dup,
            'k': k,
            'speedup': latency_base / latency_timber,
        }
    
    os.makedirs('./saves/seqlen_speed_report', exist_ok=True)
    path = './saves/seqlen_speed_report/result.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('dumped', path)

def plot():
    path = './saves/seqlen_speed_report/result.json'
    with open(path, 'r') as f:
        data = json.load(f)
    
    block_size_ks = data[list(data.keys())[0]]['block_size_ks']
    
    xss = []
    ys_bases = []
    ys_timbers = []
    ys_speedups = []
    
    for iks, block_size_k in enumerate(block_size_ks):
        xs = []
        ys_base = []
        ys_timber = []
        ys_speedup = []
        for dup in dups:
            entry = data[f's{dup * 1024}']
            xs.append(entry['seq_len'])
            ys_base.append(entry['latency_base'] / entry['batch_size'] * 1000)
            ys_timber.append(entry['latency_timbers'][iks] / entry['batch_size'] * 1000)
            ys_speedup.append(entry['latency_base'] / entry['latency_timbers'][iks])
        xss.append(xs)
        ys_bases.append(ys_bases)
        ys_timbers.append(ys_timber)
        ys_speedups.append(ys_speedup)
    
    plt.figure(figsize=(5,4))
    
    sns.lineplot(x=xs, y=ys_base, label='baseline')
    for iks, block_size_k in enumerate(block_size_ks):
        sns.lineplot(x=xs, y=ys_timbers[iks], label=f'timber (bk={block_size_k})')
    plt.legend()
    plt.title('Single Query Latency (k=512, bq=16)')
    plt.xlabel('Seq. Length')
    plt.ylabel('Latency (us)')
    plt.xlim(0, 17*1024)
    
    fig_path = './saves/seqlen_speed_report/plot_seqlen_latency'
    plt.savefig(fig_path + '.png', dpi=200, bbox_inches='tight')
    plt.savefig(fig_path + '.pdf', dpi=200, bbox_inches='tight')
    print(f'saved {fig_path}.png')
    
    plt.figure(figsize=(5,4))
    
    plt.title('Single Query Speedup (k=512, bq=16)')
    sns.lineplot(x=xs, y=[1.0,] * len(xs), label='baseline')
    for iks, block_size_k in enumerate(block_size_ks):
        sns.lineplot(x=xs, y=ys_speedups[iks], label=f'timber (bk={block_size_k})')
    plt.xlabel('Seq. Length')
    plt.ylabel('Speedup')
    plt.xlim(0, 17*1024)
    
    fig_path = './saves/seqlen_speed_report/plot_seqlen_speedup'
    plt.savefig(fig_path + '.png', dpi=200, bbox_inches='tight')
    plt.savefig(fig_path + '.pdf', dpi=200, bbox_inches='tight')
    print(f'saved {fig_path}.png')

def main():
    samples()
    plot()

if __name__ == '__main__':
    main()