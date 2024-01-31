import json
import math
import sys, subprocess, os, itertools, tqdm
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

os.environ['PYTHONPATH'] = './'

block_sizes = [1, 2, 4, 8, 16]
ks = [128, 256, 512, 1024]

def samples():
    results = {}
    for block_size, k in tqdm.tqdm(
        list(itertools.product(block_sizes, ks)), 
        desc='exam', dynamic_ncols=True
    ):
        subprocess.call([
            'python', 'src/main/llama_eval.py', 
            '--method', 'tree',
            '--stride', '4096',
            '--block_size', str(block_size),
            '--k', str(k),
        ])
        with open('./cache/llama_eval/ppl.json', 'r') as f:
            ppl = json.load(f)['ppl']
        results[f'b{block_size}_k{k}'] = {
            'block_size': block_size,
            'k': k,
            'num_blocks': math.ceil(k / block_size),
            'ppl': ppl,
        }
    
    os.makedirs('./cache/ppl_report', exist_ok=True)
    with open('./cache/ppl_report/report.json', 'w') as f:
        json.dump(results, f, indent=2)

def plots():
    baseline_ppl = 5.59
    
    with open('./cache/ppl_report/report.json', 'r') as f:
        data = json.load(f)
    
    xs = [] # num blocks
    ys = [] # ppl
    
    for entry in data.values():
        xs.append(entry['num_blocks'])
        ys.append(entry['ppl'])
    
    plt.clf()
    
    plt.title('Perplexity / Num. Blocks')
    plt.xlabel('Num. Blocks')
    plt.ylabel('PPL.')
    plt.axhline(baseline_ppl, color='lightgray', linestyle='--', linewidth=1)
    sns.scatterplot(x=xs, y=ys)
    
    plt.savefig('./cache/ppl_report/plot_ppl_report.png', dpi=200, bbox_inches='tight')
    plt.savefig('./cache/ppl_report/plot_ppl_report.pdf', dpi=200, bbox_inches='tight')
    print('saved', './cache/ppl_report/plot_ppl_report.png')

def main():
    samples()
    plots()

if __name__ == '__main__':
    main()