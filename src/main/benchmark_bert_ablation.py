from .benchmark_bert import *

def main():
    TRACE = True
    
    BSIZE = 96
    
    nbfs = [1, 2, 4, 8]
    ks = [7, 13, 25]
    ws = [32, 64, 128]
    
    data = {}
    
    latency, mem = exam_config(BenchConfig(
        method='none',
        bsize=BSIZE,
        seq_len=256,
        k=7,
        w=128,
        nbf=1,
        trace=False,
    ))
    data[f'none'] = {
        'method': 'none',
        'latency': latency * 1000, 
        'mem': mem / (1024 ** 2),
    }
    
    for nbf in nbfs:
        for k in ks:
            for w in ws:
                latency, mem = exam_config(BenchConfig(
                    method='perlin',
                    bsize=BSIZE,
                    seq_len=256,
                    k=k,
                    w=w,
                    nbf=nbf,
                    trace=TRACE,
                ))
                data[f'perlin,nbf:{nbf},k:{k},w:{w}'] = {
                    'method': 'perlin',
                    'latency': latency * 1000, 
                    'mem': mem / (1024 ** 2),
                }
    
    for baseline in BASELINES:
        if baseline == 'performer':
            for nbf in nbfs:
                latency, mem = exam_config(BenchConfig(
                    method=baseline,
                    bsize=BSIZE,
                    seq_len=256,
                    k=7,
                    w=128,
                    nbf=nbf,
                    trace=False,
                ))
                data[f'{baseline},nbf:{nbf}'] = {
                    'method': baseline,
                    'latency': latency * 1000, 
                    'mem': mem / (1024 ** 2),
                }
        elif baseline == 'cosformer':
            latency, mem = exam_config(BenchConfig(
                method=baseline,
                bsize=BSIZE,
                seq_len=256,
                k=7,
                w=128,
                nbf=1,
                trace=False,
            ))
            data[f'{baseline}'] = {
                'method': baseline,
                'latency': latency * 1000, 
                'mem': mem / (1024 ** 2),
            }
        else:
            for k in ks:
                latency, mem = exam_config(BenchConfig(
                    method=baseline,
                    bsize=BSIZE,
                    seq_len=256,
                    k=k,
                    w=128,
                    nbf=1,
                    trace=False,
                ))
                data[f'{baseline},k:{k}'] = {
                    'method': baseline,
                    'latency': latency * 1000, 
                    'mem': mem / (1024 ** 2),
                }
    
    path = './plots/main/benchmark_bert_ablation'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'data.json'), 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':
    main()