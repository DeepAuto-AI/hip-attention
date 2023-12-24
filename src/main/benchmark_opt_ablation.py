from .benchmark_bert import *
from ..models.perlin_attention import modules as perlin_modules

perlin_modules.CAUSAL_CONV_FORCE_NON_CAUSAL = False

def exam_appendix():
    fname='data_appendix.json'
    opt_model='facebook/opt-125m'
    TRACE = False
    
    nbf = 8
    k = 105
    w = 96
    seq_lens = [4096,]
    t_warmup = 1
    t_sample = 3
    
    data = {}
    
    for slen in seq_lens:
        # print(slen)
        latency, mem = exam_config(BenchConfig(
            method='perlin',
            bsize=1,
            seq_len=slen,
            k=k,
            w=w,
            nbf=nbf,
            trace=TRACE,
            causal=True,
            opt_model=opt_model,
            t_warmup=t_warmup,
            t_sample=t_sample
        ))
        name = f'perlin,nbf:{nbf},k:{k},w:{w},l:{slen}'
        entry = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
        print(name, entry)
        data[name] = entry
    
        latency, mem = exam_config(BenchConfig(
            method='none',
            bsize=1,
            seq_len=slen,
            k=64,
            w=128,
            nbf=1,
            trace=False,
            causal=True,
            n_hash=1,
            opt_model=opt_model,
        ))
        data[f'none,{slen}'] = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
        
        latency, mem = exam_config(BenchConfig(
            method='performer',
            bsize=1,
            seq_len=slen,
            k=64,
            w=128,
            nbf=nbf,
            trace=False,
            causal=True,
            opt_model=opt_model,
        ))
        data[f'performer,nbf:{nbf},slen:{slen}'] = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
    
    path = './plots/main/benchmark_opt_ablation'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, fname), 'w') as f:
        json.dump(data, f, indent=2)

def exam_table():
    fname='data_table.json'
    opt_model='facebook/opt-125m'
    TRACE = False
    
    nbf = 8
    k = 64
    w = 64
    seq_lens = [2048, 4096, 8192]
    t_warmup = 1
    t_sample = 3
    
    data = {}
    
    for slen in seq_lens:
        # print(slen)
        latency, mem = exam_config(BenchConfig(
            method='perlin',
            bsize=1,
            seq_len=slen,
            k=k,
            w=w,
            nbf=nbf,
            trace=TRACE,
            causal=True,
            opt_model=opt_model,
            t_warmup=t_warmup,
            t_sample=t_sample
        ))
        name = f'perlin,nbf:{nbf},k:{k},w:{w},l:{slen}'
        entry = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
        # print(name, entry)
        data[name] = entry
    
        latency, mem = exam_config(BenchConfig(
            method='none',
            bsize=1,
            seq_len=slen,
            k=64,
            w=128,
            nbf=1,
            trace=False,
            causal=True,
            n_hash=1,
            opt_model=opt_model,
        ))
        data[f'none,{slen}'] = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
        
        latency, mem = exam_config(BenchConfig(
            method='performer',
            bsize=1,
            seq_len=slen,
            k=64,
            w=128,
            nbf=nbf,
            trace=False,
            causal=True,
            opt_model=opt_model,
        ))
        data[f'performer,nbf:{nbf},slen:{slen}'] = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
    
    path = './plots/main/benchmark_opt_ablation'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, fname), 'w') as f:
        json.dump(data, f, indent=2)

def exam(fname='data.json', opt_model='facebook/opt-125m'):
    TRACE = os.environ.get('TRACE', '1') == '1'
    
    nbfs = [1, 2, 4, 8]
    perlin_nbf = [8]
    ks = [32, 64, 128]
    ws = [64, 128, 256, 384]
    n_hashs = [2, 4, 8, 16]
    # nbfs = [8]
    # ks = [128]
    # ws = [384]
    
    data = {}
    
    for nbf in perlin_nbf:
        for k in ks:
            for w in ws:
                latency, mem = exam_config(BenchConfig(
                    method='perlin',
                    bsize=1,
                    seq_len=2048,
                    k=k,
                    w=w,
                    nbf=nbf,
                    trace=TRACE,
                    causal=True,
                    opt_model=opt_model,
                ))
                name = f'perlin,nbf:{nbf},k:{k},w:{w}'
                entry = {
                    'latency': latency * 1000, 
                    'mem': mem / (1024 ** 2),
                }
                print(name, entry)
                data[name] = entry
    
    latency, mem = exam_config(BenchConfig(
        method='cosformer',
        bsize=1,
        seq_len=2048,
        k=64,
        w=128,
        nbf=1,
        trace=False,
        causal=True,
        n_hash=1,
        opt_model=opt_model,
    ))
    data[f'cosformer'] = {
        'latency': latency * 1000, 
        'mem': mem / (1024 ** 2),
    }
    
    latency, mem = exam_config(BenchConfig(
        method='none',
        bsize=1,
        seq_len=2048,
        k=64,
        w=128,
        nbf=1,
        trace=False,
        causal=True,
        n_hash=1,
        opt_model=opt_model,
    ))
    data[f'none'] = {
        'latency': latency * 1000, 
        'mem': mem / (1024 ** 2),
    }
    
    for nbf in nbfs:
        latency, mem = exam_config(BenchConfig(
            method='performer',
            bsize=1,
            seq_len=2048,
            k=64,
            w=128,
            nbf=nbf,
            trace=False,
            causal=True,
            opt_model=opt_model,
        ))
        data[f'performer,nbf:{nbf}'] = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
    
    for n_hash in n_hashs:
        latency, mem = exam_config(BenchConfig(
            method='reformer',
            bsize=1,
            seq_len=2048,
            k=64,
            w=128,
            nbf=nbf,
            trace=False,
            causal=True,
            n_hash=n_hash,
            opt_model=opt_model,
        ))
        data[f'reformer,n_hash:{n_hash}'] = {
            'latency': latency * 1000, 
            'mem': mem / (1024 ** 2),
        }
    
    path = './plots/main/benchmark_opt_ablation'
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, fname), 'w') as f:
        json.dump(data, f, indent=2)

def main():
    exam_appendix()
    # exam_table()
    # exam('data.json', 'facebook/opt-125m')
    # exam('data_350m.json', 'facebook/opt-350m')
    # exam('data_1.3b.json', 'facebook/opt-1.3b')
    # exam('data_2.7b.json', 'facebook/opt-2.7b')

if __name__ == '__main__':
    main()