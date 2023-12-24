"""
Benchmark the perlin bidirectional attention.
Usage: python -m src.main.tests.test_perlin_benchmark
NOTE: this test script is not under maintain.
"""

import os, tqdm, gc
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
from transformers import logging
logging.set_verbosity_error()

import torch, time, random, json
from transformers import AutoConfig
from ...models.hf_bert import BertLayer
from ...models.hf_bert import BertModel as TeacherBertModel
from ...models import perlin_bert
from ...models.perlin_bert import BertModel, BertSelfAttention
from ...models.perlin_attention.config import PerlinAttentionConfig, register_default_config
from ...utils import seed, get_bench

def main():
    get_bench().synchronize = True
    T_WARMUP = 2
    T_SAMPLE = 5
    BENCH_PRECISION = torch.float32
    BSIZE = 1
    SEQ_LEN = 4096
    layerwise = True
    BENCHMARK = True
    
    print(f"config(fp={BENCH_PRECISION}, bsize={BSIZE}, seq_len={SEQ_LEN})")
    
    if get_bench().synchronize:
        print('WARN: benchmark timer is synchronized. therefore the total latency will not be correct!')
    device = torch.device('cuda')

    config = AutoConfig.from_pretrained('bert-base-uncased')
    config.max_position_embeddings = 4096
    teacher = TeacherBertModel(config).to(device).eval()

    register_default_config(PerlinAttentionConfig(
        performer_nb_factor=8,
        lora_enabled=False,
        lora_in_approx_enabled=False,
        partial_attention_scaler=True,
        k_flatten=True,
        k=32,
    ))
    perlin = BertModel(config).to(device).eval()
    for module in perlin.modules():
        if isinstance(module, BertSelfAttention):
            module.perlin_token_merging = False
            module.perlin_token_merging_preserve_ratio = 0.2
            module.perlin_token_merging_ratio = 0.5
            module.perlin_token_merging_score_source = 'probs'
            module.attention_method = 'perlin'
        if hasattr(module, 'benchmarking'):
            module.benchmarking = BENCHMARK

    register_default_config(PerlinAttentionConfig(
        performer_nb_factor=1,
    ))
    performer = BertModel(config).to(device).eval()
    for module in performer.modules():
        if isinstance(module, BertSelfAttention):
            module.attention_method = 'performer'
        if hasattr(module, 'benchmarking'):
            module.benchmarking = BENCHMARK


    input_ids = torch.randint(0, 10000, (BSIZE, SEQ_LEN)).to(device)
    attention_mask = torch.ones((BSIZE, SEQ_LEN)).to(device)
    for i in range(attention_mask.shape[0]):
        attention_mask[i, random.randint(128, attention_mask.shape[1]-1):] = 0

    if not layerwise:
        with torch.no_grad(), torch.autocast('cuda', BENCH_PRECISION):
            output_teacher = teacher(input_ids=input_ids, attention_mask=attention_mask)
            output = perlin(input_ids=input_ids, attention_mask=attention_mask, teacher=teacher)
            output_perf = performer(input_ids=input_ids, attention_mask=attention_mask, teacher=teacher)
    torch.cuda.synchronize()

    def bench(name, fn):
        sample_count = 0
        try:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.max_memory_allocated()
            torch.cuda.synchronize()
            print(f'[{name}] warmup... ', end = '', flush=True)
            t = time.time()
            while True:
                with torch.no_grad(), torch.autocast('cuda', BENCH_PRECISION):
                    fn()
                if time.time() - t > T_WARMUP:
                    break
            torch.cuda.synchronize()
            print('benchmarking', end = '', flush=True)
            t = time.time()
            last_report = time.time()
            while True:
                with torch.no_grad(), torch.autocast('cuda', BENCH_PRECISION):
                    fn()
                sample_count += 1
                if time.time() - t > T_SAMPLE:
                    break
                if time.time() - last_report > 0.5:
                    last_report = time.time()
                    print('.', end='', flush=True)
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() - start_mem
        except torch.cuda.OutOfMemoryError as ex: #type: ignore
            mem = 0
        elapsed = time.time() - t
        interval = elapsed/(sample_count + 1e-8)
        print(f' done. sampled {sample_count}its. {interval*1000:.2f}ms/it {mem // 1024 // 1024} MB', flush=True)
        return interval, mem
    
    hidden_states = torch.randn((BSIZE, SEQ_LEN, teacher.config.hidden_size), device=device, dtype=BENCH_PRECISION)
    attention_mask_expand = attention_mask.view(BSIZE, 1, 1, -1).contiguous()
    
    def test_bert():
        if layerwise:
            layer = teacher.encoder.layer[0] # type: BertLayer
            layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)
        else:
            teacher(input_ids=input_ids, attention_mask=attention_mask)
    
    def test_perlin():
        if layerwise:
            layer = perlin.encoder.layer[0] # type: BertLayer
            layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)
        else:
            perlin(input_ids=input_ids, attention_mask=attention_mask, teacher=teacher)
    
    def test_performer():
        if layerwise:
            layer = performer.encoder.layer[0] # type: BertLayer
            layer(hidden_states=hidden_states, attention_mask=attention_mask_expand)
        else:
            performer(input_ids=input_ids, attention_mask=attention_mask, teacher=teacher)
    
    t_perlin, m_perlin = bench("perlin", test_perlin)
    t_performer, m_performer = bench("performer", test_performer)
    t_bert, m_bert = bench("bert", test_bert)

    bench_result = get_bench().todict()
    print(
        # output.last_hidden_state.shape, 
        # output_teacher.last_hidden_state.shape, 
        json.dumps({k: (v/bench_result['perlin'])*100 for k, v in bench_result.items()}, indent=2),
        f'time_bert: {t_bert*1000}ms/it, mem_bert: {m_bert // 1024 // 1024}MB', 
        f'time peformer: {t_performer*1000}ms/it, mem_performer: {m_performer // 1024 // 1024}MB', 
        f'time_perlin: {t_perlin*1000}ms/it, mem_perlin: {m_perlin // 1024 // 1024}MB', 
        f'speedup w.r.t performer: {t_performer / t_perlin}',
        f'speedup w.r.t bert: {t_bert / t_perlin}',
        f'max_mem: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB',
        # f'error: {torch.nn.functional.mse_loss(output.last_hidden_state, output_teacher.last_hidden_state)}',
        sep='\n'
    )
    
    return t_bert, t_performer, t_perlin

if __name__ == '__main__':
    seed()
    main()