import copy
import math
import os
import random
import numpy as np
import torch, transformers
from typing import List, Tuple, Dict, Optional
import tqdm
import json
from hip.dataset.calib_loft_rag import (
    prefix,
    rag_qa_pairs,
)
from hip import HiPAttentionArgs11
import matplotlib.pyplot as plt

def load_loft_rag_chat_corpus() -> List[str]:
    lines = []
    for qa in rag_qa_pairs:
        for answer in qa["answers"]:
            lines.append((f'{prefix} {qa["query_text"]}', f'Final Answer: [\'{answer}\']'))
    return lines

def format_chat_corpus(args, corpus: List[str]) -> List[str]:
    if 'llama3' in args.model:
        ret = []
        for prompt, output in corpus:
            updated = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|><|end_of_text|>"""
            ret.append(updated)
        return ret
    else:
        raise Exception()

def tokenizing_and_find_shared_prompt_on_corpus(
    tokenizer: transformers.LlamaTokenizer, 
    corpus: List[str]
):
    count = len(corpus)
    iterator = map(lambda line: tokenizer.encode(line, add_special_tokens=False), corpus)
    corpus = []
    shared_prompt = None
    for input_ids in tqdm.tqdm(iterator, total=count):
        if shared_prompt is None:
            shared_prompt = input_ids
        else:
            for i, (s, c) in enumerate(zip(shared_prompt, input_ids)):
                if s != c:
                    shared_prompt = shared_prompt[:i]
        corpus.append(input_ids)
    corpus = list(map(lambda ids: torch.tensor(ids[len(shared_prompt):]), corpus))
    shared_prompt = torch.tensor(shared_prompt)
    return shared_prompt, corpus

@torch.inference_mode
def evaluate_corpus(model, shared_output, corpus: List[torch.Tensor]):
    loss_sum = 0
    for input_ids in corpus:
        input_ids = input_ids.to(model.device, non_blocking=True)
        output = model(
            input_ids=input_ids.unsqueeze(0),
            labels=input_ids,
            past_key_values=copy.deepcopy(shared_output.past_key_values)
        )
        loss_sum += output.loss
        torch.cuda.synchronize()
    ppl = math.exp(loss_sum.item() / len(corpus))
    return ppl

from hip.models.hip_attention.attention2_draft_sampling_extend import ScanStage, dual_stage_quadratic_hip_attention

@torch.inference_mode
def job_ga(
    args, 
    model, 
    tokenizer: transformers.LlamaTokenizer,
    device, 
):
    seed = [
        [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=256,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=32,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=8,
                stage_k=8192,
                stage_stride=1,
            ),
        ] for _ in range(model.config.num_hidden_layers)
    ]
    
    def mutate_inner(p):
        p = copy.deepcopy(p)
        
        stage_job = random.choice([
            'pass', 
            'swap_layer', 
            'swap_stage',
            'copy_stage',
            'drop_stage',
        ])
        if stage_job == 'pass':
            pass
        elif stage_job == 'swap_layer':
            a = random.randint(0, len(p) - 1)
            b = random.randint(0, len(p) - 1)
            layer_a = p[a]
            layer_b = p[b]
            p[a] = layer_b
            p[b] = layer_a
        elif stage_job == 'swap_stage':
            target_layer = random.randint(0, len(p) - 1)
            layer = p[target_layer]
            if len(layer) > 2:
                a = random.randint(1, len(layer) - 1)
                b = random.randint(1, len(layer) - 1)
                stage_a = layer[a]
                stage_b = layer[b]
                layer[a] = stage_b
                layer[b] = stage_a
        elif stage_job == 'copy_stage':
            target_layer = random.randint(0, len(p) - 1)
            layer = p[target_layer] # type: list
            if len(layer) > 2:
                a = random.randint(1, len(layer) - 1)
                stage = layer[a]
                layer.insert(a, copy.deepcopy(stage))
        elif stage_job == 'drop_stage':
            target_layer = random.randint(0, len(p) - 1)
            layer = p[target_layer]
            if len(layer) > 2:
                layer.pop(-1)
        else:
            raise Exception()

        num_param_jobs = random.randint(0, 10)
        for _ in range(num_param_jobs):
            param_job = random.choice([
                'pass', 
                'block_size_q', 
                'block_stride_q', 
                'chunk_size', 
                'k', 
                'stride'
            ])
            target_layer = random.randint(0, len(p) - 1)
            layer = p[target_layer]
            target_stage = random.randint(0, len(layer) - 1)
            stage = layer[target_stage]
            
            if param_job == 'pass':
                pass
            elif param_job == 'block_size_q':
                if random.random() > 0.5:
                    stage.stage_block_size_q *= 2
                else:
                    stage.stage_block_size_q //= 2
            elif param_job == 'block_stride_q':
                if random.random() > 0.5:
                    stage.stage_block_stride_q *= 2
                else:
                    stage.stage_block_stride_q //= 2
            elif param_job == 'k':
                if stage.stage_k is not None:
                    if random.random() > 0.5:
                        stage.stage_k *= 2
                    else:
                        stage.stage_k //= 2
            elif param_job == 'chunk_size':
                if random.random() > 0.5:
                    stage.stage_chunk_size *= 2
                else:
                    stage.stage_chunk_size //= 2
            elif param_job == 'stride':
                if random.random() > 0.5:
                    stage.stage_stride *= 2
                else:
                    stage.stage_stride //= 2
            else:
                raise Exception()
        return p

    def validate(p):
        assert len(p) == model.config.num_hidden_layers, 'layer count must match'
        for layer in p:
            assert len(layer) >= 2, 'too small'
            assert len(layer) < 5, 'too large'
            assert layer[0].stage_k == None, 'first quadratic'
            assert layer[-1].stage_chunk_size <= 32, 'too large k'
            
            stage_block_size_q = 987654321
            stage_stride = 987654321
            stage_k = 987654321
            stage_chunk_size = 987654321
            
            for stage in layer:
                assert (stage.stage_k is None) or (stage.stage_k > 0)
                assert stage.stage_stride > 0
                assert stage.stage_block_size_q > 0
                assert stage.stage_block_stride_q > 0
                assert stage.stage_chunk_size > 0
                assert (stage.stage_k is None) or (stage.stage_k <= 131072)
                assert stage.stage_stride <= 16
                assert stage.stage_block_size_q <= 512
                assert stage.stage_block_stride_q <= 16
                assert stage.stage_chunk_size <= 512
                assert (stage.stage_block_size_q // stage.stage_block_stride_q) >= 16
                
                assert (stage.stage_k is None) or (stage.stage_k <= stage_k)
                assert stage.stage_stride <= stage_stride
                assert stage.stage_block_size_q <= stage_block_size_q
                assert stage.stage_chunk_size <= stage_chunk_size
                
                stage_block_size_q = stage.stage_block_size_q
                stage_stride = stage.stage_stride
                stage_chunk_size = stage.stage_chunk_size
                if stage.stage_k is not None:
                    stage_k = stage.stage_k

    def mutate(p):
        while True:
            try:
                p1 = mutate_inner(p)
                validate(p1)
                return p1
            except AssertionError as ex:
                # print(ex)
                pass
    
    def crossover(p1, p2):
        assert len(p1) == len(p2)
        pt = random.randint(0, len(p1))
        
        p1_top = p1[:pt]
        p1_bot = p2[pt:]
        
        p2_top = p1[pt:]
        p2_bot = p2[:pt]
        
        p1_new = copy.deepcopy(p1_top) + copy.deepcopy(p1_bot)
        p2_new = copy.deepcopy(p2_top) + copy.deepcopy(p2_bot)
        assert len(p1_new) == len(p1)
        assert len(p2_new) == len(p2)
        
        return p1_new, p2_new

    def apply_setting(model, setting):
        i = 0
        for m in model.modules():
            if hasattr(m, 'tree_extend_stages'):
                m.tree_extend_stages = setting[i]
                i += 1

    def evaluate_latency(model, stages):
        device = model.device
        HEAD = 32
        HEAD_KV = 8
        TDST = 8192
        TSRC = 131072
        HID = 128
        q = torch.zeros((1, TDST, HEAD, HID), dtype=torch.bfloat16, device=device)
        k = torch.zeros((1, TSRC, HEAD_KV, HID), dtype=torch.bfloat16, device=device)
        v = torch.zeros((1, TSRC, HEAD_KV, HID), dtype=torch.bfloat16, device=device)
        latency_sum = 0
        latency_count = 0
        for i in range(6):
            if i == 2:
                latency_sum = latency_count = 0
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)
            
            start.record()
            dual_stage_quadratic_hip_attention(
                q, k, v, 
                args=HiPAttentionArgs11(
                    block_size_k=64,
                    block_stride_k=1,
                    sliding_window_size=1024,
                    sink_token_size=256,
                ),
                second_stage_k=2048,
                stages=stages,
                block_sparse_block_size_q=64,
                model_context_length=131072,
                scan_extend_backend='relative',
                sa_extend_backend='streaming',
            )
            end.record()
            
            end.synchronize()
            latency_sum += start.elapsed_time(end)
            latency_count += 1
        return latency_sum / latency_count
    
    def evaluate_population(population, model, shared_output, corpus):
        scores = []
        for p in tqdm.tqdm(population, desc='eval', leave=False, dynamic_ncols=True):
            apply_setting(model, p)
            ppl = evaluate_corpus(model, shared_output, corpus)
            latency_sum = 0
            latency_count = 0
            for stage in p:
                latency_sum += evaluate_latency(model, stage)
                latency_count += 1
            scores.append((latency_sum / latency_count, ppl,))
        return scores

    # settings
    num_population = 25
    num_corpus = 50
    
    # prepare shared prompt
    corpus = load_loft_rag_chat_corpus()[:num_corpus]
    corpus = format_chat_corpus(args, corpus)
    shared_prompt, corpus = tokenizing_and_find_shared_prompt_on_corpus(
        tokenizer, corpus
    )
    
    shared_output = model(
        input_ids=shared_prompt.unsqueeze(0),
        use_cache=True
    )
    
    # run GA
    population = [seed]
    for _ in range(num_population):
        t = copy.deepcopy(seed)
        for _ in range(10):
            t = mutate(t)
        population.append(t)
    scores = evaluate_population(population, model, shared_output, corpus)
    seed_score = copy.deepcopy(scores[0])
    print('seed', seed_score)
    
    current_generation = 0
    
    while True:
        new_populations = []
        for _ in range(num_population):
            p1 = random.choice(population)
            p2 = random.choice(population)
            p1, p2 = crossover(p1, p2)
            p1 = mutate(p1)
            p2 = mutate(p2)
            new_populations.append(p1)
            new_populations.append(p2)
        new_scores = evaluate_population(new_populations, model, shared_output, corpus)
        # print(min(map(lambda x:x[1], scores)), min(map(lambda x:x[1], new_scores)), new_scores)
        
        population = population + new_populations
        scores = scores + new_scores
        
        # just check scores
        # best_args = list(map(lambda x: x[0], list(sorted(zip(range(len(scores)), scores), key=lambda x: x[1][1], reverse=False))[:num_population]))
        
        # pareto front
        import pypareto
        values = scores
        chain = pypareto.Comparison(pypareto.by_value, pypareto.MaxMinList(pypareto.MaxMin.MIN, pypareto.MaxMin.MIN,)).as_chain()
        best_scores = chain.split_by_pareto(values)
        
        plt.clf()
        for line in best_scores:
            plt.plot(
                list(map(lambda x: x[0], sorted(line, key=lambda x:x[0]))), 
                list(map(lambda x: x[1], sorted(line, key=lambda x:x[0])))
            )
        plt.scatter(x=[seed_score[0]], y=[seed_score[1]], marker='s')
        plt.grid()
        plt.savefig('dummy_pareto.png')
        
        best_scores = sum(best_scores, [])[:num_population]
        best_args = []
        for b in best_scores:
            best_args.append(scores.index(b))
        
        population = np.array(population, dtype=object)[np.array(best_args)].tolist()
        scores = np.array(scores, dtype=object)[np.array(best_args)].tolist()
        # print(population[0])
        scores = list(map(tuple, scores))
        os.makedirs('./saves/pareto', exist_ok=True)
        with open('./saves/pareto/population.json', 'w') as f:
            import dataclasses, json

            class EnhancedJSONEncoder(json.JSONEncoder):
                def default(self, o):
                    if dataclasses.is_dataclass(o):
                        return dataclasses.asdict(o)
                    return super().default(o)
            
            json.dump({
                'generation': current_generation,
                'population': population, 
                'scores': scores, 
            }, f, indent=2, cls=EnhancedJSONEncoder)
        print(scores[0])
        
        apply_setting(model, population[0])
        
        shared_output = model(
            input_ids=shared_prompt.unsqueeze(0),
            use_cache=True
        )
        current_generation += 1