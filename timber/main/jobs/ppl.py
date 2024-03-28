import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import argparse, json
from transformers import TextStreamer

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from timber.models.modeling_llama import LlamaForCausalLM, LlamaConfig
from timber.utils import seed, get_bench

def job_ppl(args, model, tokenizer, device, visualize):
    os.makedirs('./cache', exist_ok=True)
    cache_path = './cache/llama_eval.pth'
    if not os.path.exists(cache_path):
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
        torch.save(encodings, cache_path)
    else:
        encodings = torch.load(cache_path)

    max_length = model.config.max_position_embeddings
    max_length = stride = args.stride if args.stride > 0 else model.config.max_position_embeddings
    seq_len = encodings.size(1)

    nlls = []
    prev_end_loc = 0

    viz_i = 0
    sparse_sum = 0
    sparse_cnt = 0
    with tqdm(range(0, seq_len, stride)[:args.count]) as pbar:
        for begin_loc in pbar:
            if visualize and viz_i == 0:
                print("STORE FOR VISUALIZATION")
                os.environ['CHECKOUT_ENSEMBLE'] = '1'
                viz_i += 1
                
            else:
                os.environ['CHECKOUT_ENSEMBLE'] = '0'
                # print("QUIT!!!!!!!!!!!!!!")
                # return
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood.cpu())

            prev_end_loc = end_loc
            
            ppl = torch.exp(torch.stack(nlls).mean()).item()

            for layer in model.model.layers:
                sparsity = layer.self_attn.sparsity_per_layer
                if sparsity != None:
                    sparse_sum += sparsity
                else:
                    sparse_sum += 1
                sparse_cnt += 1

            pbar.set_description(f"ppl: {ppl:.3f} sparse: {sparse_sum/(sparse_cnt+1e-8):.2f}")

            if end_loc == seq_len:
                break
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    sparsity = sparse_sum/(sparse_cnt+1e-8)
    
    os.makedirs('./cache/llama_eval/', exist_ok=True)
    with open('./cache/llama_eval/ppl.json', 'w') as f:
        json.dump({'ppl': ppl}, f)

    print(f'PPL: {ppl:.4f} SPARSE: {sparsity:.3f}')