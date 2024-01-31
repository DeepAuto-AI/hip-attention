import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import argparse
from transformers import TextStreamer

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from src.models.modeling_llama import LlamaForCausalLM, LlamaConfig
from src.utils import seed, get_bench

from src.main.jobs.bench_single_layer import job_bench_single_layer
from src.main.jobs.ppl import job_ppl
from src.main.jobs.stream import job_stream
from src.main.jobs.mmlu import job_mmlu

def load_model(args):
    device = 'cuda:0'
    model_id = 'togethercomputer/LLaMA-2-7B-32K'
    
    config = LlamaConfig.from_pretrained(model_id)
    config._attn_implementation = config.attn_implementation = 'sdpa'
    
    infer_dtype = torch.bfloat16
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        config=config, 
        load_in_4bit=True,
        device_map={"" : device},
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_skip_modules=['tree_avgpool_scaler'],
            bnb_4bit_compute_dtype=infer_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=infer_dtype,
        trust_remote_code=True,
    )
    
    for m in model.modules():
        if hasattr(m, 'attention_method'):
            m.attention_method = args.method
            m.tree_k = args.k
            m.tree_block_size = args.block_size
            m.tree_using_context_avg = True
            m.tree_dense_queries = args.dense_queries
    
    if args.method != 'none' and args.checkpoint is not None:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=True,
            r=args.lora_r,
            lora_alpha=32, 
            lora_dropout=0.0,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj', 
                'gate_proj', 'up_proj', 'down_proj', 
                # 'input_layernorm', 'post_attention_layernorm'
            ],
            modules_to_save=[
                'tree_avgpool_scaler',
                'input_layernorm', 'post_attention_layernorm'
            ]
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        state_dict = torch.load(args.checkpoint, map_location='cpu')['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            x = state_dict[key]
            state_dict[key.strip('model.')] = x
            del state_dict[key]
        model.load_state_dict(state_dict)
        model = model.to(infer_dtype)
        print('lora checkpoint loaded from', args.checkpoint)
    elif args.method != 'none':
        for m in model.modules():
            if hasattr(m, 'attention_method'):
                m.tree_using_context_avg = False
    
    model = model.eval()
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer, device

def main():
    seed()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=str, default='ppl')
    parser.add_argument('--method', type=str, default='none')
    parser.add_argument('--stride', type=int, default=-1)
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--block_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--k', type=int, default=512)
    parser.add_argument('--dense_queries', type=int, default=2048)
    args = parser.parse_args()
    
    assert args.job in ['ppl', 'stream', 'mmlu', 'bench_single_layer']
    
    model, tokenizer, device = load_model(args)

    if args.job == 'ppl':
        job_ppl(args, model, tokenizer, device)
    elif args.job == 'stream':
        job_stream(args, model, tokenizer, device)
    elif args.job == 'mmlu':
        job_mmlu(args, model, tokenizer, device)
    elif args.job == 'bench_single_layer':
        job_bench_single_layer(args, model, tokenizer, device)
    else:
        raise Exception()

if __name__ == '__main__':
    main()