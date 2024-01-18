import os
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import argparse

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from src.models.modeling_llama import LlamaForCausalLM, LlamaConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='none')
    parser.add_argument('--stride', type=int, default=-1)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    device = 'cuda:0'
    model_id = 'togethercomputer/LLaMA-2-7B-32K'
    
    config = LlamaConfig.from_pretrained(model_id)
    config._attn_implementation = config.attn_implementation = 'sdpa'
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=64,
        lora_alpha=32, 
        lora_dropout=0.1,
        modules_to_save=['tree_avgpool_scaler']
    )
    
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        config=config, 
        load_in_4bit=True,
        device_map={"" : device},
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_skip_modules=['tree_avgpool_scaler'],
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    for m in model.modules():
        if hasattr(m, 'attention_method'):
            m.attention_method = args.method
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location='cpu')['state_dict']
        keys = list(state_dict.keys())
        for key in keys:
            x = state_dict[key]
            state_dict[key.strip('model.')] = x
            del state_dict[key]
        model.load_state_dict(state_dict)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

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
    for begin_loc in tqdm(range(0, seq_len, stride)[:10]):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad(), torch.autocast('cuda', torch.float16):
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.cpu())

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean()).item()

    print('PPL:', ppl)