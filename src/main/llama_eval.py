import os
import torch
import transformers
from src.models.tree_llama.modeling_flash_llama import LlamaForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse
from transformers import (
    BitsAndBytesConfig,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='none')
    parser.add_argument('--stride', type=int, default=-1)
    args = parser.parse_args()
    
    device = 'cuda:0'
    model = LlamaForCausalLM.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K", 
        load_in_4bit=True,
        device_map={"" : "cuda:0"},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
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
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K",
    )

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
    # max_length = stride = 512
    seq_len = encodings.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    print(ppl)