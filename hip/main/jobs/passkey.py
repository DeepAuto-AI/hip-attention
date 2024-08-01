import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import json
import numpy as np
from hip.dataset.passkey import Passkey

def get_numbers(s):
    lst = [c for c in s if c.isdigit()]
    # print(lst, s)
    return ''.join(lst)

def job_passkey(args, model, tokenizer, device):
    dataset = Passkey(tokenizer, batch_size=args.batch_size)
    
    accuracy = {}
    
    for j, (input_ids, target_ids) in enumerate(tqdm(dataset, dynamic_ncols=True, leave=False)):
        input_ids = input_ids.cuda()
        target_ids = target_ids.cuda()
        
        with torch.no_grad():
            output = model.generate(
                input_ids, 
                max_new_tokens=7,
                min_new_tokens=7,
                do_sample=False, 
                num_beams=1,
                attention_mask=None,
                pad_token_id=tokenizer.eos_token_id,
            )
            output = output[:, input_ids.shape[1]:]
    
        truth = tokenizer.batch_decode(target_ids)
        est = [get_numbers(s.strip())[:5] for s in tokenizer.batch_decode(output)]
        
        seq_len = input_ids.shape[1]
        acc_sum, acc_count = accuracy.get(seq_len, (0, 0))
        for x, y in zip(truth, est):
            for cx, cy in zip(x, y):
                if cx == cy:
                    acc_sum += 1
                acc_count += 1
        accuracy[seq_len] = (acc_sum, acc_count)
        
        tqdm.write(f"current accuracy { {k: f'{v[0] / v[1]*100:.2f}' for k, v in accuracy.items()} } | {truth[0]}, {est[0]}")