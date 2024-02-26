"""
cd cache/long_data_collection
wget https://huggingface.co/datasets/togethercomputer/Long-Data-Collections/resolve/main/fine-tune/booksum.jsonl.zst
"""

import os
import json
import torch
import tqdm
from torch.utils.data import Dataset
import transformers

class BookSumDataset:
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer, 
        json_path = './cache/long_data_collection/booksum.jsonl',
        max_seq_len = 32768,
        for_eval = False,
    ):
        with open(json_path, 'r') as f:
            lines = f.readlines()
        
        self.max_seq_len = max_seq_len
        
        self.data = []
        for line in lines:
            # dict_keys(['text', 'prompt', 'completion'])
            self.data.append(json.loads(line))
        
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.processed = []
        os.makedirs('./cache/long_data_collection', exist_ok=True)
        cache_path = './cache/long_data_collection/booksum.pth'
        if not os.path.exists(cache_path):
            with tqdm.tqdm(self.data, desc='tokenizing', dynamic_ncols=True, leave=False) as pbar:
                for data in pbar:
                    text_ids = tokenizer(
                        tokenizer.bos_token + data['text'] + ' ' + self.tokenizer.eos_token,
                        return_tensors='pt', 
                        truncation=True, 
                        max_length=self.max_seq_len,
                    )['input_ids'][0]
                    prompt_ids = tokenizer(
                        tokenizer.bos_token + data['prompt'] + ' ' + self.tokenizer.eos_token,
                        return_tensors='pt',
                        truncation=True, 
                        max_length=self.max_seq_len,
                    )['input_ids'][0]
                    completion_ids = tokenizer(
                        tokenizer.bos_token + data['completion'] + ' ' + self.tokenizer.eos_token,
                        return_tensors='pt',
                        truncation=True, 
                        max_length=self.max_seq_len,
                    )['input_ids'][0]
                    
                    input_ids = text_ids
                    target_ids = input_ids.clone()
                    target_ids[:-1] = input_ids[1:]
                    target_ids[-1] = -100
                    self.processed.append({
                        'input_ids': input_ids,
                        'labels': target_ids,
                        'text_ids': text_ids,
                        'prompt_ids': prompt_ids,
                        'completion_ids': completion_ids,
                    })
                    pbar.set_description(f't[{tuple(input_ids.shape)}]')
            torch.save(self.processed, cache_path)
        else:
            print('loading cache')
            self.processed = torch.load(cache_path)
            print('loaded', cache_path)
        print('loaded booksum', len(self.processed))
        self.for_eval = for_eval

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        entry = self.processed[idx]
        if self.for_eval:
            return entry['prompt_ids'], entry['completion_ids']
        else:
            return entry['input_ids'], entry['labels']

if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained('togethercomputer/LLaMA-2-7B-32K')
    ds = BookSumDataset(tokenizer)
    for idx in tqdm.tqdm(range(len(ds))):
        entry = ds[idx]
        token_length = entry['input_ids'].shape[0]
        assert token_length <= 32768, token_length