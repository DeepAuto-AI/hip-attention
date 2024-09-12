import random
import datasets
from torch.utils.data import Dataset
from tqdm import tqdm
import math

class RedPajamaDataset(Dataset):
    def __init__(self, tokenizer, stride):
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset(
            'togethercomputer/RedPajama-Data-1T',
            'arxiv',
            split='train',
            trust_remote_code=True
        )
        self.window_size = 5
        self.stride = stride

    def __len__(self):
        return len(self.dataset) // self.window_size

    def __getitem__(self, idx):
        text = []
        for i in range(self.window_size):
            entry = self.dataset[idx * self.window_size + i]
            text.append(entry['text'])
        random.shuffle(text)
        ids = self.tokenizer("\n\n".join(text), return_tensors='pt', truncation=True, max_length=self.stride).input_ids
        labels = ids.clone()
        return ids[0], labels[0]

import os
from pathlib import Path

from hip.dataset.wikitext2 import Wikitext2Dataset
import torch
import transformers 
from hip.models.modeling_llama_permute import LlamaForCausalLM as PermuteLlama
import torch.nn.functional as F

class Trainer:
    def __init__(self):
        self.model_id = 'meta-llama/Meta-Llama-3.1-8B'
        self.model_id = 'nvidia/Llama-3.1-Minitron-4B-Width-Base'
        
        self.stride = 1024
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        self.dataset = RedPajamaDataset(self.tokenizer, self.stride)
        self.eval_dataset = Wikitext2Dataset(subset='validation', tokenizer=self.tokenizer, stride=self.stride, max_length=131072)
        self.device = torch.device('cuda')
        self.teacher = transformers.AutoModelForCausalLM\
            .from_pretrained(
                self.model_id, 
                _attn_implementation='flash_attention_2', 
                attn_implementation='flash_attention_2'
            ).to(self.device)
        self.student = PermuteLlama(self.teacher.config).to(self.device)
        load_result = self.student.load_state_dict(self.teacher.state_dict(), strict=False)
        for p in self.student.parameters():
            p.requires_grad = False
        for m in self.student.modules():
            if hasattr(m, 'mark_trainable'):
                m.mark_trainable()
        print(load_result)
        
        self.optimiezr = torch.optim.adamw.AdamW(
            params=[p for p in self.student.parameters() if p.requires_grad],
            lr=1e-4,
        )
        self.scaler = torch.GradScaler()

    def evaluate(self):
        self.student.eval()
        
        losses = []
        for input_ids, labels in enumerate(tqdm.tqdm(self.eval_dataset)):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            with torch.no_grad():
                loss = self.student(input_ids, labels=labels).item()
            losses.append(loss)
        
        avg_loss = sum(losses) / (len(losses) + 1e-20)
        ppl = math.exp(avg_loss)
        
        print(f'eval: ppl = {ppl}, loss = {avg_loss}')
        return ppl

    def train_step(self, input_ids, labels):
        self.teacher.eval()
        self.student.eval()
        
        with torch.no_grad():
            output_teacher = self.teacher(input_ids, labels, output_hidden_states=True)
        output_student = self.student(input_ids, labels, output_hidden_states=True)
        
        loss_model = output_student.loss
        loss_kd_hidden = 0
        for teacher_layer, student_layer in zip(output_teacher.hidden_states, output_student.hidden_states):
            loss_kd_hidden_layer = F.mse_loss(teacher_layer, student_layer)
            loss_kd_hidden += loss_kd_hidden_layer
        loss_kd_hidden /= len(output_teacher.hidden_states)
        
        loss = loss_kd_hidden * 1.0 + loss_model * 0.1
        
        self.optimiezr.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimiezr)
        self.scaler.update()

    def train_epoch(self):
        for istep, (input_ids, labels) in enumerate(tqdm.tqdm(self.dataset)):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            
            self.current_step = istep
            self.train_step(input_ids, labels=labels)
            if ((istep + 1) % 100) == 0:
                self.evaluate()

    def main(self):
        for iepoch in range(10):
            self.current_epoch = iepoch
            self.train_epoch()
            self.evaluate()

if __name__ == '__main__':
    tr = Trainer()
    tr.main()