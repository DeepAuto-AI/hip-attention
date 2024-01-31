import os
import random
import datasets
import torch 
from torch.utils.data import Dataset
import tqdm
import transformers

HEADER = """Below are instructions that describes tasks. Write responses that appropriately completes each request.
"""

QUESTION_FORMAT = """
# Question {question_id}
{content}
"""

ALPACA_FORMAT = """### Instruction
{instruction}"""

ALPACA_FORMAT_INPUT = """### Instruction
{instruction}

### Input
{input}"""

ANSWER_FORMAT = """
# Answer {question_id}
{content}
"""

class AlpacaDataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()
        
        self.n_questions = 20
        self.tokenizer = tokenizer
        
        dataset = datasets.load_dataset('tatsu-lab/alpaca')['train']
        self.entries = []
        for entry in dataset:
            if len(entry['input']) > 0:
                question = ALPACA_FORMAT_INPUT.format(
                    instruction=entry['instruction'],
                    input=entry['input'],
                )
            else:
                question = ALPACA_FORMAT.format(
                    instruction=entry['instruction'],
                )
            self.entries.append({
                'question': question,
                'answer': entry['output']
            })
    
    def __len__(self):
        return 1000
    
    def __getitem__(self, index):
        res = HEADER
        entries = []
        for i in range(self.n_questions):
            entry = random.choice(self.entries)
            res += QUESTION_FORMAT.format(
                question_id=i+1,
                content=entry['question']
            )
            entries.append(entry)
        for i, entry in enumerate(entries):
            res += ANSWER_FORMAT.format(
                question_id=i+1,
                content=entry['answer']
            )

        ids = self.tokenizer(res, return_tensors='pt').input_ids
        labels = ids.clone()
        labels[:, :-1] = ids[:, 1:]
        labels[:, -1] = -100
        
        return ids[0], labels[0]

if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained('togethercomputer/LLaMA-2-7B-32K')
    ds = AlpacaDataset(tokenizer)
    for ids, labels in ds:
        assert ids.shape == labels.shape
        print(ids.shape)