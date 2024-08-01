import os
import math
import torch
from torch.utils.data import Dataset
import numpy as np

PREFIX = """<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

There is a pass key hidden inside a lot of irrelevant text. Find the pass key and memorize it. I will quiz you about the the pass key.

"""
FILLER_TEXT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
QUERY = """\n So now, I will ask the question. What is the five digit pass key?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Sure! I surely remember the five digits pass key. The pass key is $"""

def interpolate_passkey(k):
    keyline = f"HERE IS THE PASS KEY! The pass key is ${k}$. ${k}$ is the pass key. **the pass key is ${k}$** LOOK BEHIND FOR PASS KEY"
    return f"""

=== NOW IMPORTANT INFORMATION STARTS ===

{keyline}

REPEAT THE INFORMATION

{keyline}

REPEAT THE INFORMATION

{keyline}

REPEAT THE INFORMATION

{keyline}

REPEAT THE INFORMATION

{keyline}

=== IMPORTANT INFORMATION STOPS ===

"""

def gen_text():
    prefix_len = int(len(PREFIX[:-1].split(" ")) * 1.275)
    filler_len = int(len(FILLER_TEXT[:-1].split(" ")) * 1.275)
    query_len = int(len(QUERY[:-1].split(" ")) * 1.275)

    inputs, targets = [], []
    # prompt_lens = [2000, 4000, 8000, 16000, 32000, 64000]
    # prompt_lens = [2000, 4000, 8000, 16000, 32000]
    prompt_lens = [16000, 32000, 64000, 128000]
    # prompt_lens = [32000]
    # prompt_lens.reverse()

    for l in prompt_lens:
        n_fillers = (l - prefix_len - query_len) // filler_len + 1
        for i in range(50):

            text = [PREFIX] + [FILLER_TEXT] * n_fillers

            k = np.random.randint(10000, 100000)

            key_phrase = interpolate_passkey(k)
            target = f"{k}"

            insert_loc = np.random.randint(2, len(text) - 1)
            text = text[:insert_loc] + [key_phrase] + text[insert_loc:] + [QUERY]

            text = "".join(text)

            inputs.append(text)
            targets.append(target)

    return inputs, targets

def gen_dataset(tokenizer):
    inputs, targets = gen_text()

    x, y = [], []
    for inp, tgt in zip(inputs, targets):
        x += [
            tokenizer(
                inp, 
                return_tensors="pt", 
                truncation=False,
                add_special_tokens=False,
            ).input_ids
        ]
        y += [
            tokenizer(
                tgt,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=False,
            ).input_ids
        ]
        
    return x, y


class Passkey(Dataset):

    def __init__(self, tokenizer, batch_size=10):
        self.tokenizer = tokenizer
        self.dataset = gen_dataset(self.tokenizer)
        self.batch_size = batch_size

        self.inputs = self.dataset[0]
        self.targets = self.dataset[1]

    def __len__(self):
        return (len(self.inputs) // self.batch_size)

    def __getitem__(self, idx) -> int:
        if idx >= len(self):
            raise IndexError("Index out of range")

        inputs = self.inputs[idx * self.batch_size:(idx + 1) * self.batch_size]
        targets = self.targets[idx * self.batch_size:(idx + 1) *
                               self.batch_size]

        return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)


if __name__ == '__main__':
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        'togethercomputer/LLaMA-2-7B-32K')

    token_prefix = tokenizer(PREFIX, return_tensors="pt",
                             truncation=False).input_ids
    print(f"{token_prefix.size()=}")
    ds = Passkey(tokenizer)

    print(f"{len(ds)=}")
    for i, (x, y) in enumerate(ds):
        print(f"{i} {x.size()=} {y.size()=}")