import math
import os

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

IS_GEMMA = os.getenv("IS_GEMMA", "0") == "1"
IS_EXAONE = os.getenv("IS_EXAONE", "0") == "1"
IS_CHAT = os.getenv("IS_CHAT", "0") == "1"

if IS_EXAONE:
    PREFIX = """[BOS][|system|][|endofturn|]
[|user|]You are a helpful assistant.
There is a secret keyword hidden inside a lot of irrelevant text. Find the secret keyword and memorize it. I will quiz you about the the secret keyword.

"""
    FILLER_TEXT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    QUERY = """
In previous text, you have seen the secret keyword. You had to remember that secret keyword. What was the pass key? Just answer the secret keyword without any verbal text.[|endofturn|]
[|assistant|]"""
elif IS_GEMMA:
    PREFIX = """<start_of_turn>user

There is a pass key hidden inside a lot of irrelevant text. Find the pass key and memorize it. I will quiz you about the the pass key.

"""
    FILLER_TEXT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    QUERY = """\n So now, I will ask the question. What is the five digit pass key? Answer only 5 digit passkey.
<end_of_turn>
<start_of_turn>assistant

"""
elif IS_CHAT:
    PREFIX = """You are a helpful assistant. There is a secret keyword hidden inside a lot of irrelevant text. Find the secret keyword and memorize it. I will quiz you about the the secret keyword.

"""
    FILLER_TEXT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    QUERY = """

In previous text, you have seen the secret keyword. You had to remember that secret keyword. What was the pass key? Just answer the secret keyword without any verbal text."""
else:
    PREFIX = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

There is a secret keyword hidden inside a lot of irrelevant text. Find the secret keyword and memorize it. I will quiz you about the the secret keyword.

"""
    FILLER_TEXT = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    QUERY = """
In previous text, you have seen the secret keyword. You had to remember that secret keyword. What was the pass key? Just answer the secret keyword without any verbal text.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The secret keyword is $"""


def interpolate_passkey(k):
    keyline = f"HERE IS THE SECRET KEYWORD! The secret keyword is ${k}$. ${k}$ is the secret keyword. **the secret keyword is ${k}$** LOOK BEHIND FOR SECRET KEYWORD"
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

    step_size = int(os.getenv("PASSKEY_STEP_SIZE", "32"))
    n_samples = int(os.getenv("PASSKEY_SAMPLES", "100"))
    inputs, targets = [], []
    if os.getenv("PASSKEY_MAX_LEN", "0") != "0":
        start_seq = int(os.getenv("PASSKEY_MAX_LEN", "0"))
        prompt_lens = []
        while int(start_seq) >= 4:
            prompt_lens.append(int(start_seq) * 1000)
            if start_seq > step_size:
                start_seq = max(start_seq - step_size, step_size)
            else:
                start_seq /= 2
        print("passkey sequences", prompt_lens)
    else:
        # prompt_lens = [2000, 4000, 8000, 16000, 32000, 64000]
        # prompt_lens = [2000, 4000, 8000, 16000, 32000]
        # prompt_lens = [16000, 32000, 64000, 128000]
        # prompt_lens = [2000, 4000, 8000, 16000, 32000, 64000, 128000]
        prompt_lens = [128000, 64000, 32000, 16000, 8000, 4000]
        # prompt_lens = [32000]
        # prompt_lens.reverse()

    for l in prompt_lens:
        n_fillers = (l - prefix_len - query_len) // filler_len + 1
        for i in range(n_samples):

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
    for inp, tgt in tqdm.tqdm(
        list(zip(inputs, targets)),
        dynamic_ncols=True,
        leave=False,
        delay=3,
        desc="passkey",
    ):
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
        return len(self.inputs) // self.batch_size

    def __getitem__(self, idx) -> int:
        if idx >= len(self):
            raise IndexError("Index out of range")

        inputs = self.inputs[idx * self.batch_size : (idx + 1) * self.batch_size]
        targets = self.targets[idx * self.batch_size : (idx + 1) * self.batch_size]

        return torch.cat(inputs, dim=0), torch.cat(targets, dim=0)


if __name__ == "__main__":
    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "togethercomputer/LLaMA-2-7B-32K"
    )

    token_prefix = tokenizer(PREFIX, return_tensors="pt", truncation=False).input_ids
    print(f"{token_prefix.size()=}")
    ds = Passkey(tokenizer)

    print(f"{len(ds)=}")
    for i, (x, y) in enumerate(ds):
        print(f"{i} {x.size()=} {y.size()=}")
