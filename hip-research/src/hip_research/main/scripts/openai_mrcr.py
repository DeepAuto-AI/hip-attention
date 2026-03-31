import json
import os
from difflib import SequenceMatcher

import pandas as pd
import tiktoken
from huggingface_hub import hf_hub_download
from openai import OpenAI

# Set accordingly
MAX_CONTEXT_WINDOW = int(os.getenv("CONTEXT_LEN", "1000000"))
MODEL = "anything"

base_url = os.getenv("ENDPOINT", "http://localhost:8000/v1")

dataset = pd.read_parquet(
    hf_hub_download(
        repo_id="openai/mrcr", filename="2needle.parquet", repo_type="dataset"
    )
)
client = OpenAI(base_url=base_url, api_key="sk-hello")
enc = tiktoken.get_encoding("o200k_base")


def grade(response: str, answer: str, random_string_to_prepend: str) -> float:
    """
    Compare response and answer.
    """
    think_token = "</think>"
    if "</think>" in response:
        idx = response.index(think_token)
        response = response[idx + len(think_token) :].strip()

    # print("res:", response.replace('\n', '\\n')[:100])
    # print("ans:", answer.replace('\n', '\\n')[:100])

    if not response.startswith(random_string_to_prepend):
        return 0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def n_tokens(messages: list[dict]) -> int:
    """
    Count tokens in messages.
    """
    return sum([len(enc.encode(m["content"])) for m in messages])


scores = []
for index, row in dataset.iterrows():
    messages = json.loads(row["prompt"])
    context_len = n_tokens(messages)
    if context_len > MAX_CONTEXT_WINDOW:
        continue
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.6,
        top_p=0.95,
        max_tokens=8192,
    )
    response = completion.choices[0].message.content
    grade_score = grade(response, row["answer"], row["random_string_to_prepend"])
    scores.append(grade_score)
    print(
        f"{len(scores):>7}, {context_len:>10}, {grade_score:>7.4f}, {sum(scores) / len(scores):>7.4f}"
    )

print(scores)
print(sum(scores) / len(scores))
