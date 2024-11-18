import torch, transformers
from typing import List, Tuple, Dict, Optional
import tqdm
from hip.dataset.calib_loft_rag import (
    prefix,
    rag_qa_pairs,
)

def load_loft_rag_chat_corpus() -> List[str]:
    lines = []
    for qa in rag_qa_pairs:
        for answer in qa["answers"]:
            lines.append((f'{prefix} {qa["query_text"]}', answer))
    return lines

def format_chat_corpus(args, corpus: List[str]) -> List[str]:
    if 'llama3' in args.model:
        ret = []
        for prompt, output in corpus:
            updated = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|><|end_of_text|>"""
            ret.append(updated)
        return ret
    else:
        raise Exception()

def tokenizing_and_find_shared_prompt_on_corpus(
    tokenizer: transformers.LlamaTokenizer, 
    corpus: List[str]
):
    count = len(corpus)
    iterator = map(lambda line: tokenizer.encode(line, add_special_tokens=False), corpus)
    corpus = []
    shared_prompt = None
    for input_ids in tqdm.tqdm(iterator, total=count):
        if shared_prompt is None:
            shared_prompt = input_ids
        else:
            for i, (s, c) in enumerate(zip(shared_prompt, input_ids)):
                if s != c:
                    shared_prompt = shared_prompt[:i]
        corpus.append(input_ids)
    corpus = list(map(lambda ids: torch.tensor(ids[len(shared_prompt):]), corpus))
    shared_prompt = torch.tensor(shared_prompt)
    return shared_prompt, corpus

@torch.inference_mode
def job_ga(
    args, 
    model, 
    tokenizer: transformers.LlamaTokenizer,
    device, 
):
    corpus = load_loft_rag_chat_corpus()[:20]
    corpus = format_chat_corpus(args, corpus)
    shared_prompt, corpus = tokenizing_and_find_shared_prompt_on_corpus(
        tokenizer, corpus
    )
    
    output = model(
        input_ids=shared_prompt.unsqueeze(0),
        use_cache=True
    )
    
    torch.cuda.synchronize()