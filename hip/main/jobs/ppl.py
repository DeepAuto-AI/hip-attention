import math
import os
import pathlib
import time
import traceback
import warnings
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import argparse, json
from transformers import TextStreamer

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from hip.models.modeling_llama import LlamaForCausalLM, LlamaConfig
from hip.utils import seed, get_bench

@torch.inference_mode
def job_ppl(args, model, tokenizer: transformers.LlamaTokenizer, device, quite=False):
    try:
        from vllm import LLM, SamplingParams
    except ModuleNotFoundError:
        LLM = torch.Tensor
        warnings.warn('vllm is not installed, this may cause error when you gave vLLM LLM')
    
    outfile = f'./cache/llama_eval/{args.name}/ppl_{args.dataset}_{args.method}_{args.model}_s{args.stride}_dl{args.dense_layers}_k{args.k}_bq{args.block_size_q}_bk{args.block_size_k}_ckpt{args.checkpoint is not None}.json'
    pathlib.Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    if not quite:
        print("Will write to", outfile)
    if os.path.exists(outfile) and not args.overwrite:
        print(f'PPL already computed, skipping: {outfile}')
        return

    os.makedirs('./cache', exist_ok=True)
    cache_path = f'./cache/llama_eval_{args.dataset}_{args.model}.pth'
    PG19_BOOK_INDEX = int(os.getenv('PG19_BOOK_INDEX', '-1'))
    if PG19_BOOK_INDEX >= 0:
        cache_path = 'none'
    if not os.path.exists(cache_path):
        assert args.dataset in ['wikitext', 'pg19']
        if args.dataset == 'wikitext':
            test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            sequence = "\n\n".join(test["text"])
        elif args.dataset == 'pg19':
            test = load_dataset("emozilla/pg19-test", split="test")
            books = test["text"]
            if PG19_BOOK_INDEX >= 0:
                books = list(sorted(books, key=lambda x: tokenizer(x, return_tensors="pt").input_ids.size(1)))
                sequence = books[PG19_BOOK_INDEX]
            else:
                sequence = "\n\n".join(books)
        if isinstance(sequence, torch.Tensor):
            encodings = sequence
        else:
            encodings = tokenizer(sequence, return_tensors="pt").input_ids
        print(encodings.shape)
        if PG19_BOOK_INDEX < 0:
            torch.save(encodings, cache_path)
    else:
        encodings = torch.load(cache_path)

    max_length = model.config.max_position_embeddings if hasattr(model, 'config') else 2048
    max_length = stride = args.stride if args.stride > 0 else max_length
    seq_len = encodings.size(1)
    
    if not quite:
        print(f'[{args.dataset}] {seq_len} tokens loaded')

    nlls = []
    prev_end_loc = 0
    t = time.time()
    with tqdm(range(0, seq_len, stride)[:args.count], dynamic_ncols=True, disable=quite) as pbar:
        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
            input_ids = encodings[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():

                if isinstance(model, LLM):
                    sampling_params = SamplingParams(
                        max_tokens=1,
                        ignore_eos=True,
                        only_return_logits=True,
                    )
                    prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    outputs = model.generate(prompt, sampling_params)

                else:
                    sample_counts = int(os.getenv('_SAMPLE_COUNT', '1'))
                    samples = []
                    with tqdm(range(sample_counts), dynamic_ncols=True, position=1, disable=sample_counts <= 1) as pbar_sample:
                        for _ in pbar_sample:
                            if args.method in ['h2o', 'tova']:
                                loss_sum = 0
                                loss_count = 0
                                prompt_ids = input_ids[:, :args.k]
                                prompt_target_ids = target_ids[:, :args.k]
                                decode_ids = input_ids[:, args.k:]
                                
                                kwargs = {
                                    'output_logits': True
                                }
                                
                                past_key_values = None
                                if args.method == 'tova':
                                    from hip.models.tova.tova_cache import TOVACache
                                    past_key_values = TOVACache(args.k)
                                    del kwargs['output_logits']
                                    prompt_ids = input_ids[:, :2]
                                    prompt_target_ids = target_ids[:, :2]
                                    decode_ids = input_ids[:, 2:]
                                
                                outputs = model(
                                    prompt_ids,
                                    labels=prompt_target_ids,
                                    past_key_values=past_key_values,
                                    **kwargs,
                                )
                                loss_sum += outputs.loss * prompt_ids.shape[-1]
                                loss_count += prompt_ids.shape[-1]
                                tqdm.write(f'H2O Loss: {math.exp(loss_sum / loss_count)}')
                                for curr_idx in tqdm(range(decode_ids.shape[-1]), dynamic_ncols=True):
                                    curr_token = decode_ids[:, curr_idx:curr_idx+1]
                                    if args.method == 'tova':
                                        position_ids = torch.arange(
                                            curr_idx, 
                                            curr_idx+1, 
                                            device=curr_token.device
                                        )[None, :]
                                    elif args.method == 'h2o':
                                        position_ids = torch.arange(
                                            prompt_ids.shape[1]+curr_idx, 
                                            prompt_ids.shape[1]+curr_idx+1, 
                                            device=curr_token.device
                                        )[None, :]
                                    else:
                                        raise Exception()

                                    outputs = model(
                                        curr_token,
                                        # labels=curr_target,
                                        # output_logits=True,
                                        position_ids=position_ids,
                                        past_key_values=outputs.past_key_values,
                                        **kwargs,
                                    )
                                    loss = torch.nn.functional.cross_entropy(
                                        outputs.logits.view(-1, model.config.vocab_size), 
                                        decode_ids[:, curr_idx+1:curr_idx+2].view(-1)
                                    )
                                    loss_sum += loss * curr_token.shape[-1]
                                    loss_count += curr_token.shape[-1]
                                    tqdm.write(f'H2O Loss idx={prompt_ids.shape[1]+curr_idx+1}: {math.exp(loss_sum / loss_count)}')
                                for m in model.modules():
                                    if hasattr(m, '_clean_cache'):
                                        m._clean_cache()
                            else:
                                outputs = model(
                                    input_ids,
                                    labels=target_ids,
                                    output_logits=False,
                                )
                            samples.append(outputs.loss)
                            pbar_sample.set_description(
                                f'ppl: {torch.exp(torch.stack(nlls + [outputs.loss.cpu()]).mean()).item():.6f}'
                            )
                    if len(samples) > 1:
                        print([f'{x.item():.5f}' for x in samples])
                    neg_log_likelihood = min(samples)

            nlls.append(neg_log_likelihood.cpu())

            prev_end_loc = end_loc
            
            ppl = torch.exp(torch.stack(nlls).mean()).item()
            if not quite:
                tqdm.write(f'step {len(nlls)} PPL: {ppl:.6f}, {time.time() - t:.4f} sec')
            t = time.time()
            pbar.set_description(f"ppl: {ppl:.3f}")
            
            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).mean()).item()
    
    os.makedirs('./cache/llama_eval/', exist_ok=True)
    with open(outfile, 'w') as f:
        json.dump({'ppl': ppl}, f)

    if not quite:
        print(f'PPL: {ppl:.4f}')

    return ppl