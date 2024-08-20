import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import argparse
from transformers import TextStreamer

from transformers.models.auto import AutoTokenizer
from hip.models.modeling_llama import LlamaForCausalLM, LlamaConfig
from hip.utils import seed, get_bench

def generate_stream(
    args,
    llm,
    tokenizer, 
    prompts, 
    sampling_params,
    prompt_token_ids = None,
    use_tqdm: bool = False,
):
    from vllm import SamplingParams
    if prompts is None and prompt_token_ids is None:
        raise ValueError("Either prompts or prompt_token_ids must be "
                            "provided.")
    if isinstance(prompts, str):
        # Convert a single prompt to a list.
        prompts = [prompts]
    if (prompts is not None and prompt_token_ids is not None
            and len(prompts) != len(prompt_token_ids)):
        raise ValueError("The lengths of prompts and prompt_token_ids "
                            "must be the same.")
    if sampling_params is None:
        # Use default sampling params.
        sampling_params = SamplingParams()

    # Add requests to the engine.
    num_requests = len(prompts) if prompts is not None else len(
        prompt_token_ids)
    for i in range(num_requests):
        prompt = prompts[i] if prompts is not None else None
        token_ids = None if prompt_token_ids is None else prompt_token_ids[
            i]
        llm._add_request(
            prompt,
            sampling_params,
            token_ids,
        )
    
    # Initialize tqdm.
    if use_tqdm:
        num_requests = llm.llm_engine.get_num_unfinished_requests()
        pbar = tqdm(total=num_requests,
                    desc="Processed prompts",
                    dynamic_ncols=True)
    # Run the engine.
    outputs = [] # type: List[RequestOutput]
    istep = 0
    t = time.time()
    t_decode = 0
    n_decode = 0
    while llm.llm_engine.has_unfinished_requests():
        step_outputs = llm.llm_engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)
                if use_tqdm:
                    pbar.update(1)
        
        if istep >= args.batch_size:
            if istep == args.batch_size:
                os.system('clear')
                print('\n----- Decoding Starts -----\n')
            stream = step_outputs[0]
            token_ids = stream.outputs[-1].token_ids
            
            text = tokenizer.convert_ids_to_tokens(token_ids[-1]).replace('Ġ', ' ').replace('Ċ', '\n').replace('âĢĿ', '').replace('âĢľ', '').replace('âĢĻ', '').replace('âĢĶ', '')
            if text.startswith('▁'):
                text = ' ' + text[1:]
            if text == '<0x0A>':
                text = '\n'
            print(text, end='', flush=True)
            
            t_decode += time.time() - t
            n_decode += 1
            
            if stream.finished:
                print(f'\n\n{"="*60}\n[VLLM_ATTENTION_BACKEND={os.getenv("VLLM_ATTENTION_BACKEND", "oops")}] (model={args.model}, prompt_len={len(stream.prompt_token_ids)}, response_len={len(token_ids)})\nEnd-to-End vLLM Decoding Latency: {t_decode / n_decode*1000:.2f} ms\n{"="*60}')
        else:
            if istep == 0:
                print()
            print(f'[Prompt Processed] index = {istep + 1} / {args.batch_size}, took {time.time() - t:.2f} sec')
        t = time.time()
        istep += 1
    if use_tqdm:
        pbar.close()
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    return outputs

def job_stream_demo(args, model, tokenizer, device):
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils import config as vllm_transformers_config
    
    input_text = args.input
    assert input_text is not None
    
    if os.path.exists(input_text):
        print('loading', input_text)
        with open(input_text, 'r', encoding='utf8') as f:
            input_text = f.read()
    
    # inputs = tokenizer([input_text, ] * args.batch_size, return_tensors='pt').to(device)
    # print('input_ids', len(input_text), inputs.input_ids.shape)
    # print(inputs)

    t = time.time()
    elapsed = 0
    try:
        assert isinstance(model, LLM)
        prompts = [input_text, ] * args.batch_size
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            top_k=1000,
            max_tokens=args.max_tokens,
            # max_tokens=16,
            frequency_penalty=0.2,
            repetition_penalty=1.0,
            ignore_eos=True,
            skip_special_tokens=False,
            # max_tokens=inputs.input_ids.shape[-1] + 32,
        )
        
        outputs = generate_stream(args, model, tokenizer, prompts, sampling_params)
        elapsed = time.time() - t
        
        # n_generated = 0
        # for output in outputs:
        #     generated_text = output.outputs[0].text
        #     n_tokens = len(tokenizer([generated_text]).input_ids[0])
        #     n_generated += n_tokens
        #     if len(outputs) > 1:
        #         print(generated_text.replace('\n', '\\n')[:300] + ' [...]', n_tokens)
        #     else:
        #         print(generated_text, n_tokens)
        # print(f'{n_generated} token generated, {n_generated/elapsed:.2f} tok/sec')
    except KeyboardInterrupt:
        traceback.print_exc()
        print('Interrupted')
    if elapsed == 0:
        elapsed = time.time() - t
    print(f'elapsed {elapsed:.4f} sec')