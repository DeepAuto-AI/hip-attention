import os
import time
import traceback
import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm
import argparse
from transformers import TextStreamer

from vllm import LLM, SamplingParams
from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers.models.auto import AutoTokenizer
from timber.models.modeling_llama import LlamaForCausalLM, LlamaConfig
from timber.utils import seed, get_bench

from vllm.transformers_utils import config as vllm_transformers_config
vllm_transformers_config.FORCE_SIGNLE_LAYER = int(os.environ.get('FORCE_SINGLE_LAYER', '0'))

class BatchedStreamer(TextStreamer):
    def __init__(self, tokenizer: AutoTokenizer, skip_prompt: bool = False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.idx = 0
    
    def put(self, value):
        if self.idx == 1:
            print('prompt trace', get_bench().format_tracetree())
            get_bench().reset_trace()
            get_bench().reset_measures()
        self.idx += 1
        return super().put(value[:1])

def job_stream(args, model, tokenizer, device):
    while True:
        get_bench().reset_trace()
        get_bench().reset_measures()
        # get_bench().disabled = False
        
        if args.input is None:
            input_text = input('>>>').strip()
        else:
            input_text = args.input
        
        if len(input_text.strip()) == 0:
            continue
        
        if os.path.exists(input_text):
            print('loaded', input_text)
            with open(input_text, 'r') as f:
                input_text = f.read()
        
        inputs = tokenizer([input_text, ] * args.batch_size, return_tensors='pt').to(device)
        print('input_ids', len(input_text), inputs.input_ids.shape)

        t = time.time()
        elapsed = 0
        try:
            if isinstance(model, LLM):
                prompts = [input_text, ] * args.batch_size
                sampling_params = SamplingParams(
                    temperature=0.7, 
                    top_p=0.9,
                    top_k=500,
                    max_tokens=512,
                    # max_tokens=16,
                    frequency_penalty=0.0,
                    repetition_penalty=1.1,
                    ignore_eos=True,
                    skip_special_tokens=False,
                    # max_tokens=inputs.input_ids.shape[-1] + 32,
                )
                
                outputs = model.generate(prompts, sampling_params, use_tqdm=True)
                elapsed = time.time() - t
                
                n_generated = 0
                for output in outputs:
                    generated_text = output.outputs[0].text
                    n_tokens = len(tokenizer([generated_text]).input_ids[0])
                    n_generated += n_tokens
                    if len(outputs) > 1:
                        print(generated_text[:150] + ' [...]', n_tokens)
                    else:
                        print(generated_text, n_tokens)
                print(f'{n_generated} token generated, {n_generated/elapsed:.2f} tok/sec')
            else:
                streamer = BatchedStreamer(tokenizer, skip_prompt=True)
                
                with torch.no_grad():
                    model.generate(
                        **inputs, 
                        streamer=streamer, 
                        do_sample=True,
                        max_new_tokens=256,
                        temperature=0.5,
                        top_p=0.8,
                        top_k=1000,
                    )
        except KeyboardInterrupt:
            traceback.print_exc()
            print('Interrupted')
        if elapsed == 0:
            elapsed = time.time() - t
        tracetree = get_bench().format_tracetree().strip()
        if len(tracetree) > 0:
            print(tracetree)
        print(f'elapsed {elapsed:.4f} sec')
    
        if args.input is not None:
            return