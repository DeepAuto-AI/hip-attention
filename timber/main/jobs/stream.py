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
        get_bench().disabled = True
        
        input_text = input('>>>').strip()
        
        if os.path.exists(input_text):
            print('loaded', input_text)
            with open(input_text, 'r') as f:
                input_text = f.read()
        
        inputs = tokenizer([tokenizer.bos_token + input_text, ] * args.batch_size, return_tensors='pt').to(device)
        print('input_ids', len(input_text), inputs.input_ids.shape)

        t = time.time()
        try:
            if isinstance(model, LLM):
                prompts = [input_text, ] * args.batch_size
                sampling_params = SamplingParams(
                    temperature=0.8, 
                    top_p=0.95,
                    max_tokens=inputs.input_ids.shape[-1] + 16,
                )
                
                outputs = model.generate(prompts, sampling_params, use_tqdm=True)
                
                for output in outputs:
                    generated_text = output.outputs[0].text
                    print(generated_text)
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
        elapsed = time.time() - t
        print(get_bench().format_tracetree())
        print(f'elapsed {elapsed:.4f} sec')