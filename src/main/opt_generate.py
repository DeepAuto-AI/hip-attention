"""
Talk to perlin opt

Usage: src.main.opt_generate

NOTE: this script is WIP.
"""

import gc
import time
import torch
import cv2
import os
import argparse

import tqdm

from ..utils import batch_to, get_bench, seed
from ..models import perlin_opt
from ..models import perlin_attention
from ..trainer.perlin_trainer import OptTrainer, add_perlin_model_options, parse_perlin_model_options
from ..models.perlin_attention import modules as pmodules
pmodules.BENCHMARKING = True
from transformers import OPTForCausalLM, StoppingCriteriaList, StoppingCriteria

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
get_bench().disabled = True
# get_bench().synchronize = True

sample_text = r"""
The game takes place during the Second Europan War . 
Gallian Army Squad 422 , also known as " The Nameless " , are a penal military unit composed of 
criminals , foreign deserters , and military offenders whose real names are erased from the 
records and thereon officially referred to by numbers . Ordered by the Gallian military to 
perform the most dangerous missions that the Regular Army and Militia will not do , they are 
nevertheless up to the task , exemplified by their motto , Altaha Abilia , meaning " Always Ready . " 
The three main characters are No.7 Kurt Irving , an army officer falsely accused of treason who 
wishes to redeem himself ; Ace No.1 Imca , a female Darcsen heavy weapons specialist who seeks 
revenge against the Valkyria who destroyed her home ; and No.13 Riela Marcellis , a seemingly 
jinxed young woman who is unknowingly a descendant of the Valkyria . Together with their fellow 
squad members , these three are tasked to fight against a mysterious Imperial unit known as 
Calamity Raven , consisting of mostly Darcsen soldiers . As the Nameless officially do not exist , 
the upper echelons of the Gallian Army exploit the concept of plausible deniability in order 
to send them on missions that would otherwise make Gallia lose face in the war . While at 
times this works to their advantage , such as a successful incursion into Imperial territory , 
other orders cause certain members of the 422nd great distress . One such member , Gusurg , 
becomes so enraged that he abandons his post and defects into the ranks of Calamity Raven , 
attached to the ideal of Darcsen independence proposed by their leader , Dahau . At the same time , 
elements within Gallian Army Command move to erase the Nameless in order to protect their own interests . 
Hounded by both allies and enemies , and combined with the presence of a traitor within their ranks , 
the 422nd desperately move to keep themselves alive while at the same time fight to help the 
Gallian war effort . This continues until the Nameless 's commanding officer , Ramsey Crowe , 
who had been kept under house arrest , is escorted to the capital city of Randgriz in order to 
present evidence exonerating the weary soldiers and expose the real traitor , the Gallian General 
that had accused Kurt of Treason .""".replace('\n', '') * 4
max_length = 2048

sample_text = r"""The game takes place during the Second Europan War . """.replace('\n', '')
max_length = 128
batch_size = 1

def main(
    dataset = 'wikitext2',
    checkpoint_path = None,
    **kwargs
):
    trainer = OptTrainer(
        subset=dataset,
        **kwargs,
    )
    trainer.device = 'cpu'
    trainer.device = 'cuda'
    if checkpoint_path is None:
        # checkpoint_path = trainer.checkpoint_path()
        trainer.load()
    else:
        if os.path.exists(checkpoint_path):
            trainer.load(path=checkpoint_path)
        else:
            print('checkpoint not exists', checkpoint_path)
    
    model = trainer.model.to(trainer.device).eval() # type: perlin_opt.OPTForCausalLM
    tokenizer = trainer.tokenizer
    
    use_cache = True
    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.max_memory_allocated()
        
        t = time.time()
        
        # with torch.no_grad():
        #     generate_ids = model.generate(
        #         inputs.input_ids.to(trainer.device), 
        #         max_length=2048,
        #         use_cache=use_cache,
        #         top_k=16,
        #         temperature=0.5,
        #         do_sample=True,
        #         stopping_criteria=StoppingCriteriaList(),
        #     )
        input_ids = inputs.input_ids.to(trainer.device)
        input_ids = input_ids.repeat(batch_size, 1)
        while input_ids.size()[1] < max_length:
            output = model.generate(
                input_ids,
                max_length=max_length,
                use_cache=use_cache,
                temperature=0.95,
                do_sample=True,
                stopping_criteria=StoppingCriteriaList(),
            )
            input_ids = torch.cat([input_ids, output[:, input_ids.shape[1]:]], dim=1)
            print(f'generated {input_ids.shape[1]} / {max_length}')
        
        end_mem = torch.cuda.max_memory_allocated()
        elapsed = time.time() - t
        print(f'elapsed: {elapsed*1000:.2f} ms, {(input_ids.shape[-1] - inputs.input_ids.shape[-1]) / elapsed:.2f} wps, {(end_mem - start_mem) / 1024 / 1024:.2f} MB')
        
        generated_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return inputs, input_ids, generated_text
    
    perlin_attention.get_default_config().use_cache = use_cache
    for m in model.modules():
        if hasattr(m, 'benchmarking'):
            m.benchmarking = use_cache
    
    _, ids, generated_text = generate(sample_text)
    print('sample:', sample_text)
    print('generated:', generated_text, f'[{ids.shape[-1]}]')
    
    while args.interactive:
        print('>>> ', end='', flush=True)
        try:
            prompt = input().strip()
            if prompt in ['quit', 'exit']:
                break
        except KeyboardInterrupt as ex:
            print()
            continue
        
        inputs, generate_ids, generated_text = generate(prompt)
        
        print(generate_ids)
        print(f"```{generated_text.strip()}```")

if __name__ == '__main__':
    seed()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--model', type=str, default='opt-350m')
    parser.add_argument('--max-seq-len', type=int, default=768)
    parser.add_argument('--interactive', action='store_true')
    add_perlin_model_options(
        parser, 
        nbf=8.0,
        context_output_method='mix',
        predictor_length=64,
        k=64,
        epl=True,
    )
    
    args = parser.parse_args()
    print(args)
    
    kwargs = parse_perlin_model_options(args)
    kwargs.update({
        'model': args.model,
        'dataset': args.dataset,
        'checkpoint_path': args.checkpoint,
        'max_seq_len': args.max_seq_len,
    })
    
    main(**kwargs)