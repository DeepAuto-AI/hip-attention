from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.generation import GenerationConfig
from src.models.qwen.modeling_qwen import QWenLMHeadModel
import torch

from src.main.jobs.mmmu import job_mmmu
from src.utils import seed
from src.main.eval_args import eval_args, ArgsType

def load_qwen(args: ArgsType):
    assert args.model in ['qwen']
    
    device = 'cuda:0'
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
    
    model = QWenLMHeadModel.from_pretrained(
        "Qwen/Qwen-VL", 
        device_map="auto", 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        load_in_4bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=['visual']
        ),
        fp16=True,
    ).eval()
    
    for m in model.modules():
        if hasattr(m, 'attention_method'):
            m.attention_method = args.method
            m.tree_k = args.k
            m.tree_block_size_k = args.block_size_k
            m.tree_block_size_q = args.block_size_q
            m.tree_dense_queries = args.dense_queries

    return model, tokenizer, device

def main():
    args = eval_args(
        default_model='qwen',
        default_job='mmmu'
    )
    
    model, tokenizer, device = load_qwen(args)
    
    if args.job == 'mmmu':
        job_mmmu(args, model, tokenizer, device)
    else:
        raise Exception()

if __name__ == '__main__':
    seed()
    main()