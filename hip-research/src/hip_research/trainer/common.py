import os
import pathlib
from dataclasses import dataclass

import torch
import torch.autograd
import torch.onnx
import torch.utils.checkpoint
import transformers
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import LlamaForCausalLM

from hip_attn.models.modeling_llama import LlamaConfig, LlamaForCausalLM
from hip_attn.models.qwen.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM


@dataclass
class TrainConfig:
    disable_kd: bool = False
    using_fsdp: bool = False
    using_deepspeed: bool = False
    model_parallel: bool = False
    lr: float = 5e-5
    batch_size: int = 1
    accumulation_steps: int = 2
    lora_r: int = 32
    save_steps: int = 100
    dense_queries: int = None
    seq_len: int = 4096
    max_steps: int = 1000000
    model_checkpoint_dir: str = "./saves/dev/checkpoint"
    dataset: str = "wikitext103"
    load_from_checkpoint: str = None
    k: int = 512
    block_size_q: int = 8
    block_size_k: int = 8
    init_from_checkpoint: str = None
    method: str = "hip"
    model: str = "llama32k"
    disable_global_context: bool = False
    warmup_steps: int = 5
    sparsity_reg: float = 0.0
    dense_layers: int = 0
    name: str = "default"
    enable_sparq: bool = False
    enable_flash: bool = False
    use_sliding_window: bool = False
    local_rank: int = None


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="llama32k", type=str)
    parser.add_argument("--method", default="hip", type=str)
    parser.add_argument("--dense_queries", default=None, type=int)
    parser.add_argument("--using_fsdp", action="store_true")
    parser.add_argument("--using_deepspeed", action="store_true")
    parser.add_argument("--disable_kd", action="store_true")
    parser.add_argument("--disable_global_context", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", default=-1, type=int)
    parser.add_argument("--batch_size", default=-1, type=int)
    parser.add_argument("--lora_r", default=-1, type=int)
    parser.add_argument("--lr", default=-1, type=float)
    parser.add_argument("--max_steps", default=-1, type=int)
    parser.add_argument("--dataset", default="wikitext103", type=str)
    parser.add_argument("--seq_len", default=-1, type=int)
    parser.add_argument("--save_steps", default=-1, type=int)
    parser.add_argument("--init_checkpoint", default=None, type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--k", default=512, type=int)
    parser.add_argument("--block_size_q", default=16, type=int)
    parser.add_argument("--block_size_k", default=2, type=int)
    parser.add_argument("--warmup_steps", default=None, type=int)
    parser.add_argument("--sparsity_reg", default=None, type=float)
    parser.add_argument("--dense_layers", type=int, default=None)
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--model_parallel", default=None, type=int)
    parser.add_argument("--local-rank", default=None, type=int)

    args = parser.parse_args()

    train_config = TrainConfig(
        model_parallel=args.model_parallel,
        using_fsdp=args.using_fsdp,
        using_deepspeed=args.using_deepspeed,
        disable_kd=args.disable_kd,
        dataset=args.dataset,
        load_from_checkpoint=args.checkpoint,
        k=args.k,
        block_size_q=args.block_size_q,
        block_size_k=args.block_size_k,
        method=args.method,
        model=args.model,
        name=args.name,
        disable_global_context=args.disable_global_context,
        local_rank=args.local_rank,
    )
    if args.gradient_accumulation_steps > 0:
        train_config.accumulation_steps = args.gradient_accumulation_steps
    if args.lora_r > 0:
        train_config.lora_r = args.lora_r
    if args.lr > 0:
        train_config.lr = args.lr
    if args.batch_size > 0:
        train_config.batch_size = args.batch_size
    if args.max_steps > 0:
        train_config.max_steps = args.max_steps
    if args.seq_len > 0:
        train_config.seq_len = args.seq_len
    if args.save_steps > 0:
        train_config.save_steps = args.save_steps
    if args.dense_queries is not None:
        train_config.dense_queries = args.dense_queries
    if args.init_checkpoint is not None:
        train_config.init_from_checkpoint = args.init_checkpoint
    if args.warmup_steps is not None:
        train_config.warmup_steps = args.warmup_steps
    if args.sparsity_reg is not None:
        train_config.sparsity_reg = args.sparsity_reg
    if args.dense_layers is not None:
        train_config.dense_layers = args.dense_layers

    return train_config


MODELS = {
    "llama32k": "togethercomputer/LLaMA-2-7B-32K",
    "llama13b": "meta-llama/Llama-2-13b-hf",
    "llama13b_32k": "Yukang/Llama-2-13b-longlora-32k-ft",
    "qwen7b": "Qwen/Qwen1.5-7B-Chat",
    "qwen14b": "Qwen/Qwen1.5-14B-Chat",
    "yi6b": "01-ai/Yi-6B-200K",
    "yi34b": "01-ai/Yi-34B-200K",
    "giraffe13b": "abacusai/Giraffe-13b-32k-v3",
}


def load_model(
    train_config: TrainConfig = None,
    method="hip",
    device=None,
    is_teacher=False,
):
    device_map = "auto"
    max_memory = None

    if os.environ.get("LOCAL_RANK", None) is not None:
        train_config.local_rank = int(os.environ["LOCAL_RANK"])

    if device is None:
        if train_config.local_rank is not None:
            if train_config.model_parallel is not None:
                device_map = "auto"
                max_memory = {
                    train_config.local_rank * train_config.model_parallel + i: "70GiB"
                    for i in range(train_config.model_parallel)
                }
            else:
                device_map = {"": f"cuda:{train_config.local_rank}"}
        else:
            device_map = {"": torch.cuda.current_device()}
    if train_config.using_fsdp:
        device_map = "cpu"
    print("Device map:", device_map, "max_memory:", max_memory)

    assert train_config.model in MODELS, MODELS.keys()
    model_id = MODELS[train_config.model]

    ConfigClass = LlamaConfig
    if "qwen" in train_config.model:
        ConfigClass = Qwen2Config

    config = ConfigClass.from_pretrained(model_id)
    config._attn_implementation = config.attn_implementation = (
        "eager" if "qwen" in train_config.model else "sdpa"
    )

    quant_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=["tree_avgpool_scaler"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    if train_config.using_fsdp:
        quant_config = None

    ModelClass = LlamaForCausalLM
    if "qwen" in train_config.model:
        ModelClass = Qwen2ForCausalLM

    model = ModelClass.from_pretrained(
        model_id,
        config=config,
        device_map=device_map,
        max_memory=max_memory,
        load_in_4bit=None if quant_config is not None else True,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    for m in model.modules():
        if hasattr(m, "attention_method"):
            m.attention_method = method
            m.tree_k = train_config.k
            m.tree_block_size_q = train_config.block_size_q
            m.tree_block_size_k = train_config.block_size_k
            if train_config.dense_queries is None:
                train_config.dense_queries = train_config.k
            m.tree_dense_queries = train_config.dense_queries
            m.tree_dense_layers = list(range(train_config.dense_layers))
            m.tree_enable_sparq = train_config.enable_sparq
            m.tree_enable_flash = train_config.enable_flash
            m.tree_use_sliding_window = train_config.use_sliding_window
        if hasattr(m, "gradient_checkpointing"):
            m.gradient_checkpointing = True
            if train_config.using_fsdp:
                m._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
            elif train_config.using_deepspeed:
                import deepspeed

                m._gradient_checkpointing_func = deepspeed.checkpointing.checkpoint
            else:
                m._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

    if not is_teacher:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_config.lora_r,
            lora_alpha=train_config.lora_r // 2,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                # 'input_layernorm', 'post_attention_layernorm'
            ],
            modules_to_save=[
                "tree_avgpool_scaler",
                "input_layernorm",
                "post_attention_layernorm",
            ],
        )

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        if train_config.init_from_checkpoint is not None:

            print(f"Loading peft model from {train_config.init_from_checkpoint}")

            if pathlib.Path(train_config.init_from_checkpoint).is_dir():
                model = PeftModel.from_pretrained(
                    model, train_config.init_from_checkpoint
                )
                model.print_trainable_parameters()

            else:
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()

                print("loading from", train_config.init_from_checkpoint)
                state_dict = torch.load(
                    train_config.init_from_checkpoint, map_location="cpu"
                )
                if "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                keys = list(state_dict.keys())
                for key in keys:
                    x = state_dict[key]
                    state_dict[key.strip("model.")] = x
                    del state_dict[key]
                try:
                    result = model.load_state_dict(state_dict, strict=False)
                    print("load result", result)
                except RuntimeError as e:
                    pass
                print("lora checkpoint loaded from", train_config.init_from_checkpoint)

        else:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

    return model


def load_tokenizer(model):
    assert model in MODELS, MODELS.keys()
    model_id = MODELS[model]
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
