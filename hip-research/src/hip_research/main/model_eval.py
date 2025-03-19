import os
import pathlib
import warnings

import torch
import transformers
from hip_research.main.eval_args import ArgsType, eval_args
from hip_research.main.jobs.bench_single_layer import job_bench_single_layer
from hip_research.main.jobs.booksum import job_booksum
from hip_research.main.jobs.ga import job_ga
from hip_research.main.jobs.greedy_replace import job_greedy_replace
from hip_research.main.jobs.merge_lora import job_merge_lora
from hip_research.main.jobs.mmlu import job_mmlu
from hip_research.main.jobs.passkey import job_passkey
from hip_research.main.jobs.ppl import job_ppl
from hip_research.main.jobs.sample_diag import job_sample_diag
from hip_research.main.jobs.stream import job_stream
from hip_research.main.jobs.stream_demo import job_stream_demo
from hip_research.models.sglang_model import SglangModel
from hip_research.utils.seed import seed
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from hip_attn.models.gemma.modeling_gemma2 import Gemma2Config, Gemma2ForCausalLM
from hip_attn.models.modeling_llama import LlamaConfig, LlamaForCausalLM
from hip_attn.models.qwen.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM

MODELS = {
    "llama1b": "princeton-nlp/Sheared-LLaMA-1.3B",
    "llama3b": "princeton-nlp/Sheared-LLaMA-2.7B",
    "llama7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama32k": "togethercomputer/LLaMA-2-7B-32K",
    "llama32k_instruct": "togethercomputer/Llama-2-7B-32K-Instruct",
    "llama13b": "meta-llama/Llama-2-13b-hf",
    "llama13b_32k": "Yukang/Llama-2-13b-longlora-32k-ft",
    "llama13b_32k_instruct": "Yukang/Llama-2-13b-chat-longlora-32k-sft",
    "llama3_8b_1m": "gradientai/Llama-3-8B-Instruct-Gradient-1048k",
    "llama3.1_8b": "meta-llama/Meta-Llama-3.1-8B",
    "llama3.1_8b_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama3.2_1b": "meta-llama/Llama-3.2-1B",
    "llama3.2_3b": "meta-llama/Llama-3.2-3B",
    "llama3.2_3b_instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2_1b_instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "qwen14b": "Qwen/Qwen1.5-14B-Chat",
    "qwen7b": "Qwen/Qwen1.5-7B-Chat",
    "qwen1.5b": "Qwen/Qwen1.5-1.8B-Chat",
    "qwen0.5b": "Qwen/Qwen1.5-0.5B-Chat",
    "qwen2.5_1.5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5_3b_instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
    "gemma2_2b": "google/gemma-2-2b",
    "gemma2_9b": "google/gemma-2-9b",
    "gemma2_2b_it": "google/gemma-2-2b-it",
    "gemma2_9b_it": "google/gemma-2-9b-it",
    "exaone3.5_7.8b_instruct": "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
}

OBSOLATED_VLLM_MODELS = {
    "vllm_llama32k": "togethercomputer/LLaMA-2-7B-32K",
    "vllm_llama32k_instruct": "togethercomputer/Llama-2-7B-32K-Instruct",
    "vllm_llama128k": "NousResearch/Yarn-Llama-2-7b-128k",
    "vllm_llama13b_128k": "NousResearch/Yarn-Llama-2-13b-128k",
    "vllm_llama13b_32k": "Yukang/Llama-2-13b-longlora-32k-ft",
    "vllm_llama13b_32k_instruct": "Yukang/Llama-2-13b-chat-longlora-32k-sft",
    "vllm_llama100k": "Yukang/Llama-2-7b-longlora-100k-ft",
    "vllm_llama1b": "princeton-nlp/Sheared-LLaMA-1.3B",
    "vllm_llama7b": "meta-llama/Llama-2-7b-hf",
    "vllm_llama13b": "meta-llama/Llama-2-13b-hf",
    # 'vllm_qwen14b': 'Qwen/Qwen1.5-14B-Chat-GPTQ-Int4',
    "vllm_qwen14b_local": "./Qwen1.5-14B-Chat-GPTQ-Int4",
    "vllm_qwen14b_int8_local": "./Qwen1.5-14B-Chat-GPTQ-Int8",
    "vllm_qwen14b_noquant_local": "./Qwen1.5-14B-Chat",
    "vllm_qwen7b": "Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",
    "vllm_qwen7b_pt": "Qwen/Qwen1.5-7B",
    "vllm_qwen14b": "Qwen/Qwen1.5-14B-Chat",
    "vllm_qwen14b_gptq": "Qwen/Qwen1.5-14B-Chat-GPTQ-Int4",
    "vllm_qwen0.5b": "Qwen/Qwen1.5-0.5B-Chat",
    "vllm_pythia70m": "EleutherAI/pythia-70m",
    "vllm_yi6b": "01-ai/Yi-6B-200K",
    "vllm_yi34b": "brucethemoose/Yi-34B-200K-RPMerge",
    "vllm_mixtral8x7b": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    "vllm_gemma2b": "google/gemma-2b-it",
    "vllm_gemma7b": "google/gemma-7b-it",
    "vllm_luxia21.4b": "saltlux/luxia-21.4b-alignment-v1.1",
    "vllm_llama3_8b": "unsloth/llama-3-8b-Instruct",
    "vllm_yi1.5_9b_32k": "01-ai/Yi-1.5-9B-32K",
    "vllm_llama3.1_8b_instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "vllm_llama3.1_8b_instruct_awq": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
}


def load_vllm_model(args: ArgsType):
    from vllm import LLM

    if int(os.getenv("HIP_K", "512")) != args.k:
        warnings.warn(
            f'WARN!!! your command line argument of hip_k is {args.k} but environment variable is {os.getenv("HIP_K", "512")}. OS environment is higher priority.'
        )

    device = "cuda:0"
    if args.model.replace("vllm_", "") in MODELS:
        model_id = MODELS[args.model.replace("vllm_", "")]
    elif args.model in OBSOLATED_VLLM_MODELS:
        model_id = OBSOLATED_VLLM_MODELS[args.model]
    else:
        model_id = args.model.replace("vllm_", "")
    print(f"Loading model {model_id}")

    assert args.checkpoint is None

    seq_len = args.stride
    assert seq_len > 0
    # seq_len = 10600
    model = LLM(
        model_id,
        max_num_seqs=args.batch_size,
        max_seq_len_to_capture=seq_len,
        max_model_len=seq_len,
        swap_space=0,
        kv_cache_dtype=os.getenv("KV_CACHE_DTYPE", "fp8_e5m2"),
        dtype="half",
        gpu_memory_utilization=float(os.getenv("MEM_UTIL", "0.9")),
        tensor_parallel_size=torch.cuda.device_count(),
        enforce_eager=os.environ.get("ENFORCE_EAGER", "0") == "1",
        trust_remote_code=True,
        max_num_batched_tokens=seq_len,
        enable_chunked_prefill=False,
        # observability_config=ObservabilityConfig(
        #     collect_model_forward_time=True,
        #     collect_model_execute_time=True
        # )
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.bos_token
        tokenizer.eos_token_id = tokenizer.bos_token_id

    return model, tokenizer, device


def load_sglang_model(args: ArgsType):
    model_name = args.model.replace("sglang_", "")

    assert model_name in MODELS, f"Available Models: {list(MODELS.keys())}"

    model_id = MODELS[model_name]

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id

    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.bos_token
        tokenizer.eos_token_id = tokenizer.bos_token_id

    model = SglangModel(args.endpoint, tokenizer)

    return model, tokenizer, torch.device("cpu")


def load_model(args):
    if args.model.startswith("vllm"):
        return load_vllm_model(args)
    if args.model.startswith("sglang"):
        return load_sglang_model(args)

    device = "cuda:0"
    if args.model in MODELS:
        model_id = MODELS[args.model]
    else:
        model_id = args.model

    if args.model.startswith("qwen"):
        config = Qwen2Config.from_pretrained(model_id)
        config._attn_implementation = config.attn_implementation = "sdpa"
        ModelClass = Qwen2ForCausalLM
    elif args.model.startswith("gemma2") or ("gemma" in args.model):
        config = Gemma2Config.from_pretrained(model_id)
        config._attn_implementation = config.attn_implementation = "sdpa"
        ModelClass = Gemma2ForCausalLM
    else:
        config = LlamaConfig.from_pretrained(model_id)
        config._attn_implementation = config.attn_implementation = "sdpa"
        ModelClass = LlamaForCausalLM

    print(f"Loading model {model_id} {ModelClass} {type(config)}")

    if torch.cuda.is_bf16_supported():
        infer_dtype = torch.bfloat16
    else:
        infer_dtype = torch.float16

    if os.getenv("FORCE_FP32", "0") == "1":
        infer_dtype = torch.float32

    if args.method in ["h2o", "h2o_stream"]:
        from hip_research.models.h2o.h2o_llama import H2OLlamaForCausalLM

        ModelClass = H2OLlamaForCausalLM

        if args.method == "h2o_stream":
            args.h2o_streaming = True
        config.attention_method = args.method
        config.hh_size = args.k // 2
        config.recent_size = args.k // 2
        config._attn_implementation = config.attn_implementation = "eager"
        config.h2o_shift_q_pos = args.h2o_shift_q_pos
        config.h2o_streaming = args.h2o_streaming
        config.reduction_for_gqa = args.h2o_reduce_for_gqa
        config.tree_dense_layers = list(range(args.dense_layers))
        config.tree_k = args.k

        if args.job not in ["stream", "passkey"]:  # TODO ga?
            config.is_decoding = False
        else:
            config.is_decoding = True

    if args.method == "tova":
        from transformers.models.llama.modeling_llama import (
            LlamaForCausalLM as OriginalLlamaForCausalLM,
        )

        ModelClass = OriginalLlamaForCausalLM

    model = ModelClass.from_pretrained(
        model_id,
        config=config,
        device_map={"": device},
        quantization_config=(
            transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_skip_modules=[
                    "tree_avgpool_scaler",
                    "lm_head",
                ],
                bnb_4bit_compute_dtype=infer_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if not args.no_quantize
            else None
        ),
        torch_dtype=infer_dtype,
        # torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    if args.method == "tova":
        from hip_research.models.tova.convert_tova import enable_tova_caching

        enable_tova_caching(model)

    for m in model.modules():
        if hasattr(m, "attention_method"):
            m.attention_method = args.method
            m.tree_k = args.k
            m.tree_block_size_q = args.block_size_q
            m.tree_block_stride_q = args.block_stride_q
            m.tree_block_size_k = args.block_size_k
            m.tree_block_stride_k = args.block_stride_k
            m.tree_using_context_avg = False
            m.tree_dense_queries = args.dense_queries
            m.tree_dense_layers = list(range(args.dense_layers))
            m.tree_rope_method = args.rope_method
            m.tree_enable_sparq = not args.disable_sparq
            m.tree_enable_flash = not args.disable_flash
            m.tree_use_sliding_window = not args.disable_sliding_window
            m.tree_sampling_method = args.sampling_method

    if args.method != "none" and args.checkpoint is not None:
        if pathlib.Path(args.checkpoint).is_dir():
            # is peft checkpoint
            # Load peft pretrained
            print(f"Loading peft model from {args.checkpoint}")
            model = PeftModel.from_pretrained(model, args.checkpoint)

        else:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=True,
                r=args.lora_r,
                lora_alpha=args.lora_r // 2,
                lora_dropout=0.0,
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

            model = get_peft_model(model, peft_config)

            state_dict = torch.load(args.checkpoint, map_location="cpu")
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

        # model = model.to(infer_dtype)
        print("lora checkpoint loaded from", args.checkpoint)

    elif args.method != "none":
        for m in model.modules():
            if hasattr(m, "attention_method"):
                m.tree_using_context_avg = False

    model = model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

    return model, tokenizer, device


JOBS = {
    "ppl": job_ppl,
    "stream": job_stream,
    "mmlu": job_mmlu,
    "bench_single_layer": job_bench_single_layer,
    "booksum": job_booksum,
    "merge_lora": job_merge_lora,
    "stream_demo": job_stream_demo,
    "greedy_replace": job_greedy_replace,
    "passkey": job_passkey,
    "ga": job_ga,
    "sample_diag": job_sample_diag,
}


def main():
    args = eval_args()
    seed(seed=args.seed)

    assert args.job in JOBS.keys()

    model, tokenizer, device = load_model(args)

    JOBS[args.job](args, model, tokenizer, device)


if __name__ == "__main__":
    main()
