import argparse
import random
import torch.distributed
import tqdm
from transformers import AutoTokenizer
from hip.models.hip_attention.offload_runner.llama_model import LlamaForCausalLM, LlamaDecoderLayer, LlamaAttention
import torch, time, os
from typing import List, Optional, Dict, Union, Any, Tuple
from transformers.cache_utils import Cache, PretrainedConfig, is_torchdynamo_compiling
from transformers import BitsAndBytesConfig
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

        >>> inputs = tokenizer(text="My name is GPT2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> past_kv_length = outputs.past_key_values # access cache filled with key/values from generation
        ```
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None, share=1) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (max_batch_size, self.max_cache_len, self.num_key_value_heads, self.head_dim)
        self.share = share
        for idx_group in range(config.num_hidden_layers // share):
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=device)
            for idx_share in range(share):
                self.key_cache.append(new_layer_key_cache)
                self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        # self.key_cache[layer_idx] = self.key_cache[layer_idx].to(device=key_states.device)
        # self.value_cache[layer_idx] = self.value_cache[layer_idx].to(device=value_states.device)
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        
        if 'batch_index' in cache_kwargs:
            ibatch = cache_kwargs['batch_index']
            k_out = k_out[ibatch:ibatch+1]
            v_out = v_out[ibatch:ibatch+1]

        if cache_position is None:
            raise Exception()
            k_out.copy_(key_states, non_blocking=True)
            v_out.copy_(value_states, non_blocking=True)
        else:
            if (layer_idx % self.share) == 0:
                k_out.index_copy_(1, cache_position, key_states.to(k_out.dtype))
                v_out.index_copy_(1, cache_position, value_states.to(k_out.dtype))

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()

def convert_llama_to_vllm(model: LlamaForCausalLM):
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models.llama import LlamaMLP
    for ilayer, layer in enumerate(model.model.layers):
        layer = layer # type: LlamaDecoderLayer
        input_layernorm = RMSNorm(layer.input_layernorm.weight.shape[0]).to(model.device).half()
        input_layernorm.load_state_dict(layer.input_layernorm.state_dict())
        output_layernorm = RMSNorm(layer.post_attention_layernorm.weight.shape[0]).to(model.device).half()
        output_layernorm.load_state_dict(layer.post_attention_layernorm.state_dict())
        layer.input_layernorm = input_layernorm
        layer.post_attention_layernorm = output_layernorm
        
        mlp = LlamaMLP(
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            hidden_act=model.config.hidden_act,
            bias=getattr(model.config, "mlp_bias", False),
            prefix=f"layer{ilayer}.mlp",
        ).to(model.device).half()
        mlp.down_proj.load_state_dict(layer.mlp.down_proj.state_dict())
        mlp.gate_up_proj.load_state_dict({
            'weight': torch.cat([
                layer.mlp.gate_proj.weight,
                layer.mlp.up_proj.weight,
            ], dim=0)
        })
        layer.mlp = mlp
        
        self_attn = layer.self_attn # type: LlamaAttention
        qkv_proj = QKVParallelLinear(
            model.config.hidden_size, 
            self_attn.head_dim, 
            self_attn.num_heads, 
            self_attn.num_key_value_heads, 
            self_attn.q_proj.bias is not None, 
            params_dtype=torch.float16
        ).to(model.device).half()
        qkv_proj.load_state_dict({
            'weight': torch.cat([
                self_attn.q_proj.weight,
                self_attn.k_proj.weight,
                self_attn.v_proj.weight,
            ], dim=0)
        })
        self_attn.q_proj = None
        self_attn.k_proj = None
        self_attn.v_proj = None
        self_attn.qkv_proj = qkv_proj
    
    norm = RMSNorm(model.model.norm.weight.shape[0]).to(model.device).half()
    norm.load_state_dict(model.model.norm.state_dict())
    model.model.norm = norm
    
    return model

class CUDACapture:
    def __init__(self, model):
        self.model = model
        self.graph = None
        
    def try_capture_and_forward(self, *args, **kwargs):
        if self.need_capture():
            self.capture(*args, **kwargs)
        return self.forward(*args, **kwargs)
    
    def capture(self, *args, **kwargs):
        assert self.graph is None
        
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                self.model.forward(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self.graph_output = self.model.forward(*args, **kwargs)
        self.graph = g
        self.graph_args = args
        self.graph_kwargs = kwargs
    
    def forward(self, *args, **kwargs):
        assert self.graph is not None
        
        for src, dst in zip(args, self.graph_args):
            if isinstance(src, torch.Tensor):
                dst.copy_(src, non_blocking=True)
        
        for key in kwargs:
            if isinstance(kwargs[key], torch.Tensor):
                src = kwargs[key]
                dst = self.graph_kwargs[key]
                assert src.shape == dst.shape, key
                dst.copy_(src, non_blocking=True)
        
        self.graph.replay()
        
        return self.graph_output

    def need_capture(self):
        return self.graph is None

class Runner:
    def __init__(self, model_id, method):
        import vllm.distributed
        import torch.distributed
        
        if not torch.distributed.is_initialized():
            os.environ['MASTER_PORT'] = str(random.randint(32000, 33000))
            os.environ['MASTER_ADDR'] = 'localhost'
            torch.distributed.init_process_group(world_size=1, rank=0)
            vllm.distributed.init_distributed_environment(1, 0, local_rank=0)
            vllm.distributed.initialize_model_parallel()
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model = LlamaForCausalLM.from_pretrained(
            model_id, 
            device_map={'': 'cuda:0'}, 
            torch_dtype=torch.float16,
            attn_implementation='flash_attention_2',
        )
        for module in model.modules():
            if isinstance(module, LlamaAttention):
                module.attention_method = method
        
        self.tokenizer = tokenizer
        self.model = convert_llama_to_vllm(model.half()).eval()
        self.method = method
        self.decode_step = 0
        
        self.capture = CUDACapture(self.model)
        
        self.capture_hip_refresh = CUDACapture(self.model)
        self.capture_hip_cache = CUDACapture(self.model)
        
        self.hip_refresh_interval = 8
    
    @torch.inference_mode(True)
    def decode_forward(self, *args, **kwargs):
        if self.method == 'hip':
            if self.capture_hip_refresh.need_capture():
                for m in self.model.modules():
                    if isinstance(m, LlamaAttention):
                        m.hip_cache = None
                        m.hip_last_cache = None
                        m.hip_use_cache = False
                        m.hip_checkout_cache = True
                self.capture_hip_refresh.capture(*args, **kwargs)
                
                for m in self.model.modules():
                    if isinstance(m, LlamaAttention):
                        assert m.hip_last_cache is not None
                        m.hip_cache = m.hip_last_cache
                        m.hip_use_cache = True
                        m.hip_checkout_cache = False
                self.capture_hip_cache.capture(*args, **kwargs)
                
                for m in self.model.modules():
                    if isinstance(m, LlamaAttention):
                        assert m.hip_cache is not None
                        m.hip_cache = None
                        m.hip_last_cache = None
                        m.hip_use_cache = False
                        m.hip_checkout_cache = False
            
            if (self.decode_step % self.hip_refresh_interval) == 0:
                return self.capture_hip_refresh.forward(*args, **kwargs)
            else:
                return self.capture_hip_cache.forward(*args, **kwargs)
        else:
            return self.capture.forward(*args, **kwargs)
    
    @torch.inference_mode(True)
    def sample(self, logits: torch.Tensor):
        logits = logits[:, -1:, :] # type: torch.Tensor
        next_token_id = logits.argmax(dim=-1)
        return next_token_id
    
    @torch.inference_mode(True)
    def generate(self, text, max_tokens=256, item_repeat=24, kv_share=1):
        input_ids = self.tokenizer([text, ] * item_repeat, return_tensors="pt", padding=True).input_ids.to(self.model.device)
        bsz, context_len = input_ids.shape
        
        cache = StaticCache(
            self.model.config,
            max_batch_size=bsz,
            max_cache_len=context_len + max_tokens, 
            device=self.model.device,
            dtype=torch.float16,
            share=kv_share,
        )
        
        # compile decode step
        decode_input_ids = torch.zeros((bsz, 1), dtype=torch.long, device=self.model.device)
        decode_cache_pos = torch.zeros((1, ), dtype=torch.long, device=self.model.device)
        with torch.autocast('cuda', torch.float16):
            self.decode_forward(
                input_ids=decode_input_ids, 
                position_ids=decode_cache_pos.unsqueeze(0).expand(bsz, 1), 
                cache_position=decode_cache_pos, 
                past_key_values=cache
            )
        
        cache.reset()
        self.decode_step = 0
        
        prompt_cache_pos = torch.arange(0, context_len, dtype=torch.long, device=self.model.device)
        decode_cache_pos.fill_(context_len)
        decoded_tokens = []
        
        torch.cuda.synchronize()
        
        event_prefill_start = torch.cuda.Event(True)
        event_prefill_end = torch.cuda.Event(True)
        event_decode_start = torch.cuda.Event(True)
        event_decode_end = torch.cuda.Event(True)
        
        logits = []
        event_prefill_start.record()
        
        ibatch = 0
        for module in self.model.modules():
            if isinstance(module, LlamaAttention):
                module.prompt_batch_index = ibatch
        prompt_output = self.model(
            input_ids=input_ids[ibatch:ibatch+1], 
            position_ids=prompt_cache_pos.unsqueeze(0).expand(1, -1), 
            cache_position=prompt_cache_pos, 
            past_key_values=cache,
            num_logits_to_keep=1,
        )
        for _ in range(bsz):
            logits.append(prompt_output.logits)
        
        for ilayer in range(len(cache.key_cache)):
            for ibatch in range(bsz):
                cache.key_cache[ilayer][ibatch].copy_(cache.key_cache[ilayer][0], non_blocking=True)
                cache.value_cache[ilayer][ibatch].copy_(cache.value_cache[ilayer][0], non_blocking=True)
        
        logits = torch.cat(logits, dim=0)
        next_token = self.sample(logits)
        decoded_tokens.append(next_token)
        decode_input_ids.copy_(next_token, non_blocking=True)
        del prompt_output
        event_prefill_end.record()
        
        event_decode_start.record()
        for _ in tqdm.tqdm(range(max_tokens), dynamic_ncols=True, leave=False, desc='decode'):
            with torch.autocast('cuda', torch.float16):
                decode_output = self.decode_forward(
                    input_ids=decode_input_ids, 
                    position_ids=decode_cache_pos.unsqueeze(0).expand(bsz, 1), 
                    cache_position=decode_cache_pos, 
                    past_key_values=cache
                )
            next_token = self.sample(decode_output.logits)
            decoded_tokens.append(next_token)
            decode_input_ids.copy_(next_token, non_blocking=True)
            decode_cache_pos.add_(1)
            if (self.decode_step % 10) == 0:
                torch.cuda.synchronize()
            self.decode_step += 1
        event_decode_end.record()
        
        torch.cuda.synchronize()
        
        elapsed_prefill = event_prefill_start.elapsed_time(event_prefill_end) / 1000
        elapsed_decode = event_decode_start.elapsed_time(event_decode_end) / 1000
        
        gen_out = torch.cat(decoded_tokens, dim=-1)
        text_outs = self.tokenizer.batch_decode(gen_out, skip_special_tokens=False)
        
        print(
            f"Time taken for {tuple(input_ids.shape)}:  "
            f"{input_ids.shape[-1] / elapsed_prefill:.2f} tok/s {elapsed_prefill:.2f} s  |  "
            f"{gen_out.numel() / elapsed_decode:.2f} tok/s {elapsed_decode:.2f} s"
        )
        return text_outs
    
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--method', default='hip', type=str)
        
        args = parser.parse_args()
        
        with open('./samples/32k.md', 'r') as f:
            document = f.read()
        
        sample_input = f'''<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

Hi, can you describe about following document? Here is document, 

```
{document}
```

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

'''
        results = Runner(
            'meta-llama/Meta-Llama-3.1-8B-Instruct',
            method=args.method,
        )\
            .generate(
                sample_input,
                item_repeat=16,
                kv_share=1,
            )
        for result in results[:8]:
            result = result.replace("\n", "\\n")
            print(f'{result[:80]} [...] {len(result)}')
    finally:
        import torch.distributed
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()