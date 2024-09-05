import torch
from diffusers import FluxPipeline
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from diffusers.models.attention_processor import Attention, apply_rope
from typing import Optional
import torch.nn.functional as F

from hip import hip_attention, HiPAttentionArgs

def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

    query = attn.to_q(hidden_states)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # Apply RoPE if needed
    if image_rotary_emb is not None:
        # YiYi to-do: update uising apply_rotary_emb
        # from ..embeddings import apply_rotary_emb
        # query = apply_rotary_emb(query, image_rotary_emb)
        # key = apply_rotary_emb(key, image_rotary_emb)
        query, key = apply_rope(query, key, image_rotary_emb)

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    # TODO: add support for attn.scale when we move to Torch 2.1
    hidden_states = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    return hidden_states

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16,
).to('cuda')

for name, module in pipe.transformer.named_modules():
    if isinstance(module, Attention):
        print(module.processor)
        print(name, type(module))
        module.processor.__call__ = __call__

# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")