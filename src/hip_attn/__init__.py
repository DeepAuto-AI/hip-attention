from hip_attn.models.modeling_llama import LlamaForCausalLM as HipLlamaForCausalLM
from hip_attn.utils.attention import custom_attention
from hip_attn.v1_2.attention_extend import (
    dual_stage_quadratic_hip_attention as hip_attention,
)
from hip_attn.v1_2.attention_metadata import (
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
)

__version__ = "1.2.0"

__all__ = [
    "hip_attention",
    "HiPAttentionArgs",
    "HiPAttentionOutputMetadata",
    "custom_attention",
    "HipLlamaForCausalLM",
]
