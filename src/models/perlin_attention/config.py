import json
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim

from ..hf_bert import BertConfig

@dataclass
class PerlinAttentionConfig:
    reformer_n_hashs: int = 8
    performer_nb_factor: int = 1
    k: int = 7
    k_flatten: bool = True
    k_flatten_dim: str = 'causal_batch'
    random_lookup: bool = False
    random_lookup_count: int = 3
    attention_predictor_method: str = 'mlp'
    attention_predictor_length: int = 128
    attention_predictor_backend: str = 'performer'
    attention_predictor_comp_book_size: int = 8
    attention_predictor_comp_patch_size: int = 16
    attention_predictor_comp_patch_count: int = 16
    attention_predictor_enc_per_layer: bool = False
    layerwise: bool = False
    lora_r: int = 32
    lora_enabled: bool = False
    lora_in_approx_enabled: bool = False
    partial_attention_scaler: bool = True
    out_add_performer_context: bool = False
    v_eye_length: int = 128
    out_norm: bool = False
    causal: bool = False
    use_cache: bool = False
    compile: bool = False
    context_output_method: str = 'mix'
    k_oversample: float = 1.0
    
    def to_json(self):
        return asdict(self)
    
    def check_validity(self):
        if self.causal:
            if self.k_flatten:
                assert self.k_flatten_dim in ['causal_batch']

    def __repr__(self) -> str:
        return f"PerlinAttentionConfig({json.dumps(self.to_json())})"
    
DEFAULT_CONFIG = PerlinAttentionConfig()

def register_default_config(config: PerlinAttentionConfig):
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = config
    
def get_default_config() -> PerlinAttentionConfig:
    global DEFAULT_CONFIG
    return DEFAULT_CONFIG