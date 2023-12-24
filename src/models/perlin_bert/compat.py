import torch

BERT_SELF_ATTENTION_TO_PERLIN_SELF_ATTENTION = """
self.perlin_query_lora -> self.perlin_self_attention.perlin_query_lora
self.perlin_key_lora -> self.perlin_self_attention.perlin_key_lora
self.perlin_value_lora -> self.perlin_self_attention.perlin_value_lora
self.perlin_query_lora_for_approx_atten -> self.perlin_self_attention.perlin_query_lora_for_approx_atten
self.perlin_key_lora_for_approx_atten -> self.perlin_self_attention.perlin_key_lora_for_approx_atten
self.perlin_value_lora_for_approx_atten -> self.perlin_self_attention.perlin_value_lora_for_approx_atten
self.perlin_query_lora_for_approx_score -> self.perlin_self_attention.perlin_query_lora_for_approx_score
self.perlin_key_lora_for_approx_score -> self.perlin_self_attention.perlin_key_lora_for_approx_score
self.perlin_performer -> self.perlin_self_attention.attention.perlin_performer
self.perlin_performer_proj_updater -> self.perlin_self_attention.attention.perlin_performer_proj_updater
self.perlin_attention_predictor_enc -> self.perlin_self_attention.attention.perlin_attention_predictor_enc
self.perlin_attention_predictor_dec_row -> self.perlin_self_attention.attention.perlin_attention_predictor_dec_row
self.perlin_attention_predictor_dec_scaler -> self.perlin_self_attention.attention.perlin_attention_predictor_dec_scaler
self.perlin_attention_predictor_comp_codebook -> self.perlin_self_attention.attention.perlin_attention_predictor_comp_codebook
self.perlin_attention_predictor_comp_enc -> self.perlin_self_attention.attention.perlin_attention_predictor_comp_enc
self.perlin_attention_predictor_comp_dec_row -> self.perlin_self_attention.attention.perlin_attention_predictor_comp_dec_row
self.perlin_out -> self.perlin_self_attention.attention.perlin_out
self.perlin_out_random_lookup -> self.perlin_self_attention.attention.perlin_out_random_lookup
self.perlin_norm -> self.perlin_self_attention.attention.perlin_norm
"""

def parse_migration(defs: str):
    migrate_keys = {}
    defs = defs.strip().splitlines()
    for rule in defs:
        spl = rule.split("->")
        from_key, to_key = spl[0], spl[-1]
        migrate_keys[from_key.strip()] = to_key.strip()
    return migrate_keys

def perform_key_migration(state_dict: dict, migrate_keys: dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        for k, v in migrate_keys.items():
            if k in key:
                new_key = key.replace(k, v)
                # print(key, '->', new_key)
                key = new_key
                break
        new_state_dict[key] = value
    return new_state_dict

def migrate_state_dict_bert_to_perlin(state_dict: dict):
    return perform_key_migration(
        state_dict=state_dict, 
        migrate_keys=parse_migration(
            BERT_SELF_ATTENTION_TO_PERLIN_SELF_ATTENTION
        )
    )

def migrate_state_dict(state_dict: dict):
    # TODO: more migration...
    state_dict = migrate_state_dict_bert_to_perlin(state_dict)
    return state_dict