import torch
from timber.models.timber_attention.attention1_block_gpu import paged_timber_attention
from vllm._C import ops

def load_states():
    return torch.load('./cache/llama/vllmout.pth', map_location='cuda:0')

_PARTITION_SIZE = 512

def test_vllm():
    state = load_states()
    query = state['query']
    key_cache = state['key_cache']
    value_cache = state['value_cache']
    input_metadata = state['input_metadata']
    num_kv_heads = state['num_kv_heads']
    scale = state['scale']
    alibi_slopes = state['alibi_slopes']
    assert alibi_slopes is None
    output_truth = state['output']
    
    output = torch.empty_like(query)

    block_size = value_cache.shape[3]

    ops.paged_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        input_metadata.block_tables,
        input_metadata.context_lens,
        block_size,
        input_metadata.max_context_len,
        alibi_slopes,
        input_metadata.kv_cache_dtype,
    )
    
    error = torch.abs(output - output_truth).mean()
    print(torch.std_mean(output_truth), error)

def test_vllm_compat():
    state = load_states()
    query = state['query']
    key_cache = state['key_cache']
    value_cache = state['value_cache']
    input_metadata = state['input_metadata']
    num_kv_heads = state['num_kv_heads']
    scale = state['scale']
    alibi_slopes = state['alibi_slopes']
    assert alibi_slopes is None
    output_truth = state['output']
    
    output = paged_timber_attention(
        q=query,
        q_scale=scale,
        k=key_cache,
        v=value_cache,
        block_tables=input_metadata.block_tables,
        context_lens=input_metadata.context_lens,
        max_context_len=input_metadata.max_context_len,
    )
    
    error = torch.abs(output - output_truth).mean()
    print(torch.std_mean(output_truth), error)

def main():
    test_vllm()
    
if __name__ == '__main__':
    main()