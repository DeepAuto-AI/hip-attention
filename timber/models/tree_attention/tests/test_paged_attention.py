import torch
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

def main():
    test_vllm()
    
if __name__ == '__main__':
    main()