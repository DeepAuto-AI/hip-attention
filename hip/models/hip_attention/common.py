import torch

def load_checkouts(idx = 24, window = 1, seq_len = 4096, dtype = torch.float16, return_cos_sin = False):
    data_source = 'llama'
    device = 0
    if data_source == 'llama':
        state = torch.load('./cache/llama/qkvout.pth', map_location='cpu', weights_only=False)
        q = state['q'] / (state['q'].shape[-1] ** 0.5)
        k = state['k']
        v = state['v']
        out = state['out']
        cos = state['cos']
        sin = state['sin']
        assert cos.shape[0] == 1
        assert sin.shape[0] == 1
        N, H, T_DST, HID = q.shape
        N, H_KV, T_SRC, HID = k.shape
        q = q.view(N*H, T_DST, HID)[idx:idx+window, :seq_len].contiguous()
        k = k.view(N*H_KV, T_SRC, HID)[idx:idx+window, :seq_len].contiguous()
        v = v.view(N*H_KV, T_SRC, HID)[idx:idx+window, :seq_len].contiguous()
        out = out.view(N*H, T_DST, HID)[idx:idx+window, :seq_len].contiguous()
        cos = cos.view(-1, HID)[:, :].contiguous()
        sin = sin.view(-1, HID)[:, :].contiguous()
    else:
        q = torch.randn((1, 64, 4))
        k = torch.randn((1, 64, 4))
        v = k.clone()
        out = q.clone()
    
    q = q.to(device, dtype=dtype)
    k = k.to(device, dtype=dtype)
    v = v.to(device, dtype=dtype)
    out = out.to(device, dtype=dtype)
    
    if not return_cos_sin:
        return q, k, v, out, None, None
    else:
        cos = cos.to(device, dtype=dtype)
        sin = sin.to(device, dtype=dtype)
        return q, k, v, out, cos, sin