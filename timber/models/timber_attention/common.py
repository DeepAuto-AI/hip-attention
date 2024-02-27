import torch

def load_checkouts(idx = 24, window = 1, seq_len = 4096, dtype = torch.float16):
    data_source = 'llama'
    device = 0
    if data_source == 'llama':
        state = torch.load('./cache/llama/qkvout.pth', map_location='cpu')
        q = state['q'] / (state['q'].shape[-1] ** 0.5)
        k = state['k']
        v = state['v']
        out = state['out']
        N, H, T_DST, HID = q.shape
        N, H, T_SRC, HID = k.shape
        q = q.view(N*H, T_DST, HID)[idx:idx+window, :seq_len].contiguous() # CHECK - idx:idx+window
        k = k.view(N*H, T_SRC, HID)[idx:idx+window, :seq_len].contiguous()
        v = v.view(N*H, T_SRC, HID)[idx:idx+window, :seq_len].contiguous()
        out = out.view(N*H, T_DST, HID)[idx:idx+window, :seq_len].contiguous()
    else:
        q = torch.randn((1, 64, 4))
        k = torch.randn((1, 64, 4))
        v = k.clone()
        out = q.clone()
    
    q = q.to(device, dtype=dtype)
    k = k.to(device, dtype=dtype)
    v = v.to(device, dtype=dtype)
    out = out.to(device, dtype=dtype)
    
    return q, k, v, out