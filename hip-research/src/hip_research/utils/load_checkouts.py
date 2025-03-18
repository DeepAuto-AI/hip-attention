import torch


def load_checkouts(
    idx=24,
    window=1,
    seq_len=4096,
    dtype=torch.float16,
    return_cos_sin=False,
    derope=False,
    sm_scale=None,
    device=0,
    checkout_path="./cache/llama/qkvout.pth",
):
    q_name = "q"
    k_name = "k"
    if derope:
        q_name = "q_derope"
        k_name = "k_derope"

    data_source = "llama"
    if data_source == "llama":
        state = torch.load(checkout_path, map_location="cpu", weights_only=False)
        if sm_scale is None:
            sm_scale = 1 / (state[q_name].shape[-1] ** 0.5)
        q = state[q_name] * sm_scale
        k = state[k_name]
        v = state["v"]
        out = state["out"]
        cos = state["cos"]
        sin = state["sin"]
        assert cos.shape[0] == 1
        assert sin.shape[0] == 1
        N, H, T_DST, HID = q.shape
        N, H_KV, T_SRC, HID = k.shape
        q = q.view(N * H, T_DST, HID)[idx : idx + window, :seq_len].contiguous()
        k = k.view(N * H_KV, T_SRC, HID)[idx : idx + window, :seq_len].contiguous()
        v = v.view(N * H_KV, T_SRC, HID)[idx : idx + window, :seq_len].contiguous()
        out = out.view(N * H, T_DST, HID)[idx : idx + window, :seq_len].contiguous()
        cos = cos.view(-1, HID)[:seq_len, :].contiguous()
        sin = sin.view(-1, HID)[:seq_len, :].contiguous()
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
