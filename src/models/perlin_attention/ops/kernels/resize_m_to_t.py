import os
import random
import torch, math
import torch.nn.functional as F

def resize_from_m_to_t(
    x: torch.Tensor, 
    masked_fill_value: float, 
    attention_mask: torch.Tensor, 
    target_width: int=None, 
    training=False,
    is_causal=True,
    k=None,
    oversampled=None,
):
    assert masked_fill_value is not None
    N, H, T1, T_M = x.shape
    _N, _H, _TQ, _TK = attention_mask.shape
    assert _H == 1
    
    if target_width is not None:
        T2 = target_width
    else:
        T2 = T1
    
    if is_causal:
        assert attention_mask.shape[-2:] == (T1, T2)
        assert attention_mask.ndim == 4
        assert attention_mask.shape == (N, 1, T1, T2)
    else:
        assert attention_mask.shape[-1:] == (T2,), f"{attention_mask.shape} == {T2}"
        assert attention_mask.ndim == 4
        assert attention_mask.shape == (N, 1, 1, T2)
        attention_mask = attention_mask.expand(N, 1, T1, T2)
    
    mask = (attention_mask > -1).float()
    mask_cs = mask.cumsum(-1)
    token_length = mask_cs[:, :, :, -1].unsqueeze(-1) 
    if training:
        if random.random() < 0.1:
            mask_cs = torch.clamp(
                mask_cs + (torch.rand_like(mask_cs) * 1.5 - 0.75), 
                torch.ones((1,1,1,1,), device=x.device), 
                mask_cs.max(dim=-1, keepdim=True)[1]
            )
    token_index_x = torch.floor(((mask_cs - 1) + 0.5) / token_length * T_M - 1e-4).to(torch.long) + ((1 - mask) * T_M).to(torch.long)
    token_index_x = torch.clamp(token_index_x, 0, T_M)
    token_index_x = token_index_x.expand(N, H, T1, T2)
    
    grid_input = F.pad(x, pad=(0, 1), value=masked_fill_value)
    assert grid_input.shape[-1] == (T_M + 1)
    output = grid_input.gather(dim=-1, index=token_index_x)
    
    if oversampled is not None:
        # if oversampled in compressed one, we should undersample on uncompressed one.
        # so mask out some pixels in perticular X index.
        assert isinstance(oversampled, (float, int))
        assert isinstance(k, (int, float))
        
        N, H, T1, T2 = output.shape
        
        xs = torch.arange(0, T2, device=token_length.device).view(1, 1, 1, T2)
        ws = token_length # current width
        ps = torch.clamp_min(torch.round(token_length / oversampled), 1) # how many total pixels
        
        # rounding error is smaller than ws pixel in ps
        oys = torch.clamp(token_length, round(k), round(k*oversampled)) / k
        # print(oys)
        mask = torch.abs(((xs + 1) / ws * ps) - torch.round((xs + 1) / ws * ps)) <= ((1 / oys) * 0.5 + 1e-4)
        # print(mask)
        output.masked_fill_(~mask, value=masked_fill_value)
    
    return output

def test_main():
    N = 4
    H = 12
    MIN_T_SRC = 16
    T_SRC = 128
    T_DST = 128
    T_M = 32
    FP_MIN = torch.finfo(torch.float16).min * 0.5
    device = 0
    
    def rand_perlin_2d(shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])
        
        grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim = -1) % 1
        angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1)
        gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
        
        tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
        dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
        
        n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
        n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
        n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
        n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
        t = fade(grid[:shape[0], :shape[1]])
        return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])
    image = F.interpolate(rand_perlin_2d((128, 128), (16, 16)).view(1, 1, 128, 128), (T_DST, T_M)).expand(N, H, T_DST, T_M).contiguous().to(device)
    
    # test resize m to t in BERT
    attention_mask = torch.full((N, 1, 1, T_SRC), FP_MIN, device=device)
    for i in range(N):
        attention_mask[i, :, :, :random.randint(MIN_T_SRC, T_SRC)] = 0
    
    resized_score = resize_from_m_to_t(
        x=image,
        masked_fill_value=7,
        attention_mask=attention_mask,
        target_width=T_SRC,
        training=True,
        is_causal=False,
    )
    
    os.makedirs('./saves/tests/ops/resize_m_to_t', exist_ok=True)
    torch.save({
        'image': image,
        'mask': attention_mask,
        'resized':resized_score,
    }, './saves/tests/ops/resize_m_to_t/state.pth')
    print('saved sample ./saves/tests/ops/resize_m_to_t/state.pth')

if __name__ == '__main__':
    test_main()