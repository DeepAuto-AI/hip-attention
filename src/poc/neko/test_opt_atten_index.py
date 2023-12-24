from ...models import perlin_opt
import transformers
import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm
import torch.nn.functional as F
import math, tqdm

model_config = 'facebook/opt-125m'

model = perlin_opt.OPTForCausalLM.from_pretrained(model_config)
for module in model.modules():
    if isinstance(module, perlin_opt.OPTAttention):
        module.attention_method = 'none'
        # module.attention_method = 'reformer'
        module.checkout_intermediates = True

tokenizer = transformers.AutoTokenizer.from_pretrained(model_config)
input_ids = tokenizer(
    """
The cat (Felis catus) is a domestic species of small carnivorous mammal.[1][2]
It is the only domesticated species in the family Felidae and is commonly referred 
to as the domestic cat or house cat to distinguish it from the wild members of the family.
[4] Cats are commonly kept as house pets but can also be farm cats or feral cats; the 
feral cat ranges freely and avoids human contact.[5] Domestic cats are valued by humans 
for companionship and their ability to kill vermin. About 60 cat breeds are recognized by 
various cat registries.[6] The cat is similar in anatomy to the other felid species: 
it has a strong flexible body, quick reflexes, sharp teeth, and retractable claws adapted 
to killing small prey like mice and rats. Its night vision and sense of smell are well developed. 
Cat communication includes vocalizations like meowing, purring, trilling, hissing, 
growling, and grunting as well as cat-specific body language. Although the cat is a 
social species, it is a solitary hunter. As a predator, it is crepuscular, i.e. most 
active at dawn and dusk. It can hear sounds too faint or too high in frequency for human ears, 
such as those made by mice and other small mammals.[7] It also secretes and perceives pheromones.[8]
Female domestic cats can have kittens from spring to late autumn, with litter 
sizes often ranging from two to five kittens.[9] Domestic cats are bred and shown at 
events as registered pedigreed cats, a hobby known as cat fancy. Population control 
of cats may be achieved by spaying and neutering, but their proliferation and the 
abandonment of pets has resulted in large numbers of feral cats worldwide, 
contributing to the extinction of entire bird, mammal, and reptile species.[10] It was 
long thought that cat domestication began in ancient Egypt, where cats were venerated 
from around 3100 BC,[11][12] but recent advances in archaeology and genetics 
have shown that their domestication occurred in the Near East around 7500 BC.[13]
As of 2021, there were an estimated 220 million owned and 480 million stray cats 
in the world.[14][15] As of 2017, the domestic cat was the second most popular 
pet in the United States, with 95.6 million cats owned[16][17] and around 42 million 
households owning at least one cat.[18] In the United Kingdom, 26%% of adults 
have a cat, with an estimated population of 10.9 million pet cats as of 2020.[19]""".replace('\n', '')*4, 
    truncation=True, 
    max_length=128, 
    return_tensors='pt'
).input_ids

print(input_ids.shape)

batch = {
    'input_ids': input_ids,
    'labels': input_ids,
    'output_attentions': True,
    'output_hidden_states': True,
}

with torch.no_grad():
    output = model(**batch)

root = './saves/neko/tests/test_perlin_opt/'
os.makedirs(root, exist_ok=True)

batch.update({
    'hidden_states': output.hidden_states,
    'attentions': output.attentions,
})
torch.save(batch, os.path.join(root, 'io.pth'))

def imsave(img: torch.Tensor, filename, gamma=0.2):
    img = img.cpu().numpy()
    
    def convert_to_colormap(arr: np.ndarray):
        def norm(x):
            return (x / (np.max(x) + 1e-12)) ** gamma
        T, T = arr.shape
        im = Image.fromarray((norm(cm.gist_earth((arr-np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-12)))*255).astype(np.uint8))
        arr = np.asarray(im)[:, :, :3]
        # arr = cv2.resize(arr, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        border = np.ones((arr.shape[0]+2, arr.shape[1]+2, arr.shape[2]), dtype=np.uint8)
        border = border * 255
        border[1:-1, 1:-1, :] = arr
        return border
    img = convert_to_colormap(img)
    
    path = os.path.join(root, filename)
    cv2.imwrite(path, img)
    
    print('saved', path)

render_attention = True
if render_attention:
    for ilayer, atten_layer in enumerate(output.attentions):
        # if ilayer >= 1: break
        layer = model.model.decoder.layers[ilayer] # type: perlin_opt.OPTDecoderLayer
        atten = layer.self_attn
        q, k, v, mask = atten.last_q, atten.last_k, atten.last_v, atten.last_attention_mask
        N_H, T, HID = q.shape
        mask = mask.view(1, T, T)
        print(q.shape, k.shape, v.shape, mask.shape)
        
        q_raw = q
        k_raw = k
        
        # q_std, q_mean = torch.std_mean(q, dim=-2, keepdim=True)
        # k_std, k_mean = torch.std_mean(q, dim=-2, keepdim=True)
        # q = (q - q_mean) / q_std
        # k = (k - k_mean) / k_std
        
        # idx_q = torch.topk(torch.abs(q) - torch.abs(q).mean(dim=-2, keepdim=True), k=16, dim=-1).indices #type: torch.Tensor
        # idx_k = torch.topk(torch.abs(k) - torch.abs(k).mean(dim=-2, keepdim=True), k=16, dim=-1).indices #type: torch.Tensor
        # q_top_mask = torch.zeros_like(q).scatter_(dim=-1, index=idx_q, value=1)
        # k_top_mask = torch.zeros_like(k).scatter_(dim=-1, index=idx_k, value=1)
        
        # # diff = (idx_q.unsqueeze(2) - idx_k.unsqueeze(1)).square().sum(-1)
        
        # q_sign = ((q > 0).float() - 0.5) * 2
        # k_sign = ((k > 0).float() - 0.5) * 2
        
        # q_norm = torch.norm(q_raw, p=2, dim=-1, keepdim=True).sqrt()
        # k_norm = torch.norm(k_raw, p=2, dim=-1, keepdim=True).sqrt()
        
        # diff = torch.bmm(q_sign * q_top_mask * q_norm, (k_sign * k_top_mask * k_norm).transpose(-1, -2))
        # # diff = torch.bmm(q_sign, k_sign.transpose(-1, -2))
        # # diff = torch.bmm(q_sign * q_norm, (k_sign * k_norm).transpose(-1, -2))
        # # diff = torch.bmm(q_norm, k_norm.transpose(-1, -2))
        
        # q = q - torch.min(q)
        # k = k - torch.min(k)
        # q = q ** 3
        # k = k ** 3
        
        q_abs = torch.abs(q)
        k_abs = torch.abs(k)
        atten_abs = torch.softmax(torch.bmm(q_abs, k_abs.transpose(-1, -2)) + mask, dim=-1)
        
        # mask = mask > -1
        # diff = diff * mask + (~mask) * torch.min((diff.masked_fill(mask, 987654321)).view(-1))
        
        q = q ** 7
        k = k ** 7
        diff = torch.zeros_like(atten_layer[0])
        sign = lambda x: 1 if x >= 0 else -1
        for ihead in range(min(12, N_H)):
            S = 20
            queries_raw = q[ihead]
            queries = torch.abs(q[ihead])
            # queries = q[ihead]
            keys_raw = k[ihead]
            keys = torch.abs(k[ihead])
            # keys = k[ihead]
            cs = torch.cumsum(keys, dim=0)
            for i in tqdm.tqdm(range(T)):
                c = cs[i]
                query = queries[i]
                s = torch.round((c * query) / torch.sum(c*query) * S)
                Ls, L_indices = torch.sort(keys[:i+1], dim=0, descending=True) #type: torch.Tensor
                # print(Ls.shape)
                dist = {}
                for j in range(HID):
                    sampled = 0
                    for ji in range(min(i+1, 1000)):
                        x_ij = Ls[ji, j]
                        counter = dist.get(L_indices[ji, j].item(), 0)
                        counts = max(math.ceil(s[j]*x_ij/c[j]), 1)
                        # counts = max(math.ceil(s[j]*x_ij/c[j]), 1) * sign(keys_raw[L_indices[ji, j], j]) * sign(queries_raw[i, j])
                        # print(counts)
                        counts = counts
                        counter += counts
                        sampled = sampled + counts
                        if sampled > s[j]: break
                        dist[L_indices[ji, j].item()] = counter
                for ji, v in dist.items():
                    # if v > 0:
                    diff[ihead, i, ji] = v
                # print(s, s.long(), dist)
                # print(len(dist))
                # input()
        
        for ihead, atten_head in enumerate(atten_layer[0]):
            os.makedirs(os.path.join(root, f'attention/l{ilayer}'), exist_ok=True)
            imsave(atten_head, f'attention/l{ilayer}/h{ihead}.png')
            imsave(diff[ihead], f'attention/l{ilayer}/d{ihead}.png', gamma=0.3)
            imsave(atten_abs[ihead], f'attention/l{ilayer}/habs{ihead}.png', gamma=0.2)

test_generation = False
if test_generation:
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids, max_length=2048)
    decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(decoded)