"""
Testing OPT

Usage: python -m src.main.tests.test_perlin_opt
"""

from ...models import perlin_opt, perlin_attention
import transformers
import torch
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import cm

model_config = 'facebook/opt-125m'

perlin_attention.get_default_config().k = 64
model = perlin_opt.OPTForCausalLM.from_pretrained(model_config)
for module in model.modules():
    if isinstance(module, perlin_opt.OPTAttention):
        # module.attention_method = 'none'
        # module.attention_method = 'reformer'
        module.attention_method = 'perlin'

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
    max_length=2048, 
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

root = './saves/tests/test_perlin_opt/'
os.makedirs(root, exist_ok=True)

batch.update({
    'hidden_states': output.hidden_states,
    'attentions': output.attentions,
})
torch.save(batch, os.path.join(root, 'io.pth'))

def imsave(img: torch.Tensor, filename):
    img = img.cpu().numpy()
    
    def convert_to_colormap(arr: np.ndarray):
        def norm(x):
            return (x / (np.max(x) + 1e-12)) ** 0.2
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
        for ihead, atten_head in enumerate(atten_layer[0]):
            os.makedirs(os.path.join(root, f'attention/l{ilayer}'), exist_ok=True)
            imsave(atten_head, f'attention/l{ilayer}/h{ihead}.png')

test_generation = False
if test_generation:
    prompt = "Hey, are you conscious? Can you talk to me?"
    inputs = tokenizer(prompt, return_tensors="pt")

    generate_ids = model.generate(inputs.input_ids, max_length=2048)
    decoded = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(decoded)