"""
Validate token gernations between using cache or not. 
Please do not use this script to benchmark generation speed. generation optimization is WIP

Usage: python -m src.main.tests.test_perlin_opt_cache --k 32 --predictor-length 64
            ^ put proper k and predictor length

Example Output:

accuracy 1.0  <-- this means the token accuracy. how many tokens are same with non-cached output. this should be 1.0 or 0.9x (enougly high)
                                                         vvv Error @ sequence index INDEX. this should be lower than 0 or -0.5 for safety.
   vvv Checked temp_buffer names                             However error in here are acceptable.
   ERROR=(x-y).abs().sum().log10()      | INDEX:            0,           1,           2,          10,          20,          30,          40,          -1
 - q                                    | ERROR:      -5.0280,     -4.8412,     -4.8891,     -4.8618,     -4.9202,     -4.9735,     -4.9047,     -4.9767
 - k                                    | ERROR:      -4.0735,     -3.8530,     -3.8873,     -3.8861,     -3.9289,     -3.9863,     -3.9259,     -4.0593
 - v_for_atten                          | ERROR:      -5.0252,     -4.9480,     -4.9437,     -4.9292,     -4.9208,     -4.9448,     -4.8942,     -4.9337
 - performer_context_layer              | ERROR:      -3.9410,     -4.1299,     -4.1825,     -4.5160,     -4.6093,     -4.6535,     -4.6473,     -4.7752
 - performer_value                      | ERROR:      -3.9067,     -4.0685,     -4.1131,     -4.3742,     -4.4367,     -4.4741,     -4.4524,     -4.5462
 - t_attention_predictor                | ERROR:      -3.8124,     -3.8011,     -3.7459,     -3.7808,     -3.8090,     -3.8025,     -3.7881,     -3.7606
 - estimated_attention_score_dec_row    | ERROR:      -4.8571,     -4.8560,     -4.8414,     -4.8523,     -4.8864,     -4.8495,     -4.8496,     -4.8470
 - estimated_attention_score            | ERROR:      -3.3189,     -3.3498,     -3.3077,     -3.3108,     -3.5085,     -3.5438,     -3.4404,     -3.5651
 - estimated_attention_probs            | ERROR:      -5.1139,     -5.1358,     -5.1781,     -5.1314,     -5.3887,     -5.3652,     -5.3033,     -5.3355
 - partial_attention_mask_before_interp | ERROR:         -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf
 - partial_attention_mask               | ERROR:         -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf
 - partial_attention_scores             | ERROR:         -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf,        -inf
 - estimated_scales                     | ERROR:      -6.1501,     -6.0072,     -5.9879,     -6.1224,     -6.0011,     -6.1958,     -6.2234,     -6.0134
 - average_scale                        | ERROR:      -6.7476,     -7.2247,     -7.0486,     -6.5257,     -6.9237,     -7.2247,     -7.5257,     -6.7476
 - average_context_layer                | ERROR:      -5.0252,     -5.1285,     -5.2015,     -5.4289,     -5.5386,     -5.5839,     -5.6234,     -5.6636
 - partial_context_layer_sparse         | ERROR:      -5.1280,     -5.2156,     -5.2653,     -5.4711,     -5.4849,     -5.6106,     -5.5324,     -5.5906
 - normalized_partial_context_layer     | ERROR:      -3.8719,     -3.8293,     -3.8622,     -3.8867,     -3.9708,     -4.0119,     -3.9690,     -3.9888
 - partial_context_layer                | ERROR:      -3.8543,     -3.8112,     -3.8469,     -3.8767,     -3.9594,     -4.0033,     -3.9517,     -3.9767
 - logits                               | ERROR:      -1.0928,     -1.1591,     -1.1043,     -1.2141,     -1.2202,     -1.2087,     -1.2146,     -1.2232
"""

import os, tqdm, gc
import flax
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"
import numpy as np
import torch
from .common_opt import init
from ...models import perlin_attention
from ...utils import get_bench, strify
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

TRACKING_BUFFERS = [
    'q',
    'k',
    'v_for_atten',
    'performer_context_layer',
    'performer_value',
    't_attention_predictor',
    'estimated_attention_score_dec_row',
    'estimated_attention_score',
    'estimated_attention_probs',
    'partial_attention_mask_before_interp',
    'partial_attention_mask',
    'partial_attention_scores',
    'estimated_scales',
    'average_scale',
    'average_context_layer',
    'partial_context_layer_sparse',
    'normalized_partial_context_layer',
    'partial_context_layer',
]

BUFFER_ACCUMULATE = {
    'q', 
    'performer_context_layer',
    'performer_value',
    't_attention_predictor',
    'estimated_attention_score_dec_row',
    'estimated_attention_score',
    'estimated_attention_probs',
    'partial_attention_scores',
    'partial_attention_mask_before_interp',
    'partial_attention_mask',
    'partial_context_layer',
    'estimated_scales',
    'average_scale',
    'average_context_layer',
    'partial_context_layer_sparse',
    'normalized_partial_context_layer',
    'logits',
}

DST_SOURCE_BUFFER = {
    'partial_attention_mask',
    'partial_attention_scores',
}

MASK_BUFFER = {
    'partial_attention_scores',
    'partial_attention_mask_before_interp',
    'partial_attention_mask',
}

INDEX_LAYER = 0
MAX_SEQ_LEN = 128

def main():
    use_cache = True
    bench = get_bench()
    bench.disabled = False
    bench.activate_temp_buffers = True
    
    # trainer, model, tokenizer = init(skip_init_loaders=True, checkpoint_path='./saves/trainer/opt_trainer/opt-125m_wikitext2_kf1_lw0_perlin_k64_full_copy/checkpoint.pth')
    trainer, model, tokenizer = init(skip_init_loaders=True)
    model.eval()
    
    input_ids = tokenizer(
        "Famitsu enjoyed the story , and were particularly pleased with the improvements to gameplay . Japanese gaming site Game Watch <unk> , despite negatively noting its pacing and elements recycled from previous games , was generally positive about its story and characters , and found its gameplay entertaining despite off @-@ putting difficulty spikes . <unk> writer <unk> <unk> , in a Play Test article based on the game 's <unk> demo , felt that Valkyria Chronicles III provided a profound feeling of closure for the Valkyria Chronicles series . He praised its gameplay despite annoying limitations to aspects such as special abilities , and positively noted its shift in story to a tone similar to the first game . PlayStation Official Magazine - UK praised the story 's <unk> of Gallia 's moral standing , art style , and most points about its gameplay , positively noting the latter for both its continued quality and the tweaks to balance and content . Its one major criticism were multiple difficulty spikes , something that had affected the previous games . Heath Hindman of gaming website PlayStation <unk> praised the addition of non @-@ linear elements and improvements or removal of mechanics from Valkyria Chronicles II in addition to praising the returning gameplay style of previous games . He also positively noted the story 's serious tone . Points criticized in the review were recycled elements , awkward cutscenes that seemed to include all characters in a scene for no good reason , pacing issues , and occasional problems with the game 's AI ",
        return_tensors="pt"
    ).input_ids.to(trainer.device) # type: torch.Tensor
    input_ids = input_ids[:,:min(input_ids.shape[-1], MAX_SEQ_LEN)]
    
    # sample dense
    with torch.no_grad():
        output = model(input_ids)
    
    buffers_truth = {}
    for name in TRACKING_BUFFERS:
        # sample only first layer
        if name in bench.buffers:
            buffers_truth[name] = bench.get_temp_buffer(name, index=INDEX_LAYER)
    buffers_truth['logits'] = output.logits
    bench.reset_temp_buffers()
    
    dense_output = torch.argmax(output.logits, dim=-1)[0].cpu().numpy()
    dense_text = tokenizer.batch_decode(dense_output.reshape(1, -1), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(dense_output, dense_text)
    
    # sample with cache
    bench.reset_measures()
    bench.reset_trace()
    past_key_values = None
    output_ids = []
    perlin_attention.get_default_config().use_cache = use_cache
    # os.environ['PERLIN_HOTFIX_STATEFUL'] = '1'
    # for module in model.modules():
    #     if hasattr(module, 'benchmarking'):
    #         module.benchmarking = True
    #         print(type(module))
    buffers = {}
    for i in tqdm.tqdm(range(input_ids.shape[-1])):
        if use_cache:
            ids_slice = input_ids[:, i:i+1]
            with torch.no_grad():
                with get_bench().region("sample"):
                    output = model(
                        input_ids=ids_slice,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
            past_key_values = output.past_key_values
            token_id = torch.argmax(output.logits, dim=-1).item()
            
            for name in TRACKING_BUFFERS:
                if name in bench.buffers:
                    buf = bench.get_temp_buffer(name, index=INDEX_LAYER)
                    if not name in buffers:
                        buffers[name] = buf
                    else:
                        if name in BUFFER_ACCUMULATE:
                            if name in DST_SOURCE_BUFFER:
                                buffers[name] = F.pad(buffers[name], pad=(0, buf.shape[-1] - buffers[name].shape[-1]), mode='constant', value=-32000)
                            assert buffers[name].shape[-1] == buf.shape[-1], f"{name}: {buffers[name].shape}, {buf.shape}"
                            assert buffers[name].shape[:-2] == buf.shape[:-2], f"{name}: {buffers[name].shape}, {buf.shape}"
                            buffers[name] = torch.cat([buffers[name], buf], dim=-2)
                        else:
                            buffers[name] = buf
            
            buf = output.logits
            if not 'logits' in buffers:
                buffers['logits'] = buf
            else:
                if 'logits' in BUFFER_ACCUMULATE:
                    buffers['logits'] = torch.cat([buffers['logits'], buf], dim=-2)
                else:
                    buffers['logits'] = buf
            
            bench.reset_temp_buffers()
        else:
            ids_slice = input_ids[:, :i+1]
            with torch.no_grad():
                output = model(
                    input_ids=ids_slice,
                    use_cache=False
                )
            token_id = torch.argmax(output.logits[:,-1,:], dim=-1).item()
            
            for name in TRACKING_BUFFERS:
                if name in bench.buffers:
                    buf = bench.get_temp_buffer(name, index=INDEX_LAYER)
                    # buffers[name] = buf
                    if name in BUFFER_ACCUMULATE:
                        buf = buf[...,-1:,:]
                    if not name in buffers:
                        buffers[name] = buf
                    else:
                        if name in BUFFER_ACCUMULATE:
                            buffers[name] = torch.cat([buffers[name], buf], dim=-2)
                        else:
                            buffers[name] = buf
            
            # buffers['logits'] = output.logits
            buf = output.logits[...,-1:,:]
            if not 'logits' in buffers:
                buffers['logits'] = buf
            else:
                buffers['logits'] = torch.cat([buffers['logits'], buf], dim=-2)
            
            bench.reset_temp_buffers()
        output_ids.append(token_id)
    
    print(bench.format_tracetree())
    
    cached_output = np.array(output_ids)
    cached_text = tokenizer.batch_decode(cached_output.reshape(1, -1), skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    print(cached_output, cached_text)
    print('accuracy', ((cached_output == dense_output) * 1.0).mean())
    
    # print('truth', strify(buffers_truth))
    # print('buffers', strify(buffers))
    os.makedirs('./saves/tests/test_perlin_opt_cache', exist_ok=True)
    torch.save({
        'truth': buffers_truth, 
        'buffers': buffers
    }, './saves/tests/test_perlin_opt_cache/buf.pth')
    for name in buffers_truth.keys():
        assert buffers[name].shape == buffers_truth[name].shape, f"{name}: {buffers[name].shape} == {buffers_truth[name].shape}"
    
    def preproc(buffers):
        for name in buffers.keys():
            if name in MASK_BUFFER:
                buffers[name] = (buffers[name] > -1).float()
    preproc(buffers)
    preproc(buffers_truth)
    
    CHECK_INDEX = [0, 1, 2, 10, 20, 30, 40, -1]
    JUST_WIDTH = 12
    print(f'   {"ERROR=(x-y).abs().sum().log10()".ljust(JUST_WIDTH*3)} | INDEX: {",".join([str(i).rjust(JUST_WIDTH) for i in CHECK_INDEX])}')
    for name in TRACKING_BUFFERS + ['logits']:
        if name in buffers_truth and name in buffers:
            truth = buffers_truth[name]
            mine = buffers[name]
            losses = []
            for idx in CHECK_INDEX:
                def error(x, y):
                    x = x.to(torch.float64)
                    y = y.to(torch.float64)
                    return (x - y).abs().sum().log10()
                loss = error(truth[...,idx,:], mine[...,idx,:]).item()
                losses.append(loss)
            def deco_error(str, e):
                if e < -3:
                    return f"\033[92m{str}\033[0m"
                return f"\033[91m{str}\033[0m"
            print(f' - {name.ljust(JUST_WIDTH*3)} | ERROR: {",".join([deco_error(f"{loss:.4f}".rjust(JUST_WIDTH), loss) for loss in losses])}')

if __name__ == '__main__':
    main()