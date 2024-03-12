from timber.trainer.timber_trainer import load_model
import torch
import os

def ensemble_random_pruning(
    ks : torch.Tensor,
    q_timber: torch.Tensor,
    k: torch.Tensor,
    v : torch.Tensor,
    mask_k : int,
    block_size_q : int,
    block_size_k : int,

    ensemble : bool,
    ensemble_model_setting : str,
    ensemble_method : str, 
    ensemble_method_final : str,
    ensemble_per_layer_n : int,
    ensemble_per_attn_iter_n : int,
    ensemble_model_n : int,
    ensemble_particular_layer : int,
    ensemble_attn_mask_per_layer : list, 

    layer_id : int,
    ):
    N_H, TDST, HID = q_timber.shape
    _, TSRC, _ = k.shape
    # indices : [40, 128, 256] = [N*H, TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K]
    assert ensemble_method in ['final_attn']
    assert ensemble_method_final in ['all_agree', 'more_sparse', 'same_sparse', 'less_sparse', 
                                    'avg', 'max', 'min', 'med',]

    origin_sparsity = (torch.sum((torch.stack(ensemble_attn_mask_per_layer, dim=0) < TSRC))//ensemble_model_n).item()
    if ensemble_method == "final_attn":
        if ensemble_method_final == 'all_agree':
            '''
            [40, 128, 256] * 5
            package in one batch; 
            in batch; output of attentions
            '''
            ensemble_attn_mask_per_layer = torch.cat(ensemble_attn_mask_per_layer, dim=-1)
            # ensemble_attn_mask_per_layer : [40, 128, 1280] = [N*H, TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K * ensemble_model_n]
            ensemble_attn_mask_per_layer = ensemble_attn_mask_per_layer.view(-1, ensemble_attn_mask_per_layer.shape[-1])
            # ensemble_attn_mask_per_layer : [N*H * TDST//BLOCK_SIZE_Q, mask_k//BLOCK_SIZE_K * ensemble_model_n]
            per_query_token_cnt_diclist = []
            for r in ensemble_attn_mask_per_layer:
                unique_ensemble, counts = torch.unique(r, return_counts=True)
                per_query_token_cnt_diclist.append(dict(zip(unique_ensemble.tolist(), counts.tolist())))
                    
            ensembled_indices = torch.full((N_H, TDST//block_size_q, mask_k//block_size_k), 32000)
            for idx, token_dict in enumerate(per_query_token_cnt_diclist): # N_H * (TDST//block_size_q)
                n_h_idx = idx // (TDST // block_size_q)
                tdst_idx = idx % (TDST // block_size_q)
                
                selected_indices = [key for key, value in token_dict.items() if value == ensemble_model_n and key < TSRC]
                
                for key_idx, key in enumerate(selected_indices):
                    if key_idx < (mask_k // block_size_k):
                        ensembled_indices[n_h_idx, tdst_idx, key_idx] = key

        elif ensemble_method_final == "more_sparse":
            pass

        elif ensemble_method_final == "same_sparse":
            pass

        elif ensemble_method_final == "less_sparse":
            pass

        elif ensemble_method_final == 'avg':
            pass
        elif ensemble_method_final == "max":
            pass
        elif ensemble_method_final == "min":
            pass
        elif ensemble_method_final == "med":
            pass
        else:
            raise Exception('self.ensemble_method_final : ', ensemble_method_final)

    sparsity_per_layer = torch.sum(ensembled_indices!=32000).item()
    sparsity_ratio = (sparsity_per_layer/origin_sparsity)
    print('origin sparsity : ', origin_sparsity)
    print(f'l_{layer_id} sparsity: ', sparsity_per_layer)
    print(f'sparsity ratio {(sparsity_per_layer/origin_sparsity)} ')

    print("PATH: hardcoded to llama 13b chat")
    if os.environ.get('CHECKOUT_ENSEMBLE', '0') == '1':
        os.makedirs(f'./cache/ensemble/llama13b_chat/method/{ensemble_model_setting}_{ensemble_method}_{ensemble_method_final}', exist_ok=True)
        torch.save({
            'ks' : ks,
            'q_timber': q_timber,
            'k': k,
            'v': v,
            'mask_k':mask_k,
            'block_size_q':block_size_q,
            'block_size_k':block_size_k,
            'ensemble': ensemble,
            'ensemble_model_setting' : ensemble_model_setting,
            'ensemble_method' : ensemble_method,
            'ensemble_method_final' : ensemble_method_final,
            'ensemble_per_layer_n' : ensemble_per_layer_n,
            'ensemble_per_attn_iter_n' : ensemble_per_attn_iter_n,
            'ensemble_model_n' : ensemble_model_n,
            'ensemble_particular_layer' : ensemble_particular_layer,
            'layer_id' : layer_id,

            'ensemble_attn_mask_per_layer': ensemble_attn_mask_per_layer,
            'per_query_token_cnt_diclist': per_query_token_cnt_diclist,
            'ensembled_indices' : ensembled_indices,
            'origin_sparsity' : origin_sparsity,
            'sparsity_per_layer' : sparsity_per_layer,
            'sparse_ratio' : sparsity_ratio,

        }, f'./cache/ensemble/llama13b_chat/method/{ensemble_model_setting}_{ensemble_method}_{ensemble_method_final}/l_{layer_id}_m_{ensemble_model_n}_pl_{ensemble_per_layer_n}_pat{ensemble_per_attn_iter_n}_ln{ensemble_particular_layer}.pth')
        print(">>> STORED.")
        # input('stored. press enter to continue >>> ')
    return ensembled_indices, sparsity_ratio