import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.models.bert.configuration_bert import BertConfig

from .vae import *

#TODO : permute_bert_modules에서 Trainer input value hard coding된 것들 check

class LearnablePermutation(nn.Module):
    def __init__(self, hidden_size=768, dropout=0.1, outch=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.outch = outch
        
        # self.net = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, hidden_size * 4),
        #     nn.LayerNorm(hidden_size * 4),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size * 4, hidden_size * 4),
        #     nn.LayerNorm(hidden_size * 4),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size * 4, hidden_size * 4),
        #     nn.LayerNorm(hidden_size * 4),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size * 4, hidden_size * 4),
        #     nn.LayerNorm(hidden_size * 4),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(hidden_size * 4, outch),
        #     nn.LayerNorm(outch),
        # )
        
        # self.net = nn.Sequential(
        #     # nn.LayerNorm(hidden_size, elementwise_affine=False),
        #     nn.Linear(hidden_size, 16, bias=False),
        #     nn.Linear(16, outch, bias=True),
        #     # nn.Linear(hidden_size, outch),
        #     nn.Dropout(dropout),
        # )
        
        self.using_vae = False
        self.net_vae = VAEModel(hidden_size, outch, 384, 192)
        
        bottleneck = 64
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.LeakyReLU(0.2),
            
            nn.Dropout(dropout),
            nn.Linear(bottleneck, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.LeakyReLU(0.2),
            
            nn.Dropout(dropout),
            nn.Linear(bottleneck, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.LeakyReLU(0.2),
            
            # nn.Dropout(dropout),
            # nn.Linear(bottleneck, bottleneck),
            # nn.LayerNorm(bottleneck),
            # nn.LeakyReLU(0.2),
            
            # nn.Dropout(dropout),
            # nn.Linear(bottleneck, bottleneck),
            # nn.LayerNorm(bottleneck),
            # nn.LeakyReLU(0.2),
            
            nn.Linear(bottleneck, outch),
            nn.Dropout(dropout),
        )
        
        # self.norm = nn.LayerNorm(hidden_size)
        # self.norm_unperm = nn.LayerNorm(hidden_size)
        self.norm = nn.Identity()
        self.norm_unperm = nn.Identity()
        
        self.temperature = 0.05
        self.sinkhorn_iteration = 0
        
        self.last_permutation_prob = None
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        # hidden_state: Tensor([N, TOK, HID])
        # attention_mask: Tensor([N, 1, 1, TOK]) in [-10000, 0]
        
        N, TOK, HID = hidden_states.shape
        if attention_mask is None:
            attention_mask = torch.zeros((N, 1, 1, TOK), dtype=hidden_states.dtype, device=hidden_states.device)
        # print(attention_mask.shape)
        N_, _ONE, __ONE, TOK_ = attention_mask.shape
        assert _ONE == 1 and __ONE == 1
        assert N_ == N
        assert TOK_ == TOK
        attention_mask = attention_mask.squeeze(1).squeeze(1)
        
        x = (hidden_states - hidden_states.mean(dim=1, keepdim=True)) / (hidden_states.std(dim=1, keepdim=True) + 1e-6)
        if self.using_vae:
            x, self.last_vae_mean, self.last_vae_log_var = self.net_vae(x)
        else:
            x = self.net(x)
        # TODO: 서로 다른 토큰으로 맵핑되로록 regulation, minimize inner dot --> 0
        x = x.unsqueeze(-1)
        # x: [N, TOK, TOKMAX, 1]
        
        grid = (attention_mask > -1) * 1.0
        # grid: [N, TOK]
        grid_y = torch.cumsum(grid, dim=1)
        grid_y = grid_y / torch.sum(grid, dim=1, keepdim=True) * 2 - 1.0
        grid_y = grid_y.unsqueeze(-1).unsqueeze(-1)
        grid_x = torch.zeros_like(grid_y)
        grid = torch.cat([grid_x, grid_y], dim=-1)
        assert grid.shape == (N, TOK, 1, 2)
        # grid: [N, TOK, 1, 2]
        
        x = torch.nn.functional.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        # x: [N, TOK, TOK, 1]
        x = x + attention_mask.unsqueeze(1).unsqueeze(-1)
        temperature = self.temperature
        
        # old softmax
        # x = torch.softmax(x / temperature + attention_mask.unsqueeze(-1).unsqueeze(1), dim=2)
        # # x = torch.sigmoid(x)
        # x = x * ((attention_mask > -1) * 1.0).unsqueeze(-1).unsqueeze(1) #for just sure
        # x = x * ((attention_mask > -1) * 1.0).unsqueeze(-1).unsqueeze(2) #for just sure
        # # x = torch.softmax(x / (temperature**2) + attention_mask.unsqueeze(-1).unsqueeze(2), dim=1)
        # # x = x * ((attention_mask > -1) * 1.0).unsqueeze(-1).unsqueeze(1)
        # # x = x * ((attention_mask > -1) * 1.0).unsqueeze(-1).unsqueeze(2)
        
        # new softmax
        # x = x / temperature + attention_mask.unsqueeze(-1).unsqueeze(1) + attention_mask.unsqueeze(-1).unsqueeze(2)
        # N, TOK, TOK_, _ = x.shape
        # x = x - torch.max(x.view(N, TOK*TOK_), dim=1)[0].view(-1 ,1, 1, 1)
        # x = torch.exp(x)
        # x = x / (torch.sum(x, dim=1, keepdim=True) + torch.sum(x, dim=2, keepdim=True) - x + 1e-7)
        # # x = x / temperature + attention_mask.unsqueeze(-1).unsqueeze(1) + attention_mask.unsqueeze(-1).unsqueeze(2)
        # # x = torch.softmax(x.view(N, TOK*TOK_), dim=1).view(N, TOK, TOK_, 1)
        # # x = x * ((attention_mask > -1) * 1.0).unsqueeze(-1).unsqueeze(1) #for just sure
        # # x = x * ((attention_mask > -1) * 1.0).unsqueeze(-1).unsqueeze(2) #for just sure
        
        #gumbel sinkhorn, https://arxiv.org/pdf/1802.08665.pdf... quite interestingly same as my idea :(
        x = x + attention_mask.unsqueeze(-1).unsqueeze(1) + attention_mask.unsqueeze(-1).unsqueeze(2)
        N, TOK, TOK_, _ = x.shape
        x = x - torch.max(x.view(N, TOK*TOK_), dim=1)[0].view(-1 ,1, 1, 1)
        x = x / temperature
        x = torch.exp(x)
        if self.sinkhorn_iteration > 0:
            for _ in range(self.sinkhorn_iteration):
                x = x / (torch.sum(x, dim=1, keepdim=True) + 1e-6)
                x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-6)
        else:
            x = x / (torch.sum(x, dim=2, keepdim=True) + 1e-6)
        
        x = x / ((x.square().sum(2, keepdim=True) + 1e-8).sqrt() + 1e-6) #for unit vector
        # x2 = x / ((x.square().sum(1, keepdim=True) + 1e-8).sqrt() + 1e-6) #for unit vector
        # x = (x1 + x2) / 2
        
        # if not torch.is_grad_enabled() or self.hard_mode:
        #     x = (x >= x.max(2, keepdim=True)[0]) * 1.0
        
        x = self.dropout(x)
        
        permutation_prob = x
        self.last_permutation_prob = permutation_prob
        self.last_attention_mask = attention_mask
        
        # print(temperature, x[1, :3, :10, 0])
        
        #h = hidden_states.unsqueeze(2)
        # # h: [N, TOK, 1, HID]
        # x = h * x
        # x = torch.sum(x, dim=1)
        
        h = hidden_states
        
        x = self.permute(h, x)
        
        return x, permutation_prob

    def permute(self, hidden_states: torch.Tensor, permutation_prob: torch.Tensor):
        h = hidden_states
        x = permutation_prob
        x = x / (x.sum(dim=1, keepdim=True) + 1e-6)
        x = torch.bmm(x.squeeze(-1).transpose(1, 2), h)
        x = self.norm(x)
        return x
    
    def calc_loss(self):
        loss = 0
        if self.last_permutation_prob is not None:
            p = self.last_permutation_prob.squeeze(-1)
            mask = (self.last_attention_mask > -1) * 1.0
            
            # p = p / (((p * p).sum(2, keepdim=True) + 1e-5).sqrt() + 1e-5)
            # # TODO(): Compare to minimize MSE(PxPt, I)
            # loss = (torch.bmm(p, p.transpose(1, 2)).square() * mask.unsqueeze(1) * mask.unsqueeze(2))
            # loss = loss * (1 - torch.eye(loss.shape[1], device=loss.device, dtype=loss.dtype).unsqueeze(0).expand(*loss.shape))
            # # print(loss)
            # loss = loss.view(loss.shape[0], -1).sum(1) / (mask.sum(1).square() - mask.sum(1) + 1e-5)
            # loss = loss.mean()
            
            #L2 Loss
            N, TOK, _ = p.shape
            error = torch.bmm(p, p.transpose(1, 2)) - torch.eye(TOK, device=p.device, dtype=p.dtype).view(1, TOK, TOK)
            error = error.square()
            # error = error.abs()
            error = error * mask.unsqueeze(1) * mask.unsqueeze(2)
            error = error.view(error.shape[0], -1).sum(1) / (mask.sum(1).square() + 1e-6)
            loss = error.mean()
            
            #BCE Loss
            # N, TOK, _ = p.shape
            # error = torch.nn.functional.binary_cross_entropy(
            #     torch.bmm(p, p.transpose(1, 2)),
            #     torch.eye(TOK, device=p.device, dtype=p.dtype).view(1, TOK, TOK) * mask.unsqueeze(1) * mask.unsqueeze(2),
            #     reduction='none'
            # )
            # # error = error.abs()
            # error = error * mask.unsqueeze(1) * mask.unsqueeze(2)
            # error = error.view(error.shape[0], -1).sum(1) / (mask.sum(1).square() + 1e-6)
            # loss = error.mean() * 0.5
            
            if self.using_vae:
                loss_vae = self.net_vae.kl_loss(self.last_vae_mean, self.last_vae_log_var)
                # print(loss, loss_vae)
                loss = loss + loss_vae
            
        # self.last_permutation_prob = None
        return loss
    
    def unpermute(self, hidden_states, permutation_prob):
        x = permutation_prob
        h = hidden_states
        x = x / (x.sum(dim=2, keepdim=True) + 1e-6)
        x = torch.bmm(x.squeeze(-1), h)
        x = self.norm_unperm(x)
        return x

class DenseAttention(nn.Module):
    def __init__(self, max_seq_len, d_k, d_hid = 64, attn_dropout = 0.1):
        #d_hid = 8*(128/8)/2
        super(DenseAttention, self).__init__()
        self.w_1 = nn.Linear(d_k, d_hid)
        self.w_2 = nn.Linear(d_hid, max_seq_len)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, v, len_q, mask=None):

        # b x n x lq x dq -> b x n x lq x lq #
        dense_attn = self.w_2(self.relu(self.w_1(q)))[:,:,:,:len_q]
        # print('DenseAttn: ', dense_attn)
        # print('Attn: ', dense_attn.shape)
        # print('Mask: ', mask.shape)
        # print('V: ', v.shape)
        
        ### TODO check all of the mask==0 parts
        if mask is not None:
            dense_attn = dense_attn.masked_fill(mask != 0, -1e9)

        dense_attn = self.dropout(F.softmax(dense_attn, dim=-1))
        output = torch.matmul(dense_attn, v)
        
        return output, dense_attn

class FactorizedDenseAttention(nn.Module):
    def __init__(self, max_seq_len, d_k, f, attn_dropout = 0.1):
        #d_hid = 8*(128/8)/2
        super(FactorizedDenseAttention,self).__init__() ## TODO changed 
        self.f = f
        self.max_seq_len = max_seq_len
        self.f_a = nn.Linear(d_k, f)
        self.f_b = nn.Linear(d_k, max_seq_len//f) ##TODO changed / to // check
        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, v, len_q, mask=None, factorize=False):

        h_a = torch.repeat_interleave(self.f_a(q), self.max_seq_len//self.f, -1)[:,:,:,:len_q] #TODO changed / to //
        h_b = torch.repeat_interleave(self.f_b(q), self.f, -1)[:,:,:,:len_q]
        dense_attn = torch.matmul(h_a, h_b.transpose(2, 3))

        if mask is not None:
            dense_attn = dense_attn.masked_fill(mask != 0, -1e9)

        dense_attn = self.dropout(F.softmax(dense_attn, dim=-1))
        output = torch.matmul(dense_attn, v)
        
        return output, dense_attn

class RandomAttention(nn.Module):
    def __init__(self, batch_size, n_head, max_seq_len, attn_dropout = 0.1):
        super(RandomAttention, self).__init__()
        #device = torch.device("GPU"),
        self.register_buffer('random_attn', None, persistent=True)
        self.random_attn = torch.randn(1, n_head, max_seq_len, max_seq_len, requires_grad = True)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, v, len_q, mask=None):

        # b x n x max_len x max_len -> b x n x lq x lq
        random_attn = self.random_attn[:mask.shape[0],:,:len_q,:len_q]
        # random_attn = random_attn.to(torch.device('cuda' if mask.is_cuda else 'cpu'))

        if mask is not None:
            random_attn = random_attn.masked_fill(mask != 0, -1e9)

        random_attn = self.dropout(F.softmax(random_attn, dim=-1))
        output = torch.matmul(random_attn, v)
        
        return output, random_attn

class FactorizedRandomAttention(nn.Module):
    def __init__(self, batch_size, n_head, f,  max_seq_len, attn_dropout = 0.1):
        super(FactorizedRandomAttention, self).__init__()
        #device = torch.device("GPU"),
        self.random_attn_1 = torch.randn(1, n_head, max_seq_len, f, requires_grad = True)
        self.random_attn_2 = torch.randn(1, n_head, f, max_seq_len, requires_grad = True)
        self.dropout = nn.Dropout(attn_dropout)
    def forward(self, v, len_q, mask=None, factorize=False):
        
        
        
        # b x n x max_len x max_len -> b x n x lq x lq #[:,:,:len_q,:len_q]
        random_attn = torch.matmul(self.random_attn_1, self.random_attn_2)[:mask.shape[0],:,:len_q,:len_q]

        if mask is not None:
            random_attn = random_attn.masked_fill(mask != 0, -1e9)
            
        random_attn = self.dropout(F.softmax(random_attn, dim=-1))
        output = torch.matmul(random_attn, v)
        
        return output, random_attn


class BertSelfAttention(nn.Module):
    permutations: List[Tuple[LearnablePermutation, nn.Module, nn.Module, nn.Module]]
    
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        

        from sinkhorn_transformer.sinkhorn_transformer import SinkhornCausalAttention, SinkhornAttention

        from performer_pytorch import FastAttention
        ### performer
        self.performer_enabled = False
        self.performer_type = 'performer'
        self.performer_attn = FastAttention(dim_heads=self.attention_head_size, nb_features=config.hidden_size, causal=False)
        ### sinknorn_permutation_attn
        self.sinkhorn_permutation_attn = SinkhornAttention(
            bucket_size=1,
            dim=768,
            dim_heads=768//12,
            heads=12,
            max_seq_len=512,
            temperature=0.02,
            n_top_buckets=32,
        )
        
        ### permutation
        self.permutation_enabled = False
        self.permutations = nn.ModuleList([
            nn.ModuleList([
                LearnablePermutation(hidden_size=self.attention_head_size), 
                nn.Identity(self.attention_head_size),
                nn.Identity(self.attention_head_size),
                nn.Identity(self.attention_head_size),
            ]) for i in range(self.num_attention_heads)
        ])
        
        self.permutation_master_enabled = False

        self.permutation_master = LearnablePermutation(hidden_size=self.attention_head_size)
        
        ### synthesizer
        self.synthesizer_enabled = False

        self.running_type="head_permutation" # TODO!! have to get from trainer <<<< hardcoded
        
        # print("RUNNING TYPE::: ",self.running_type)
        # if self.running_type=="head_permutation":
        #     print("RUNNING TYPE IS head_permutation :: bert_modules")

        if self.running_type=="DenseAttention":
            d_k=config.hidden_size//config.num_attention_heads
            self.dense_attn = DenseAttention(config.max_position_embeddings, d_k)
        elif self.running_type=="FactrDenseAttention":
            d_k=config.hidden_size//config.num_attention_heads # TODO
            self.fdense_attn = FactorizedDenseAttention(config.max_position_embeddings, d_k, 4)
        elif self.running_type=="RandomAttention": # TODO check batch size
            self.random_attn = RandomAttention(1, config.num_attention_heads, config.max_position_embeddings)
        elif self.running_type=="FactrRandomAttention":
            self.frandom_attn = FactorizedRandomAttention(1, config.num_attention_heads, 4, config.max_position_embeddings)
        else:
            # raise Exception(f"unknown type, {self.running_type}")
            print(f"Not using synthesizer, rather using [{self.running_type}]")

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        
        ### TODO : Longformer
        
        ### TODO : Bigbird
        
        ### TODO : Reformer
        
        # linear_attention_mask for permutation
        self.linear_attention_mask_enabled = True
        self.register_buffer('linear_attention_mask', None, persistent=False)
        MAX_LEN = 512
        self.linear_attention_mask = torch.empty((1, 1, MAX_LEN, MAX_LEN))
        self.linear_attention_mask.fill_(-10000)
        for i in range(MAX_LEN):
            for j in range(MAX_LEN):
                if abs(i-j) < 15:
                    self.linear_attention_mask[0,0,i,j] = 0
                # else:
                #     self.linear_attention_mask[0,0,i,j] = -10000
                    
        self.dynamic_attention_pattern_enabled = True
        self.register_buffer('dynamic_attention_pattern', None, persistent=False)
        self.dynamic_attention_pattern = torch.empty((1, 1, MAX_LEN, MAX_LEN, 20))
        self.dynamic_attention_pattern.fill_(-10000)
        pattern_id = 0
        for k in [4, 6, 8, 12, 16]:
            for i in range(MAX_LEN):
                for j in range(MAX_LEN):
                    if abs(i-j) <= k:
                        self.dynamic_attention_pattern[:, :, i, j, pattern_id] = 0
                    # else:
                    #     self.dynamic_attention_pattern[:, :, i, j, pattern_id] = -10000
            pattern_id += 1
        for k in [8, 12, 16, 24, 32]:
            # self.dynamic_attention_pattern[:, :, :, :, pattern_id] = -10000
            for i in range(0, MAX_LEN, k):
                self.dynamic_attention_pattern[:, :, i:min(MAX_LEN, i+k), i:min(MAX_LEN, i+k), pattern_id] = 0
            pattern_id += 1
        #global tokens
        self.dynamic_attention_pattern[:,:,:,:,pattern_id:pattern_id*2] = self.dynamic_attention_pattern[:,:,:,:,:pattern_id]
        self.dynamic_attention_pattern[:,:,:,:4,pattern_id:pattern_id*2] = 0
        pattern_id += 10
        self.dynamic_attention_decision_net_for_permutation = nn.Sequential(
            nn.LayerNorm(self.attention_head_size),
            nn.Dropout(0.1),
            nn.Linear(self.attention_head_size, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2),
            
            nn.Dropout(0.1),
            nn.Linear(128, 20),
        )
        self.dynamic_attention_decision_dropout = nn.Dropout(0.1)
        # print(self.linear_attention_mask)
        self.mask_temperature = 0.05
        
        self.last_attention_probs = None

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3) # batch head token hid

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer_pre_permute = query_layer = self.transpose_for_scores(mixed_query_layer)
        
        permutation_probs = []
        if self.permutation_enabled and not self.performer_enabled:
            if not self.permutation_master_enabled:
                queries = []
                keys = []
                values = []
                # hidden_layer = self.transpose_for_scores(hidden_states)
                for ihead, (permutation, norm_q, norm_k, norm_v) in enumerate(self.permutations):
                    query = query_layer[:, ihead, :, :]
                    key = key_layer[:, ihead, :, :]
                    value = value_layer[:, ihead, :, :]
                    hidden = query_layer[:, ihead, :, :]
                    _, permutation_prob = permutation(hidden, attention_mask)
                    permutation_probs.append(permutation_prob)
                    query = norm_q(permutation.permute(query, permutation_prob))
                    key = norm_k(permutation.permute(key, permutation_prob))
                    value = norm_v(permutation.permute(value, permutation_prob))
                    queries.append(query.unsqueeze(1))
                    keys.append(key.unsqueeze(1))
                    values.append(value.unsqueeze(1))
                query_layer = torch.cat(queries, dim=1)
                key_layer = torch.cat(keys, dim=1)
                value_layer = torch.cat(values, dim=1)
            else:
                N, H, T, HID = query_layer.shape
                query_layer = query_layer.reshape(N*H, T, HID)
                key_layer = key_layer.reshape(N*H, T, HID)
                value_layer = value_layer.reshape(N*H, T, HID)
                
                query_layer, permutation_prob = self.permutation_master(query_layer, attention_mask.repeat(1, H, 1, 1).view(N*H, 1, 1, -1))
                key_layer = self.permutation_master.permute(key_layer, permutation_prob)
                value_layer = self.permutation_master.permute(value_layer, permutation_prob)
                
                query_layer = query_layer.view(N, H, T, HID)
                key_layer = key_layer.view(N, H, T, HID)
                value_layer = value_layer.view(N, H, T, HID)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)
        
        ### performer
        if self.performer_enabled:
            attention_probs = None
            assert query_layer.shape[1] == 12, query_layer.shape
            
            if self.performer_type == 'performer':
                context_layer = self.performer_attn(
                    query_layer, key_layer, value_layer
                )
            elif self.performer_type == 'sinkhorn':
                N, H, T, HID_HEAD = query_layer.shape
                context_layer = self.sinkhorn_permutation_attn(
                    query_layer, key_layer, value_layer, ((attention_mask == 0)).view(N, T)
                )
            else:
                raise Exception()
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        ### synthesizer
        elif self.synthesizer_enabled:
            attention_probs = None
            # assert query_layer.shape[1] == 12, query_layer.shape
            if self.running_type=="DenseAttention":
                context_layer = self.dense_attn(
                    query_layer, 
                    value_layer, 
                    value_layer.shape[-2], 
                    attention_mask
                )[0] ### TODO check attention mask - for all if elif cases
            elif self.running_type=="FactrDenseAttention":
                context_layer = self.fdense_attn(
                    query_layer,
                    value_layer,
                    value_layer.shape[-2],
                    attention_mask,
                    factorize=True)[0]
            elif self.running_type=="RandomAttention":
                context_layer = self.random_attn(
                    value_layer,
                    value_layer.shape[-2],
                    attention_mask)[0]
            elif self.running_type=="FactrRandomAttention":
                context_layer = self.frandom_attn(
                    value_layer,
                    value_layer.shape[-2],
                    attention_mask,
                    factorize=True)[0]
            
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        ### TODO : Longformer
        
        ### TODO : Bigbird
        
        ### TODO : Reformer
        
        ### bert
        # Take the dot product between "query" and "key" to get the raw attention scores.
        else:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
                query_length, key_length = query_layer.shape[2], key_layer.shape[2]
                if use_cache:
                    position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                        -1, 1
                    )
                else:
                    position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
                position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
                distance = position_ids_l - position_ids_r

                positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
                positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

                if self.position_embedding_type == "relative_key":
                    relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores
                elif self.position_embedding_type == "relative_key_query":
                    relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                    relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                    attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
                attention_scores = attention_scores + attention_mask
            if self.linear_attention_mask_enabled:
                if not self.dynamic_attention_pattern_enabled:
                    N, H, T, T_ = attention_scores.shape
                    attention_scores = attention_scores + self.linear_attention_mask[:, :, :T, :T_]
                else:
                    N, H, T, T_ = attention_scores.shape
                    x = self.dynamic_attention_decision_net_for_permutation(query_layer_pre_permute[:,:,0,:])
                    if self.training:
                        x = torch.softmax(x / self.mask_temperature, dim=-1)
                    else:
                        x = torch.softmax(x / self.mask_temperature, dim=-1)
                    x = self.dynamic_attention_decision_dropout(x)
                    # x: [N, H, TYPE]
                    x = x.view(N, H, 1, 1, -1)
                    attention_scores = attention_scores.unsqueeze(-1) + self.dynamic_attention_pattern[:, :, :T, :T_, :]
                    attention_scores_decision = x

            # Normalize the attention scores to probabilities.
            if attention_scores.ndim == 4:
                attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            elif attention_scores.ndim == 5:
                attention_probs = nn.functional.softmax(attention_scores, dim=-2)
                attention_probs = attention_probs * attention_scores_decision
                attention_probs = attention_probs.sum(-1)
            self.last_attention_probs = attention_probs

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            attention_probs = self.dropout(attention_probs)

            # Mask heads if we want to
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            context_layer = torch.matmul(attention_probs, value_layer)
            
            # unpermutation
            if self.permutation_enabled:
                if not self.permutation_master_enabled:
                    contexts = []
                    for ihead, (permutation, norm_q, norm_k, norm_v) in enumerate(self.permutations):
                        context = context_layer[:, ihead, :, :]
                        permutation_prob = permutation_probs[ihead]
                        context = permutation.unpermute(context, permutation_prob)
                        contexts.append(context.unsqueeze(1))
                    context_layer = torch.cat(contexts, dim=1)
                else:
                    N, H, T, HID = context_layer.shape
                    context_layer = context_layer.view(N*H, T, HID)
                    context_layer = self.permutation_master.unpermute(context_layer, permutation_prob)
                    context_layer = context_layer.view(N, H, T, HID)

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

if __name__ == '__main__':
    from ..utils import seed
    # seed()
    perm = LearnablePermutation(hidden_size=4, dropout=0.0)
    x = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    mask = torch.zeros((2, 4))
    mask[1, 2:] = -10000
    mask = mask.unsqueeze(1).unsqueeze(1)
    y, prob = perm(x, mask)
    x_ = perm.unpermute(y, prob)
    
    print(x[:], y[:], x_[:], perm.calc_loss(), sep='\n')

class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.permutation_enabled = False
        self.permutation = LearnablePermutation()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        hidden_states_residual = hidden_states
        if self.permutation_enabled:
            # hidden_states_permuted, permutation_prob = self.permutation(
            #     hidden_states = hidden_states[:, 1:, :],
            #     attention_mask = attention_mask[:, :, :, 1:]
            # )
            # hidden_states = torch.cat([hidden_states[:, 0:1, :], hidden_states_permuted], dim=1)
            hidden_states, permutation_prob = self.permutation(
                hidden_states = hidden_states[:, :, :],
                attention_mask = attention_mask[:, :, :, :]
            )
            
            #TODO: sparsify attention mask
            sparse_attention_mask  = attention_mask
            assert encoder_attention_mask is None, "decoder is not supported yet"
        
        self_outputs = self.self(
            hidden_states,
            sparse_attention_mask if self.permutation_enabled else attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        if self.permutation_enabled:
            self_hidden_states = self_outputs[0]
            # self_hidden_states_permuted = self.permutation.unpermute(
            #     hidden_states=self_hidden_states[:, 1:, :], 
            #     permutation_prob=permutation_prob
            # )
            # self_hidden_states = torch.cat([self_hidden_states[:, 0:1, :], self_hidden_states_permuted], dim=1)
            self_hidden_states = self.permutation.unpermute(
                hidden_states=self_hidden_states[:, :, :], 
                permutation_prob=permutation_prob
            )
            self_outputs = (self_hidden_states, *self_outputs[1:])
        attention_output = self.output(self_outputs[0], hidden_states_residual)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
