import os
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import transformers
from datasets import load_dataset, load_metric
import random, copy
import torch
# torch.autograd.set_detect_anomaly(True)

from transformers.models.bert import modeling_bert as berts
from ..models import permute_bert as pberts
from ..utils.get_optimizer import get_optimizer
from ..utils import batch_to, seed
from ..dataset.wikitext import WikitextBatchLoader

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_epochs = {
    "cola": 100,
    "mnli": 4,
    "mrpc": 200,
    "qnli": 3,
    "qqp":  4,
    "rte":  200,
    "sst2": 15,
    "stsb": 200,
    "wnli": 200,
    "bert": 200,
}

task_to_batch_size = {
    "cola": 64,
    "mnli": 4,
    "mrpc": 32,
    "qnli": 4,
    "qqp":  16,
    "rte":  8,
    "sst2": 16,
    "stsb": 16,
    "wnli": 32,
    "bert": 4,
}

task_to_valid = {
    "cola": "validation",
    "mnli": "validation_matched",
    "mrpc": "test",
    "qnli": "validation",
    "qqp": "validation",
    "rte": "validation",
    "sst2": "validation",
    "stsb": "validation",
    "wnli": "validation",
    "bert": "validation",
}

def get_dataloader(subset, tokenizer, batch_size, split='train'):
    if subset == 'bert':
        subset = "cola" #return dummy set
    
    dataset = load_dataset('glue', subset, split=split, cache_dir='./cache/datasets')
    
    sentence1_key, sentence2_key = task_to_keys[subset]

    def encode(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=True, max_length=256, truncation=True)
        # result = tokenizer(*args, padding="max_length", max_length=512, truncation=True)
        # Map labels to IDs (not necessary for GLUE tasks)
        # if label_to_id is not None and "label" in examples:
        #     result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    if split.startswith('train'): #shuffle when train set
        dataset = dataset.sort('label')
        dataset = dataset.shuffle(seed=random.randint(0, 10000))
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True, batch_size=64)
    dataset = dataset.map(encode, batched=True, batch_size=64)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
    return dataloader

def get_base_model(dataset, only_tokenizer=False):
    checkpoint = {
        "cola": "textattack/bert-base-uncased-CoLA",
        "mnli": "yoshitomo-matsubara/bert-base-uncased-mnli",
        "mrpc": "textattack/bert-base-uncased-MRPC",
        # "mrpc": "M-FAC/bert-tiny-finetuned-mrpc",
        "qnli": "textattack/bert-base-uncased-QNLI",
        "qqp": "textattack/bert-base-uncased-QQP",
        "rte": "textattack/bert-base-uncased-RTE",
        "sst2": "textattack/bert-base-uncased-SST-2",
        "stsb": "textattack/bert-base-uncased-STS-B",
        "wnli": "textattack/bert-base-uncased-WNLI",
        "bert": "bert-base-uncased",
    }[dataset]

    model = {
        "cola": berts.BertForSequenceClassification,
        "mnli": berts.BertForSequenceClassification,
        "mrpc": berts.BertForSequenceClassification,
        "qnli": berts.BertForSequenceClassification,
        "qqp": berts.BertForSequenceClassification,
        "rte": berts.BertForSequenceClassification,
        "sst2": berts.BertForSequenceClassification,
        "stsb": berts.BertForSequenceClassification,
        "wnli": berts.BertForSequenceClassification,
        "bert": berts.BertForSequenceClassification,
    }[dataset]
    
    tokenizer = transformers.BertTokenizerFast.from_pretrained(checkpoint)
    if only_tokenizer:
        return None, tokenizer
    
    bert = model.from_pretrained(checkpoint, cache_dir='./cache/huggingface/')
    return bert, tokenizer

class Trainer:
    def __init__(
        self, 
        subset='mrpc',
        token_permutation_enabled=False,
        permutation_temperature=0.25,
        permutation_sinkhorn_iteration=7,
        performer_enabled=False,
        performer_type='performer',
        head_permutation_enabled=True,
        head_permutation_master_enabled=True,
        linear_pattern_enabled=True,
        dynamic_linear_pattern_enabled=True,
        dynamic_linear_pattern_temperature=0.05,
        synthesizer_enabled=False,
        running_type="head_permutation",
    ) -> None:
        seed()
        
        self.subset = subset
        
        self.amp_enabled = True
        self.device = 0
        
        self.batch_size = task_to_batch_size[self.subset] #* 2
        
        self.epochs = 100
        self.lr = 1e-5
        self.wd = 1e-2
        
        self.base_model, self.tokenizer = get_base_model(subset)
        self.base_model.to(self.device)
        
        self.reset_trainloader()
        self.valid_loader = get_dataloader(subset, self.tokenizer, self.batch_size, split=task_to_valid[self.subset])
        
        self.model = pberts.BertForSequenceClassification(self.base_model.config).to(self.device)

        self.load_state_from_base()

        
        for module in self.model.modules():
            if isinstance(module, pberts.BertAttention):
                # token permutation

                module.permutation_enabled = token_permutation_enabled

            if isinstance(module, pberts.LearnablePermutation):
                module.temperature = permutation_temperature
                module.sinkhorn_iteration = permutation_sinkhorn_iteration
            if isinstance(module, pberts.BertSelfAttention):
                # for sinkhorn and performer
                module.performer_enabled = performer_enabled
                """
                우리꺼는 sinkhorn에다가 permutation matrix에 regularization을 걸었음
                """
                module.performer_type = performer_type
                # per head permutation
                module.permutation_enabled = head_permutation_enabled
                # shared permutation layer
                module.permutation_master_enabled = head_permutation_master_enabled
                module.linear_attention_mask_enabled = linear_pattern_enabled
                module.dynamic_attention_pattern_enabled = dynamic_linear_pattern_enabled
                module.mask_temperature = dynamic_linear_pattern_temperature
                
                ### synthesizer enabled case
                module.synthesizer_enabled=synthesizer_enabled
                module.running_type=running_type#"FactrRandomAttention"
                self.running_type=running_type#"FactrRandomAttention" #TODO 빼야???

            # if isinstance(module, torch.nn.Dropout):
            #     module.p = 0.5
            
            
        
        for param_name, param in self.model.named_parameters():
            if 'permutation' in param_name:
                param.requires_grad = True
            else:
                param.requires_grad = True if self.subset != 'bert' else False
        
        self.optimizer = self.get_optimizer(self.model, lr=self.lr, weight_decay=self.wd)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.model_unwrap = self.model
        # self.model = torch.compile(self.model)
    
    def reset_trainloader(self):
        if self.subset != 'bert':
            self.train_loader = get_dataloader(self.subset, self.tokenizer, self.batch_size, split='train')
        else:
            self.train_loader = WikitextBatchLoader(self.batch_size)
    
    def load_state_from_base(self):
        load_result = self.model.load_state_dict(self.base_model.state_dict(), strict=False)
        for it in load_result.unexpected_keys:
            print('Trainer.init: unexpected', it)
        for it in load_result.missing_keys:
            if not ('performer' in it or 'permutation' in it):
                print('Trainer.init: missing', it)
    
    def get_optimizer(
        self,
        model:torch.nn.Module, 
        optimizer_type:str='AdamW',
        lr:float=1e-4,
        weight_decay:float=1e-3,
        no_decay_keywords=[]
    ):
        param_optimizer = list(model.named_parameters())
        no_decay = [
            'bias', 
            'LayerNorm.bias', 
            'LayerNorm.weight', 
            'BatchNorm1d.weight', 
            'BatchNorm1d.bias', 
            'BatchNorm1d',
            'bnorm',
        ]
        high_lr = ['permutation']
        if no_decay_keywords is not None and len(no_decay_keywords) > 0:
            no_decay += no_decay_keywords
        set_normal = set([p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))])
        set_normal_no_wd = set([p for n, p in param_optimizer if any(nd in n for nd in no_decay)])
        set_high = set([p for n, p in param_optimizer if any(nk in n for nk in high_lr) and (not any(nd in n for nd in no_decay))])
        set_high_no_wd = set([p for n, p in param_optimizer if any(nk in n for nk in high_lr) and any(nd in n for nd in no_decay)])
        set_normal = set_normal - set_high
        set_normal_no_wd = set_normal_no_wd - set_high_no_wd
        params = [
            {'params': list(set_normal), 'weight_decay': weight_decay, 'lr': lr},
            {'params': list(set_normal_no_wd), 'weight_decay': 0.0, 'lr': lr},
            {'params': list(set_high), 'weight_decay': weight_decay, 'lr': lr*10},
            {'params': list(set_high_no_wd), 'weight_decay': 0.0, 'lr': lr*10},
        ]
        # print('1', [k.shape for k in set_normal])
        # print('2', [k.shape for k in set_normal_no_wd])
        # print('3', [k.shape for k in set_high])
        # print('4', [k.shape for k in set_high_no_wd])

        kwargs = {
            'lr':lr,
            'weight_decay':weight_decay,
        }
        
        if optimizer_type == 'AdamW':
            optim_cls = torch.optim.AdamW
        elif optimizer_type == 'Adam':
            optim_cls = torch.optim.Adam
        else: raise Exception()
        
        return optim_cls(params, **kwargs)
    
    def train_step(self, batch):
        self.optimizer.zero_grad()
        
        with torch.autocast('cuda', torch.float16, enabled=self.amp_enabled):
            batch['output_hidden_states'] = True
            batch['output_attentions'] = True
            output = self.model(**batch) # type: pberts.SequenceClassifierOutput
            with torch.no_grad():
                output_base = self.base_model(**batch) # type: pberts.SequenceClassifierOutput
        
        if not self.subset == 'bert':
            loss_model = output.loss
        else:
            loss_model = 0.0
        
        loss_kd = 0
        for ilayer in range(len(output_base.hidden_states)):
            loss_kd += torch.nn.functional.mse_loss(output_base.hidden_states[ilayer], output.hidden_states[ilayer])
        loss_kd = loss_kd / len(output_base.hidden_states) * 10
        assert len(output_base.hidden_states) > 0
        
        loss_perm = 0
        loss_perm_count = 0
        for module in self.model.modules():
            if isinstance(module, pberts.LearnablePermutation):
                t = module.calc_loss()
                if isinstance(t, torch.Tensor):
                    loss_perm += t
                    loss_perm_count += 1
        if loss_perm_count > 0:
            loss_perm = loss_perm / loss_perm_count * 10
        
        loss = loss_model + loss_perm + loss_kd
        
        self.scaler.scale(loss).backward()
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.loss = loss.item()
        self.loss_details = {
            'loss': loss.item(), 
            'loss_perm': loss_perm.item() if isinstance(loss_perm, torch.Tensor) else loss_perm, 
            'loss_model': loss_model.item() if isinstance(loss_model, torch.Tensor) else loss_model,
            'loss_kd': loss_kd.item()
        }
        # print(self.loss_details)
    
    def train_epoch(self):
        # self.model = torch.compile(self.model_unwrap)
        self.reset_trainloader()
        
        self.model.train()
        self.base_model.eval()
        
        smooth_loss_sum = 0
        smooth_loss_count = 0
        
        with tqdm.tqdm(self.train_loader) as pbar:
            for istep, batch in enumerate(pbar):
                batch = batch_to(batch, self.device)
                # for module in self.model.modules():
                #     if isinstance(module, pberts.LearnablePermutation):
                #         module.temperature = 0.15
                self.train_step(batch)
                # for module in self.model.modules():
                #     if isinstance(module, pberts.LearnablePermutation):
                #         module.temperature = 0.01
                # self.train_step(batch)
                
                smooth_loss_sum += self.loss
                smooth_loss_count += 1
                pbar.set_description(
                    f'[{self.epoch+1}/{self.epochs}] '
                    f'({self.running_type}) '
                    f'L:{smooth_loss_sum/smooth_loss_count:.6f}({self.loss:.6f}) '
                    f'Lperm:{self.loss_details["loss_perm"]:.6f} '
                    f'Lkd:{self.loss_details["loss_kd"]:.6f}'
                )
                
                if ((istep+1) % 1500) == 0:
                # if ((istep+1) % 500) == 0:
                    self.evaluate()
                    self.debug_plot_perm()
                    self.save()
                    self.model.train()
                    self.base_model.eval()
    
    def evaluate(self, max_step=123456789, show_messages=True, model=None, split='valid'):
        if self.subset == 'bert':
            return {'accuracy': 0.0}
        
        # seed()
        if model is None:
            model = self.model
        model.eval()
        
        if self.subset == 'bert':
            metric = load_metric('glue', 'cola')
        else:
            metric = load_metric('glue', self.subset)
        
        loader = self.valid_loader
        if split == 'train':
            loader = self.train_loader
        for i, batch in enumerate(tqdm.tqdm(loader, desc=f'({self.subset}[{split}])')):
            if i > max_step: break

            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch['labels']
            del batch['labels']
            
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.amp_enabled):
                outputs = model(**batch)
            predictions = outputs[0]

            if self.subset != 'stsb': 
                predictions = torch.argmax(predictions, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)
        
        score = metric.compute()
        self.last_metric_score = score
        if show_messages:
            print('metric score', score)
        return score

    def save(self):
        os.makedirs('./saves/trainer/kd_trainer/', exist_ok=True)
        path = f'./saves/trainer/kd_trainer/checkpoint_{self.subset}.pth'
        print(f'Trainer: save {path}')
        torch.save({
            'model': self.model.state_dict(),
            'base_model': self.base_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path=None):
        try:
            if path is None:
                path = f'./saves/trainer/kd_trainer/checkpoint_{self.subset}.pth'
            print(f'Trainer: load {path}')
            state = torch.load(path, map_location='cpu')
            self.model.load_state_dict(state['model'])
            self.base_model.load_state_dict(state['base_model'])
            # self.optimizer.load_state_dict(state['optimizer'])
            del state
        except Exception as ex:
            print('error while load', ex)
    
    def debug_plot_perm(self):
        os.makedirs('./saves/trainer/kd_trainer/permM/', exist_ok=True)
        idx = 0
        idx_attn = 0
        for module in self.model.modules():
            if isinstance(module, pberts.LearnablePermutation):
                if module.last_permutation_prob is not None:
                    img = module.last_permutation_prob[0, :, :, 0]
                    img = img.detach().cpu().numpy()
                    
                    idx += 1
                    
                    plt.clf()
                    plt.imshow(img)
                    plt.colorbar()
                    plt.savefig(f'./saves/trainer/kd_trainer/permM/{idx}.png', dpi=160)
                    
                    plt.clf()
                    plt.imshow(np.matmul(img, np.transpose(img)))
                    plt.colorbar()
                    plt.savefig(f'./saves/trainer/kd_trainer/permM/{idx}_i.png', dpi=160)
            if isinstance(module, pberts.BertSelfAttention):
                if module.last_attention_probs is not None:
                    img = module.last_attention_probs[0, 0, :, :]
                    img = img.detach().cpu().numpy()
                    idx_attn += 1
                    
                    plt.clf()
                    plt.imshow(img)
                    plt.colorbar()
                    plt.savefig(f'./saves/trainer/kd_trainer/permM/attn_{idx}.png', dpi=160)
    
    def main(self):
        self.epoch = 0
        self.step = 0
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()
            self.evaluate()
            self.debug_plot_perm()
            self.evaluate(split='train')
            self.save()

if __name__ == '__main__':
    trainer = Trainer(
        subset='mnli'
    )
    trainer.load('./saves/trainer/kd_trainer/checkpoint_mnli_ep48.pth')
    # trainer.load('./saves/trainer/kd_trainer/checkpoint_mnli_ep5_sinkhorn.pth')
    # trainer.load_state_from_base()
    # trainer.load('./saves/trainer/kd_trainer/checkpoint_mnli.pth')
    trainer.main()