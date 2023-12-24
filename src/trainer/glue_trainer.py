import os
import traceback
import warnings
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import transformers
from datasets import load_dataset, load_metric
from ..dataset.glue import get_dataloader, TASK_TO_VALID
import random, copy
import torch

# torch.autograd.set_detect_anomaly(True)

# from transformers.models.bert import modeling_bert as berts
from ..models import hf_bert as berts
from ..utils.get_optimizer import get_optimizer
from ..utils import batch_to, seed
from ..dataset.wikitext import WikitextBatchLoader
import torch.nn.functional as F

import wandb

task_to_epochs = {
    "cola": 100,
    "mnli": 20,
    "mrpc": 100,
    "qnli": 20,
    "qqp":  20,
    "rte":  100,
    "sst2": 100,
    "stsb": 100,
    "wnli": 100,
    "bert": 100,
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

    # NOTE: this bert models has special hooks
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

BF16 = torch.float16 if torch.cuda.is_bf16_supported() else torch.float16
from ..utils import Metric

class Trainer:
    def __init__(
        self, 
        subset='mrpc',
        model_cls=berts.BertForSequenceClassification,
        high_lr_names=[],
        amp_enabled = True,
        trainer_name = 'bert_glue_trainer',
        running_type = None,
        using_kd = True,
        using_loss = True,
        eval_steps = 1500,
        lr = 1e-5,
        epochs = 100,
        load_ignore_keys = ['perlin', 'pbert', 'permute'],
        gradient_checkpointing = False,
        gradient_accumulation_steps = 1,
        wandb_configs = {}
    ) -> None:
        seed()
        
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.load_ignore_keys = load_ignore_keys
        self.running_type = running_type
        self.exp_name = trainer_name
        self.subset = subset
        self.high_lr_names = high_lr_names
        self.using_kd = using_kd
        self.using_loss = using_loss
        self.wandb_configs = wandb_configs
        
        self.amp_enabled = amp_enabled
        self.device = 0
        
        self.batch_size = task_to_batch_size[self.subset]
        
        self.eval_steps = eval_steps * gradient_accumulation_steps
        self.epochs = epochs if epochs is not None else task_to_epochs[subset]
        self.lr = lr
        self.wd = 1e-2
        
        self.base_model, self.tokenizer = get_base_model(subset)
        self.base_model.to(self.device)
        
        self.reset_trainloader()
        self.valid_loader = get_dataloader(subset, self.tokenizer, self.batch_size, split=TASK_TO_VALID[self.subset])
        
        assert model_cls is not None
        self.model = model_cls(self.base_model.config)
        for module in self.model.modules():
            if hasattr(module, 'gradient_checkpointing') and isinstance(getattr(module, 'gradient_checkpointing', None), bool):
                module.gradient_checkpointing = gradient_checkpointing
                if gradient_checkpointing: print('gradient-checkpoint patched')
        self.model.to(self.device)

        self.load_state_from_base()
        
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
            if not any([k in it for k in self.load_ignore_keys]):
                print('Trainer.init: missing', it)
    
    def get_optimizer(
        self,
        model:torch.nn.Module, 
        optimizer_type:str='AdamW',
        lr:float=1e-4,
        weight_decay:float=1e-3,
        no_decay_keywords=[]
    ):
        param_optimizer = list([(n, p) for n, p in model.named_parameters() if p.requires_grad])
        no_decay = [
            'bias', 
            'LayerNorm.bias', 
            'LayerNorm.weight', 
            'BatchNorm1d.weight', 
            'BatchNorm1d.bias', 
            'BatchNorm1d',
            'bnorm',
        ]
        high_lr = self.high_lr_names
        if no_decay_keywords is not None and len(no_decay_keywords) > 0:
            no_decay += no_decay_keywords
        set_normal = set([(n, p) for n, p in param_optimizer if (not any(nd in n for nd in no_decay))])
        set_normal_no_wd = set([(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)])
        set_high = set([(n, p) for n, p in param_optimizer if any(nk in n for nk in high_lr) and (not any(nd in n for nd in no_decay))])
        set_high_no_wd = set([(n, p) for n, p in param_optimizer if any(nk in n for nk in high_lr) and any(nd in n for nd in no_decay)])
        set_normal = set_normal - set_high
        set_normal_no_wd = set_normal_no_wd - set_high_no_wd
        
        psort = lambda lst: list([item[1] for item in sorted(list(lst), key=lambda it: it[0])])
        
        params = [
            {'params': psort(set_normal), 'weight_decay': weight_decay, 'lr': lr},
            {'params': psort(set_normal_no_wd), 'weight_decay': 0.0, 'lr': lr},
            {'params': psort(set_high), 'weight_decay': weight_decay, 'lr': lr*10},
            {'params': psort(set_high_no_wd), 'weight_decay': 0.0, 'lr': lr*10},
        ]
        
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
        with torch.autocast('cuda', BF16, enabled=self.amp_enabled):
            batch['output_hidden_states'] = True
            batch['output_attentions'] = True
            with torch.no_grad():
                output_base = self.base_model(**batch)
            batch['teacher'] = self.base_model
            output = self.model(**batch)
        
        if not self.subset == 'bert' and self.using_loss:
            if self.using_kd:
                loss_model = output.loss * 0.1
            else:
                loss_model = output.loss
        else:
            loss_model = 0.0
        
        loss_kd = 0
        if self.using_kd:
            for ilayer in range(len(output_base.hidden_states)):
                loss_kd += F.mse_loss(output_base.hidden_states[ilayer], output.hidden_states[ilayer])
            loss_kd = loss_kd / len(output_base.hidden_states) * 10
            assert len(output_base.hidden_states) > 0
            loss_kd = loss_kd + F.kl_div(
                F.log_softmax(output.logits, dim=-1), 
                F.softmax(output_base.logits, dim=-1),
                reduction='batchmean',
            ) * 0.1
        
        loss_special = 0
        if hasattr(self.model, 'calc_loss_special'):
            # warnings.warn('special loss found!')
            loss_special = self.model.calc_loss_special()
        
        loss = loss_model + loss_kd + loss_special
        
        self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
        
        # print(loss, flush=True)
        
        if ((int(self.step) + 1) % self.gradient_accumulation_steps) == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.optimizer.zero_grad()
            self.scaler.update()
        
            for module in self.model.modules():
                if hasattr(module, 'redraw_projections'):
                    module.redraw_projections(self.device)
        
        self.loss = loss.item()
        self.loss_details = {
            'loss': loss.item(), 
            'loss_sp': loss_special.item() if isinstance(loss_special, torch.Tensor) else loss_special, 
            'loss_model': loss_model.item() if isinstance(loss_model, torch.Tensor) else loss_model,
            'loss_kd': loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd
        }
        
        return loss.item()
    
    def train_epoch(self):
        # self.model = torch.compile(self.model_unwrap)
        self.reset_trainloader()
        
        self.model.train()
        self.base_model.eval()
        
        smooth_loss_sum = 0
        smooth_loss_count = 0
        
        m = Metric()
        
        with tqdm.tqdm(self.train_loader, dynamic_ncols=True) as pbar:
            for istep, batch in enumerate(pbar):
                batch = batch_to(batch, self.device)
                self.train_step(batch)
                
                smooth_loss_sum += self.loss
                smooth_loss_count += 1
                pbar.set_description(
                    f'[{self.epoch+1}/{self.epochs}] '
                    f'({self.running_type}) ' if self.running_type is not None else ''
                    f'L:{smooth_loss_sum/smooth_loss_count:.6f}({m.update(self.loss, "loss"):.4f}) '
                    f'Lsp:{m.update(self.loss_details["loss_sp"], "loss_sp"):.4f} '
                    f'Lkd:{m.update(self.loss_details["loss_kd"], "loss_kd"):.4f}'
                )
                
                if ((istep+1) % self.eval_steps) == 0:
                    score = self.evaluate()
                    self.save()
                    self.model.train()
                    self.base_model.eval()
                    m = Metric()
                    
                    wandb.log({'eval/score': score}, step=int(self.step))
                
                if ((istep + 1) % 15) == 0:
                    wandb_data = {}
                    for k, v in self.loss_details.items():
                        wandb_data[f'train/{k}'] = v
                    wandb_data['train/epoch'] = (istep / len(pbar)) + self.epoch
                    wandb.log(wandb_data, step=int(self.step))
                
                self.step += 1 / self.gradient_accumulation_steps
    
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
        for i, batch in enumerate(tqdm.tqdm(loader, desc=f'({self.subset}[{split}])', dynamic_ncols=True)):
            if i > max_step: break

            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch['labels']
            del batch['labels']
            
            with torch.no_grad(), torch.autocast('cuda', BF16, enabled=self.amp_enabled):
                self.base_model(**batch)
                batch['teacher'] = self.base_model
                outputs = model(**batch)
            predictions = outputs[0]

            if self.subset != 'stsb': 
                predictions = torch.argmax(predictions, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)
        
        score = metric.compute()
        self.last_metric_score = score
        if show_messages:
            tqdm.tqdm.write(f'metric score {score}')
        return score

    def checkpoint_path(self):
        os.makedirs(f'./saves/trainer/bert_glue_trainer/{self.exp_name}/', exist_ok=True)
        path = f'./saves/trainer/bert_glue_trainer/{self.exp_name}/checkpoint.pth'
        return path
    
    def save(self):
        path = self.checkpoint_path()
        print(f'Trainer: save {path}')
        torch.save({
            'model': self.model.state_dict(),
            'base_model': self.base_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'step': self.step,
        }, path)
    
    def migrate_state_dict(self, state_dict):
        pass
    
    def load(self, path=None):
        try:
            if path is None:
                path = self.checkpoint_path()
            state = torch.load(path, map_location='cpu')
            print(f'Trainer: load {path} @ {state["step"] if "step" in state else None}')
            model_state_dict = state['model']
            model_state_dict = self.migrate_state_dict(model_state_dict)
            self.model.load_state_dict(model_state_dict)
            self.base_model.load_state_dict(state['base_model'])
            try:
                self.optimizer.load_state_dict(state['optimizer'])
            except Exception as ex:
                traceback.print_exc()
                warnings.warn(f"optimizer load is failed due to {ex}, however we will just ignore it")
            del state
        except Exception as ex:
            print('error while load', ex)
    
    def main(self):
        self.epoch = 0
        self.step = 0
        
        from ..utils.secrets import WANDB_KEY, USER_NAME
        os.environ['WANDB_API_KEY'] = WANDB_KEY
        wandb.init(
            # Set the project where this run will be logged
            project=f"[{USER_NAME}] perlin-glue" if USER_NAME is not None else "perlin-glue",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                "subset": self.subset,
                "epochs": self.epochs,
                "global_configs": self.wandb_configs,
            }
        )
        # wandb.watch(self.model, log='all')
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_epoch()
            valid_score = self.evaluate()
            train_score = self.evaluate(split='train', max_step=1000)
            wandb.log({
                'eval/score': valid_score,
                'train/score': train_score,
            }, step=int(self.step))
            self.save()

if __name__ == '__main__':
    trainer = Trainer(
        subset='mnli'
    )
    trainer.main()