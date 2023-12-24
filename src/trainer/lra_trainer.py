import random
import time
import transformers, os
import torch
from torch import nn, optim
import tqdm
import wandb
from ..models import hf_bert as berts
from ..dataset.lra_benchmarks import get_loaders
from ..utils.get_optimizer import get_optimizer
from ..utils import batch_to
from ..dataset.lra_benchmarks.list_ops import get_tokenizer as get_tokenizer_listops
from ..dataset.lra_benchmarks.text import get_tokenizer as get_tokenizer_text
from ..dataset.lra_benchmarks.image import get_tokenizer as get_tokenizer_image
from ..utils import Metric, seed

BF16 = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

LRA_TASKS = {
    'listops': {
        'batch_size': 32,
        'dataloader_fn': lambda bs: get_loaders('listops', bs),
        'lr': 2e-3,
        'wd': 1e-1,
        'epochs': 30,
        'eval_steps': 6000,
        'wandb_steps': 10,
        'gradient_accumulation_steps': 8,
        'config': berts.BertConfig(
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=6,
            hidden_size=512,
            intermediate_size=2048,
            num_labels=10,
            vocab_size=get_tokenizer_listops().vocab_size,
        )
    },
    'text': {
        'batch_size': 16,
        'dataloader_fn': lambda bs: get_loaders('text', bs),
        'lr': 1e-5,
        'wd': 1e-1,
        'epochs': 30,
        'eval_steps': 12000,
        'wandb_steps': 10,
        'gradient_accumulation_steps': 2,
        'config': berts.BertConfig(
            max_position_embeddings=1024,
            num_attention_heads=4,
            num_hidden_layers=4,
            hidden_size=256,
            intermediate_size=1024,
            num_labels=2,
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.2,
            vocab_size=get_tokenizer_text().vocab_size,
        )
    },
    'image': {
        'batch_size': 256,
        'dataloader_fn': lambda bs: get_loaders('image', bs),
        'lr': 1e-3,
        'wd': 0.0,
        'epochs': 500,
        'eval_steps': 12000,
        'wandb_steps': 10,
        'gradient_accumulation_steps': 256//256,
        'config': berts.BertConfig(
            max_position_embeddings=1024,
            num_attention_heads=1,
            num_hidden_layers=1,
            hidden_size=32,
            intermediate_size=64,
            num_labels=10,
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.2,
            vocab_size=get_tokenizer_image().vocab_size,
        )
    }
}

class Trainer:
    def __init__(
        self,
        exp_name: str = 'listops',
        subset: str = 'listops',
        
        model_cls: berts.BertForSequenceClassification = berts.BertForSequenceClassification,
        gradient_checkpointing = False,
        gradient_accumulation_steps = 1,
        using_kd: bool = False,
        kd_checkpoint: str = None,
        
        amp_enabled: bool = True,
        device: int = 0,
    ) -> None:
        seed()
        
        task_desc = LRA_TASKS[subset]
        
        self.exp_name = exp_name
        self.subset = subset
        self.batch_size = task_desc['batch_size']
        self.epochs = task_desc['epochs']
        self.lr = task_desc['lr']
        self.wd = task_desc['wd']
        self.eval_steps = task_desc['eval_steps']
        self.wandb_steps = task_desc['wandb_steps']
        self.device = device
        self.amp_enabled = amp_enabled
        self.gradient_checkpointing = gradient_checkpointing
        assert not gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps * task_desc['gradient_accumulation_steps']
        assert (self.batch_size % gradient_accumulation_steps) == 0
        self.batch_size = self.batch_size // self.gradient_accumulation_steps
        
        self.using_kd = using_kd
        self.kd_checkpoint = kd_checkpoint
        if self.kd_checkpoint is None:
            self.kd_checkpoint = f'./saves/trainer/lra_trainer/{subset}/checkpoint.pth'
        
        self.train_loader, self.test_loader = task_desc['dataloader_fn'](self.batch_size)
        
        self.model = model_cls(task_desc['config'])
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.model, lr=self.lr, weight_decay=self.wd)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.epoch = 0
        self.step = 0
        
        self.base_model = None
        if self.using_kd:
            state = torch.load(self.kd_checkpoint, map_location='cpu')
            self.base_model = berts.BertForSequenceClassification(self.model.config)
            self.base_model.load_state_dict(state['model'])
            self.base_model.to(self.device)
            del state
            print('loaded base model from', self.kd_checkpoint)
    
    def train_step(self, batch):
        base_model = self.base_model
        model = self.model
        
        model.train()
        if base_model is not None: base_model.eval()
        
        with torch.autocast('cuda', BF16, enabled=self.amp_enabled):
            batch['output_hidden_states'] = True
            batch['output_attentions'] = True
            if self.using_kd:
                with torch.no_grad():
                    output_teacher = base_model(**batch)
                batch['teacher'] = base_model
            output = model(**batch)
            loss = output.loss
        
        loss_details = {'loss': loss, 'loss_model': loss}
        
        if self.using_kd:
            loss_model = loss * 0.1
            
            loss_kd = 0
            for ilayer in range(len(output_teacher.hidden_states)):
                loss_kd += torch.nn.functional.mse_loss(
                    output_teacher.hidden_states[ilayer], 
                    output.hidden_states[ilayer]
                )
            loss_kd = loss_kd / len(output_teacher.hidden_states) * 10
            assert len(output_teacher.hidden_states) > 0
            
            loss_special = 0
            if hasattr(self.model, 'calc_loss_special'):
                # warnings.warn('special loss found!')
                loss_special = self.model.calc_loss_special()
            
            loss = loss_model + loss_kd + loss_special
            
            loss_details['loss'] = loss.item()
            loss_details['loss_model'] = loss_model.item()
            loss_details['loss_kd'] = loss_kd.item()
            loss_details['loss_sp'] = loss_special.item()
        
        self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
        
        if ((self.step + 1) % self.gradient_accumulation_steps) == 0:
            self.scaler.step(self.optimizer)
            self.optimizer.zero_grad()
            self.scaler.update()
        
        self.step += 1
        
        return loss, loss_details
    
    def train_epochs(self):
        m = Metric()
        with tqdm.tqdm(self.train_loader, dynamic_ncols=True) as pbar:
            for istep, batch in enumerate(pbar):
                batch = batch_to(batch, self.device)
                loss, loss_details = self.train_step(batch)
                
                if (self.step % self.eval_steps) == 0:
                    metric = self.evaluate()
                    self.save()
                    m = Metric()
                    wandb.log({'eval/metric': metric}, step=self.step)
                
                if (self.step % self.wandb_steps) == 0:
                    wandb_dict = {f'train/{k}': v for k, v in loss_details.items()}
                    wandb_dict['train/epoch'] = istep / len(pbar) + self.epoch
                    wandb.log(wandb_dict, step=self.step)
                
                pbar.set_description((
                    f'[{self.epoch}/{self.epochs}] '
                    f'L:{m.update(loss.item(), "l"):.4f}({m.update(loss_details.get("loss_model", 0.0), "lm"):.4f}) '
                    f'Lsp:{m.update(loss_details.get("loss_sp", 0.0), "lsp"):.4f} '
                    f'Lkd:{m.update(loss_details.get("loss_kd", 0.0), "lkd"):.4f}'
                ).strip())
    
    def evaluate(self):
        model = self.model
        base_model = self.base_model
        model.eval()
        if base_model is not None: base_model.eval()
        
        acc_sum = acc_count = 0
        for batch in tqdm.tqdm(self.test_loader, dynamic_ncols=True):
            batch = batch_to(batch, self.device)
            batch['output_hidden_states'] = True
            batch['output_attentions'] = True
            with torch.no_grad(), torch.autocast('cuda', BF16, enabled=self.amp_enabled):
                if base_model is not None:
                    base_model(**batch)
                    batch['teacher'] = base_model
                output = model(**batch)
                logits = output.logits
            acc = ((torch.argmax(logits, dim=-1) == batch['labels'])*1.0).sum()
            acc_sum += acc.item()
            acc_count += len(batch['input_ids'])
        
        acc_sum = acc_sum / (acc_count + 1e-8)
        print('accuracy:', acc_sum)
        
        return acc_sum
    
    def checkpoint_path(self):
        dir = f'./saves/trainer/lra_trainer/{self.exp_name}'
        os.makedirs(dir, exist_ok=True)
        return f'{dir}/checkpoint.pth'
    
    def save(self, path=None):
        if path is None: path = self.checkpoint_path()
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
        }, path)
        print('saved', path)
    
    def load(self, path=None):
        if path is None: path = self.checkpoint_path()
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model'])
        try:
            self.optimizer.load_state_dict(state['optimizer'])
            self.scaler.load_state_dict(state['scaler'])
        except Exception as ex:
            print('except while load', ex)
        del state
        print('loaded', path)
    
    def main(self):
        from ..utils.secrets import WANDB_KEY, USER_NAME
        os.environ['WANDB_API_KEY'] = WANDB_KEY
        wandb.init(
            project=f"[{USER_NAME}] perlin-lra",
            name=f"{self.exp_name}-{int(time.time()*1000 % 1000)}",
            config={
                "lr": self.lr,
                "subset": self.subset,
                "epochs": self.epochs,
            }
        )
        wandb.watch(self.model, log='all')
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            
            self.train_epochs()
            metric = self.evaluate()
            wandb.log({'eval/metric': metric}, step=self.step)
            self.save()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', default='listops', type=str)
    args = parser.parse_args()
    
    t = Trainer(
        subset=args.subset,
        exp_name=args.subset,
    )
    t.main()