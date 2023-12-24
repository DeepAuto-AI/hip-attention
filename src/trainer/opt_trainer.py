from dataclasses import dataclass, field, asdict
import math
import traceback
import warnings

import tqdm
from ..utils import raise_if_nan, seed, batch_to, Metric
raise_if_nan = lambda x: x
from ..models import hf_opt as opt
from typing import List, Dict, Optional, Tuple, Callable
import torch
import wandb
import transformers
from torch import nn, optim
import os
from ..dataset.wikitext2 import get_dataloader
import gc
import torch.nn.functional as F
from ..utils import strify
import torch.distributed

CHECKPOINT_REPOSITORY = os.environ.get('CHECKPOINT_REPOSITORY', './saves')

default = lambda x, y: x if x is not None else y

@dataclass
class TrainerConfig:
    # trainer metadata
    experiment_name: str = 'opt_wikitext2'
    eval_steps: int = 2000
    wandb_steps: int = 20
    
    # optimization flags
    # TODO grad checkpointing is not correct...
    gradient_checkpointing: bool = False
    kd_checkpointing: bool = False
    gradient_accumulation_steps: int = 8
    amp_enabled: bool = True
    
    # experiment settings
    dataset: str = 'wikitext2'
    teacher_model_cls: opt.OPTForCausalLM = opt.OPTForCausalLM
    model_cls: opt.OPTForCausalLM = opt.OPTForCausalLM
    model_config: str = 'Aalaa/opt-125m-wikitext2'
    # model_config: str = 'lnair/opt-350m-wikitext2'
    lr: float = 1e-5
    lr_high_scale: float = 10.0
    lr_low_scale: float = 1.0
    wd: float = 1e-2
    num_steps: int = 100000
    batch_size: int = 1
    load_ignore_keys: List[str] = field(default_factory=lambda: ['perlin'])
    high_lr_names: List[str] = field(default_factory=lambda: ['perlin'])
    using_kd: bool = True
    using_loss: bool = True
    # NOTE decrease this only for DEBUG!!, this should be larger than 2048 on OPT
    max_seq_len: int = 32000
    additional_config: dict = field(default_factory=lambda: {})
    
    on_model_init: Optional[Callable] = None
    
# BF_16 = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
BF_16 = torch.float16

def gc_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
import deepspeed
import deepspeed.comm

class KDWrapperModel(nn.Module):
    def __init__(self, config, device, swap_out_device, model, base_model, using_deepspeed):
        super().__init__()
        
        self.config = config
        self.device = device
        self.swap_out_device = swap_out_device
        self.model = model
        self.base_model = base_model
        self.using_deepspeed = using_deepspeed
    
        self.lazy_attention = True
        if self.lazy_attention:
            for m in self.base_model.modules():
                if hasattr(m, 'lazy_checkout'):
                    m.lazy_checkout = True
    
    def forward(self, batch):
        # print('cc')

        if self.lazy_attention:
            batch['output_attentions'] = False
        
        self.base_model.eval()
        
        # print('dd')
        
        swap_in_device = self.device
        swap_out_device = self.swap_out_device
        # swap_out_device = torch.device('cpu')
        
        for m in self.base_model.modules():
            if hasattr(m, 'swap_out_device'):
                if self.using_deepspeed:
                    m.swap_out_device = torch.device('cpu')
                else:
                    m.swap_out_device = swap_out_device
        
        for m in self.model.modules():
            if hasattr(m, 'swap_out_device'):
                m.swap_out_device = swap_out_device
            if hasattr(m, 'swap_in_device'):
                m.swap_in_device = swap_in_device
        
        # print('fi')
        
        with torch.no_grad(), torch.autocast('cuda', BF_16, enabled=self.config.amp_enabled):
            # print('*'*20, 'base', torch.cuda.max_memory_allocated() // 1024 // 1024)
            output_teacher = self.base_model(**batch)
            if batch['output_hidden_states']:
                output_teacher.hidden_states = batch_to(output_teacher.hidden_states, swap_out_device)
            if batch['output_attentions']:
                output_teacher.attentions = batch_to(output_teacher.attentions, swap_out_device)
            output_teacher.logits = batch_to(output_teacher.logits, swap_out_device)
        
        # print('gg')
        
        # print('*'*20, 'base2', torch.cuda.max_memory_allocated() // 1024 // 1024)
        with torch.autocast('cuda', BF_16, enabled=self.config.amp_enabled):
            batch['teacher'] = self.base_model
            # print('*'*20, 'model', torch.cuda.max_memory_allocated() // 1024 // 1024)
            output_student = self.model(**batch)
            # print('*'*20, 'model2', torch.cuda.max_memory_allocated() // 1024 // 1024)
        
        if self.config.using_loss:
            if self.config.using_kd:
                loss_model = output_student.loss * 0.1
            else:
                loss_model = output_student.loss
        else:
            loss_model = 0.0
        if float(os.environ.get('__TASK_LOSS', '0')) > 0.01:
            loss_model = float(os.environ.get('__TASK_LOSS', '0')) * output_student.loss
        
        loss_kd = 0
        if self.config.using_kd:
            for ilayer, teacher_value in enumerate(output_teacher.hidden_states):
                teacher_value = batch_to(teacher_value, self.device)
                raise_if_nan(teacher_value)
                student_value = batch_to(output_student.hidden_states[ilayer], self.device)
                raise_if_nan(student_value)
                _loss_kd_layer = F.mse_loss(batch_to(teacher_value, torch.float32), batch_to(student_value, torch.float32))
                loss_kd += _loss_kd_layer
                raise_if_nan(loss_kd)
            loss_kd = loss_kd / len(output_teacher.hidden_states) * 5
            raise_if_nan(loss_kd)
            assert len(output_teacher.hidden_states) > 0
            teacher_logit = batch_to(output_teacher.logits, self.device).view(-1, output_student.logits.shape[-1])
            raise_if_nan(teacher_logit)
            student_logit = batch_to(output_student.logits, self.device).view(-1, output_student.logits.shape[-1])
            raise_if_nan(student_logit)
            loss_kd = loss_kd + F.kl_div(
                F.log_softmax(student_logit, dim=-1, dtype=torch.float32), 
                F.softmax(teacher_logit, dim=-1, dtype=torch.float32),
                reduction='batchmean',
            ) * 0.2
            raise_if_nan(loss_kd)
        
        loss_special = 0
        if hasattr(self.model, 'calc_loss_special'):
            loss_special = self.model.calc_loss_special()
        # assert loss_special.requires_grad, loss_special.requires_grad
        
        if not os.environ.get('IGNORE_KD_LOSS', '0') == '1':
            loss = loss_model + loss_kd + loss_special
        else:
            warnings.warn('kd loss ignored!')
            loss = output_student.loss
            if os.environ.get('KD_SELF_TEACHER', '0') == '1':
                warnings.warn('using self teacher!')
                loss = loss + loss_special
        
        # assert loss.requires_grad, f"{loss_model.requires_grad}, {loss_kd.requires_grad}, {loss_special.requires_grad}"
        
        loss_py = loss.item()
        loss_details = {
            'loss': loss_py,
            'loss_sp': loss_special.item() if isinstance(loss_special, torch.Tensor) else loss_special, 
            'loss_model': loss_model.item() if isinstance(loss_model, torch.Tensor) else loss_model,
            'loss_kd': loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd,
            'student_model_loss': output_student.loss,
        }
        
        # print('ff')
        
        return loss, loss_py, loss_details

class Trainer:
    def __init__(self, config: TrainerConfig = None, skip_init_loaders = False, deepspeed=False, cmd_args=None) -> None:
        import deepspeed as ds
        
        self.config = config if config is not None else TrainerConfig()
        if cmd_args is None:
            self.device = 0
        else:
            self.device = 0 if cmd_args.local_rank < 0 else cmd_args.local_rank
        self.local_rank = max(0, cmd_args.local_rank if cmd_args != None else 0)
        torch.cuda.set_device(self.device)
        seed(42 + self.local_rank)
        
        if deepspeed: 
            print('deepspeed enabeld, start dist server')
            ds.init_distributed()
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_size = 1
        print(f'DDP: {self.local_rank} / {self.world_size}')
        if self.config.kd_checkpointing:
            self.swap_out_device = torch.device('cpu')
            warnings.warn("using cpu offload for KD buffers. this will save a lot of memory, but slow down A--LOT!")
            raise Exception()
        else:
            self.swap_out_device = self.device
        
        self.deepspeed = deepspeed
        self.cmd_args = cmd_args
        
        if self.deepspeed:
            ds_config = ds.DeepSpeedConfig(self.cmd_args.deepspeed_config, None)
            old_steps = self.config.gradient_accumulation_steps
            self.config.gradient_accumulation_steps = int(ds_config.train_batch_size / (ds_config.train_micro_batch_size_per_gpu * self.world_size))
            warnings.warn(f"--gradient-accumulation-steps={old_steps} is ignored, cacluated grad. acc. steps using deepspeed config. inferenced={self.config.gradient_accumulation_steps}")
        
        self.init_model()
        if self.config.on_model_init is not None: self.config.on_model_init()
        if not skip_init_loaders: self.init_loader()
        self.init_optimizer()
        self.deepspeed_inited = False
        if not os.environ.get('LAZY_DEEPSPEED', '0') == '1':
            self.init_deepspeed()
        
        self.wandb_inited = False
    
    def init_model(self):
        teacher = self.config.teacher_model_cls.from_pretrained(
            self.config.model_config
        ).eval()
        
        student = self.config.model_cls(teacher.config)
        try:
            missing_keys, unexpected_keys = student.load_state_dict(teacher.state_dict(), strict=False)
            missing_keys = [k for k in missing_keys if not any([s in k for s in self.config.load_ignore_keys])]
            unexpected_keys = [k for k in unexpected_keys if not any([s in k for s in self.config.load_ignore_keys])]
            if len(missing_keys) > 0: 
                print('during init model, missing keys are:', missing_keys)
            if len(unexpected_keys) > 0: 
                print('during init model, unexpected keys are:', unexpected_keys)
        except Exception as ex:
            print(ex)
        
        # compatible with GPT2
        if hasattr(student.config, 'n_positions'):
            max_length = student.config.n_positions
        else:
            max_length = student.config.max_position_embeddings
        self.max_seq_len = min(default(self.config.max_seq_len, 32000), max_length)
        
        if self.config.gradient_checkpointing:
            for m in student.modules():
                if hasattr(m, 'gradient_checkpointing'):
                    print(f'patch gradient checkpointing {type(m)}')
                    m.gradient_checkpointing = True
                    if hasattr(m, "config"):
                        m.config.use_cache = False
                if hasattr(m, 'use_deepspeed'):
                    m.use_deepspeed = self.deepspeed
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.model_config, use_fast=True)
        
        self.kd_model = KDWrapperModel(
            self.config, 
            self.device, 
            self.swap_out_device,
            student, 
            teacher,
            self.deepspeed,
        )
        # self.kd_model = self.kd_model.to(self.device)
        self.base_model = self.kd_model.base_model
        self.model = self.kd_model.model
    
    def init_loader(self):
        if self.config.dataset == 'wikitext2':
            self.train_loader = get_dataloader(
                subset='train', 
                tokenizer=self.tokenizer, 
                batch_size=self.config.batch_size, 
                max_length=self.max_seq_len,
                local_rank=self.local_rank,
                world_size=self.world_size,
            )
            self.valid_loader = get_dataloader(
                subset='valid', 
                tokenizer=self.tokenizer, 
                batch_size=self.config.batch_size, 
                max_length=self.max_seq_len,
                local_rank=self.local_rank,
                world_size=self.world_size,
            )
        else:
            raise Exception()
    
    def get_optimizer(
        self,
        model:torch.nn.Module, 
        optimizer_type:str='AdamW',
        lr:float=1e-4,
        weight_decay:float=1e-3,
        no_decay_keywords=[]
    ):
        lr_high = lr * self.config.lr_high_scale
        lr_low = lr * self.config.lr_low_scale
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
        high_lr = self.config.high_lr_names
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
            {'params': psort(set_normal), 'weight_decay': weight_decay, 'lr': lr_low},
            {'params': psort(set_normal_no_wd), 'weight_decay': 0.0, 'lr': lr_low},
            {'params': psort(set_high), 'weight_decay': weight_decay, 'lr': lr_high},
            {'params': psort(set_high_no_wd), 'weight_decay': 0.0, 'lr': lr_high},
        ]

        kwargs = {
            'lr':lr,
            'weight_decay':weight_decay,
        }
        
        if optimizer_type == 'AdamW':
            if self.deepspeed:
                config = deepspeed.DeepSpeedConfig(self.cmd_args.deepspeed_config, None)
                if config.zero_enabled and (config.zero_config.stage == 3 or config.zero_config.offload_optimizer):
                    optim_cls = deepspeed.ops.adam.DeepSpeedCPUAdam
                    # optim_cls = torch.optim.AdamW
                else:
                    optim_cls = torch.optim.AdamW
                print(f'selected optim cls {optim_cls}')
            else:
                optim_cls = torch.optim.AdamW
        elif optimizer_type == 'Adam':
            optim_cls = torch.optim.Adam
        else: raise Exception()
        
        return optim_cls(params, **kwargs)
    
    def init_optimizer(self):
        self._istep = 0
        self.step = 0
        self.epoch = 0
        self.backward_performed = 0
        
        self.optimizer = self.get_optimizer(
            model=self.model, 
            lr=self.config.lr, 
            weight_decay=self.config.wd
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp_enabled)
        self.optimizer.zero_grad()
    
    def init_deepspeed(self):
        if self.deepspeed_inited: return
        
        if self.deepspeed:
            engine, optimizer, _, _ = deepspeed.initialize(
                args=self.cmd_args,
                model=self.kd_model,
                optimizer=self.optimizer,
            )
            
            self.ds_engine = engine
            self.ds_optimizer = optimizer
        else:
            self.kd_model.to(self.device)
        
        self.deepspeed_inited = True
    
    def train_step(self, batch) -> Tuple[float, Dict[str, float]]:
        batch = batch_to(batch, self.device)
        del batch['trg_len']
        batch.update({
            'output_hidden_states': True,
            'output_attentions': False,
        })
        
        if not self.deepspeed:
            loss, loss_py, loss_details = self.kd_model(batch)
            
            if not torch.isnan(loss).item():
                self.scaler.scale(loss / self.config.gradient_accumulation_steps).backward()
                self.backward_performed += 1
            
            if ((self._istep + 1) % self.config.gradient_accumulation_steps) == 0:
                assert self.backward_performed > 0, self.backward_performed
                self.backward_performed = 0
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                for module in self.model.modules():
                    if hasattr(module, 'redraw_projections'):
                        module.redraw_projections(self.device)
        else:
            self.ds_engine.train()
            loss, loss_py, loss_details = self.ds_engine(batch)
            self.ds_engine.backward(loss)
            self.ds_engine.step()
        
        return loss_py, loss_details
    
    def train_epoch(self):
        if not self.deepspeed:
            self.model.train()
            self.base_model.eval()
        else:
            self.ds_engine.train()
        
        m = Metric()
        
        train_loader_len = len(self.train_loader)
        total_steps_len = (int(self.config.num_steps * self.config.gradient_accumulation_steps) + 1) - self._istep
        done = train_loader_len >= total_steps_len
        with tqdm.tqdm(self.train_loader, dynamic_ncols=True, total=min(train_loader_len, total_steps_len), disable=self.local_rank != 0) as pbar:
            for istep, batch in enumerate(pbar):
                wandb_dict = {}
                try:
                    loss, loss_details = self.train_step(batch)
                except torch.cuda.OutOfMemoryError as ex: # type: ignore
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    print(torch.cuda.memory_summary())
                    for obj in gc.get_objects():
                        try:
                            if isinstance(obj, torch.Tensor) and (not isinstance(obj, torch.nn.Parameter)) and (torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data))):
                                if obj.device != torch.device('cpu'):
                                    obj = obj #type: torch.Tensor
                                    print('obj', type(obj), obj.size(), obj.device, obj.element_size() * obj.numel())
                                    referrers = gc.get_referrers(obj)
                                    print('ref', len(referrers), [type(r) for r in referrers])
                        except OSError:
                            pass
                    raise ex
                wandb_dict['train/loss'] = loss
                wandb_dict['train/epoch'] = self.epoch + istep / train_loader_len
                wandb_dict.update({
                    f'trian/loss/{k}': v for k, v in loss_details.items()
                })
                
                pbar.set_description(
                    f'L:{m.update(loss, "l"):.4f} '\
                    f'Lsp:{m.update(loss_details["loss_sp"], "sp"):.4f} '\
                    f'Lkd:{m.update(loss_details["loss_kd"], "kd"):.4f} '\
                    f'Lm:{m.update(loss_details["loss_model"], "md"):.4f} '\
                    f'M:{torch.cuda.max_memory_allocated()/1024/1024:.1f}'\
                )
                
                self._istep += 1
                self.step = self._istep // self.config.gradient_accumulation_steps
                
                reported = False
                if ((self.step + 1) % self.config.eval_steps) == 0 and (self._istep % self.config.gradient_accumulation_steps) == 0:
                    gc_cuda()
                    if self.deepspeed:
                        self.ds_engine.empty_partition_cache()
                    score = self.evaluate()
                    # if self.local_rank == 0: 
                    #     score = self.evaluate()
                    # else:
                    #     score = 0
                    if self.deepspeed:
                        deepspeed.comm.barrier()
                    gc_cuda()
                    wandb_dict['eval/score'] = score
                    self.save()
                    
                    if not self.deepspeed:
                        self.model.train()
                        self.base_model.eval()
                    else:
                        self.ds_engine.train()
                    
                    reported = True
                
                if ((self.step % self.config.wandb_steps) == 0 and (self._istep % self.config.gradient_accumulation_steps) == 0) or reported:
                    if self.wandb_inited: wandb.log(wandb_dict, step=self.step)
                
                if self.step > self.config.num_steps:
                    done = True
                    break
        
        return done
    
    def evaluate(self, on_step=None, quite=False):
        gc.collect()
        torch.cuda.empty_cache()

        if not self.deepspeed:
            self.model.eval()
            self.base_model.eval()
        else:
            self.ds_engine.eval()
        
        # nlls = []
        nll_sum = torch.tensor(0.0, device=self.device)
        nll_count = torch.tensor(0.0, device=self.device)
        for batch in tqdm.tqdm(self.valid_loader, dynamic_ncols=True, desc='evaluate', disable=self.local_rank != 0):
            batch = batch_to(batch, self.device)
            trg_len = batch['trg_len']
            del batch['trg_len']
            batch.update({
                'output_hidden_states': True,
                'output_attentions': False,
            })
            if not self.deepspeed:
                # with torch.no_grad(), torch.autocast('cuda', BF_16, enabled=self.config.amp_enabled):
                #     self.base_model(**batch)
                with torch.no_grad(), torch.autocast('cuda', BF_16, enabled=self.config.amp_enabled):
                    batch['teacher'] = self.base_model
                    output_student = self.model(**batch)
                    student_loss = output_student.loss.item()
            else:
                with torch.no_grad():
                    # print('aa')
                    _, _, loss_details = self.ds_engine(batch)
                    # print('bb')
                    student_loss = loss_details['student_model_loss']
                    # print(student_loss)
            print(student_loss, trg_len)
            neg_log_likelihood = student_loss * trg_len.float().mean().item()
            
            # torch.cuda.synchronize()
            # gc.collect()
            # torch.cuda.empty_cache()
            
            # nlls.append(neg_log_likelihood)
            nll_sum += neg_log_likelihood
            nll_count += batch['input_ids'].shape[-1]

            # for debugging
            if on_step is not None: on_step()
        
        if self.world_size > 1:
            print(f"worker {self.local_rank}: nll_sum={nll_sum}, nll_count={nll_count}")
            torch.distributed.all_reduce(nll_sum)
            torch.distributed.all_reduce(nll_count)
        if self.local_rank == 0:
            print(f"master worker: nll_sum={nll_sum}, nll_count={nll_count}({self.valid_loader.dataset.seq_len})")
        
        ppl = math.exp(nll_sum.item() / nll_count.item())
        if not quite: print(f'[{self.step}/{self.config.num_steps}] PPL:', ppl)
        return ppl
    
    def checkpoint_path(self):
        os.makedirs(f'{CHECKPOINT_REPOSITORY}/trainer/opt_trainer/{self.config.experiment_name}/', exist_ok=True)
        path = f'{CHECKPOINT_REPOSITORY}/trainer/opt_trainer/{self.config.experiment_name}/checkpoint.pth'
        if os.environ.get('FORCE_OPENWEBTEXT', '0') == '1':
            path += 'owt.pth'
        return path
    
    def save(self, path=None):
        if os.environ.get('NO_SAVE', '0') == '1':
            print('skip saving')
            return
        
        if path is None: 
            path = self.checkpoint_path()
        
        save_prefix = os.environ.get('__SAVE_PREFIX', '')
        if save_prefix != '':
            path = path[:-4] + save_prefix + path[-4:]
        
        if not self.deepspeed:
            torch.save({
                'step': self.step,
                '_istep': self._istep,
                'epoch': self.epoch,
                'model': self.model.state_dict(),
                'base_model': self.base_model.state_dict(),
                'scaler': self.scaler.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'config': asdict(self.config),
            }, path)
        else:
            path = path[:-4]
            os.makedirs(path, exist_ok=True)
            self.ds_engine.save_checkpoint(path, tag='deepspeed')
        print('saved', path)
    
    def load(self, path=None):
        if path is None: path = self.checkpoint_path()
        
        load_prefix = os.environ.get('__LOAD_PREFIX', '')
        if load_prefix != '':
            path = path[:-4] + load_prefix + path[-4:]
        
        if not self.deepspeed or not self.deepspeed_inited:
            if os.path.exists(path):
                print(f'load from {path}')
                state = torch.load(path, map_location='cpu')
                try:
                    result = self.model.load_state_dict(state['model'], strict=False)
                    print(result)
                except RuntimeError as ex:
                    print(ex)
                if 'scaler' in state and len(state['scaler']) > 0: self.scaler.load_state_dict(state['scaler'])
                try:
                    self.optimizer.load_state_dict(state['optimizer'])
                except Exception as ex:
                    traceback.print_exc()
                    print('error during load optimizer', ex)
                if 'step' in state:
                    step = state['step']
                else:
                    step = -1
                if 'epoch' in state:
                    epoch = state['epoch']
                else:
                    epoch = -1
                if 'epochs' in state:
                    epochs = state['config']['epochs']
                else:
                    epochs = -1
                del state
                print(f'loaded {path} ({step}@[{epoch}/{epochs}])')
            else:
                path = path[:-4]
                print(f'try to load from {path}@{"deepspeed"}')
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                try:
                    try:
                        state = get_fp32_state_dict_from_zero_checkpoint(path, tag='deepspeed')
                    except FileNotFoundError:
                        mp_state_path = os.path.join(path, 'deepspeed', 'mp_rank_00_model_states.pt')
                        if not os.path.exists(mp_state_path):
                            print('not found', mp_state_path)
                            return
                        state = torch.load(mp_state_path, map_location='cpu')['module']
                except RuntimeError as ex:
                    print(ex)
                try:
                    result = self.kd_model.load_state_dict(state, strict=False)
                    print(result)
                except RuntimeError as ex:
                    print(ex)
                del state
                print(f'loaded {path}')
        else:
            path = path[:-4]
            print(f'try to load from {path}@{"deepspeed"}')
            if int(os.environ.get('DS_LOAD_OPTIM', '1')) == 1:
                self.ds_engine.load_checkpoint(path, tag='deepspeed', load_module_strict=False)
            else:
                self.ds_engine.load_checkpoint(path, tag='deepspeed', load_optimizer_states=False, load_module_strict=False)
            print(f'loaded {path} ({-1}@[{-1}/{-1}])')
    
    def main(self):
        if os.environ.get('LAZY_DEEPSPEED', '0') == '1':
            self.init_deepspeed()
        
        torch.set_float32_matmul_precision('high')
        warnings.warn("using TF32 if available, so be caution...")
        
        from ..utils.secrets import WANDB_KEY, USER_NAME
        os.environ['WANDB_API_KEY'] = WANDB_KEY
        if self.local_rank == 0:
            try:
                config = asdict(self.config)
            except Exception as ex:
                print('failed to pickle')
                config = {'oops': f'{ex}'}
            wandb.init(
                project=f"[{USER_NAME}] perlin-opt" if USER_NAME is not None else "perlin-opt",
                config=config
            )
            self.wandb_inited = True
            print('save path', self.checkpoint_path())
        
        epoch = 0
        while True:
            self.epoch = epoch
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            done = self.train_epoch()
            
            gc_cuda()
            if self.local_rank == 0:
                score = self.evaluate()
            else:
                score = 0
            if self.deepspeed:
                deepspeed.comm.barrier()
            gc_cuda()
            
            if self.wandb_inited: wandb.log({'eval/score': score, 'train/epoch': self.epoch+1}, step=self.step)
            self.save()
            
            if done: break
            epoch += 1
        
        if self.world_size > 1:
            torch.distributed.destroy_process_group()

if __name__ == '__main__':
    t = Trainer()
    t.main()
