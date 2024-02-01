import gc
import os
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.onnx
import lightning as pl
import transformers
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch.utils.checkpoint
from deepspeed.ops.adam import DeepSpeedCPUAdam
import deepspeed.checkpoint
import deepspeed
import torch.autograd

# torch.autograd.set_detect_anomaly(True)

from src.models.modeling_llama import LlamaForCausalLM, LlamaConfig, LlamaDecoderLayer

from src.utils import seed
from src.dataset.labdataset import LabDataset
from src.dataset.booksum import BookSumDataset
from src.dataset.alpaca import AlpacaDataset
from src.dataset.openwebtext import OpenWebTextDataset
from torch.utils.data import DataLoader, random_split

torch.set_float32_matmul_precision('high')

@dataclass
class TrainConfig:
    disable_kd: bool = False
    using_fsdp: bool = False
    lr: float = 5e-5
    batch_size: int = 1
    accumulation_steps: int = 2
    lora_r: int = 32
    save_steps: int = 100
    dense_queries: int = None
    seq_len: int = 4096
    max_steps: int = -1
    model_checkpoint_dir: str = "./saves/dev/checkpoint"
    dataset: str = 'wikitext103'
    load_from_checkpoint: str = None
    k: int = 512
    block_size: int = 8
    init_from_checkpoint: str = None
    method: str = 'tree'

class LabDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config: TrainConfig,
        num_workers: int = 0,
        data_dir: Path ="data",
        download: bool = True,
        train_size: float = 0.9,
    ):
        super().__init__()
        self.config = config
        self.data_dir = data_dir
        self.block_size = config.seq_len
        self.download = download
        self.num_workers = num_workers
        self.train_size = train_size
        self.dataset = None
        self.tokenizer = load_tokenizer()
        self.bsize = config.batch_size
    
    def prepare_data(self):
        if self.config.dataset in ['wikitext2', 'wikitext103']:
            self.dataset = LabDataset(
                data_dir=self.data_dir,
                block_size=self.block_size,
                download=self.download,
                tokenizer=self.tokenizer,
                dataset=self.config.dataset,
            )
        elif self.config.dataset in ['alpaca']:
            self.dataset = AlpacaDataset(
                tokenizer=self.tokenizer,
            )
        elif self.config.dataset in ['booksum']:
            self.dataset = BookSumDataset(
                tokenizer=self.tokenizer,
            )
        elif self.config.dataset in ['openwebtext']:
            self.dataset = OpenWebTextDataset(
                tokenizer=self.tokenizer,
                stride=self.block_size,
            )
        else:
            raise Exception()
    
    def setup(self, stage: str):
        if self.config.dataset in ['wikitext2', 'wikitext103']:
            if stage == "fit" or stage is None:
                test_size = min(100, len(self.dataset) * (1 - self.train_size))
                train_size = int(len(self.dataset) - test_size)
                self.train_data, self.val_data = random_split(self.dataset, lengths=[train_size, test_size])
            if stage == "test" or stage is None:
                self.test_data = self.val_data
        elif self.config.dataset in ['booksum', 'alpaca', 'openwebtext']:
            if stage == "fit" or stage is None:
                def train_val_dataset(dataset, val_split=0.05):
                    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
                    train = Subset(dataset, train_idx)
                    valid = Subset(dataset, val_idx)
                    return train, valid
                self.train_data, self.val_data = train_val_dataset(self.dataset)
            if stage == "test" or stage is None:
                self.test_data = self.val_data
        else:
            raise Exception()

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers, batch_size=self.bsize)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=self.num_workers, batch_size=self.bsize)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers, batch_size=self.bsize)

from peft import LoraConfig, TaskType
from peft import get_peft_model, prepare_model_for_kbit_training

def load_model(
    train_config: TrainConfig = None, 
    method = 'tree', 
    device = 'cuda:0',
    model_id = 'togethercomputer/LLaMA-2-7B-32K',
):
    if train_config.using_fsdp:
        device = 'cpu'
    
    config = LlamaConfig.from_pretrained(model_id)
    config._attn_implementation = config.attn_implementation = 'sdpa'
    
    quant_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_skip_modules=['tree_avgpool_scaler'],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    if train_config.using_fsdp:
        quant_config = None
    
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        config=config, 
        device_map={"" : device} if device != 'cpu' else 'cpu',
        load_in_4bit=quant_config is not None,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    for m in model.modules():
        if hasattr(m, 'attention_method'):
            m.attention_method = method
            m.tree_k = train_config.k
            m.tree_block_size = train_config.block_size
            if train_config.dense_queries is None:
                train_config.dense_queries = train_config.k
            m.tree_dense_queries = train_config.dense_queries
        if hasattr(m, 'gradient_checkpointing'):
            m.gradient_checkpointing = True
            if train_config.using_fsdp:
                # m._gradient_checkpointing_func = deepspeed.checkpointing.checkpoint
                m._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
            else:
                m._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
    
    if method != 'none':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_config.lora_r,
            lora_alpha=train_config.lora_r//2, 
            lora_dropout=0.05,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj', 
                'gate_proj', 'up_proj', 'down_proj', 
                # 'input_layernorm', 'post_attention_layernorm'
            ],
            modules_to_save=[
                'tree_avgpool_scaler',
                'input_layernorm', 'post_attention_layernorm'
            ]
        )
        
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        if train_config.init_from_checkpoint is not None:
            print('loading from', train_config.init_from_checkpoint)
            state_dict = torch.load(train_config.init_from_checkpoint, map_location='cpu')['state_dict']
            keys = list(state_dict.keys())
            for key in keys:
                x = state_dict[key]
                state_dict[key.strip('model.')] = x
                del state_dict[key]
            model.load_state_dict(state_dict)
            print('lora checkpoint loaded from', train_config.init_from_checkpoint)
    
    return model

def load_tokenizer():
    model_id = 'togethercomputer/LLaMA-2-7B-32K'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return tokenizer

class LabModule(pl.LightningModule):
    def __init__(self, config: TrainConfig):
        super().__init__()
        
        self.model = load_model(train_config=config, method=config.method)
        if not config.disable_kd:
            self.teacher = load_model(train_config=config, method='none')
        else:
            self.teacher = None
        
        self.validation_preds = []
        self.validation_targets = []
        
        self.config = config

    def forward(self, inputs, target, output_hidden_states=False):
        return self.model(
            inputs, 
            target, 
            output_hidden_states=output_hidden_states
        )

    def training_step(self, batch, batch_idx):
        if self.teacher is not None:
            self.teacher.eval()
        self.model.train()
        
        inputs, target = batch
        
        if not self.config.disable_kd:
            with torch.no_grad(): #, torch.autocast('cuda', torch.bfloat16):
                output_teacher = self.teacher(inputs, output_hidden_states=not self.config.disable_kd)
        # with torch.autocast('cuda', torch.bfloat16):
        output = self(inputs, target, output_hidden_states=not self.config.disable_kd)
        logits = output.logits
        
        loss_model = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]).to(torch.float32),
            target.view(-1)
        )
        
        loss_kd_hidden = 0
        loss_kd_logits = 0
        if not self.config.disable_kd:
            for teacher_layer, student_layer in zip(output_teacher.hidden_states, output.hidden_states):
                loss_kd_hidden += torch.nn.functional.mse_loss(student_layer.to(torch.float32), teacher_layer.to(torch.float32))
            loss_kd_hidden = loss_kd_hidden / len(output_teacher.hidden_states)
            
            loss_kd_logits = torch.nn.functional.kl_div(
                output.logits.view(-1, logits.shape[-1]).to(torch.float32).log_softmax(-1),
                output_teacher.logits.view(-1, logits.shape[-1]).to(torch.float32).softmax(-1),
                reduction='batchmean',
            )
        
        loss = loss_model * 0.1 + (loss_kd_hidden + loss_kd_logits) * 2.5
        
        self.log("training/loss_model", loss_model.item())
        if not self.config.disable_kd:
            if loss_kd_hidden > 0:
                self.log("training/loss_kd_hidden", loss_kd_hidden.item())
            if loss_kd_logits > 0:
                self.log("training/loss_kd_logits", loss_kd_logits.item())
        self.log("training/loss", loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        
        inputs, target = batch
        with torch.no_grad(): #, torch.autocast('cuda', torch.bfloat16):
            # print('asdfasdf', inputs.shape, target.shape, flush=True)
            output = self(inputs, target).logits
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, output.shape[-1]), 
                target.view(-1)
            )
        self.log("val/loss", loss.item())
        
        self.validation_preds.append(output.cpu())
        self.validation_targets.append(target.cpu())
    
    def on_validation_epoch_end(self):
        from torchmetrics.text.perplexity import Perplexity
        with torch.no_grad():
            device = 'cpu'
            if self.config.using_fsdp:
                device = 'cuda'
            calculator = Perplexity(ignore_index=-100).to(device)
            for preds, target in zip(self.validation_preds, self.validation_targets):
                calculator.update(preds.to(device), target.to(device))
            ppl = calculator.compute()
        ppl = ppl.item()
        print('val/ppl', ppl)
        self.log("val/ppl", ppl)
        
        self.validation_preds.clear()
        self.validation_targets.clear()
        
    def configure_optimizers(self):
        params = []
        for name, p in self.model.named_parameters():
            # print(name, p.requires_grad, p.shape, p.dtype)
            if p.requires_grad:
                params.append(p)
        if self.config.using_fsdp:
            return DeepSpeedCPUAdam(params, lr=self.config.lr)
        # return DeepSpeedCPUAdam(params, lr=self.config.lr)
        return torch.optim.AdamW(params, lr=self.config.lr)

from lightning.pytorch.strategies import FSDPStrategy
from lightning.pytorch.strategies import DeepSpeedStrategy

def main(config: TrainConfig):
    os.makedirs('./saves/dev/wandb', exist_ok=True)
    os.makedirs('./saves/dev/checkpoint', exist_ok=True)
    
    if config.using_fsdp:
        devices = "1"
        policy = {LlamaDecoderLayer}
        # strategy = FSDPStrategy(
        #     auto_wrap_policy=policy,
        #     activation_checkpointing_policy=policy,
        #     cpu_offload=True,
        # )
        # strategy = 'deepspeed_stage_3'
        deepspeed_config = {
            "zero_allow_untested_optimizer": True,
            "zero_optimization": {
                "stage": 3,
                "offload_param": {"device": "cpu"},
                "offload_optimizer": {"device": "cpu"},
                "max_live_parameters": 5e8,
                "max_reuse_distance": 1e8,
                "contiguous_gradients": True,
                "overlap_comm": False, 
                "allgather_bucket_size": 1e7,
                "reduce_bucket_size": 1e7,
            },
        }
        strategy = DeepSpeedStrategy(config=deepspeed_config)
    else:
        devices = "1"
        strategy = "auto"
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="step",
        mode="max",
        dirpath=config.model_checkpoint_dir,
        filename=f"llama32k-{config.dataset}-{config.seq_len}-block{config.block_size}-k{config.k}-{{epoch:02d}}-{{step}}",
        every_n_train_steps=config.save_steps,
        enable_version_counter=False,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = '-'
    checkpoint_callback.FILE_EXTENSION = '.pth'
    
    trainer = pl.Trainer(
        log_every_n_steps=1,
        devices=devices,
        accelerator="gpu",
        strategy=strategy,
        precision=16,
        default_root_dir=config.model_checkpoint_dir,
        accumulate_grad_batches=config.accumulation_steps,
        max_epochs=20,
        max_steps=config.max_steps,
        logger=WandbLogger(
            save_dir="saves/dev/wandb", 
            project="tree-attention"
        ),
        enable_checkpointing=True,
        callbacks=[
            checkpoint_callback
        ],
    )
    
    datamodule = LabDataModule(config=config)
    model = LabModule(config=config)
    kwargs = dict(
        model=model,
        datamodule=datamodule, 
    )
    if config.load_from_checkpoint is not None:
        kwargs['ckpt_path'] = config.load_from_checkpoint
    trainer.fit(**kwargs)

if __name__ == "__main__":
    seed()
    
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--using_fsdp', action='store_true')
    parser.add_argument('--disable_kd', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=-1, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--lora_r', default=-1, type=int)
    parser.add_argument('--lr', default=-1, type=float)
    parser.add_argument('--max_steps', default=-1, type=int)
    parser.add_argument('--dataset', default='wikitext103', type=str)
    parser.add_argument('--seq_len', default=-1, type=int)
    parser.add_argument('--save_steps', default=-1, type=int)
    parser.add_argument('--init_checkpoint', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--k', default=512, type=int)
    parser.add_argument('--block_size', default=8, type=int)
    parser.add_argument('--method', default='tree', type=str)
    
    args = parser.parse_args()
    
    train_config = TrainConfig(
        using_fsdp=args.using_fsdp,
        disable_kd=args.disable_kd,
        dataset=args.dataset,
        load_from_checkpoint=args.checkpoint,
        k=args.k,
        block_size=args.block_size,
        method=args.method,
    )
    if args.gradient_accumulation_steps > 0:
        train_config.accumulation_steps = args.gradient_accumulation_steps
    if args.lora_r > 0:
        train_config.lora_r = args.lora_r
    if args.lr > 0:
        train_config.lr = args.lr
    if args.batch_size > 0:
        train_config.batch_size = args.batch_size
    if args.max_steps > 0:
        train_config.max_steps = args.max_steps
    if args.seq_len > 0:
        train_config.seq_len = args.seq_len
    if args.save_steps > 0:
        train_config.save_steps = args.save_steps
    
    main(train_config)