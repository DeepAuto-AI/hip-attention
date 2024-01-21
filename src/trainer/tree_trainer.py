import gc
import os
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path

# import mlflow
import torch
import torch.onnx
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler

import transformers

from src.models.modeling_llama import LlamaForCausalLM, LlamaConfig, LlamaDecoderLayer

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.labdataset import LabDataset
from torch.utils.data import DataLoader, random_split

torch.set_float32_matmul_precision('high')

@dataclass
class TrainConfig:
    disable_kd: bool = False
    using_fsdp: bool = False
    lr: float = 5e-5
    batch_size: int = 1
    accumulation_steps: int = 16
    lora_r: int = 32
    seq_len: int = 4096
    model_checkpoint_dir: str = "./saves/dev/checkpoint"

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
        self.dataset = LabDataset(
            data_dir=self.data_dir,
            block_size=self.block_size,
            download=self.download,
            tokenizer=self.tokenizer
        )
    
    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            test_size = min(100, len(self.dataset) * (1 - self.train_size))
            train_size = int(len(self.dataset) - test_size)
            self.train_data, self.val_data = random_split(self.dataset, lengths=[train_size, test_size])
        if stage == "test" or stage is None:
            self.test_data = self.val_data

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
        load_in_4bit=True,
        device_map={"" : device},
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    for m in model.modules():
        if hasattr(m, 'attention_method'):
            m.attention_method = method
    
    if method != 'none':
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_config.lora_r,
            lora_alpha=32, 
            lora_dropout=0.1,
            modules_to_save=['tree_avgpool_scaler']
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model

def load_tokenizer():
    model_id = 'togethercomputer/LLaMA-2-7B-32K'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return tokenizer

class LabModule(pl.LightningModule):
    def __init__(self, config: TrainConfig):
        super().__init__()
        
        self.model = load_model(train_config=config)
        self.teacher = load_model(train_config=config, method='none')
        
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
        self.teacher.eval()
        self.model.train()
        
        inputs, target = batch
        
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
            output_teacher = self.teacher(inputs, output_hidden_states=False)
        with torch.autocast('cuda', torch.bfloat16):
            output = self(inputs, target, output_hidden_states=False)
        logits = output.logits
        
        loss_model = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]).to(torch.float32),
            target.view(-1)
        )
        
        loss_kd_hidden = 0
        # for teacher_layer, student_layer in zip(output_teacher.hidden_states, output.hidden_states):
        #     loss_kd_hidden += torch.nn.functional.mse_loss(student_layer.to(torch.float32), teacher_layer.to(torch.float32))
        # loss_kd_hidden = loss_kd_hidden / len(output_teacher.hidden_states)
        
        loss_kd_logits = 0
        if not self.config.disable_kd:
            loss_kd_logits = torch.nn.functional.kl_div(
                output.logits.view(-1, logits.shape[-1]).to(torch.float32).log_softmax(-1),
                output_teacher.logits.view(-1, logits.shape[-1]).to(torch.float32).softmax(-1),
                reduction='batchmean',
            )
        
        loss = loss_model + (loss_kd_hidden + loss_kd_logits) * 0.1
        
        self.log("training/loss_model", loss_model.item())
        if not self.config.disable_kd:
            # self.log("training/loss_kd_hidden", loss_kd_hidden.item())
            self.log("training/loss_kd_logits", loss_kd_logits.item())
        self.log("training/loss", loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        
        inputs, target = batch
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
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
        calculator = Perplexity(ignore_index=-100)
        for preds, target in zip(self.validation_preds, self.validation_targets):
            calculator.update(preds, target)
        ppl = calculator.compute()
        ppl = ppl.item()
        print('val/ppl', ppl)
        self.log("val/ppl", ppl)
        
        self.validation_preds.clear()
        self.validation_targets.clear()
        
    def configure_optimizers(self):
        params = []
        for name, p in self.model.named_parameters():
            print(name, p.requires_grad, p.shape, p.dtype)
            if p.requires_grad:
                params.append(p)
        return torch.optim.AdamW(params, lr=self.config.lr)

from lightning.pytorch.strategies import FSDPStrategy

def main(config: TrainConfig):
    os.makedirs('./saves/dev/wandb', exist_ok=True)
    os.makedirs('./saves/dev/checkpoint', exist_ok=True)
    
    if config.using_fsdp:
        policy = {LlamaDecoderLayer}
        devices = "auto"
        strategy = FSDPStrategy(
            auto_wrap_policy=policy,
            activation_checkpointing_policy=policy,
            cpu_offload=True,
        )
    else:
        devices = "1"
        strategy = "auto"
        
        # policy = {LlamaDecoderLayer}
        # strategy = FSDPStrategy(
        #     auto_wrap_policy=policy,
        #     activation_checkpointing_policy=policy,
        #     # cpu_offload=True,
        # )
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="step",
        mode="max",
        dirpath=config.model_checkpoint_dir,
        filename=f"llama32k-wikitext2-{config.seq_len}-{{epoch:02d}}-{{step}}",
        every_n_train_steps=25,
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
    trainer.fit(model=model, datamodule=datamodule) 

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--using_fsdp', action='store_true')
    parser.add_argument('--disable_kd', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=-1, type=int)
    parser.add_argument('--lora_r', default=-1, type=int)
    args = parser.parse_args()
    
    train_config = TrainConfig(
        using_fsdp=args.using_fsdp,
        disable_kd=args.disable_kd,
    )
    if args.gradient_accumulation_steps > 0:
        train_config.accumulation_steps = args.gradient_accumulation_steps
    if args.lora_r > 0:
        train_config.lora_r = args.lora_r
    
    main(train_config)
