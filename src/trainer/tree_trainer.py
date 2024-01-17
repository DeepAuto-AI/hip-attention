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

from src.models.modeling_llama import LlamaForCausalLM, LlamaConfig

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.labdataset import LabDataset
from torch.utils.data import DataLoader, random_split

@dataclass
class TrainConfig:
    lr: float = 1e-4
    batch_size: int = 256
    model_checkpoint_dir: str = "./saves/dev"

class LabDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: int = 2,
        data_dir: Path ="data",
        block_size: int = 35,
        download: bool = True,
        train_size: float = 0.8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.block_size = block_size
        self.download = download
        self.num_workers = num_workers
        self.train_size = train_size
        self.dataset = None
        self.tokenizer = load_tokenizer()
    
    def prepare_data(self):
        self.dataset = LabDataset(
            data_dir=self.data_dir,
            block_size=self.block_size,
            download=self.download,
            tokenizer=self.tokenizer
        )
    
    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            train_size = int(len(self.dataset) * self.train_size)
            test_size = len(self.dataset) - train_size
            self.train_data, self.val_data = random_split(self.dataset, lengths=[train_size, test_size])
        if stage == "test" or stage is None:
            self.test_data = self.val_data

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers)

from peft import LoraConfig, TaskType
from peft import get_peft_model

def load_model(method = 'tree', device = 'cuda:0'):
    model_id = 'togethercomputer/LLaMA-2-7B-32K'
    config = LlamaConfig.from_pretrained(model_id)
    config._attn_implementation = config.attn_implementation = 'sdpa'
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=32,
        lora_alpha=32, 
        lora_dropout=0.1
    )
    
    model = LlamaForCausalLM.from_pretrained(
        model_id,
        config=config, 
        load_in_4bit=True,
        device_map={"" : device},
        quantization_config=transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        ),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    for m in model.modules():
        if hasattr(m, 'attention_method'):
            m.attention_method = method
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def load_tokenizer():
    model_id = 'togethercomputer/LLaMA-2-7B-32K'
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return tokenizer

class LabModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = load_model()

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        self.log("training-loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        self.log("val-loss", loss)
        
    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.0001)

def main(config: TrainConfig):
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy="auto",
        precision="32-true",
        enable_checkpointing=True,
        # callbacks=EarlyStopping(monitor="val-loss", mode="min"),
        logger=WandbLogger(name="textlab-demo", save_dir="saves/dev/wandb"),
        # profiler=PyTorchProfiler(dirpath="cache/torch_profiler"),
    )
    
    # instantiate the datamodule
    datamodule = LabDataModule()
    # instantiate the model
    model = LabModule() 
    # call .fit
    trainer.fit(model=model, datamodule=datamodule) 

if __name__ == "__main__":
    train_config = TrainConfig()
    main(train_config)