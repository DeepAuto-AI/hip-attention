import os
from pathlib import Path
from typing import Any, Dict, Union

import datasets
import torch
import torch.autograd
import torch.onnx
import torch.utils.checkpoint
from hip_research.dataset.alpaca import AlpacaDataset
from hip_research.dataset.booksum import BookSumDataset
from hip_research.dataset.labdataset import LabDataset
from hip_research.dataset.lmsys import LmsysChatDataset
from hip_research.dataset.openwebtext import OpenWebTextDataset
from hip_research.trainer.common import (
    TrainConfig,
    load_model,
    load_tokenizer,
    parse_args,
)
from hip_research.utils.seed import seed
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("high")


class LabDataModule:
    def __init__(
        self,
        config: TrainConfig,
        num_workers: int = 0,
        data_dir: Path = "data",
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
        self.tokenizer = load_tokenizer(config.model)
        self.bsize = config.batch_size

    def get_dataset(self):
        if self.config.dataset in ["wikitext2", "wikitext103"]:
            dataset = LabDataset(
                data_dir=self.data_dir,
                block_size=self.block_size,
                download=self.download,
                tokenizer=self.tokenizer,
                dataset=self.config.dataset,
            )
        elif self.config.dataset in ["alpaca"]:
            dataset = AlpacaDataset(
                tokenizer=self.tokenizer,
            )
        elif self.config.dataset in ["booksum"]:
            dataset = BookSumDataset(
                tokenizer=self.tokenizer,
            )
        elif self.config.dataset in ["openwebtext"]:
            dataset = OpenWebTextDataset(
                tokenizer=self.tokenizer,
                stride=self.block_size,
            )
        elif self.config.dataset == "lmsys":
            dataset = LmsysChatDataset(
                tokenizer=self.tokenizer,
            )
        else:
            raise Exception()
        return dataset

    def prepare_data(self):
        self.get_dataset()

    def setup(self, stage: str):
        self.dataset = self.get_dataset()
        if self.config.dataset in ["wikitext2", "wikitext103"]:
            if stage == "fit" or stage is None:
                test_size = min(100, len(self.dataset) * (1 - self.train_size))
                train_size = int(len(self.dataset) - test_size)
                self.train_data, self.val_data = random_split(
                    self.dataset, lengths=[train_size, test_size]
                )
            if stage == "test" or stage is None:
                self.test_data = self.val_data
        elif self.config.dataset in ["booksum", "alpaca", "openwebtext", "lmsys"]:
            if stage == "fit" or stage is None:

                def train_val_dataset(dataset, val_split=0.05):
                    train_idx, val_idx = train_test_split(
                        list(range(len(dataset))), test_size=val_split
                    )
                    train = Subset(dataset, train_idx)
                    valid = Subset(dataset, val_idx)
                    return train, valid

                self.train_data, self.val_data = train_val_dataset(self.dataset)
            if stage == "test" or stage is None:
                self.test_data = self.val_data
        else:
            raise Exception()

    def train_dataloader(self):
        return DataLoader(
            self.train_data, num_workers=self.num_workers, batch_size=self.bsize
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, num_workers=self.num_workers, batch_size=self.bsize
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, num_workers=self.num_workers, batch_size=self.bsize
        )


def get_hf_dataset(ds):
    def gen():
        for idx in range(len(ds)):
            inputs, targets = ds[idx]
            yield {"input_ids": inputs, "labels": targets}

    return datasets.IterableDataset.from_generator(gen)


class Trainer(Seq2SeqTrainer):
    def __init__(
        self,
        config=None,
        model=None,
        teacher=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        self.model = model
        self.teacher = teacher

        self.validation_preds = []
        self.validation_targets = []

        self.config = config
        self.pad_token_id = tokenizer.pad_token_id

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        result = super().training_step(model, inputs)
        return result

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.teacher is not None:
            self.teacher.eval()
        self.model.train()

        inputs, target = inputs["input_ids"], inputs["labels"]
        inputs = inputs[..., : self.config.seq_len]
        target = target[..., : self.config.seq_len]

        if not self.config.disable_kd:
            with torch.no_grad():  # , torch.autocast('cuda', torch.bfloat16):
                output_teacher = self.teacher(
                    inputs, output_hidden_states=not self.config.disable_kd
                )
        # with torch.autocast('cuda', torch.bfloat16):
        output = self.model(
            inputs,
            attention_mask=(inputs.ne(self.pad_token_id)).to(inputs.dtype),
            # labels=target,
            output_hidden_states=not self.config.disable_kd,
            output_attn_sparsity_loss=self.config.sparsity_reg != 0,
        )
        logits = output.logits

        loss_model = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).to(torch.float32), target.reshape(-1)
        )

        loss_kd_hidden = 0
        loss_kd_logits = 0
        if not self.config.disable_kd:
            for teacher_layer, student_layer in zip(
                output_teacher.hidden_states, output.hidden_states
            ):
                loss_kd_hidden += torch.nn.functional.mse_loss(
                    student_layer.to(torch.float32), teacher_layer.to(torch.float32)
                )
            loss_kd_hidden = loss_kd_hidden / len(output_teacher.hidden_states)

            loss_kd_logits = torch.nn.functional.kl_div(
                output.logits.view(-1, logits.shape[-1])
                .to(torch.float32)
                .log_softmax(-1),
                output_teacher.logits.view(-1, logits.shape[-1])
                .to(torch.float32)
                .softmax(-1),
                reduction="batchmean",
            )

        if not self.config.disable_kd:
            loss = loss_model * 0.1 + (loss_kd_hidden + loss_kd_logits) * 2.5
        else:
            loss = loss_model

        sparsity_loss = None
        if self.config.sparsity_reg != 0:
            sparsity_loss = sum(
                layer_sparsity.mean()
                for layer_sparsity in output.attn_sparsity_loss
                if layer_sparsity is not None
            ) / len(output.attn_sparsity_loss)
            loss = loss + self.config.sparsity_reg * sparsity_loss

        log_dict = dict()
        log_dict["training/loss_model"] = loss_model.item()
        if not self.config.disable_kd:
            if loss_kd_hidden > 0:
                log_dict["training/loss_kd_hidden"] = loss_kd_hidden.item()
            if loss_kd_logits > 0:
                log_dict["training/loss_kd_logits"] = loss_kd_logits.item()
        if sparsity_loss is not None:
            log_dict["training/sparsity_loss"] = sparsity_loss.item()
        log_dict["training/loss"] = loss.item()
        self.log(log_dict)

        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()

        inputs, target = batch
        inputs = inputs[..., : self.config.seq_len]
        target = target[..., : self.config.seq_len]

        with torch.no_grad():  # , torch.autocast('cuda', torch.bfloat16):
            # print('asdfasdf', inputs.shape, target.shape, flush=True)
            output = self.model(
                inputs,
                attention_mask=(inputs != self.pad_token_id).to(inputs.dtype),
                labels=target,
            ).logits
            loss = torch.nn.functional.cross_entropy(
                output.view(-1, output.shape[-1]), target.view(-1)
            )
        self.log({"val/loss": loss.item()})

        self.validation_preds.append(output.cpu())
        self.validation_targets.append(target.cpu())

    def on_validation_epoch_end(self):
        from torchmetrics.text.perplexity import Perplexity

        with torch.no_grad():
            device = "cpu"
            if self.config.using_fsdp:
                device = "cuda"
            calculator = Perplexity(ignore_index=-100).to(device)
            for preds, target in zip(self.validation_preds, self.validation_targets):
                calculator.update(preds.to(device), target.to(device))
            ppl = calculator.compute()
        ppl = ppl.item()
        print("val/ppl", ppl)
        self.log({"val/ppl": ppl})

        self.validation_preds.clear()
        self.validation_targets.clear()


def main(config: TrainConfig):
    os.environ["WANDB_PROJECT"] = "hip-attention"

    os.makedirs("./saves/dev/wandb", exist_ok=True)
    os.makedirs("./saves/dev/checkpoint", exist_ok=True)

    if config.method == "hip":
        filename = f"{config.model}-{config.dataset}-{config.name}-{config.seq_len}-bq{config.block_size_q}-bk{config.block_size_k}-k{config.k}-sp{config.sparsity_reg}"
    elif config.method == "none":
        filename = f"{config.model}-{config.dataset}-{config.name}-{config.seq_len}"
    elif config.method == "reformer":
        filename = f"{config.model}-{config.method}-{config.name}-{config.dataset}-{config.seq_len}-k{config.k}"
    elif config.method == "performer":
        filename = f"{config.model}-{config.method}-{config.name}-{config.dataset}-{config.seq_len}"
    else:
        raise Exception()

    config.model_checkpoint_dir = config.model_checkpoint_dir + "/" + filename

    model = load_model(train_config=config, method=config.method)
    if not config.disable_kd:
        teacher = load_model(train_config=config, method="none", is_teacher=True)
    else:
        teacher = None

    datamodule = LabDataModule(config=config)
    datamodule.setup("fit")

    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e8,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e8,
            "stage3_max_reuse_distance": 1e8,
            "stage3_gather_16bit_weights_on_model_save": True,
            "zero_hpz_partition_size": torch.cuda.device_count(),
        },
        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": True,
            "contiguous_memory_optimization": False,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        },
        "train_micro_batch_size_per_gpu": config.batch_size,
        "gradient_accumulation_steps": config.accumulation_steps,
        "gradient_clipping": 1.0,
    }

    trainer_config = Seq2SeqTrainingArguments(
        logging_steps=1,
        bf16=True,
        output_dir=config.model_checkpoint_dir,
        gradient_accumulation_steps=config.accumulation_steps,
        max_steps=config.max_steps,
        report_to=["wandb"],
        gradient_checkpointing=True,
        save_total_limit=3,
        save_steps=config.save_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.lr,
        ignore_data_skip=True,
        warmup_steps=config.warmup_steps,
        local_rank=config.local_rank,
        deepspeed=ds_config if config.using_deepspeed else None,
    )

    trainer = Trainer(
        config=config,
        model=model,
        teacher=teacher,
        args=trainer_config,
        train_dataset=get_hf_dataset(datamodule.train_data),
        eval_dataset=get_hf_dataset(datamodule.val_data),
        tokenizer=datamodule.tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=datamodule.tokenizer,
            padding="longest",
            pad_to_multiple_of=config.seq_len,
        ),
    )

    trainer.train(resume_from_checkpoint=config.load_from_checkpoint)


def run():
    seed()
    main(parse_args())


if __name__ == "__main__":
    run()
