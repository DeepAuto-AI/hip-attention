import math
import os
import random

import datasets
import torch
import torch.nn.functional as F
import transformers
from hip_research.dataset.wikitext2 import Wikitext2Dataset
from hip_research.models.modeling_llama_permute import LlamaForCausalLM as PermuteLlama
from torch.utils.data import Dataset
from tqdm import tqdm


class RedPajamaDataset(Dataset):
    def __init__(self, tokenizer, stride):
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset(
            "togethercomputer/RedPajama-Data-1T",
            "arxiv",
            split="train",
            trust_remote_code=True,
        )
        self.window_size = 5
        self.stride = stride

    def __len__(self):
        return len(self.dataset) // self.window_size

    def __getitem__(self, idx):
        text = []
        for i in range(self.window_size):
            entry = self.dataset[idx * self.window_size + i]
            text.append(entry["text"])
        random.shuffle(text)
        ids = self.tokenizer(
            "\n\n".join(text),
            return_tensors="pt",
            truncation=True,
            max_length=self.stride,
        ).input_ids
        labels = ids.clone()
        return ids[0], labels[0]


class Trainer:
    def __init__(
        self,
        model: str,
        stride: int,
        name: str,
        lr: float,
    ):
        self.model_id = {
            "llama3.1_8b": "meta-llama/Meta-Llama-3.1-8B",
            "llama3.1_4b": "nvidia/Llama-3.1-Minitron-4B-Width-Base",
            "llama2_1b": "princeton-nlp/Sheared-LLaMA-1.3B",
        }[model]

        self.name = name
        self.stride = stride
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        self.dataset = RedPajamaDataset(self.tokenizer, self.stride)
        self.eval_dataset = Wikitext2Dataset(
            subset="validation",
            tokenizer=self.tokenizer,
            stride=self.stride,
            max_length=self.stride,
        )
        self.device = torch.device("cuda")
        self.teacher = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.student = PermuteLlama(self.teacher.config).to(self.device)
        for m in self.student.modules():
            if hasattr(m, "gradient_checkpointing"):
                from torch.utils.checkpoint import checkpoint

                m._gradient_checkpointing_func = checkpoint
                m.gradient_checkpointing = True
        load_result = self.student.load_state_dict(
            self.teacher.state_dict(), strict=False
        )
        print(load_result)
        for p in self.student.parameters():
            p.requires_grad = False
        for m in self.student.modules():
            if hasattr(m, "mark_trainable"):
                m.mark_trainable()

        self.optimiezr = torch.optim.AdamW(
            params=[p for p in self.student.parameters() if p.requires_grad],
            lr=lr,
        )
        self.scaler = torch.GradScaler()

    def evaluate(self):
        return

        self.student.eval()

        losses = []
        for istep, (input_ids, labels) in enumerate(tqdm(self.eval_dataset)):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
                labels = labels.unsqueeze(0)

            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                loss = self.student(input_ids, labels=labels, use_cache=False).item()
            losses.append(loss)

        avg_loss = sum(losses) / (len(losses) + 1e-20)
        ppl = math.exp(avg_loss)

        print(f"eval: ppl = {ppl}, loss = {avg_loss}")
        return ppl

    def train_step(self, input_ids, labels):
        self.teacher.eval()
        self.student.train()

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            labels = labels.unsqueeze(0)

        kd_hidden = False

        # print('start', input_ids.shape, labels.shape)
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            output_teacher = self.teacher(
                input_ids,
                labels=labels,
                output_hidden_states=kd_hidden,
                use_cache=False,
            )
        # print('teacher done')
        with torch.autocast("cuda", torch.float16):
            output_student = self.student(
                input_ids,
                labels=labels,
                output_hidden_states=kd_hidden,
                use_cache=False,
            )
        # print('student done', output_student.loss)

        loss_model = output_student.loss
        loss_kd_hidden = 0
        if kd_hidden:
            for teacher_layer, student_layer in zip(
                output_teacher.hidden_states, output_student.hidden_states
            ):
                loss_kd_hidden_layer = F.mse_loss(teacher_layer, student_layer)
                loss_kd_hidden += loss_kd_hidden_layer
            loss_kd_hidden /= len(output_teacher.hidden_states)

        loss_kd_logit = F.kl_div(
            torch.log_softmax(
                output_student.logits.view(-1, output_student.logits.shape[-1]), dim=-1
            ),
            output_teacher.logits.view(-1, output_student.logits.shape[-1]).softmax(
                dim=-1
            ),
            reduce="batchmean",
        )

        loss = loss_kd_logit * 1.0 + loss_kd_hidden * 1.0 + loss_model * 0.1

        grad_acc = 8

        self.scaler.scale(loss / grad_acc).backward()

        if ((self.current_step + 1) % grad_acc) == 0:
            self.scaler.step(self.optimiezr)
            self.scaler.update()
            self.optimiezr.zero_grad()

            tqdm.write(
                f"loss: {loss.item():.6f} {loss_kd_logit.item():.6f} {loss_kd_hidden:.6f} {loss_model.item():.6f}"
            )

    def save(self):
        os.makedirs(f"./saves/checkpoint/{self.name}", exist_ok=True)
        path = f"./saves/checkpoint/{self.name}/checkpoint.pth"
        torch.save(
            {
                "current_step": self.current_step,
                "current_epoch": self.current_epoch,
                "optimizer": self.optimiezr.state_dict(),
                "student": self.student.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            path,
        )
        print("saved", path)

    def train_epoch(self):
        for istep, (input_ids, labels) in enumerate(tqdm(self.dataset)):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            self.current_step = istep
            self.train_step(input_ids, labels=labels)
            if ((istep + 1) % 1000) == 0:
                self.save()
                self.evaluate()

    def main(self):
        for iepoch in range(10):
            self.current_epoch = iepoch
            self.train_epoch()
            self.save()
            self.evaluate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--model", type=str, default="llama2_1b")
    parser.add_argument("--name", type=str, default="dev")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    print(args)

    tr = Trainer(
        model=args.model,
        stride=args.stride,
        name=args.name,
        lr=args.lr,
    )
    tr.main()
