import os
from typing import TypedDict

import matplotlib.pyplot as plt
import torch
import tqdm
import transformers
from hip_research.main.jobs.ppl import job_ppl

from hip_attn.models.modeling_llama import LlamaConfig, LlamaForCausalLM


def log(*args):
    tqdm.tqdm.write(" ".join(map(lambda x: str(x), args)))


class HookArgs(TypedDict):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_derope: torch.Tensor
    k_derope: torch.Tensor
    out: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    layer_idx: int


class Hook:
    def __init__(self, model: torch.nn.Module):
        self.num_layers = model.config.num_hidden_layers
        self.device = model.device
        self.sink_token_size = 64
        self.window = 16384
        seq_len = model.config.max_position_embeddings
        self.prob_sum = torch.zeros(
            (self.num_layers, model.config.num_attention_heads, seq_len),
            dtype=torch.float32,
            device=self.device,
        )
        self.prob_count = torch.zeros_like(self.prob_sum, dtype=torch.int64)

    def __call__(self, args: HookArgs):
        q = args["q"]
        k = args["k"]

        # if args['layer_idx'] != 0:
        #     return

        N, HEAD, TDST, HID = q.shape
        N, HEAD_KV, TSRC, HID = k.shape
        window = self.window
        sink = self.sink_token_size

        prob_sum = torch.zeros((HEAD, TSRC), dtype=torch.float32, device=q.device)
        prob_count = torch.zeros_like(prob_sum, dtype=torch.int64)

        for i in tqdm.tqdm(range(window), leave=False):
            i_tdst = TDST - window + i
            query = q[0, :, i_tdst : i_tdst + 1, :]
            key = (
                k[0, :, : i_tdst + 1, :].permute(0, 2, 1).repeat(HEAD // HEAD_KV, 1, 1)
            )
            scores = query @ key
            scores[:, :, :sink].fill_(-32000.0)
            # scores[:, :, -sink:].fill_(-32000.0)
            probs = scores.softmax(dim=-1)[:, 0, :] ** 0.1
            probs = torch.nn.functional.max_pool1d(
                probs.unsqueeze(1), kernel_size=31, stride=1, padding=15
            ).squeeze(1)
            prob_sum[:, -probs.shape[-1] :] += probs
            prob_count[:, :] += 1
            # prob_count[:, -probs.shape[-1]:] += 1

        self.prob_sum[args["layer_idx"]][:, -prob_sum.shape[-1] :] += prob_sum
        self.prob_count[args["layer_idx"]][:, -prob_count.shape[-1] :] += prob_count

    def finalize(self):
        prob_avg = (self.prob_sum / self.prob_count) * (self.prob_count > 0)
        os.makedirs("./saves/sample_diag", exist_ok=True)
        torch.save(
            {
                "prob_avg": prob_avg,
            },
            "./saves/sample_diag/diag_info.pth",
        )


@torch.inference_mode
def job_sample_diag(
    args,
    model: LlamaForCausalLM,
    tokenizer: transformers.LlamaTokenizer,
    device,
    quite=os.getenv("HIP_QUITE", "0") == "1",
):
    assert args.method == "fa2"
    hook = Hook(model=model)
    model.register_hook_on("attention", hook)
    job_ppl(args, model, tokenizer, device)
    hook.finalize()
