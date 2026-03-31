import json
import logging
import os
import pathlib
import subprocess

import torch
from hip_research.dataset.booksum import BookSumDataset
from hip_research.utils.seed import seed
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList


class StopAfterStringIsGenerated(LogitsProcessor):
    def __init__(self, base_len: int, tokenizer):
        super().__init__()

        self.base_len = base_len
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.size(1) > self.base_len:
            decoded = self.tokenizer.batch_decode(input_ids[:, self.base_len :])
            ends_with_answer = torch.tensor(
                [s.endswith("</s>") for s in decoded], device=scores.device
            )
            forced_eos = torch.full(
                (scores.size(1),), -float("inf"), device=scores.device
            )
            forced_eos[self.tokenizer.eos_token_id] = 0

            # Force generation of EOS after a space
            scores[ends_with_answer] = forced_eos
        return scores


PROMPT_FIRST_ONLY = os.getenv("PROMPT_FIRST_ONLY", "1") == "1"


def generate_summary(args, model, tokenizer, device, idx, item, out_dir):
    PROMPT_ALWAYS_FLASH = os.environ.get("PROMPT_ALWAYS_FLASH", "0") == "1"
    LAST_DENSE = os.environ.get("LAST_DENSE", "0") == "1"

    inputs, completion = item

    if (out_dir / f"out_{idx}.txt").exists() and not args.overwrite:
        with open(out_dir / f"out_{idx}.txt", "r") as f:
            return f.read()

    tokenizer.truncation_side = "left"

    assert hasattr(model, "config")
    assert hasattr(model.config, "max_position_embeddings")

    if not args.disable_prompt:
        if "llama13b_32k" in args.model.lower():
            prompt = f"[INST] <<SYS>>\nSummarize the following text in about 300 words\n<</SYS>>[INST] {tokenizer.decode(inputs, skip_special_tokens=True)} [/INST]"
        else:
            if PROMPT_FIRST_ONLY:
                prompt = f"Summarize the following text in about 300 words:\n\n{tokenizer.decode(inputs, skip_special_tokens=True)}"
            else:
                prompt = f"Summarize the following text in about 300 words:\n\n{tokenizer.decode(inputs, skip_special_tokens=True)} The summary of previously given text is following."
    else:
        prompt = tokenizer.decode(inputs, skip_special_tokens=True)
    if prompt.endswith("</s>"):
        prompt = prompt[:-4]

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=model.config.max_position_embeddings - args.max_tokens,
        truncation=True,
    )["input_ids"][0]

    seq_len = inputs.shape[-1]
    print(f"seq_len: {seq_len}")

    if PROMPT_ALWAYS_FLASH:
        for m in model.modules():
            if hasattr(m, "attention_method"):
                m.tree_dense_queries = seq_len - 1
    if LAST_DENSE:
        for m in model.modules():
            if hasattr(m, "attention_method"):
                m.tree_last_dense_queries = -1

    additional_args = {}
    if not args.no_sample:
        additional_args = dict(
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )

    output = model.generate(
        inputs=inputs.unsqueeze(0).cuda(),
        attention_mask=torch.ones(
            (1, inputs.shape[-1]), dtype=torch.long, device="cuda"
        ),
        max_new_tokens=args.max_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        logits_processor=LogitsProcessorList(
            [StopAfterStringIsGenerated(inputs.shape[-1], tokenizer)]
        ),
        **additional_args,
    )
    output: str = tokenizer.decode(
        output[0][seq_len:].data.cpu(),
        skip_special_tokens=True,
    )
    if output.endswith("</s>"):
        output = output[:-4]
    output = output.strip()

    return output


def install_rogue():
    logger = logging.getLogger()

    ROUGE_HOME = os.environ.get("ROUGE_HOME", "cache/ROUGE-1.5.5")
    if "ROUGE_HOME" not in os.environ:
        logger.info("ROUGE_HOME not set, using default location %s", ROUGE_HOME)

    if not os.path.exists(ROUGE_HOME):
        logger.info("ROUGE_HOME=%s not a directory.", ROUGE_HOME)
        try:
            logger.info(
                "Installing rouge Perl script to {ROUGE_HOME} - this will take a few seconds"
            )
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "https://github.com/Yale-LILY/SummEval/tarball/7e4330d",
                    "-o",
                    "project.tar.gz",
                    "-s",
                ]
            )
            subprocess.run(["tar", "-xzf", "project.tar.gz"])
            subprocess.run(
                [
                    "mv",
                    "Yale-LILY-SummEval-7e4330d/evaluation/summ_eval/ROUGE-1.5.5/",
                    ROUGE_HOME,
                ]
            )
            subprocess.run(["rm", "project.tar.gz"])
            subprocess.run(["rm", "-rf", "Yale-LILY-SummEval-7e4330d/"])
        except subprocess.CalledProcessError as err:
            logger.error(
                "Failed to install the rouge Perl script; please install manually and set the ROUGE_HOME environment variable."
            )
            raise err

    return ROUGE_HOME


def generate_samples(args, model, tokenizer, device, out_dir):
    from vllm import LLM, SamplingParams

    is_vllm = isinstance(model, LLM)
    if is_vllm:
        # we do not access to tokenizer.
        tokenizer = None

    dataset = BookSumDataset(
        tokenizer=tokenizer,
        for_eval=True,
        need_tokenization=not is_vllm,
    )
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=0.05)
    test_dataset = Subset(dataset, test_idx)

    outputs = []

    for idx, item in enumerate(
        tqdm(test_dataset, dynamic_ncols=True, leave=True, desc="booksum")
    ):
        inputs, completion = item

        if is_vllm:
            assert isinstance(inputs, str)

            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                max_tokens=args.max_tokens,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                ignore_eos=False,
                skip_special_tokens=True,
            )

            if "qwen" in args.model.lower():
                # Qwen 1.5
                prompt = (
                    f"<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
                    f"<|im_start|>user\nSummarize the following text in about 300 words:\n\n{inputs}\n<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            elif (
                "llama32k" in args.model.lower()
            ) and "instruct" not in args.model.lower():
                # llama2 7b
                if PROMPT_FIRST_ONLY:
                    prompt = (
                        f"Summarize the following text in about 300 words:\n\n{inputs}"
                    )
                else:
                    prompt = f"Summarize the following text in about 300 words:\n\n{inputs} The summary of previously given text is following."
            elif ("llama32k" in args.model.lower) and (
                "instruct" in args.model.lower()
            ):
                # llama2 7b chat
                prompt = f"[INST] <<SYS>>\nSummarize the following text in about 300 words\n<</SYS>>[INST] {inputs} [/INST]"
            elif "llama13b_32k" in args.model.lower():
                # llama2 13b chat
                prompt = f"[INST] <<SYS>>\nSummarize the following text in about 300 words\n<</SYS>>[INST] {inputs} [/INST]"
            else:
                raise Exception(args.model)
            vllm_outputs = model.generate(
                prompt,
                sampling_params,
                use_tqdm=False,
            )
            output = vllm_outputs[0].outputs[0].text
        else:
            output = generate_summary(
                args, model, tokenizer, device, idx, item, out_dir
            )

        output_summary = output.replace("\n", "\\n")[:200]
        tqdm.write(f"[{idx:<7}] Summary: {output_summary}[...]")
        with open(out_dir / f"out_{idx}.txt", "w") as f:
            f.write(output)
        outputs.append(output)

        with open(f"saves/llama_eval/booksum/reference/ref_{idx}.txt", "w") as f:
            if isinstance(completion, str):
                f.write(completion)
            else:
                f.write(tokenizer.decode(completion, skip_special_tokens=True))


MAX_NEW_TOKENS = 256


def evaluate_rouge(args, model, tokenizer, device, out_dir: pathlib.Path):
    for node in out_dir.glob("*"):
        if node.is_file():
            content = node.read_text()
            ids = tokenizer(content, truncation=True, max_length=256).input_ids
            content = tokenizer.decode(ids, skip_special_tokens=True)
            node.write_text(content)

    rouge_dir = install_rogue()

    from pyrouge import Rouge155

    r = Rouge155(rouge_dir=rouge_dir)
    r.system_dir = out_dir  # "system" is the one we want to measure
    r.model_dir = "saves/llama_eval/booksum/reference"  # "model" is the gold standard (i.e. human summaries)
    r.system_filename_pattern = r"out_(\d+)\.txt"
    r.model_filename_pattern = r"ref_#ID#\.txt"

    output = r.convert_and_evaluate()
    print("R: Recall, P: Precision, F: F1 score")
    print(output)
    output_dict = r.output_to_dict(output)
    with open(out_dir / "rouge_scores.json", "w") as f:
        json.dump(output_dict, f, indent=2)


@torch.no_grad()
def job_booksum(args, model, tokenizer, device):
    seed()

    out_dir = pathlib.Path(
        f"saves/llama_eval/booksum/{args.name}_{args.model}_{args.method}_bq{args.block_size_q}"
        f"_bk{args.block_size_k}_k{args.k}_gl{args.max_tokens}_ns{args.no_sample}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pathlib.Path("saves/llama_eval/booksum/reference").mkdir(
        parents=True, exist_ok=True
    )

    generate_samples(args, model, tokenizer, device, out_dir)
    evaluate_rouge(args, model, tokenizer, device, out_dir)
    print(args)
