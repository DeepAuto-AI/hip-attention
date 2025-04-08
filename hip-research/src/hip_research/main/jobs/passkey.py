import os

import torch
from hip_research.dataset.passkey import Passkey
from hip_research.models.sglang_model import SglangModel
from tqdm import tqdm
from vllm import LLM, SamplingParams


def get_numbers(s, cnt):
    lst = [c for c in s if c.isdigit()]
    # print(lst, s)
    if len(lst) < cnt:
        lst += ["_"] * (cnt - len(lst))
    return "".join(lst)


def job_passkey(args, model, tokenizer, device):
    dataset = Passkey(tokenizer, batch_size=args.batch_size)

    accuracy = {}

    for j, (input_ids, target_ids) in enumerate(
        tqdm(dataset, dynamic_ncols=True, leave=False)
    ):
        if isinstance(model, LLM):
            input_ids = input_ids.cuda()
            target_ids = target_ids.cuda()

            prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            sampling_params = SamplingParams(
                n=1,
                temperature=0.0,
                top_k=1,
                max_tokens=20,
            )

            outputs = model.generate(prompts, sampling_params, use_tqdm=False)
            output = []
            for item in outputs:
                output.append(item.outputs[0].text)
        elif isinstance(model, SglangModel):
            input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
            IS_CHAT = os.getenv("IS_CHAT", "0") == "1"
            if IS_CHAT:
                output = [
                    model.generate(
                        input_text=input_text[0],
                        max_tokens=1024,
                        need_chat_prompt=True,
                        system_message=None,
                        handle_deepseek=True,
                        verbose=False,
                    )
                ]
                print(output)
            else:
                output = [model.generate(input_text=input_text, max_tokens=20)]
        else:
            input_ids = input_ids.cuda()
            target_ids = target_ids.cuda()

            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                output = model.generate(
                    input_ids,
                    max_new_tokens=20,
                    min_new_tokens=5,
                    do_sample=False,
                    num_beams=1,
                    attention_mask=None,
                    # pad_token_id=tokenizer.eos_token_id,
                )
                for m in model.modules():
                    if hasattr(m, "_clean_cache"):
                        m._clean_cache()
                output = output[:, input_ids.shape[1] :]
                tqdm.write(f"{tokenizer.batch_decode(output)}")

        # tqdm(tokenizer.batch_decode(output))
        truth = tokenizer.batch_decode(target_ids)
        est = [
            get_numbers(s.strip(), 5)[:5]
            for s in (
                output if isinstance(output[0], str) else tokenizer.batch_decode(output)
            )
        ]

        t = tokenizer.batch_decode(input_ids)[0]  # type: str
        e = truth[0]  # type: str
        idx = t.find(e)

        location = idx / len(t)
        location = int(location / 0.2) / 5

        seq_len = input_ids.shape[1]
        accuracy_key = (seq_len, location)
        acc_sum, acc_count = accuracy.get(accuracy_key, (0, 0))
        for x, y in zip(truth, est):
            for cx, cy in zip(x, y):
                if cx == cy:
                    acc_sum += 1
                acc_count += 1
        accuracy[accuracy_key] = (acc_sum, acc_count)

        accuracy_key = (seq_len,)
        acc_sum, acc_count = accuracy.get(accuracy_key, (0, 0))
        for x, y in zip(truth, est):
            for cx, cy in zip(x, y):
                if cx == cy:
                    acc_sum += 1
                acc_count += 1
        accuracy[accuracy_key] = (acc_sum, acc_count)

        tqdm.write(
            f"current accuracy { {k: f'{v[0] / (v[1] + 1e-20)*100:.2f}' for k, v in accuracy.items()} } | {truth[0]}, {est[0]}"
        )
