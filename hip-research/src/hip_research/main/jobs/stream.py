import os
import time
import traceback

import torch
from hip_research.models.sglang_model import SglangModel
from transformers import TextStreamer
from transformers.models.auto import AutoTokenizer

from hip_attn.utils.benchmarking import get_bench


class BatchedStreamer(TextStreamer):
    def __init__(
        self, tokenizer: AutoTokenizer, skip_prompt: bool = False, **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.idx = 0

    def put(self, value):
        if self.idx == 1:
            # print('prompt trace', get_bench().format_tracetree())
            get_bench().reset_trace()
            get_bench().reset_measures()
        self.idx += 1
        return super().put(value[:1])


def job_stream(args, model, tokenizer, device):
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils import config as vllm_transformers_config

    vllm_transformers_config.FORCE_SIGNLE_LAYER = int(
        os.environ.get("FORCE_SINGLE_LAYER", "0")
    )

    while True:
        get_bench().reset_trace()
        get_bench().reset_measures()
        # get_bench().disabled = False

        if args.input is None:
            input_text = input(">>>").strip()
        else:
            input_text = args.input

        if len(input_text.strip()) == 0:
            continue

        if os.path.exists(input_text):
            print("loading", input_text)
            with open(input_text, "r", encoding="utf8") as f:
                input_text = f.read()

        inputs = tokenizer(
            [
                input_text,
            ]
            * args.batch_size,
            return_tensors="pt",
        ).to(device)
        print("input_ids", len(input_text), inputs.input_ids.shape)
        # print(inputs)

        t = time.time()
        elapsed = 0
        try:
            if isinstance(model, SglangModel):
                output_texts = [
                    model.generate(input_text=input_text, max_tokens=args.max_tokens)
                ]
            elif isinstance(model, LLM):
                prompts = [
                    input_text,
                ]
                sampling_params = SamplingParams(
                    n=args.batch_size,
                    temperature=0.7,
                    top_p=0.9,
                    top_k=1000,
                    max_tokens=args.max_tokens,
                    # max_tokens=16,
                    frequency_penalty=0.0,
                    repetition_penalty=1.0,
                    ignore_eos=True,
                    skip_special_tokens=False,
                    # max_tokens=inputs.input_ids.shape[-1] + 32,
                )

                outputs = model.generate(prompts, sampling_params, use_tqdm=True)
                elapsed = time.time() - t

                n_generated = 0
                output_texts = []
                for output in outputs:
                    for item in output.outputs:
                        output_texts.append(item.text)
                for generated_text in output_texts:
                    n_tokens = len(tokenizer([generated_text]).input_ids[0])
                    n_generated += n_tokens
                    if len(output_texts) > 1:
                        print(
                            generated_text.replace("\n", "\\n")[:200] + " [...]",
                            n_tokens,
                        )
                    else:
                        print(generated_text, n_tokens)
                print(
                    f"{n_generated} token generated, {n_generated/elapsed:.2f} tok/sec"
                )
            else:
                without_cache = os.environ.get("STREAM_WITHOUT_CACHE", "0") == "1"
                if without_cache:
                    last_output_text = ""
                    with torch.no_grad():
                        import triton

                        input_ids_len = inputs["input_ids"].shape[-1]
                        target_index = input_ids_len - 1
                        pad_size = 128
                        padded_inputs = torch.zeros(
                            (input_ids_len + pad_size * 4,),
                            dtype=torch.long,
                            device=inputs["input_ids"].device,
                        )
                        padded_inputs[:input_ids_len] = inputs["input_ids"][0]
                        output = model(
                            use_cache=False,
                            output_logits=True,
                            num_logits_to_keep=-target_index,
                            input_ids=padded_inputs[
                                : target_index + pad_size
                            ].unsqueeze(0),
                        )
                        output_tokens = [output.logits[0, -1, :].topk(k=1).indices]
                        new_output_text = tokenizer.batch_decode(
                            torch.cat(output_tokens).cpu().unsqueeze(0),
                            skip_special_tokens=False,
                        )[0]
                        print(
                            new_output_text[len(last_output_text) :].replace(
                                "\n", "\\n\n"
                            ),
                            end="",
                            flush=True,
                        )
                        last_output_text = new_output_text
                        for i in range(256):
                            target_index += 1
                            padded_inputs[
                                input_ids_len : input_ids_len + len(output_tokens)
                            ] = torch.cat(output_tokens)
                            output = model(
                                input_ids=padded_inputs[
                                    : target_index + pad_size
                                ].unsqueeze(0),
                                use_cache=False,
                                output_logits=True,
                                num_logits_to_keep=-target_index,
                            )
                            output_tokens += [output.logits[0, -1, :].topk(k=1).indices]
                            new_output_text = tokenizer.batch_decode(
                                torch.cat(output_tokens).cpu().unsqueeze(0),
                                skip_special_tokens=False,
                            )[0]
                            print(
                                new_output_text[len(last_output_text) :].replace(
                                    "\n", "\\n\n"
                                ),
                                end="",
                                flush=True,
                            )
                            last_output_text = new_output_text
                            # print(output_tokens[-1], )
                else:
                    streamer = BatchedStreamer(
                        tokenizer, skip_prompt=False, skip_special_tokens=False
                    )

                    with torch.no_grad():
                        model.generate(
                            **inputs,
                            streamer=streamer,
                            do_sample=True,
                            max_new_tokens=256,
                            temperature=0.7,
                            top_p=0.9,
                            top_k=10,
                            repetition_penalty=1.0,
                            cache_implementation="static",
                        )
        except KeyboardInterrupt:
            traceback.print_exc()
            print("Interrupted")
        if elapsed == 0:
            elapsed = time.time() - t
        tracetree = get_bench().format_tracetree().strip()
        if len(tracetree) > 0:
            print(tracetree)
        print(f"elapsed {elapsed:.4f} sec")

        if args.input is not None:
            return
