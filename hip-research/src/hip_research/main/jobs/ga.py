import copy
import dataclasses
import gc
import json
import math
import os
import random
import threading
import traceback
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import transformers
import triton
import wandb
from hip_research.dataset.calib_loft_rag import rag_prefix, rag_qa_pairs
from hip_research.dataset.calib_loft_retrieval import (
    retrieval_pid_to_id,
    retrieval_prefix,
    retrieval_qa_pairs,
)

from hip_attn import HiPAttentionArgs11
from hip_attn.v1_1.attention2_draft_sampling_extend import (
    ScanStage,
    dual_stage_quadratic_hip_attention,
)


def load_loft_rag_chat_corpus() -> List[Tuple[str, str]]:
    lines = []
    for qa in rag_qa_pairs:
        for answer in qa["answers"][:1]:
            lines.append((f'{rag_prefix}{qa["query_text"]}', f"{answer}"))
    return lines


def load_loft_retrieval_chat_corpus() -> List[Tuple[str, str]]:
    lines = []
    for qa in retrieval_qa_pairs:
        for answer in qa["answers"][:1]:
            pid = answer[0]
            doi = retrieval_pid_to_id[pid]
            lines.append((f'{retrieval_prefix}{qa["query_text"]}', f"{doi}"))
    return lines


from datasets import Features, Sequence, Value, load_dataset


def load_infinite_bench_subset(split: str, tokenizer, seq_len: int, count: int):
    ft = Features(
        {
            "id": Value("int64"),
            "context": Value("string"),
            "input": Value("string"),
            "answer": Sequence(Value("string")),
            "options": Sequence(Value("string")),
        }
    )
    dataset = load_dataset("xinrongzhang2022/InfiniteBench", features=ft)

    lines = []
    for idx in range(min(count, len(dataset[split]))):
        entry = dataset[split][random.randint(0, len(dataset[split]) - 1)]
        context = f"""You are a helpful assistant.

Please read given text careful and answer user\'s query after it.

------------------------------------------------------
- From here, document is started.
------------------------------------------------------

{entry["context"]}

------------------------------------------------------
- The document is ended. Now, please answer user's query.
------------------------------------------------------

"""
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        if len(context_ids) > seq_len:
            context_ids = context_ids[: seq_len // 2] + context_ids[-(seq_len // 2) :]
        context = tokenizer.decode(context_ids)
        context = (
            context
            + f"""

Before you start, here is rules that you have to follow.
- Please be concise.
- Only answer that I asked. Do not put any words except the answer. For example, if I ask some values or passkey, just answer only passkey and values.
{entry["input"]}
Now, answer my question: {entry["input"]}"""
        )
        lines.append((context, entry["answer"][0]))

    return lines


def format_chat_corpus(args, corpus: List[str]) -> List[str]:
    if "llama3" in args.model:
        ret = []
        for prompt, output in corpus:
            updated = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024

<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            ret.append((updated, "\n\n" + output + "<|eot_id|>"))
        return ret
    else:
        raise Exception()


def tokenizing_and_find_shared_prompt_on_corpus(
    tokenizer: transformers.LlamaTokenizer,
    corpus: List[str],
    shared_prompt_max_len: int = 999999999,
):
    count = len(corpus)
    iterator = map(
        lambda line: (
            tokenizer.encode(line[0], add_special_tokens=False),
            tokenizer.encode(line[1], add_special_tokens=False),
        ),
        corpus,
    )
    corpus = []
    shared_prompt = None
    for input_ids, output_ids in tqdm.tqdm(iterator, total=count, dynamic_ncols=True):
        if shared_prompt is None:
            shared_prompt = input_ids
        else:
            for i, (s, c) in enumerate(zip(shared_prompt, input_ids)):
                if s != c:
                    shared_prompt = shared_prompt[:i]
                    break
        corpus.append([input_ids, output_ids])
    shared_prompt = shared_prompt[:shared_prompt_max_len]
    corpus = list(
        map(
            lambda ids: (
                torch.tensor(ids[0][len(shared_prompt) :], dtype=torch.int64),
                torch.tensor(ids[1], dtype=torch.int64),
            ),
            corpus,
        )
    )
    shared_prompt = torch.tensor(shared_prompt)
    return shared_prompt, corpus


STORAGE_LOCK = threading.Lock()
STORAGE = {}


@torch.inference_mode
def evaluate_corpus(
    stream: torch.cuda.Stream,
    model: torch.nn.Module,
    tokenizer: transformers.LlamaTokenizer,
    shared_prompt: torch.Tensor,
    corpus: List["TextCorpus"],
    evaluate_method: Literal["kd", "output"] = "output",
):
    if evaluate_method == "kd":
        samples = []
        for line in corpus:
            samples.append(
                (
                    torch.cat([shared_prompt, line.input_ids, line.output_ids]),
                    line.output_ids,
                )
            )

        def set_method(method: str):
            for m in model.modules():
                if hasattr(m, "attention_method"):
                    m.attention_method = method

        loss_sum = 0
        loss_count = 0
        for i_sample, (sample, output_ids) in enumerate(samples):
            with torch.cuda.stream(stream), torch.autocast("cuda", torch.bfloat16):
                set_method("fa2")
                logit_truth = None
                hash_id = hash(str(sample.numpy().tolist())) % 1000000000
                with STORAGE_LOCK:
                    if hash_id in STORAGE:
                        logit_truth = STORAGE[hash_id].to(stream.device)
                if logit_truth is None:
                    output_truth = model(
                        input_ids=sample.to(stream.device, non_blocking=True).unsqueeze(
                            0
                        ),
                        use_cache=False,
                        num_logits_to_keep=16384,
                    )
                    logit_truth = output_truth.logits.view(
                        -1, output_truth.logits.shape[-1]
                    )
                    with STORAGE_LOCK:
                        STORAGE[hash_id] = logit_truth.cpu()
                else:
                    assert isinstance(logit_truth, torch.Tensor)
                    assert logit_truth.device == stream.device

                set_method("hip")
                output_student = model(
                    input_ids=sample.to(stream.device, non_blocking=True).unsqueeze(0),
                    use_cache=False,
                    num_logits_to_keep=len(output_ids),
                )
                logit_student = output_student.logits.view(
                    -1, output_student.logits.shape[-1]
                )

                logit_truth = logit_truth[-logit_student.shape[0] :, :]

                loss = torch.nn.functional.kl_div(
                    input=logit_student.float().log_softmax(dim=-1),
                    target=logit_truth.float().softmax(dim=-1),
                    reduction="batchmean",
                )
                if stream.device_index == 0:
                    tqdm.tqdm.write(f"({i_sample}) L: {loss.item():.6f}")
                loss_sum += loss
                loss_count += 1
        loss = loss_sum.item() / loss_count
        return loss
    elif evaluate_method == "output":
        with torch.cuda.stream(stream), torch.autocast("cuda", torch.bfloat16):
            shared_output = model(
                input_ids=shared_prompt.to(stream.device, non_blocking=True).unsqueeze(
                    0
                ),
                use_cache=True,
                num_logits_to_keep=1,
            )

        loss_sum = 0
        loss_count = 0
        for line in corpus:
            input_ids = line.input_ids
            output_ids = line.output_ids
            input_ids = input_ids.to(stream.device, non_blocking=True)
            output_ids = output_ids.to(stream.device, non_blocking=True)
            state = copy.deepcopy(shared_output.past_key_values)

            with torch.cuda.stream(stream), torch.autocast("cuda", torch.bfloat16):
                try:
                    output = model(
                        input_ids=input_ids.unsqueeze(0),
                        past_key_values=state,
                        num_logits_to_keep=1,
                    )
                except triton.runtime.errors.OutOfResources as ex:
                    print(ex)
                    return 99999999
                except RuntimeError as ex:
                    traceback.print_exc()
                    print(ex)
                    return 99999999
            state = output.past_key_values

            # stream.synchronize()

            logits = []
            step_size = 1  # output_ids.shape[-1]
            for i in range(0, output_ids.shape[-1], step_size):
                with torch.cuda.stream(stream), torch.autocast("cuda", torch.bfloat16):
                    output = model(
                        input_ids=output_ids[i : i + step_size].unsqueeze(0),
                        past_key_values=state,
                        num_logits_to_keep=step_size,
                    )
                    state = output.past_key_values
                    logits.append(output.logits.view(-1, output.logits.shape[-1]))
                # stream.synchronize()

            with torch.cuda.stream(stream):
                logits = torch.cat(logits, dim=0)

                loss_ce = torch.nn.functional.cross_entropy(
                    logits[:-1].to(torch.float32), output_ids[1:]
                )

                # probs = logits[:-1].to(torch.float32).softmax(dim=-1).to(torch.float64)
                # probs = probs.gather(dim=-1, index=output_ids[1:].unsqueeze(-1))
                # seq_probs = probs.cumprod(0)[-1]
                # loss_probs = -torch.log(seq_probs)

                if stream.device_index == 0:
                    # tqdm.tqdm.write(f'{loss_probs.item():.5f} ({seq_probs.item():.5f}), {loss_ce.item():.5f}, {tokenizer.decode(logits[:-1].argmax(dim=-1))}, {tokenizer.decode(output_ids[1:])}')
                    tqdm.tqdm.write(
                        f"{loss_ce.item():.5f}, {tokenizer.decode(logits[:-1].argmax(dim=-1))}, {tokenizer.decode(output_ids[1:])}"
                    )

                loss_sum += loss_ce
                loss_count += 1

            stream.synchronize()
        loss = math.exp(loss_sum.item() / loss_count)
        return loss
    else:
        raise Exception(evaluate_method)


@dataclasses.dataclass
class TextCorpus:
    input_ids: torch.Tensor
    output_ids: torch.Tensor


class TextDataset:
    shared_input_ids: torch.Tensor
    corpus: List[TextCorpus]

    def __repr__(self):
        return f"TextDataset(shared_input_ids={self.shared_input_ids.shape}, corpus=TextCorpus[{len(self.corpus)}])"

    @staticmethod
    def from_corpus(args, tokenizer, corpus):
        ds = TextDataset()

        corpus = format_chat_corpus(args, corpus)
        shared_prompt, corpus = tokenizing_and_find_shared_prompt_on_corpus(
            tokenizer, corpus
        )

        assert shared_prompt.dtype == torch.int64

        ds.shared_input_ids = shared_prompt
        ds.corpus = []
        for input_ids, output_ids in corpus:
            assert input_ids.dtype == torch.int64, input_ids.shape
            assert output_ids.dtype == torch.int64
            ds.corpus.append(
                TextCorpus(
                    input_ids=input_ids,
                    output_ids=output_ids,
                )
            )

        return ds


@torch.inference_mode
def job_ga(
    args,
    model,
    tokenizer: transformers.LlamaTokenizer,
    device,
):
    model.eval()

    # seed = [
    #     {
    #         'second_stage_k': 2048,
    #         'sliding_window_size': 1024,
    #         'sink_token_size': 256,
    #         'sa_extend_backend': 'streaming',
    #         'stages': [
    #             ScanStage(
    #                 stage_block_size_q=64,
    #                 stage_block_stride_q=4,
    #                 stage_chunk_size=256,
    #                 stage_k=None,
    #                 stage_stride=1,
    #             ),
    #             ScanStage(
    #                 stage_block_size_q=64,
    #                 stage_block_stride_q=4,
    #                 stage_chunk_size=32,
    #                 stage_k=32768,
    #                 stage_stride=1,
    #             ),
    #             ScanStage(
    #                 stage_block_size_q=64,
    #                 stage_block_stride_q=1,
    #                 stage_chunk_size=8,
    #                 stage_k=8192,
    #                 stage_stride=1,
    #             ),
    #         ]
    #     } for _ in range(model.config.num_hidden_layers)
    # ]

    seed = [
        {
            "second_stage_k": 2048,
            "sliding_window_size": 1024,
            "sink_token_size": 256,
            "sa_extend_backend": "streaming",
            "stages": [
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=4,
                    stage_chunk_size=256,
                    stage_k=None,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=4,
                    stage_chunk_size=32,
                    stage_k=32768,
                    stage_stride=1,
                ),
                ScanStage(
                    stage_block_size_q=64,
                    stage_block_stride_q=1,
                    stage_chunk_size=8,
                    stage_k=8192,
                    stage_stride=1,
                ),
            ],
        }
        for _ in range(2)
    ]

    def mutate_inner(p):
        p = copy.deepcopy(p)

        stage_job = random.choice(
            [
                "pass",
                "swap_layer",
                "swap_stage",
                "copy_stage",
                "drop_stage",
            ]
        )
        if stage_job == "pass":
            pass
        elif stage_job == "swap_layer":
            a = random.randint(0, len(p) - 1)
            b = random.randint(0, len(p) - 1)
            layer_a = p[a]
            layer_b = p[b]
            p[a] = layer_b
            p[b] = layer_a
        elif stage_job == "swap_stage":
            target_layer = random.randint(0, len(p) - 1)
            layer = p[target_layer]["stages"]
            if len(layer) > 2:
                a = random.randint(1, len(layer) - 1)
                b = random.randint(1, len(layer) - 1)
                stage_a = layer[a]
                stage_b = layer[b]
                layer[a] = stage_b
                layer[b] = stage_a
        elif stage_job == "copy_stage":
            target_layer = random.randint(0, len(p) - 1)
            layer = p[target_layer]["stages"]  # type: list
            if len(layer) > 2:
                a = random.randint(1, len(layer) - 1)
                stage = layer[a]
                layer.insert(a, copy.deepcopy(stage))
        elif stage_job == "drop_stage":
            target_layer = random.randint(0, len(p) - 1)
            layer = p[target_layer]["stages"]
            if len(layer) > 2:
                layer.pop(-1)
        else:
            raise Exception()

        num_param_jobs = random.randint(0, 10)
        for _ in range(num_param_jobs):
            param_job = random.choice(
                [
                    "pass",
                    "sliding_window_size",
                    "sink_token_size",
                    "sa_extend_backend",
                    "stage_extend_backend",
                    "second_stage_k",
                    "block_size_q",
                    "block_stride_q",
                    "chunk_size",
                    "k",
                    "stride",
                ]
            )
            target_layer = random.randint(0, len(p) - 1)
            layer = p[target_layer]["stages"]
            target_stage = random.randint(0, len(layer) - 1)
            stage = layer[target_stage]

            if param_job == "pass":
                pass
            elif param_job == "sliding_window_size":
                if random.random() > 0.5:
                    p[target_layer]["sliding_window_size"] *= 2
                else:
                    p[target_layer]["sliding_window_size"] //= 2
            elif param_job == "sink_token_size":
                if random.random() > 0.5:
                    p[target_layer]["sink_token_size"] *= 2
                else:
                    p[target_layer]["sink_token_size"] //= 2
            elif param_job == "sa_extend_backend":
                p[target_layer]["sa_extend_backend"] = random.choice(
                    ["streaming", "dynamic_extend"]
                )
            elif param_job == "stage_extend_backend":
                stage.stage_extend_backend = random.choice(
                    ["streaming", "dynamic_extend", "relative"]
                )
            elif param_job == "second_stage_k":
                if random.random() > 0.5:
                    p[target_layer]["second_stage_k"] *= 2
                else:
                    p[target_layer]["second_stage_k"] //= 2
            elif param_job == "block_size_q":
                if random.random() > 0.5:
                    stage.stage_block_size_q *= 2
                else:
                    stage.stage_block_size_q //= 2
            elif param_job == "block_stride_q":
                if random.random() > 0.5:
                    stage.stage_block_stride_q *= 2
                else:
                    stage.stage_block_stride_q //= 2
            elif param_job == "k":
                if stage.stage_k is not None:
                    if random.random() > 0.5:
                        stage.stage_k *= 2
                    else:
                        stage.stage_k //= 2
            elif param_job == "chunk_size":
                if random.random() > 0.5:
                    stage.stage_chunk_size *= 2
                else:
                    stage.stage_chunk_size //= 2
            elif param_job == "stride":
                if random.random() > 0.5:
                    stage.stage_stride *= 2
                else:
                    stage.stage_stride //= 2
            else:
                raise Exception()
        return p

    def validate(p):
        assert len(p) in [2, model.config.num_hidden_layers], "layer count must match"
        for layer in p:
            layer_meta = layer
            layer = layer["stages"]
            assert layer_meta["second_stage_k"] > 0
            assert (
                layer_meta["second_stage_k"] <= 8192
            )  # we have to restrict this to prevent full dense attention...
            assert layer_meta["sliding_window_size"] >= 64
            assert layer_meta["sliding_window_size"] <= 8192
            assert layer_meta["sink_token_size"] >= 4
            assert layer_meta["sink_token_size"] <= 8192
            assert len(layer) >= 2, "too small"
            assert len(layer) <= 7, "too large"
            assert layer[0].stage_k == None, "first quadratic"
            assert layer[-1].stage_chunk_size <= 32, "too large k"
            assert layer[-1].stage_chunk_size > 0, "too large k"
            assert (layer_meta["second_stage_k"] % layer[-1].stage_chunk_size) == 0

            stage_block_size_q = 987654321
            stage_stride = 987654321
            stage_k = 987654321
            stage_chunk_size = 987654321

            for stage in layer:
                assert (stage.stage_k is None) or (stage.stage_k > 0)
                assert stage.stage_stride > 0
                assert stage.stage_block_size_q > 0
                assert stage.stage_block_stride_q > 0
                assert stage.stage_chunk_size > 0
                assert (stage.stage_k is None) or (stage.stage_k <= 131072)
                assert stage.stage_stride <= 16
                assert stage.stage_block_size_q <= 512
                assert stage.stage_block_stride_q <= 16
                assert stage.stage_chunk_size <= 512
                assert (stage.stage_block_size_q // stage.stage_block_stride_q) >= 16
                assert (stage.stage_k is None) or (
                    (stage.stage_k % stage.stage_chunk_size) == 0
                )

                assert (stage.stage_k is None) or (stage.stage_k <= stage_k)
                assert stage.stage_stride <= stage_stride
                assert stage.stage_block_size_q <= stage_block_size_q
                assert stage.stage_chunk_size <= stage_chunk_size

                stage_block_size_q = stage.stage_block_size_q
                stage_stride = stage.stage_stride
                stage_chunk_size = stage.stage_chunk_size
                if stage.stage_k is not None:
                    stage_k = stage.stage_k

    def mutate(p):
        while True:
            try:
                p1 = mutate_inner(p)
                validate(p1)
                return p1
            except AssertionError as ex:
                # print(ex)
                pass

    def crossover(p1, p2):
        assert len(p1) == len(p2)
        pt = random.randint(0, len(p1))

        p1_top = p1[:pt]
        p1_bot = p2[pt:]

        p2_top = p1[pt:]
        p2_bot = p2[:pt]

        p1_new = copy.deepcopy(p1_top) + copy.deepcopy(p1_bot)
        p2_new = copy.deepcopy(p2_top) + copy.deepcopy(p2_bot)
        assert len(p1_new) == len(p1)
        assert len(p2_new) == len(p2)

        return p1_new, p2_new

    def apply_setting(model, setting):
        for m in model.modules():
            if hasattr(m, "tree_extend_stages"):
                idx = m.layer_idx
                if setting is None:
                    m.attention_method = "fa2"
                else:
                    m.attention_method = "hip"
                    if len(setting) == 2:
                        if idx < 4:
                            m.tree_extend_stages = setting[0]
                        else:
                            m.tree_extend_stages = setting[1]
                    else:
                        m.tree_extend_stages = setting[idx]

    latency_cache = {}

    def evaluate_latency(model, stages):
        hash_id = str(stages)
        if hash_id in latency_cache:
            return latency_cache[hash_id]
        else:
            device = 0
            HEAD = 32
            HEAD_KV = 8
            TDST = 8192
            TSRC = 131072
            HID = 128
            q = torch.empty((1, TDST, HEAD, HID), dtype=torch.bfloat16, device=device)
            k = torch.empty(
                (1, TSRC, HEAD_KV, HID), dtype=torch.bfloat16, device=device
            )
            v = torch.empty(
                (1, TSRC, HEAD_KV, HID), dtype=torch.bfloat16, device=device
            )
            latency_sum = 0
            latency_count = 0
            for i in range(10):
                if i == 2:
                    latency_sum = latency_count = 0
                start = torch.cuda.Event(True)
                end = torch.cuda.Event(True)

                start.record()
                dual_stage_quadratic_hip_attention(
                    q,
                    k,
                    v,
                    args=HiPAttentionArgs11(
                        block_size_k=64,
                        block_stride_k=1,
                        sliding_window_size=stages["sliding_window_size"],
                        sink_token_size=stages["sink_token_size"],
                    ),
                    second_stage_k=stages["second_stage_k"],
                    stages=stages["stages"],
                    block_sparse_block_size_q=64,
                    model_context_length=131072,
                    scan_extend_backend="relative",
                    sa_extend_backend=stages["sa_extend_backend"],
                )
                end.record()

                end.synchronize()
                latency_sum += start.elapsed_time(end)
                latency_count += 1
            latency = latency_sum / latency_count
            latency_cache[hash_id] = latency
            return latency

    def evaluate_latency_of_candidate(model, p):
        latency_sum = 0
        latency_count = 0
        if len(p) == 2:
            latency_sum += evaluate_latency(model, p[0]) * 4
            latency_count += 4
            latency_sum += evaluate_latency(model, p[1]) * 28
            latency_count += 28
        else:
            for stage in p:
                latency_sum += evaluate_latency(model, stage)
                latency_count += 1
        return latency_sum / latency_count

    def evaluate_population(population, model, evaluate_ds: List[TextDataset]):
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        ret = evaluate_population_inner(population, model, evaluate_ds)

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        return ret

    def evaluate_population_inner(population, model, evaluate_ds: List[TextDataset]):
        import threading

        n_threads = torch.cuda.device_count()
        jobs = [
            [(i, p) for i, p in enumerate(population) if (i % n_threads) == tid]
            for tid in range(n_threads)
        ]
        results = []
        models = [(torch.cuda.Stream(device=0), model)]
        for tid in range(1, n_threads):
            models.append((torch.cuda.Stream(device=tid), copy.deepcopy(model).to(tid)))

        def thread_main(tid, args, jobs):
            stream, model = args
            for jid, job in tqdm.tqdm(
                jobs, position=tid, dynamic_ncols=True, leave=False
            ):
                try:
                    # with lock:
                    #     torch.set_default_device(tid)
                    #     torch.cuda.set_device(tid)
                    with torch.cuda.stream(stream):
                        apply_setting(model, job)
                        ppls = []
                        for ds in evaluate_ds:
                            ppl = evaluate_corpus(
                                stream, model, tokenizer, ds.shared_input_ids, ds.corpus
                            )
                            ppls.append(ppl)
                        ppl = sum(ppls) / len(ppls)
                    # stream.synchronize()
                    results.append((jid, ppl))
                except Exception as ex:
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                    traceback.print_exc()
                    print("thread main failed due to", ex)
                    results.append((jid, 999999999999))

        torch.cuda.synchronize()
        threads = [
            threading.Thread(
                target=thread_main, args=(tid, models[tid], jobs[tid]), daemon=True
            )
            for tid in range(n_threads)
        ]
        list(map(lambda x: x.start(), threads))
        list(map(lambda x: x.join(), threads))
        torch.cuda.synchronize()

        torch.set_default_device(0)
        torch.cuda.set_device(0)

        ppls = list(map(lambda x: x[1], sorted(results, key=lambda x: x[0])))

        latencies = []
        for p in tqdm.tqdm(
            population, desc="eval latency", leave=False, dynamic_ncols=True
        ):
            apply_setting(model, p)
            latency = evaluate_latency_of_candidate(model, p)
            latencies.append(latency)
        scores = list(zip(latencies, ppls))
        return scores

    # settings
    num_population = 25

    # prepare shared prompt
    # num_corpus = 20
    # corpus_rag = load_loft_rag_chat_corpus()[:num_corpus // 2]
    # corpus_ret = load_loft_retrieval_chat_corpus()[:num_corpus // 2]
    # corpus = corpus_rag + corpus_ret
    # corpus = load_loft_rag_chat_corpus()[:num_corpus]
    # corpus = load_loft_retrieval_chat_corpus()[:num_corpus]
    # corpus = format_chat_corpus(args, corpus)
    # shared_prompt, corpus = tokenizing_and_find_shared_prompt_on_corpus(
    #     tokenizer, corpus
    # )

    evaluate_ds = [
        TextDataset.from_corpus(
            args,
            tokenizer,
            load_loft_retrieval_chat_corpus()[:15] + load_loft_rag_chat_corpus()[:15],
        ),
        # TextDataset.from_corpus(
        #     args, tokenizer,
        #     load_infinite_bench_subset('passkey', tokenizer, 131072, 4),
        # ),
        # TextDataset.from_corpus(
        #     args, tokenizer,
        #     load_infinite_bench_subset('kv_retrieval', tokenizer, 131072, 4),
        # ),
        TextDataset.from_corpus(
            args,
            tokenizer,
            load_infinite_bench_subset("longbook_qa_eng", tokenizer, 131072, 30),
        ),
    ]

    print("shared prompt size", evaluate_ds)

    # run GA
    load_population = os.getenv("LOAD_POPULATION", "0") == "1"
    population = [seed]
    if not load_population:
        for _ in range(num_population):
            t = copy.deepcopy(seed)
            for _ in range(10):
                t = mutate(t)
            population.append(t)
    scores = evaluate_population(population, model, evaluate_ds)
    seed_score = copy.deepcopy(scores[0])
    seed_latency, seed_loss = seed_score
    print(population[0])
    print(scores)
    print("seed", seed_score)

    if load_population:
        # {
        #     'generation': current_generation,
        #     'best_idx': best_idx,
        #     'best_candidate': population[best_idx],
        #     'scores': scores,
        #     'population': population,
        # }
        checkpoint = "./saves/pareto/population.json"
        with open(checkpoint, "r") as f:
            state = json.load(f)
        population = state["population"]
        for p in population:
            for l in p:
                for i in range(len(l["stages"])):
                    ss = l["stages"][i]
                    l["stages"][i] = ScanStage(
                        stage_block_size_q=ss["stage_block_size_q"],
                        stage_block_stride_q=ss["stage_block_stride_q"],
                        stage_chunk_size=ss["stage_chunk_size"],
                        stage_k=ss["stage_k"],
                        stage_stride=ss["stage_stride"],
                    )
        scores = state["scores"]

    run = wandb.init(
        project="hip-ga",
        config={
            "num_population": num_population,
            "corpus_setting": f"{evaluate_ds}",
        },
    )

    current_generation = 0

    while True:
        new_populations = []

        # population = list(map(lambda x:x[1], sorted(zip(scores, population), key=lambda x:x[0][1])))
        # scores = list(map(lambda x:x[0], sorted(zip(scores, population), key=lambda x:x[0][1])))

        for _ in range(num_population):
            # more elites
            # p1, p2 = random.sample(population, counts=(len(population) - np.arange(0, len(population))).tolist(), k=2)
            p1, p2 = random.sample(
                population,
                counts=[
                    1,
                ]
                * len(population),
                k=2,
            )
            p1, p2 = crossover(p1, p2)
            for _ in range(
                random.randint(0, 2) if random.random() < 0.5 else random.randint(0, 10)
            ):
                p1 = mutate(p1)
            for _ in range(
                random.randint(0, 2) if random.random() < 0.5 else random.randint(0, 10)
            ):
                p2 = mutate(p2)

            p1_latency = evaluate_latency_of_candidate(model, p1)
            p2_latency = evaluate_latency_of_candidate(model, p2)
            if p1_latency < (seed_latency * 2):
                new_populations.append(p1)
            if p2_latency < (seed_latency * 2):
                new_populations.append(p2)
        new_scores = evaluate_population(new_populations, model, evaluate_ds)

        population = population + new_populations
        scores = scores + new_scores

        # kill way too slow candidates
        survived_populations = []
        survived_scores = []
        for p, s in zip(population, scores):
            if s[0] <= (seed_score[0] * 2):
                survived_populations.append(p)
                survived_scores.append(s)
        # print(f'{len(new_scores) - len(survived_scores)} candidates are killed')
        population = survived_populations
        scores = survived_scores

        # just check scores
        # best_args = list(map(lambda x: x[0], list(sorted(zip(range(len(scores)), scores), key=lambda x: x[1][1], reverse=False))[:num_population]))

        # pareto front
        import pypareto

        values = list(
            map(
                lambda x: (
                    x[0],
                    x[1],
                ),
                scores,
            )
        )
        chain = pypareto.Comparison(
            pypareto.by_value,
            pypareto.MaxMinList(
                pypareto.MaxMin.MIN,
                pypareto.MaxMin.MIN,
            ),
        ).as_chain()
        best_values = chain.split_by_pareto(values)

        best_values = sum(best_values, [])[:num_population]
        best_args = []
        best_scores = []
        for b in best_values:
            best_args.append(values.index(b))
            best_scores.append(scores[values.index(b)])

        print(best_scores)
        os.makedirs("./saves/pareto", exist_ok=True)
        plt.clf()
        plt.title(f"Gen {current_generation}")
        # for line in best_scores:
        plt.scatter(
            list(map(lambda x: x[0], sorted(best_scores, key=lambda x: x[0]))),
            list(map(lambda x: x[1], sorted(best_scores, key=lambda x: x[0]))),
            label="population",
        )
        plt.scatter(x=[seed_score[0]], y=[seed_score[1]], marker="s", label="seed")

        population = np.array(population, dtype=object)[np.array(best_args)].tolist()
        scores = np.array(scores, dtype=object)[np.array(best_args)].tolist()
        # print(population[0])
        scores = list(map(tuple, scores))

        s = np.array(scores, dtype=np.float32)
        seed_latency = seed_score[0]

        s_cand = np.argsort(np.abs(s[:, 0] - seed_latency))[:5]
        s_cand_loss = np.argsort(s[s_cand, 1])[0]
        best_idx = s_cand[s_cand_loss].item()

        plt.scatter(
            x=[
                scores[best_idx][0],
            ],
            y=[
                scores[best_idx][1],
            ],
            marker="s",
            label="best",
        )
        plt.legend()
        plt.grid()
        plt.savefig("dummy_pareto.png")
        if (current_generation % 1) == 0:
            plt.savefig(f"./saves/pareto/dummy_pareto_{current_generation}.png")

        json_data = {
            "generation": current_generation,
            "best_idx": best_idx,
            "best_candidate": population[best_idx],
            "scores": scores,
            "population": population,
        }

        class EnhancedJSONEncoder(json.JSONEncoder):
            def default(self, o):
                if dataclasses.is_dataclass(o):
                    return dataclasses.asdict(o)
                return super().default(o)

        with open("./saves/pareto/population.json", "w") as f:
            json.dump(json_data, f, indent=2, cls=EnhancedJSONEncoder)

        if (current_generation % 1) == 0:
            with open(
                f"./saves/pareto/population_gen{current_generation}.json", "w"
            ) as f:
                json.dump(json_data, f, indent=2, cls=EnhancedJSONEncoder)
        print(population[best_idx])
        print("=====> gen", current_generation, scores[best_idx], "<=====")

        latencies = list(map(lambda x: [x[0]], scores))
        losses = list(map(lambda x: [x[1]], scores))

        latency_table = wandb.Table(data=latencies, columns=["latency"])
        latency_hist = wandb.plot.histogram(latency_table, "latency")

        loss_table = wandb.Table(data=losses, columns=["loss"])
        loss_hist = wandb.plot.histogram(loss_table, "loss")

        best_latency, best_loss = scores[best_idx]
        avg_loss = np.mean(np.array(losses)).item()

        wandb.log(
            {
                "ga/best_latency": best_latency,
                "ga/best_loss": best_loss,
                "ga/avg_loss": avg_loss,
                "ga/generation": current_generation,
                "ga/loss_hist": loss_hist,
                "ga/latency_hist": latency_hist,
            },
            step=current_generation,
            commit=True,
        )

        apply_setting(model, population[best_idx])
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        current_generation += 1
