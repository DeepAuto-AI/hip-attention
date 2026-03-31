import json
import os
import time

import numpy as np
import torch
import tqdm
import transformers
from datasets import load_dataset
from hip_research.utils.seed import seed

MMLU_FORMAT = """> The following are multiple choice questions (with answers) about {subject_name}.

{number}. {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer:"""

MMLU_SUBJECTS = [
    "business_ethics",
    "clinical_knowledge",
    "medical_genetics",
    "high_school_us_history",
    "high_school_physics",
    "high_school_world_history",
    "virology",
    "high_school_microeconomics",
    "econometrics",
    "college_computer_science",
    "high_school_biology",
    "abstract_algebra",
    "professional_accounting",
    "philosophy",
    "professional_medicine",
    "nutrition",
    "global_facts",
    "machine_learning",
    "security_studies",
    "public_relations",
    "professional_psychology",
    "prehistory",
    "anatomy",
    "human_sexuality",
    "college_medicine",
    "high_school_government_and_politics",
    "college_chemistry",
    "logical_fallacies",
    "high_school_geography",
    "high_school_european_history",  # 9600
    "elementary_mathematics",
    "human_aging",
    "college_mathematics",
    "high_school_psychology",
    "formal_logic",
    "high_school_statistics",
    "international_law",
    "high_school_mathematics",
    "high_school_computer_science",
    "conceptual_physics",
    "miscellaneous",
    "high_school_chemistry",
    "marketing",
    "professional_law",
    "management",
    "college_physics",
    "jurisprudence",
    "world_religions",
    "sociology",
    "us_foreign_policy",
    "high_school_macroeconomics",
    "computer_security",
    "moral_scenarios",
    "moral_disputes",
    "electrical_engineering",
    "astronomy",
    "college_biology",
]


def format_mmlu(question, number, subject_name):
    """
    {'input': 'A "dished face" profile is often associated with',
    'A': 'a protruding mandible due to reactivation of the condylar cartilage by acromegaly.',
    'B': 'a recessive maxilla due to failure of elongation of the cranial base.',
    'C': 'an enlarged frontal bone due to hydrocephaly.',
    'D': 'defective development of the maxillary air sinus.',
    'target': 'B'}
    """

    return MMLU_FORMAT.format(
        subject_name=subject_name,
        number=number,
        question=question["input"],
        choice_a=question["A"],
        choice_b=question["B"],
        choice_c=question["C"],
        choice_d=question["D"],
    )


def exam_mmlu(model, tokenizer: transformers.PreTrainedTokenizer, text):
    PROMPT_ALWAYS_FLASH = os.environ.get("PROMPT_ALWAYS_FLASH", "0") == "1"
    LAST_DENSE = os.environ.get("LAST_DENSE", "0") == "1"

    def gather_token_ids(candidates):
        ids = []
        for cand in candidates:
            ids.append(tokenizer(cand).input_ids[-1])
        return ids

    tokens_a = gather_token_ids(["A", ":A", ": A", ":  A", "a", ":a", ": a", ":  a"])
    tokens_b = gather_token_ids(["B", ":B", ": B", ":  B", "b", ":b", ": b", ":  b"])
    tokens_c = gather_token_ids(["C", ":C", ": C", ":  C", "c", ":c", ": c", ":  c"])
    tokens_d = gather_token_ids(["D", ":D", ": D", ":  D", "d", ":d", ": d", ":  d"])

    tokenizer.truncation_side = "left"

    assert hasattr(model, "config")
    assert hasattr(model.config, "max_position_embeddings")

    inputs = tokenizer(
        [text],
        return_tensors="pt",
        max_length=model.config.max_position_embeddings,
        truncation=True,
    )
    # print(inputs.input_ids.shape)
    seq_len = inputs.input_ids.shape[-1]
    with torch.no_grad():
        if PROMPT_ALWAYS_FLASH:
            for m in model.modules():
                if hasattr(m, "attention_method"):
                    m.tree_dense_queries = seq_len - 1
        if LAST_DENSE:
            for m in model.modules():
                if hasattr(m, "attention_method"):
                    m.tree_last_dense_queries = -1

        output = model(output_logits=True, **inputs).logits
        output = torch.softmax(output, dim=-1).cpu()

    prob_a = max([output[0, -1, token].item() for token in tokens_a])
    prob_b = max([output[0, -1, token].item() for token in tokens_b])
    prob_c = max([output[0, -1, token].item() for token in tokens_c])
    prob_d = max([output[0, -1, token].item() for token in tokens_d])
    probs = [("A", prob_a), ("B", prob_b), ("C", prob_c), ("D", prob_d)]
    probs = list(sorted(probs, key=lambda x: x[1], reverse=True))
    return probs, seq_len


def evaluate_mmlu(args, model, tokenizer, subject_name):
    dataset = load_dataset("lukaemon/mmlu", subject_name, trust_remote_code=True)

    few_shots = []
    for question in dataset["train"]:
        text = format_mmlu(question, len(few_shots) + 1, subject_name)
        choice = question["target"]
        text += choice
        few_shots.append(text)
    for question in dataset["validation"]:
        text = format_mmlu(question, len(few_shots) + 1, subject_name)
        choice = question["target"]
        text += choice
        few_shots.append(text)
    few_shots = few_shots[:20]

    t_start = time.time()
    results = []
    n_correct = 0
    seq_len_sum = 0
    for question in tqdm.tqdm(
        dataset["test"], dynamic_ncols=True, leave=True, desc=subject_name
    ):
        text = format_mmlu(question, len(few_shots) + 1, subject_name)
        truth = question["target"]
        text = "\n\n".join(
            few_shots
            + [
                text,
            ]
        )
        estimations, seq_len = exam_mmlu(model, tokenizer, text)
        estimation = estimations[0][0]
        correct = truth == estimation
        # print(truth, estimations, seq_len)
        if correct:
            n_correct += 1
        seq_len_sum += seq_len
        results.append(
            {
                # 'text': text,
                "truth": truth,
                "estimations": estimations,
                "estimation": estimation,
                "correct": correct,
                "seq_len": seq_len,
            }
        )
        for m in model.modules():
            if hasattr(m, "_clean_cache"):
                m._clean_cache()

    elapsed = time.time() - t_start

    accuracy = (n_correct / len(results)) * 100
    avg_seq_len = seq_len_sum / len(results)
    print(
        f"{subject_name} = Accuracy: {accuracy:.4f} %, avg_seq_len: {avg_seq_len:.2f}. elapsed: {elapsed:.1f} s"
    )

    folder = f"./saves/llama_eval/mmlu/{args.name}_{args.model}_{args.method}"
    if args.method == "hip":
        folder = f"./saves/llama_eval/mmlu/{args.name}_{args.model}_{args.method}_bq{args.block_size_q}_bk{args.block_size_k}_k{args.k}"
    os.makedirs(folder, exist_ok=True)
    json_path = f"{folder}/{subject_name}.json"

    with open(json_path, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "avg_seq_len": avg_seq_len,
                "elapsed": elapsed,
                "k": args.k,
                "model": args.model,
                "block_size_q": args.block_size_q,
                "block_size_k": args.block_size_k,
                "dense_queries": args.dense_queries,
                "results": results,
            },
            f,
            indent=2,
        )
        print("dumped", json_path)

    return accuracy


def job_mmlu(args, model, tokenizer, device):
    seed()

    # evaluate_mmlu(args, model, tokenizer, 'anatomy')
    # return

    accuracies = []
    for subjects in MMLU_SUBJECTS:
        acc = evaluate_mmlu(args, model, tokenizer, subjects)
        accuracies.append(acc)

    accuracy = np.array(accuracies).mean()
    print(f"MMLU AVG. ACC: {accuracy}")
