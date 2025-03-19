import json
import os
import time

import datasets
import torch
import tqdm
import transformers
from hip_research.main.eval_args import ArgsType

from hip_attn.utils.benchmarking import get_bench

MMMU_SUBJECT = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]

"""
{'id': 'validation_Accounting_1',
 'question': '<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?',
 'options': "['$6', '$7', '$8', '$9']",
 'explanation': '',
 'image_1': <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=733x237>,
 'image_2': None,
 'image_3': None,
 'image_4': None,
 'image_5': None,
 'image_6': None,
 'image_7': None,
 'img_type': "['Tables']",
 'answer': 'B',
 'topic_difficulty': 'Medium',
 'question_type': 'multiple-choice',
 'subfield': 'Managerial Accounting'}
"""


def convert_to_inputs(entry, with_answer):
    res = []

    def add_image(image_id):
        if image_id in entry and entry[image_id] is not None:
            root = f'./cache/mmmu_imgs/{entry["subfield"].replace(" ", "_").lower()}'
            os.makedirs(root, exist_ok=True)
            cache_path = os.path.join(root, f'{entry["id"]}_{image_id}.png')
            if not os.path.exists(cache_path):
                entry["image_1"].save(cache_path)
            res.append({"text": f"Image {image_id}: \n"})
            res.append({"image": cache_path})

    add_image("image_1")
    add_image("image_2")
    add_image("image_3")
    add_image("image_4")
    add_image("image_5")
    add_image("image_6")
    add_image("image_7")

    if isinstance(entry["options"], str):
        # x = entry['options'].replace('"', '\\"').replace("'", '"')
        x = entry["options"]
        # print(x)
        # entry['options'] = json.loads(x)
        entry["options"] = eval(x)

    res.append({"text": entry["question"] + "\n"})
    for i in range(len(entry["options"])):
        marker = {
            0: "A",
            1: "B",
            2: "C",
            3: "D",
            4: "E",
            5: "F",
            6: "G",
            7: "H",
            8: "I",
            9: "J",
            10: "K",
        }[i]
        res.append({"text": f' {marker}: {entry["options"][i]}\n'})

    if with_answer:
        res.append({"text": f'Answer:{entry["answer"]}\n'})
    else:
        res.append({"text": f"Answer:"})

    return res


def get_logit_prob(
    probs: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer, letter: str
):
    gathered = []
    for prefix in ["", ":", ": ", " ", "  ", ":  "]:
        token_id = tokenizer(prefix + letter).input_ids[-1]
        gathered.append(probs[token_id].item())

        token_id = tokenizer((prefix + letter).lower()).input_ids[-1]
        gathered.append(probs[token_id].item())

    # print(token_id)
    return letter, max(gathered)


def evaluate_subject(args: ArgsType, model, tokenizer, subject):
    # print(subject)
    ds = datasets.load_dataset(f"./cache/MMMU/", subject)
    # ds = datasets.load_dataset('./cache/MMLU/', subject)

    few_shots = []
    for entry in ds["dev"]:
        if entry["question_type"] != "multiple-choice":
            continue
        few_shots.append(convert_to_inputs(entry, True))

    for i, entry in enumerate(ds["validation"]):
        if i >= 3:
            break
        if entry["question_type"] != "multiple-choice":
            continue
        few_shots.append(convert_to_inputs(entry, True))

    few_shots = few_shots[:7]
    few_shots = sum(few_shots, start=[])

    results = []
    seq_lens = []
    corrects = 0
    count = 0
    choices = 0
    t_start = time.time()
    get_bench().reset_measures()
    get_bench().reset_trace()
    get_bench().disabled = False
    for i, entry in enumerate(
        tqdm.tqdm(ds["validation"], desc=subject, dynamic_ncols=True, leave=True)
    ):
        if i < 3:
            continue
        if entry["question_type"] != "multiple-choice":
            continue

        question_inputs = convert_to_inputs(entry, False)

        inputs = few_shots + question_inputs

        # print(inputs)

        inputs = tokenizer.from_list_format(inputs)
        inputs = tokenizer(inputs, return_tensors="pt")
        inputs = inputs.to(model.device)

        # print(inputs.input_ids.shape)

        with torch.no_grad():
            logits = model(**inputs).logits
            logits = torch.softmax(logits, -1)[0, -1]
            # print(logits.shape)

        probs = [
            get_logit_prob(logits, tokenizer, "A"),
            get_logit_prob(logits, tokenizer, "B"),
            get_logit_prob(logits, tokenizer, "C"),
            get_logit_prob(logits, tokenizer, "D"),
            get_logit_prob(logits, tokenizer, "E"),
            get_logit_prob(logits, tokenizer, "F"),
            get_logit_prob(logits, tokenizer, "G"),
            get_logit_prob(logits, tokenizer, "H"),
            get_logit_prob(logits, tokenizer, "I"),
            get_logit_prob(logits, tokenizer, "J"),
            get_logit_prob(logits, tokenizer, "K"),
        ]
        probs = probs[: len(entry["options"])]
        probs = list(sorted(probs, key=lambda x: x[1], reverse=True))
        correct = probs[0][0] == entry["answer"]
        count += 1
        corrects += 1 if correct else 0
        results.append(
            {
                "seq_len": inputs.input_ids.shape[-1],
                "correct": correct,
                "probs": probs,
                "answer": entry["answer"],
                "id": entry["id"],
            }
        )
        seq_lens.append(inputs.input_ids.shape[-1])
        choices += len(entry["options"])

        for m in model.modules():
            if hasattr(m, "_clean_cache"):
                m._clean_cache()

    # print(get_bench().format_tracetree())
    benchmarks = get_bench().todict()
    tick_vit = benchmarks["qwen.vit.encode"] * 1000
    tick_decoder = benchmarks["qwen.decoder"] * 1000
    elapsed = time.time() - t_start
    bogo_accuracy = count / choices * 100
    accuracy = corrects / count * 100
    avg_seq_len = sum(seq_lens) / len(seq_lens)

    os.makedirs("./saves/qwen_eval/mmmu", exist_ok=True)
    json_path = f"./saves/qwen_eval/mmmu/{subject}_{args.model}_{args.method}.json"
    if args.method == "hip":
        json_path = f"./saves/qwen_eval/mmmu/{subject}_{args.model}_{args.method}_bq{args.block_size_q}_bk{args.block_size_k}_k{args.k}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "bogo_accuracy": bogo_accuracy,
                "avg_seq_len": avg_seq_len,
                "elapsed": elapsed,
                "tick_vit": tick_vit,
                "tick_decoder": tick_decoder,
                "results": results,
            },
            f,
            indent=2,
        )
        print(
            f"{json_path}: {accuracy:.2f} (seq: {avg_seq_len:.2f}) ({elapsed:.2f} sec, {tick_vit:.4f}, {tick_decoder:.4f})"
        )

    return accuracy


def job_mmmu(args: ArgsType, model, tokenizer, device):
    accs = []
    results = {}
    for subject in MMMU_SUBJECT:
        acc = evaluate_subject(args, model, tokenizer, subject)
        results[subject] = acc
        accs.append(acc)

    accuracy = sum(accs) / len(accs)

    os.makedirs("./saves/qwen_eval/mmmu", exist_ok=True)
    json_path = f"./saves/qwen_eval/mmmu/{args.model}_{args.method}.json"
    if args.method == "hip":
        json_path = f"./saves/qwen_eval/mmmu/{args.model}_{args.method}_bq{args.block_size_q}_bk{args.block_size_k}_k{args.k}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"MMMU (Avg.): {accuracy} ({json_path})")
