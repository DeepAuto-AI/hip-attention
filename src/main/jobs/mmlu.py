import json
import os
import torch
import transformers
from datasets import load_dataset
import tqdm

from src.utils import seed, get_bench

MMLU_FORMAT = """> The following are multiple choice questions (with answers) about {subject_name}.

{number}. {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Answer: """

MMLU_SUBJECTS = [
    'high_school_european_history', 
    'business_ethics', 
    'clinical_knowledge', 
    'medical_genetics', 
    'high_school_us_history', 
    'high_school_physics', 
    'high_school_world_history', 
    'virology', 
    'high_school_microeconomics', 
    'econometrics', 
    'college_computer_science', 
    'high_school_biology', 
    'abstract_algebra', 
    'professional_accounting', 
    'philosophy', 
    'professional_medicine', 
    'nutrition', 
    'global_facts', 
    'machine_learning', 
    'security_studies', 
    'public_relations', 
    'professional_psychology', 
    'prehistory', 
    'anatomy', 
    'human_sexuality', 
    'college_medicine', 
    'high_school_government_and_politics', 
    'college_chemistry', 
    'logical_fallacies', 
    'high_school_geography', 
    'elementary_mathematics', 
    'human_aging', 
    'college_mathematics', 
    'high_school_psychology', 
    'formal_logic', 
    'high_school_statistics', 
    'international_law', 
    'high_school_mathematics', 
    'high_school_computer_science', 
    'conceptual_physics', 
    'miscellaneous', 
    'high_school_chemistry', 
    'marketing', 
    'professional_law', 
    'management', 
    'college_physics', 
    'jurisprudence', 
    'world_religions', 
    'sociology', 
    'us_foreign_policy', 
    'high_school_macroeconomics', 
    'computer_security', 
    'moral_scenarios', 
    'moral_disputes', 
    'electrical_engineering', 
    'astronomy', 
    'college_biology'
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
        subject_name = subject_name,
        number = number,
        question = question['input'],
        choice_a = question['A'],
        choice_b = question['B'],
        choice_c = question['C'],
        choice_d = question['D'],
    )

def exam_mmlu(model, tokenizer: transformers.PreTrainedTokenizer, text):
    token_a = tokenizer.convert_tokens_to_ids("A")
    token_b = tokenizer.convert_tokens_to_ids("B")
    token_c = tokenizer.convert_tokens_to_ids("C")
    token_d = tokenizer.convert_tokens_to_ids("D")
    
    inputs = tokenizer([text], return_tensors='pt')
    # print(inputs.input_ids.shape)
    seq_len = inputs.input_ids.shape[-1]
    with torch.no_grad():
        output = model(**inputs).logits
    prob_a = output[0, -1, token_a].item()
    prob_b = output[0, -1, token_b].item()
    prob_c = output[0, -1, token_c].item()
    prob_d = output[0, -1, token_d].item()
    probs = [('A', prob_a), ('B', prob_b), ('C', prob_c), ('D', prob_d)]
    probs = list(sorted(probs, key=lambda x: x[1], reverse=True))
    return probs, seq_len

def evaluate_mmlu(args, model, tokenizer, subject_name):
    dataset = load_dataset('lukaemon/mmlu', subject_name)
    
    few_shots = []
    for question in dataset['train']:
        text = format_mmlu(question, len(few_shots) + 1, subject_name)
        choice = question['target']
        text += choice
        few_shots.append(text)
    for question in dataset['validation']:
        text = format_mmlu(question, len(few_shots) + 1, subject_name)
        choice = question['target']
        text += choice
        few_shots.append(text)
    
    results = []
    n_correct = 0
    seq_len_sum = 0
    for question in tqdm.tqdm(dataset['test'], dynamic_ncols=True, leave=False, desc=subject_name):
        text = format_mmlu(question, len(few_shots) + 1, subject_name)
        truth = question['target']
        text = "\n\n".join(few_shots + [text,])
        estimations, seq_len = exam_mmlu(model, tokenizer, text)
        estimation = estimations[0][0]
        correct = truth == estimation
        # print(truth, estimations, seq_len)
        if correct:
            n_correct += 1
        seq_len_sum += seq_len
        results.append({
            'text': text,
            'truth': truth,
            'estimations': estimations,
            'estimation': estimation,
            'correct': correct,
            'seq_len': seq_len,
        })
    
    accuracy = (n_correct / len(results)) * 100
    print(f'{subject_name} = {accuracy:.4f} %')
    print(f'avg_seq_len: {seq_len_sum / len(results)}')
    
    os.makedirs('./saves/llama_eval/mmlu/', exist_ok=True)
    json_path = f'./saves/llama_eval/mmlu/{subject_name}_{args.method}.json'
    with open(json_path, 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'k': args.k,
            'block_size': args.block_size,
            'dense_queries': args.dense_queries,
            'results': results,
        }, f, indent=2)
        print('dumped', json_path)

def job_mmlu(args, model, tokenizer, device):
    seed()
    
    evaluate_mmlu(args, model, tokenizer, 'anatomy')
    # for subjects in MMLU_SUBJECTS:
    #     evaluate_mmlu(args, model, tokenizer, subjects)