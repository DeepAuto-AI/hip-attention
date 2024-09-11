import torch
import datasets
import transformers
from torch.utils.data import Dataset


class LongBookSumDataset(Dataset):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                 min_seq_len=32768, max_seq_len=128000,
                 split="train", for_eval=False, need_tokenization=True):
        self.for_eval = for_eval
        self.need_tokenization = need_tokenization

        cache_dir = f'./cache/long_booksum_{split}_{tokenizer.name_or_path.replace("/", "_")}_{min_seq_len}_{max_seq_len}'
        if need_tokenization:
            cache_dir += "_tokenized"

        try:
            self.dataset = datasets.load_from_disk(cache_dir)
        except FileNotFoundError:
            print("Loading dataset")
            if split == "validation+test":
                self.dataset = datasets.concatenate_datasets([
                    datasets.load_dataset('kmfoda/booksum', split='validation'),
                    datasets.load_dataset('kmfoda/booksum', split='test')
                ])
            else:
                self.dataset = datasets.load_dataset('kmfoda/booksum', split=split)

            def tokenize(item):
                chapter = item['chapter']
                prompt = (f"Summarize the following text in about 300 words:\n\n{chapter}"
                          f"\n\nThe summary of the previously given text is the following:\n")
                prompt_ids = [tokenizer.bos_token_id] + tokenizer(
                    prompt,
                    add_special_tokens=False,
                )['input_ids']
                completion_ids = tokenizer(
                    item['summary_text'],
                    add_special_tokens=False,
                )['input_ids'] + [tokenizer.eos_token_id]
                return {
                    'prompt_ids': prompt_ids,
                    'completion_ids': completion_ids,
                    'total_length': len(prompt_ids) + len(completion_ids),
                }

            self.dataset = self.dataset.map(tokenize)
            self.dataset = self.dataset.filter(lambda x: min_seq_len <= x['total_length'] <= max_seq_len)

            def untokenize(item):
                prompt = tokenizer.decode(item['prompt_ids'], skip_special_tokens=True)
                completion = tokenizer.decode(item['completion_ids'], skip_special_tokens=True)
                return {
                    'prompt': prompt,
                    'completion': completion,
                }

            if not self.need_tokenization:
                self.dataset = self.dataset.map(untokenize)

            # Cache dataset
            print(f"Caching dataset to {cache_dir}")
            self.dataset.save_to_disk(cache_dir)

        print("Number of examples:", len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.for_eval:
            if self.need_tokenization:
                prompt_ids = self.dataset[idx]['prompt_ids']
                completion_ids = self.dataset[idx]['completion_ids']
                return prompt_ids, completion_ids
            else:
                prompt = self.dataset[idx]['prompt']
                completion = self.dataset[idx]['completion']
                return prompt, completion
        else:
            prompt_ids = self.dataset[idx]['prompt_ids']
            completion_ids = self.dataset[idx]['completion_ids']
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
            return input_ids[:-1], labels[1:]


if __name__ == '__main__':
    tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B')
    print(tokenizer.is_fast)
    dataset = LongBookSumDataset(tokenizer, for_eval=True, split="validation+test", need_tokenization=False)
    prompt, output = dataset[0]
    print("Prompt: ", len(prompt), "Output: ", len(output))
    print("'''" + prompt + "'''")
    print("="*80 + "\n\n\n")
    print("'''" + output + "'''")
    #print("'''" + tokenizer.decode(prompt, skip_special_tokens=False) + "'''")
    #print("="*80 + "\n\n\n")
    #print("'''" + tokenizer.decode(output, skip_special_tokens=False) + "'''")

    #dataset = LongBookSumDataset(tokenizer, for_eval=False, split="validation+test")
    #inputs, labels = dataset[0]
    #print("inputs: ", len(inputs), "labels: ", len(labels))
    #labels = [x if x != -100 else 0 for x in labels]
    #print("'''" + tokenizer.decode(inputs, skip_special_tokens=False) + "'''")
    #print("="*80 + "\n\n\n")
    #print("'''" + tokenizer.decode(labels, skip_special_tokens=False) + "'''")
