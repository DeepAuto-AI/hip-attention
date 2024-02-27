import peft


def job_merge_lora(args, model: peft.PeftModelForCausalLM, tokenizer, device):
    assert args.output is not None

    print('Merging Lora weights')
    # Merge lora weights
    model.merge_and_unload()

    print(f'Saving model to {args.output}')
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
