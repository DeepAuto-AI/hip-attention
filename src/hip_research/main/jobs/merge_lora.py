import peft


def job_merge_lora(args, model: peft.PeftModelForCausalLM, tokenizer, device):
    assert args.output is not None

    print("Merging Lora weights")
    # Merge lora weights
    model = model.merge_and_unload()

    state_dict = model.state_dict()

    removed_cnt = 0
    for key in list(state_dict.keys()):
        if "tree_avgpool_scaler" in key:
            state_dict.pop(key)
            removed_cnt += 1
    print(f"Removed {removed_cnt} keys")

    print(f"Saving model to {args.output}")
    model.save_pretrained(args.output, state_dict=state_dict)
    tokenizer.save_pretrained(args.output)
