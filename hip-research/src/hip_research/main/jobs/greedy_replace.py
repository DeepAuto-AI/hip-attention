from hip_research.main.jobs import ppl


def job_greedy_replace(args, model, tokenizer, device):
    num_layers = len(model.model.layers)
    dense_layers = list(range(num_layers))

    args.overwrite = True
    num_dense_layers = args.dense_layers

    while len(dense_layers) > num_dense_layers:
        min_ppl = 99999
        min_dense_layers = None
        for i_target in range(num_layers):
            if i_target not in dense_layers:
                continue

            new_dense_layer = dense_layers.copy()
            new_dense_layer.remove(i_target)
            for m in model.modules():
                if hasattr(m, "attention_method"):
                    m.tree_dense_layers = new_dense_layer

            new_ppl = ppl.job_ppl(args, model, tokenizer, device, quite=True)
            if new_ppl < min_ppl:
                min_ppl = new_ppl
                min_dense_layers = new_dense_layer
            print(new_ppl, new_dense_layer)
        dense_layers = min_dense_layers

    print("-" * 80)
    print("final", dense_layers)
    print("-" * 80)

    args.count = -1

    for m in model.modules():
        if hasattr(m, "attention_method"):
            m.tree_dense_layers = dense_layers

    ppl.job_ppl(args, model, tokenizer, device)
