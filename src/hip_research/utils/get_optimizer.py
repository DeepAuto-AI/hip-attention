import torch
from torch import nn, optim


def get_optimizer(
    model: nn.Module,
    optimizer_type: str = "AdamW",
    lr: float = 1e-4,
    weight_decay: float = 1e-3,
    no_decay_keywords=[],
):
    param_optimizer = list(model.named_parameters())
    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "BatchNorm1d.weight",
        "BatchNorm1d.bias",
        "BatchNorm1d",
        "bnorm",
    ]
    if no_decay_keywords is not None and len(no_decay_keywords) > 0:
        no_decay += no_decay_keywords
    params = [
        {
            "params": [
                p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # print('nodecay', params[1])

    kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
    }

    if optimizer_type == "AdamW":
        optim_cls = optim.AdamW
    elif optimizer_type == "Adam":
        optim_cls = optim.Adam
    else:
        raise Exception()

    return optim_cls(params, **kwargs)
