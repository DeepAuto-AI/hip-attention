import os
import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

ddp_rank = 0
ddp_world_size = 1
ddp_disabled = False


def setup(rank, world_size, port):
    global ddp_rank, ddp_world_size

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    ddp_rank = rank
    ddp_world_size = world_size


def cleanup():
    dist.destroy_process_group()


def spawn(entry, args=(), n_gpus=None, join=True):
    """
    Entry function should be `entry(rank, world_size, ddp_port, *args)`
    """
    global ddp_disabled

    if n_gpus is None:
        n_gpus = 1024
    n_gpus = min(n_gpus, torch.cuda.device_count())
    port = random.randint(32000, 37000)
    if n_gpus == 1:
        print(f"DDP: No need to DDP, using single process")
        ddp_disabled = True
        entry(0, 1, port, *args)
    elif n_gpus > 1:
        print(f"DDP: Setup DDP with {n_gpus} devices")
        mp.spawn(
            entry,
            args=(
                n_gpus,
                port,
            )
            + tuple(args),
            nprocs=n_gpus,
            daemon=False,
            join=join,
        )
    else:
        raise Exception("no gpu")


def barrier():
    return dist.barrier()


def printable():
    global ddp_disabled, ddp_rank, ddp_world_size
    return ddp_world_size == 1 or (ddp_rank == 0) or ddp_disabled


def wrap_model(model, find_unused_paramters=False):
    global ddp_rank, ddp_world_size
    if ddp_world_size == 1:
        return MimicDDP(model)
    else:
        print("DDP: Model wrapped", ddp_rank, find_unused_paramters)
        return DDP(
            model, device_ids=[ddp_rank], find_unused_parameters=find_unused_paramters
        )


class MimicDDP(nn.Module):
    def __init__(self, module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
