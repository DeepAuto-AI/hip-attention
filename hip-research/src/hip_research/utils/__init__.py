import copy
import gc
import os
import random
import threading
from typing import Dict, List, TextIO, Tuple

import numpy as np
import torch

# from .profiler import Profiler


def indent_string(s):
    return ("  " + s).replace("\n", "\n  ")


def strify(d):
    newline = "\n"
    if isinstance(d, torch.Tensor):
        return f"Tensor({d.shape})"
    elif isinstance(d, (float, int)):
        return f"Num({d})"
    elif isinstance(d, (str)):
        return f"Str({d})"
    elif isinstance(d, (list, tuple)):
        return f'[\n{indent_string(f", {newline}".join([strify(i) for i in d]))}\n]({len(d)})'
    elif isinstance(d, dict):
        return f'{{\n{indent_string(f", {newline}".join([k + ": " + strify(i) for k, i in d.items()]))}\n}}({len(d)})'
    elif hasattr(d, "strify"):
        return d.strify()
    else:
        return f"{type(d)}"


cuda_copy_lock = None  # type: threading.RLock


def register_copy_lock(lock):
    global cuda_copy_lock
    assert cuda_copy_lock is None

    cuda_copy_lock = lock


def tensor_buffer_to(v, device):
    if isinstance(v, torch.Tensor):
        if not isinstance(device, torch.device) and not isinstance(device, torch.dtype):
            device = torch.device(device)
        op_type = "device" if isinstance(device, torch.device) else "dtype"

        if op_type == "device":
            if v.device == device:
                return v

            acquired = False
            if cuda_copy_lock is not None:
                cuda_copy_lock.acquire()
                acquired = True

            new_v = v.to(device, non_blocking=True)

            if acquired:
                cuda_copy_lock.release()

            return new_v
        elif op_type == "dtype":
            if v.dtype == device:
                return v
            return v.to(device)
        else:
            raise Exception()
    elif isinstance(v, list):
        return list([tensor_buffer_to(i, device) for i in v])
    elif isinstance(v, dict):
        return dict({k: tensor_buffer_to(vv, device) for k, vv in v.items()})
    elif isinstance(v, (float, int, str)):
        return v
    elif v is None:
        return v
    else:
        raise Exception(type(v))


def batch_to(batch, device):
    if isinstance(batch, dict):
        new_batch = {}
        for k, v in batch.items():
            new_batch[k] = tensor_buffer_to(v, device)
        return new_batch
    elif isinstance(batch, tuple) or isinstance(batch, list):
        new_batch = []
        for v in batch:
            new_batch.append(tensor_buffer_to(v, device))
        if isinstance(batch, tuple):
            new_batch = tuple(new_batch)
        return new_batch
    elif isinstance(batch, torch.Tensor):
        return tensor_buffer_to(batch, device)
    else:
        raise Exception(type(batch))


def get_device_name(device):
    name = torch.cuda.get_device_name(device=device)
    name = name.lower()
    name = name.replace("nvidia", "").replace("geforce", "").replace("rtx", "")
    name = name.strip()
    name = name.replace(" ", "_")
    return name


def model_hash(model):
    import hashlib

    flt = hashlib.shake_256()
    for name, param in sorted(model.named_parameters(), key=lambda x: x[0]):
        flt.update(name.encode())
        flt.update(
            param.data.view(-1)[: min(16, param.data.numel())].cpu().numpy().tobytes()
        )
    return flt.hexdigest(16)


class NanException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__("NaN occure")

        self.args = args


nan_check = True


def set_global_nan_check(v):
    global nan_check
    nan_check = v


def raise_if_nan(tensor):
    global nan_check
    if not nan_check:
        return tensor

    if torch.isinf(tensor).any():
        torch.save(tensor, "utils_inf.pth")
        print("Has INF in here raise_if_nan. check utils_inf.pth")
        raise NanException()

    if torch.isnan(tensor).any():
        # print('Has NAN in here', tensor)
        torch.save(tensor, "utils_nan.pth")
        print("Has NAN in here raise_if_nan. check utils_nan.pth")
        raise NanException()
    return tensor


import io
import pickle


def module_clone(module: torch.nn.Module, device=None):
    is_training = copy.deepcopy(module.training)
    if device is None:
        device = copy.deepcopy(get_module_device(module))

    # cloned_module = pickle.loads(pickle.dumps(module)) # type: torch.nn.Module
    buffer = io.BytesIO()
    try:
        torch.save(
            module, buffer, _use_new_zipfile_serialization=False, pickle_protocol=5
        )
        buffer.seek(0)
        cloned_module = torch.load(buffer, map_location="cpu")  # type: torch.nn.Module
    finally:
        buffer.close()
    # cloned_module = copy.deepcopy(module)

    cloned_module = cloned_module.train(is_training)
    if device is not None:
        cloned_module = cloned_module.to(device)
    return cloned_module


def human_readable(value, unit="B", unit_size=1024):
    if value < 0:
        return f"-{human_readable(-value)}"
    if value >= (unit_size**5):
        return f"{value / (unit_size**5):.1f} P{unit}"
    elif value >= (unit_size**4):
        return f"{value / (unit_size**4):.1f} T{unit}"
    elif value >= (unit_size**3):
        return f"{value / (unit_size**3):.1f} G{unit}"
    elif value >= (unit_size**2):
        return f"{value / (unit_size**2):.1f} M{unit}"
    elif value >= unit_size:
        return f"{value / unit_size:.1f} K{unit}"
    else:
        return f"{value:.1f} {unit}"


def compute_internal_fragmentation():
    # https://github.com/pytorch/pytorch/issues/29554
    snapshot = torch.cuda.memory_snapshot()
    return 1 - (
        sum(b["allocated_size"] for b in snapshot if b["allocated_size"] > 0)
        / sum(b["total_size"] for b in snapshot if b["allocated_size"] > 0)
    )


def compact_cuda_memory():
    """
    https://github.com/pytorch/pytorch/issues/31252
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    acquired = False
    if cuda_copy_lock is not None:
        cuda_copy_lock.acquire()
        acquired = True

    try:
        # wait for all computation is finished
        torch.cuda.synchronize()

        # Run a full garbage collect first so any dangling tensors are released
        gc.collect()

        # Then move all tensors to the CPU
        locations = {}  # type: Dict[torch.Tensor, str]
        cpu_device = torch.device("cpu")
        for obj in gc.get_objects():
            if not isinstance(obj, torch.Tensor):
                continue

            obj_device = obj.device
            if obj_device != cpu_device:
                locations[obj] = obj_device
                # obj.device = cpu_device
                obj.data = obj.data.to(cpu_device, non_blocking=True)
                assert obj.device != obj_device
                if (isinstance(obj, torch.nn.Parameter)) and obj.grad is not None:
                    obj.grad.data = obj.grad.data.to(cpu_device, non_blocking=True)

        torch.cuda.synchronize()
        gc.collect()
        # Now empty the cache to flush the allocator
        torch.cuda.empty_cache()

        # Finally move the tensors back to their associated GPUs
        for tensor, device in locations.items():
            # tensor.device = device
            tensor.data = tensor.data.to(device, non_blocking=True)
            assert tensor.data.device == tensor.device
            if (isinstance(tensor, torch.nn.Parameter)) and tensor.grad is not None:
                tensor.grad.data = tensor.grad.data.to(device, non_blocking=True)
                assert tensor.grad.device == tensor.data.device

        torch.cuda.synchronize()
    finally:
        if acquired:
            cuda_copy_lock.release()


def get_module_device(m: torch.nn.Module):
    dev = None
    for param in m.parameters():
        new_dev = param.device
        if dev is not None:
            assert dev == new_dev
        else:
            dev = new_dev
    return dev


warmup_finished = False


def warmup_torch_stream(job, warmup_steps=3, force_rewarmup=False):
    global warmup_finished
    if (not force_rewarmup) and warmup_finished:
        return
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup_steps):
            job()
    torch.cuda.current_stream().wait_stream(s)
    warmup_finished = True


# default_profiler = Profiler()
# def get_profiler():
#     return default_profiler


def copy_batch(from_batch, to_batch):
    if isinstance(from_batch, torch.Tensor):
        to_batch.copy_(from_batch)
    elif isinstance(from_batch, list):
        for i, item in enumerate(from_batch):
            copy_batch(from_batch[i], to_batch[i])
    elif isinstance(from_batch, dict):
        for i in from_batch.keys():
            copy_batch(from_batch[i], to_batch[i])
    elif isinstance(from_batch, bool):
        pass
    else:
        raise Exception(type(from_batch))


def unzip(lst, dim=0):
    return [i[dim] for i in lst]


import contextlib
import sys

import tqdm


class TqdmWriteDummyFile(object):
    file = None

    def __init__(self, file: TextIO):
        self.file = file

    def write(self, x):
        # # Avoid print() second call (useless \n)
        # if len(x.rstrip()) > 0:
        #     tqdm.tqdm.write(x, file=self.file)
        tqdm.tqdm.write(x, file=self.file, end="")

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


@contextlib.contextmanager
def using_tqdm_write():
    save_stdout = sys.stdout
    sys.stdout = TqdmWriteDummyFile(sys.stdout)
    yield
    sys.stdout = save_stdout


import multiprocessing as mp


def __query_available_devices(q):
    import torch

    num_gpus = torch.cuda.device_count()
    available_devices = []
    avail_mem = []
    for i in range(num_gpus):
        free_mem, total_mem = torch.cuda.mem_get_info(i)
        available = (free_mem / (total_mem + 1e-8)) > 0.5
        if available:
            available_devices.append(i)
        avail_mem.append(
            f"[{i}({available})]={free_mem / (total_mem + 1e-8) * 100:.1f}%"
        )
    print("QueryDevice: Available Memories,", *avail_mem)
    q.put(available_devices)


def query_available_devices() -> list[int]:
    q = mp.Queue()
    cuda_process = mp.Process(target=__query_available_devices, args=(q,), daemon=True)
    cuda_process.start()
    cuda_process.join()
    available_devices = q.get()
    q.close()
    return available_devices


class Metric:
    def __init__(self, method="moving_average", window_size=50):
        self.method = method
        self.window_size = window_size

        self.sum = {}
        self.count = {}
        self.buf = {}

    def update(self, x, name="", weight=1):
        if isinstance(x, torch.Tensor):
            x = x.item()

        if self.method == "mean":
            if not name in self.sum:
                self.sum[name] = 0
                self.count[name] = 0
            self.sum[name] += x * weight
            self.count[name] += weight
        elif self.method == "moving_average":
            buf = self.buf.get(name, [])
            buf.append(x)
            if len(buf) > self.window_size:
                buf.pop(0)
            self.buf[name] = buf
        return self.get(name)

    def get(self, name=""):
        if self.method == "mean":
            return self.sum[name] / self.count[name]
        elif self.method == "moving_average":
            return sum(self.buf[name]) / len(self.buf[name])
        raise Exception(self.method)

    def to_dict(self):
        r = {}
        for key in self.sum:
            r[key] = self.get(key)
        return r


def trace_referrers(obj, live_count=2):
    if live_count <= 0:
        return str(type(obj))
    parent = gc.get_referrers(obj)
    return f"[{[trace_referrers(p, live_count=live_count-1) for p in parent]}]({type(obj)})"


def get_all_allocated_tensors() -> List[Tuple[torch.Tensor, list]]:
    tensors = []
    for obj in gc.get_objects():
        try:
            if (
                isinstance(obj, torch.Tensor)
                and (not isinstance(obj, torch.nn.Parameter))
                and (
                    torch.is_tensor(obj)
                    or (hasattr(obj, "data") and torch.is_tensor(obj.data))
                )
            ):
                if obj.device != torch.device("cpu"):
                    obj = obj  # type: torch.Tensor
                    # referrers = gc.get_referrers(obj)
                    referrers = trace_referrers(obj)
                    tensors.append((obj, referrers))
        except OSError:
            pass
    return tensors


def setup_seaborn(
    label_fontsize=8,
    legend_fontsize=7,
    axes_label_fontsize=6,
    font_weight=500,
    axes_label_weight=600,
    axis_below=False,
):
    import seaborn as sns

    sns.set_theme(
        context="paper",
        style="whitegrid",
        palette="bright",
        font="Noto Sans, Sans Serif",
        font_scale=1.0,
        color_codes=True,
        rc={
            "axes.titlesize": str(label_fontsize),
            "font.weight": font_weight,
            "axes.labelweight": axes_label_weight,
            "axes.titleweight": "600",
            "legend.fontsize": str(legend_fontsize),
            "axes.grid.which": "both",
            "ytick.labelsize": str(axes_label_fontsize),
            "xtick.labelsize": str(axes_label_fontsize),
            "axes.labelsize": str(label_fontsize),
            "ytick.major.pad": "1.0",
            "xtick.major.pad": "1.0",
            "axes.axisbelow": axis_below,
        },
    )
