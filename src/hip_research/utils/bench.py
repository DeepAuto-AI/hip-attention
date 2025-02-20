import gc
import time

import torch

from hip_attn.utils.benchmarking import get_bench


def bench(name, fn, t_warmup, t_sample, timeunit="ms", tracetree=False):
    sample_count = 0
    bench_sync = get_bench().synchronize
    bench_disabled = get_bench().disabled

    try:
        torch.cuda.synchronize()
        print(f"[{name}] warmup... ", end="", flush=True)
        t = time.time()
        while True:
            with torch.no_grad():
                fn()
            if time.time() - t > t_warmup:
                break
        if tracetree:
            get_bench().synchronize = True
            get_bench().disabled = False
            get_bench().reset_measures()
            get_bench().reset_temp_buffers()
            get_bench().reset_trace()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.max_memory_allocated()
        torch.cuda.synchronize()
        print("benchmarking", end="", flush=True)
        elapsed = 0
        t = time.time()
        last_report = time.time()
        while True:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                fn()
            end.record()
            torch.cuda.synchronize()
            elapsed += start.elapsed_time(end) / 1000

            sample_count += 1
            if time.time() - t > t_sample:
                break
            if time.time() - last_report > 0.5:
                last_report = time.time()
                print(".", end="", flush=True)
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() - start_mem
    except torch.cuda.OutOfMemoryError as ex:  # type: ignore
        mem = 0
        elapsed = 0
    interval = elapsed / (sample_count + 1e-8)
    if timeunit == "ms":
        print(
            f" done. sampled {sample_count}its. {interval*1000:.2f}ms/it {mem / 1024 / 1024:.2f} MB",
            flush=True,
        )
    elif timeunit == "us":
        print(
            f" done. sampled {sample_count}its. {interval*1000*1000:.2f}us/it {mem / 1024 / 1024:.2f} MB",
            flush=True,
        )
    else:
        raise Exception()
    if tracetree:
        msg = get_bench().format_tracetree()
        if len(msg) > 0:
            print(msg)
        get_bench().synchronize = bench_sync
        get_bench().disabled = bench_disabled
    return interval, mem
