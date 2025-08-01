import dataclasses
import os
from typing import List, Optional

import torch

try:
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        model_parallel_is_initialized,
        tensor_model_parallel_all_gather,
    )

    SGLANG_DIST_AVAILABLE = True

except:
    SGLANG_DIST_AVAILABLE = False


def get_local_rank():
    if SGLANG_DIST_AVAILABLE:
        return (
            get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0
        )
    else:
        return 0


@dataclasses.dataclass
class CaptureEvents:
    start: torch.cuda.Event
    end: torch.cuda.Event
    handle: "capture"
    _elapsed: Optional[int] = None

    def elapsed(self):
        if self._elapsed is not None:
            return self._elapsed
        else:
            self.end.synchronize()
            self._elapsed = self.start.elapsed_time(self.end)
            return self._elapsed


class capture(object):
    buffers: List[CaptureEvents] = []
    call_depth: int = 0

    @classmethod
    def report(cls):
        last_elapsed_sum = {}
        last_depth = 0
        for depth, event in capture.buffers:
            if depth < last_depth:
                print(
                    "--" * last_depth,
                    f"[level {last_depth}] took {last_elapsed_sum.get(last_depth, 0)} ms",
                    sep="",
                )
                last_elapsed_sum[last_depth] = 0
            last_depth = depth

            elapsed = event.elapsed()

            if not depth in last_elapsed_sum:
                last_elapsed_sum[depth] = 0
            last_elapsed_sum[depth] += elapsed

            print(
                "--" * depth, f"{event.handle.callback} took {elapsed:.2f} ms", sep=""
            )

        if len(capture.buffers) > 0:
            allocated = torch.cuda.memory_allocated()
            print(f"{allocated / 1024 / 1024:.2f} MB allocated")

        capture.buffers.clear()

    @classmethod
    def add_event(cls, depth: int, event: CaptureEvents):
        capture.buffers.append((depth, event))
        while len(capture.buffers) > 32:
            capture.buffers.pop(0)

    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, ex_typ, ex_val, traceback):
        return True

    def __call__(self, *args, **kwargs):
        run_benchmark = (
            (not torch.cuda.is_current_stream_capturing())
            and (kwargs["q"].shape[1] > 1 if "q" in kwargs else True)
            and os.getenv("HIP_DEBUG_BENCH", "0") == "1"
            and os.getenv("HIP_DEBUG_CAPTURE_DECORATOR", "1") == "1"
            and (get_local_rank() == 0)
        )

        if run_benchmark:
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()

        my_call_depth = capture.call_depth
        capture.call_depth += 1
        ret = self.callback(*args, **kwargs)
        capture.call_depth -= 1

        if run_benchmark:
            end.record()

            capture.add_event(
                my_call_depth, CaptureEvents(handle=self, start=start, end=end)
            )

        return ret
