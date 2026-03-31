import time

import torch


class BenchmarkRegion:
    def __init__(self, benchmark: "Benchmark", name: str) -> None:
        self.benchmark = benchmark
        self.name = name

        self.parent = None
        self.children = []

    def __enter__(self):
        if self.benchmark.disabled:
            return
        # if self.benchmark.synchronize: torch.cuda.synchronize()
        self.t = time.time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.start.record()

        if self.benchmark.tracking_callstack:
            if self.benchmark.current_region_context is None:
                self.benchmark.current_region_context = self
            else:
                self.parent = self.benchmark.current_region_context
                self.parent.children.append(self)
                self.benchmark.current_region_context = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.benchmark.disabled:
            return

        self.end = torch.cuda.Event(enable_timing=True)
        if self.benchmark.synchronize:
            self.end.record()

            def measure():
                torch.cuda.synchronize()
                return self.start.elapsed_time(self.end) / 1000

            self.elapsed = measure
        else:
            self.elapsed = time.time() - self.t
        self.benchmark.add_data(self.name, self.elapsed)

        if self.benchmark.tracking_callstack:
            if self.parent is None:
                self.benchmark.tracking_callstack = False
                self.benchmark.current_region_context = None
                self.benchmark.traced_callstack = self
            else:
                self.benchmark.current_region_context = self.parent

    def format_tree(self, indent=0):
        spaces = "  " * indent
        messages = [f"{spaces}> {self.name}"]
        for child in self.children:
            messages.append(child.format_tree(indent + 1))
        return "\n".join(messages)


class BenchmarkMemRegion:
    def __init__(self, benchmark: "Benchmark", name: str) -> None:
        self.benchmark = benchmark
        self.name = name

    def __enter__(self):
        if self.benchmark.disabled:
            return

        # if self.benchmark.synchronize: torch.cuda.synchronize()
        self.t = torch.cuda.memory_allocated()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.benchmark.disabled:
            return

        # if self.benchmark.synchronize: torch.cuda.synchronize()
        self.t = torch.cuda.memory_allocated() - self.t
        # print(self.name, self.t // 1024)
        # self.benchmark.add_data(self.name, self.t)


class Benchmark:
    current_region_context: BenchmarkRegion
    traced_callstack: BenchmarkRegion

    def __init__(self):
        self.synchronize = True
        self.disabled = True
        self.activate_temp_buffers = False
        self.buffers = {}
        self.data = {}

        self.tracking_callstack = True
        self.current_region_context = None
        self.traced_callstack = None

    def add_data(self, name, t):
        count, sum = self.data.get(name, (0, 0))
        if isinstance(t, (int, float, torch.Tensor)):
            self.data[name] = (count + 1, sum + t)
        else:
            if sum == 0:
                sum = []
            self.data[name] = (count + 1, sum + [t])

    def reset_trace(self):
        self.tracking_callstack = True
        self.current_region_context = None
        self.traced_callstack = None

    def reset_measures(self):
        self.data = {}

    def region(self, name):
        return BenchmarkRegion(benchmark=self, name=name)

    def mem_region(self, name):
        return BenchmarkMemRegion(benchmark=self, name=name)

    def todict(self):
        data = {}
        for key, (c, s) in self.data.items():
            if isinstance(s, list):
                s = [i() for i in s]
                s = sum(s)
            data[key] = s / (c + 1e-10)
        return data

    def register_temp_buffer(self, name, v, lazy=None):
        if not self.activate_temp_buffers:
            return
        buffer = self.buffers.get(name, [])
        if (v is None) and (lazy is not None):
            v = lazy()
        buffer.append(v)
        self.buffers[name] = buffer

    def get_temp_buffer(self, name, index=-1):
        return self.buffers[name][index]

    def reset_temp_buffers(self):
        self.buffers = {}

    def format_tracetree(self):
        data = self.todict()
        root = self.traced_callstack
        if root is None:
            return ""

        total_time = data[root.name]

        def format_tree_percent(item, indent=0):
            spaces = ""
            if indent == 1:
                spaces = "╰─"
            elif indent > 1:
                spaces = "  " * (indent - 1) + "╰─"
            messages = [
                f"{spaces}> {item.name} ({data[item.name] * 1000:.2f} ms, {data[item.name] / total_time * 100:.2f}%)"
            ]
            for child in item.children:
                messages.append(format_tree_percent(child, indent + 1))
            return "\n".join(messages)

        return format_tree_percent(root)


BENCHMARK = Benchmark()


def get_bench() -> Benchmark:
    global BENCHMARK
    return BENCHMARK
