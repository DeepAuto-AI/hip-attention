import cupy
import nvtx
import torch

if __name__ == "__main__":
    x = torch.randn((32, 1024, 512), device=0)

    @nvtx.annotate("torch")
    @torch.inference_mode
    def torch_median(x: torch.Tensor):
        # x.argsort(stable=False, dim=-1)
        # center = x.shape[-1] // 2
        # median = x.gather(index=indices[:, :, center:center+1], dim=-1)
        # return median
        # x.median(-1)
        x.median()

    @nvtx.annotate("cupy")
    @torch.inference_mode
    def cupy_median(c: cupy.ndarray):
        cupy.median(c, axis=-1)
        # cupy.argsort(c, axis=-1)
        # print(c)

    from cupy.cuda.memory import MemoryPointer, UnownedMemory

    x = x.view(-1, x.shape[-1])
    byte_size = x.numel() * x.element_size()
    mem = UnownedMemory(x.data_ptr(), byte_size, None, x.device.index)
    memptr = MemoryPointer(mem, 0)
    c = cupy.ndarray(
        x.size(),
        dtype=cupy.float32,
        memptr=memptr,
        strides=[s * x.element_size() for s in x.stride()],
    )

    print("warmup torch")
    for _ in range(10):
        torch_median(x)
    print("warmup cupy")
    for _ in range(10):
        cupy_median(c)

    print("start sampling cupy")
    SAMPLES = 100

    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    torch.cuda.synchronize()
    elapsed_cupy = 0
    for _ in range(SAMPLES):
        start.record()
        cupy_median(c)
        end.record()
        torch.cuda.synchronize()
        elapsed_cupy += start.elapsed_time(end)

    print("start sampling torch")

    torch.cuda.synchronize()
    elapsed_torch = 0
    for _ in range(SAMPLES):
        start.record()
        torch_median(x)
        end.record()
        torch.cuda.synchronize()
        elapsed_torch += start.elapsed_time(end)

    print("torch", elapsed_torch / SAMPLES, "cupy", elapsed_cupy / SAMPLES)
