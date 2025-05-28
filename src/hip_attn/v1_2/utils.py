import torch
import os

try:
    from sglang.srt.distributed import (
        get_tensor_model_parallel_world_size,
        get_tensor_model_parallel_rank,
        tensor_model_parallel_all_gather,
        model_parallel_is_initialized,
    )

    SGLANG_DIST_AVAILABLE = True
    
except:
    SGLANG_DIST_AVAILABLE = False

def get_local_rank():
    if SGLANG_DIST_AVAILABLE:
        return get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0
    else:
        return 0

class capture(object):

    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, ex_typ, ex_val, traceback):
        return True

    def __call__(self, *args, **kwargs):
        run_benchmark = (
            (not torch.cuda.is_current_stream_capturing()) and
            (kwargs['q'].shape[1] > 1 if 'q' in kwargs else True) and
            os.getenv('HIP_DEBUG_BENCH', '0') == '1' and
            (get_local_rank() == 0)
        )

        if run_benchmark:
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            # hip_args = kwargs['args']
            # hip_args.rope_cos = hip_args.rope_cos.to(torch.bfloat16)
            # hip_args.rope_sin = hip_args.rope_sin.to(torch.bfloat16)
            # hip_args.disable_flashdecode = True
            # hip_args.online_update_cache = False
            # hip_args.v_hidden_dim = 128
            # hip_args.rope_is_neox_style = True

            start.record()
        
        ret = self.callback(*args, **kwargs)

        if run_benchmark:
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            # print(args[0].dtype)
            # print(kwargs['args'].pretty())
            print(f'{self.callback} took {elapsed:.2f} ms')
        
        return ret