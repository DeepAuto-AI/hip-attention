## Eval for QWEN
```bash
python qwen_eval.py --method timber --job mmmu --k 512 --block_size_k 2 --block_size_q 16
```


## Eval for LLaVA

### References

1. llava using huggingface protocol ([link](https://github.com/haotian-liu/LLaVA/tree/main))

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
```

2. eval script for llava from MMMU official repo ([link](https://github.com/MMMU-Benchmark/MMMU/tree/main/eval#run-llava))
This code provided `llava-v1.5-13b` model's results.