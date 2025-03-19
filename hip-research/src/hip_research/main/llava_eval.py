from hip_research.main.eval_args import ArgsType, eval_args
from hip_research.main.jobs.mmmu import job_mmmu
from hip_research.utils import seed
from llava.mm_utils import get_model_name_from_path

from hip_attn.models.llava.builder import load_pretrained_model

"""
        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=prompt, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "\nUSER: What's the content of the image?\nASSISTANT: The image features a stop sign on a street corner"
        ```"""


def load_llava(args: ArgsType):
    assert args.model in ["llava"]

    device = "cuda:0"

    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)

    # llava_config = LlavaConfig.from_pretrained(args.model)
    # llava_config._attn_implementation = llava_config.attn_implementation = 'sdpa'
    # model = LlavaForConditionalGeneration.from_pretrained(args.model,
    #                                                       config=llava_config,)
    # processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        moel_path=args.model,
        model_base=None,
        model_name=get_model_name_from_path(args.model),
    )

    # model = QWenLMHeadModel.from_pretrained(
    #     "Qwen/Qwen-VL",
    #     device_map="auto",
    #     trust_remote_code=True,
    #     torch_dtype=torch.float16,
    #     load_in_4bit=True,
    #     quantization_config=BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.float16,
    #         llm_int8_skip_modules=['visual']
    #     ),
    #     fp16=True,
    # ).eval()

    for m in model.modules():
        if hasattr(m, "attention_method"):
            m.attention_method = args.method
            m.tree_k = args.k
            m.tree_block_size_k = args.block_size_k
            m.tree_block_size_q = args.block_size_q
            m.tree_dense_queries = args.dense_queries

    return tokenizer, model, image_processor


def main():
    args = eval_args(default_model="lava-hf/llava-1.5-7b-hf", default_job="mmmu")

    tokenizer, model, image_processor = load_llava(args)

    if args.job == "mmmu":
        job_mmmu(args, model, tokenizer, image_processor)
    else:
        raise Exception()


if __name__ == "__main__":
    seed()
    main()
