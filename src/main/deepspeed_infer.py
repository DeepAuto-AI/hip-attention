import mii


pipe = mii.pipeline("meta-llama/Llama-2-7b-hf")
response = pipe(["DeepSpeed is", "Seattle is"], max_new_tokens=128)
print(response)
