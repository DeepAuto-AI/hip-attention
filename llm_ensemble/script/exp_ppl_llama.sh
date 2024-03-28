
# llama13b_32k de
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=6 python -m timber.main.llama_eval --model llama13b_32k --stride 8192 --method timber --k 512 --block_size_q 32 --block_size_k 2 --job ppl --dense_queries 0

# ensemble
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 CUDA_VISIBLE_DEVICES=6 python -m timber.main.llama_eval --model llama13b_32k --stride 8192 --method timber --k 512 --block_size_q 32 --block_size_k 2 --job ppl --dense_queries 0 --ensemble