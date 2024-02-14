# TimberAttention

CUDA_VISIBLE_DEVICES=0 python src/trainer/timber_trainer.py --batch_size 1 --gradient_accumulation_steps 2 --dataset wikitext2 --lora_r 512 --max_steps 10000 --block_size 4 --k 256

python src/trainer/timber_trainer.py --disable_kd --lora_r 512 --batch_size 1 --block_size 8 --k 512 --init_checkpoint ./saves/dev/llama32k-wikitext103-4096-block8-k512-epoch-00-step-8400.pth --dataset booksum --using_fsdp --max_steps 10000