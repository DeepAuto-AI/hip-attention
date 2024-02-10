# TimberAttention

w   p   k   s   ppl
64  32  256 2   16.67

32  32  256 2   16.75

64  32  128 2   18.72

64  16  256 2   19.94
64  64  256 2   12.52
64  128 256 2   8.975

64  32  256 1.5 55.97 <-- seems bug
64  32  256 3   34.73 <-- seems bug
64  32  256 4   13.41

64  16  256 4   17.77
64  64  256 4   9.714

w   p   k   s   ppl     latency(4k)
64  32  256 4   13.41   11.54
64  64  256 4   9.714   8.868
64  32  256 2   16.67   2.146
64  64  256 2   12.52   2.441
64  128 256 2   9.013   3.397 <-- definatly go to this
torch                   9.093

64
128
256
512
1024
2048
4096

7*256

CUDA_VISIBLE_DEVICES=0 python src/trainer/timber_trainer.py --batch_size 1 --gradient_accumulation_steps 2 --dataset wikitext2 --lora_r 512 --max_steps 10000 --block_size 4 --k 256

python src/trainer/timber_trainer.py --disable_kd --lora_r 512 --batch_size 1 --block_size 8 --k 512 --init_checkpoint ./saves/dev/llama32k-wikitext103-4096-block8-k512-epoch-00-step-8400.pth --dataset booksum --using_fsdp --max_steps 10000