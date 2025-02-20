#!/bin/bash

echo MODEL=${MODEL}
echo METHOD=${METHOD}

EVAL_TOKENS=16777216

for seq in 131072 65536 32768 16384 8192
do
count=$((EVAL_TOKENS / seq))

echo "--------------------------------------------------------------------------------"
echo -e " PG19 ${MODEL}\t${METHOD}\tT=${seq}"
echo "--------------------------------------------------------------------------------"

if [ "$METHOD" == "hip" ]; then
echo run hip
python hip/main/model_eval.py --method hip --model llama3.1_8b --stride $seq --dataset pg19 --block_size_q 64 --block_stride_q 2 --block_size_k 2 --block_stride_k 1 --k 512 --overwrite
elif [ "${METHOD}" == "fa2" ]; then
echo run fa2
python hip/main/model_eval.py --method fa2 --model llama3.1_8b --stride $seq --dataset pg19 --overwrite
elif [ "${METHOD}" == "streaming_llm" ]; then
echo run streaming llm
python hip/main/model_eval.py --method streaming_llm --model llama3.1_8b --stride $seq --dataset pg19 --overwrite
elif [ "${METHOD}" == "hyper_attention" ]; then
echo run hyper attention
if [ "${DENSE_LAYERS}" == "" ]; then
DENSE_LAYERS=3
fi
python hip/main/model_eval.py --method hyper_attention --model llama3.1_8b --stride $seq --dataset pg19 --overwrite --dense_layers $DENSE_LAYERS
elif [ "${METHOD}" == "bigbird" ]; then
echo run bigbird
HIP_RANDOM_MASK=1 python hip/main/model_eval.py --method hip --model llama3.1_8b --stride $seq --dataset pg19 --block_size_q 64 --block_stride_q 2 --block_size_k 2 --block_stride_k 1 --k 512 --overwrite
else
echo "*******************"
echo " UNKNOWN METHOD \"${METHOD}\", please set METHOD and MODEL env-var"
echo "*******************"
break
fi

done
