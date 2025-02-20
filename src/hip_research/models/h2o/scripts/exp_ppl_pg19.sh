#!/bin/bash
MODEL="llama3.1_8b"

echo MODEL=${MODEL}
echo METHOD=${METHOD}

EVAL_TOKENS=16777216

for seq in 16384 32768 65536 131072 # 8192
do
count=$((EVAL_TOKENS / seq))

echo "--------------------------------------------------------------------------------"
echo -e " PG19 ${MODEL}\t${METHOD}\tT=${seq}"
echo "--------------------------------------------------------------------------------"

elif [ "${METHOD}" == "h2o" ]; then
echo run h2o
python -m hip.main.model_eval --method h2o --model llama3.1_8b --stride $seq --dataset pg19 --overwrite --count $count
elif [ "${METHOD}" == "h2o_stream" ]; then
echo run h2o stream
python -m hip.main.model_eval --method h2o_stream --model llama3.1_8b --stride $seq --dataset pg19 --overwrite --streaming --count $count
else
echo "*******************"
echo " UNKNOWN METHOD \"${METHOD}\", please set METHOD and MODEL env-var"
echo "*******************"
break
fi

done
