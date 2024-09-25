export PYTHONPATH=.
export HIP_DISABLE_AUTOTUNE=1
export SA_BLOCK_BK=16
#export ENFORCE_EAGER=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL_NAME="vllm_llama3.1_8b_instruct"
TOKENIZER_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
FT_CKPT_PATH="$HOME/hip-ft-ckpts/meta-llama_Meta-Llama-3.1-8B-Instruct-default-hip_orig_mask-rp-32768-512-1/checkpoint-200"
MAX_TOKENS=2048

# Set DIVISOR to 1 if environment variable DIVISOR is unset
DIVISOR=${DIVISOR:-1}

SUFFIX=""
if [ "${DIVISOR}" -ne 1 ]; then
  echo "DIVISOR: ${DIVISOR}"
  SUFFIX="_div${DIVISOR}"
fi

if [ -z "${SKIP_FLASH}" ]; then
  echo "FLASH_ATTN"
  VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  python hip/main/model_eval.py --job long_booksum --model "${MODEL_NAME}" --stride 131072 --max_tokens "${MAX_TOKENS}" --method none --name longbs_llama31_none --overwrite
fi

if [ -z "${SKIP_BIGBIRD}" ]; then
  echo "BigBird"
  echo "DIVISOR: ${DIVISOR}"
  VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_PREFILL_BQ=64 HIP_PREFILL_BK=2 HIP_SW=$((1024 / DIVISOR)) HIP_NSINK=16 HIP_K=$((4096 / DIVISOR)) \
  HIP_USING_SNAP_KV=0 HIP_BK_AFTER_MASK=-1 HIP_RANDOM_MASK=1 \
  python hip/main/model_eval.py --job long_booksum --model "${MODEL_NAME}" --stride 131072 --max_tokens "${MAX_TOKENS}" --method hip --name "longbs_llama31_bigbird${SUFFIX}" --overwrite
fi

if [ -z "${SKIP_AVD}" ]; then
  echo "AVD Baseline"
  echo "DIVISOR: ${DIVISOR}"
  VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_PREFILL_BQ=64 HIP_PREFILL_BK=2 HIP_SW=$((4096 / DIVISOR)) HIP_NSINK=16 HIP_K=16 \
  HIP_USING_SNAP_KV=1 HIP_SNAP_KV_VERT_K=$((4096 / DIVISOR)) HIP_SNAP_KV_DIAG_K=$((4096 / DIVISOR)) HIP_BK_AFTER_MASK=-1 HIP_RANDOM_MASK=0 \
  python hip/main/model_eval.py --job long_booksum --model "${MODEL_NAME}" --stride 131072 --max_tokens "${MAX_TOKENS}" --method hip --name "longbs_llama31_avd${SUFFIX}" --overwrite
fi

if [ -z "${SKIP_HIP_PNP}" ]; then
  echo "HIP P&P"
  echo "DIVISOR: ${DIVISOR}"
  VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_PREFILL_BQ=64 HIP_PREFILL_BK=2 HIP_SW=$((1024 / DIVISOR)) HIP_NSINK=16 HIP_K=$((2048 / DIVISOR)) \
  HIP_USING_SNAP_KV=0 HIP_BK_AFTER_MASK=8 HIP_RANDOM_MASK=0 HIP_DECODE_ALWAYS_DENSE=0 \
  python hip/main/model_eval.py --job long_booksum --model "${MODEL_NAME}" --stride 131072 --max_tokens "${MAX_TOKENS}" --method hip --name "longbs_llama31_hip_pnp${SUFFIX}" --overwrite
fi

if [ -z "${SKIP_HIP_SNAPKV}" ]; then
  echo "HIP + SNAPKV P&P"
  echo "DIVISOR: ${DIVISOR}"
  SA_BLOCK_BK=2 \
  VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_PREFILL_BQ=64 HIP_PREFILL_BK=2 HIP_SW=$((1024 / DIVISOR)) HIP_NSINK=16 HIP_K=$((2048 / DIVISOR)) HIP_USING_SNAP_KV=1 \
  HIP_SNAP_KV_VERT_K=$((2048 / DIVISOR)) HIP_SNAP_KV_DIAG_K=$((1024 / DIVISOR)) HIP_BK_AFTER_MASK=16 HIP_RANDOM_MASK=0 HIP_DECODE_ALWAYS_DENSE=0 \
  python hip/main/model_eval.py --job long_booksum --model "${MODEL_NAME}" --stride 131072 --max_tokens "${MAX_TOKENS}" --method hip --name "longbs_llama31_hip_snapkv_pnp${SUFFIX}" --overwrite
fi

if [ -z "${SKIP_HIP_FT}" ]; then
  echo "HIP Fine-Tuned"
  echo "DIVISOR: ${DIVISOR}"
  VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_PREFILL_BQ=64 HIP_PREFILL_BK=2 HIP_SW=$((1024 / DIVISOR)) HIP_NSINK=16 HIP_K=$((2048 / DIVISOR)) \
  HIP_USING_SNAP_KV=0 HIP_BK_AFTER_MASK=8 HIP_RANDOM_MASK=0 HIP_DECODE_ALWAYS_DENSE=0 \
  python hip/main/model_eval.py --job long_booksum --model "vllm_${FT_CKPT_PATH}" --tokenizer-id "${TOKENIZER_ID}" --stride 131072 --max_tokens "${MAX_TOKENS}" --method hip --name "longbs_llama31_hip_ft${SUFFIX}" --overwrite
fi

if [ -z "${SKIP_BIGBIRD_8K}" ]; then
  echo "BigBird-TRUNC8K"
  echo "DIVISOR: ${DIVISOR}"
  VLLM_ATTENTION_BACKEND=HIP_ATTN HIP_PREFILL_BQ=64 HIP_PREFILL_BK=2 HIP_SW=$((1024 / DIVISOR)) HIP_NSINK=16 HIP_K=$((4096 / DIVISOR)) \
  HIP_USING_SNAP_KV=0 HIP_BK_AFTER_MASK=-1 HIP_RANDOM_MASK=1 \
  python hip/main/model_eval.py --job long_booksum --model "${MODEL_NAME}" --stride 131072 --max_tokens "${MAX_TOKENS}" --truncate-size 8192 --method hip --name "longbs_llama31_bigbird_trunc8k${SUFFIX}" --overwrite
fi
