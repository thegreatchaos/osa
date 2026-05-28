source env.sh
set -x
cd ${ROOT}/app.src;
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_WORKER_MULTIPROC_METHOD=spawn  \
    python3 -m vllm.entrypoints.openai.api_server \
    --model /home/chaos/prjs/models/gemma4e4b \
    --enforce-eager --port 8000 --host 0.0.0.0  \
    --trust-remote-code  \
    --gpu-memory-util=0.4 \
    --enable-prefix-caching \
    --max-num-batched-tokens=8192 \
    --max-model-len=10240 \
    --block-size 64 \
    --quantization fp8
