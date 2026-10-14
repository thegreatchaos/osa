#!/bin/bash
source csVars.sh
I_LEN=1024
O_LEN=512
MNBT=2048
if [ $# -eq 0 ]; then
    echo "a session name required"
    exit 1;
fi
set -x
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
dirRoot=${DIR_PREFIX}_$1
# ---------- warmup ----------
echo ">>> warmup start"
vllm bench serve \
    --model "${M_PATH}" \
    --dataset-name random \
    --random-input-len="${I_LEN}" \
    --random-output-len="${O_LEN}" \
    --ignore-eos \
    --num-prompts 2 \
    --trust_remote_code \
    --request-rate inf \
    --backend vllm \
    --host 127.0.0.1 \
    --port "${VLLM_SERVER_PORT}" > /dev/null 2>&1
echo ">>> warmup done"

# ---------- bench ----------
echo ">>> bench start"
vllm bench serve \
    --model "${M_PATH}" \
    --dataset-name random \
    --random-input-len="${I_LEN}" \
    --random-output-len="${O_LEN}" \
    --ignore-eos \
    --num-prompts 1 \
    --trust_remote_code \
    --request-rate inf \
    --backend vllm \
    --host 127.0.0.1 \
    --port "${VLLM_SERVER_PORT}" 2>&1 | tee "${dirRoot}/vllm_bench.log"
echo ">>> bench done"
