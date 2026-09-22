#!/bin/bash
I_LEN=1
O_LEN=1
MNBT=2048
M_PATH=/home/chaos/prjs/models/Qwen3.8-27B
if [ $# -eq 0 ]; then
    echo "a session name required"
    exit 1;
fi
UNITRACE_SESSION=${UNITRACE_SERVING_MODEL}$1
set -x
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unitrace --resume "${UNITRACE_SESSION}"
vllm bench serve \
	--model  "${M_PATH}"  \
	--dataset-name random \
	--random-input-len=${I_LEN} \
	--random-output-len=${O_LEN} \
	--ignore-eos  \
	--num-prompt 1 \
	--trust_remote_code \
	--request-rate inf  \
	--backend vllm \
	--host 127.0.0.1 \
	--port ${UNITRACE_PORT} 2>&1 | tee /home/chaos/tmp/${UNITRACE_SESSION}/vllm_bench.log
unitrace --stop "${UNITRACE_SESSION}"
