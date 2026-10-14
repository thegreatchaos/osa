#!/bin/bash
set -x
source csVars.sh
Q_TYPE="compressed-tensors"
if [ $# -eq 0 ]; then
    echo "a session name required"
    exit 1;
fi
dirRoot=${DIR_PREFIX}_$1
runSvr(){
    mkdir -p ${dirRoot}

    vllm serve \
    --port "${VLLM_SERVER_PORT}" --host 127.0.0.1 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 2048 \
    --max-model-len 2000 \
    --max-num-seqs 32 \
    --block-size 64 \
    --dtype float16 \
    --model "${M_PATH}" \
    --served-model-name "${M_PATH}" \
    --tensor-parallel-size 1 \
    --quantization "${Q_TYPE}" \
    --enforce-eager \
    --trust-remote-code 2>&1 | tee ${dirRoot}/vllm_serve.log
}
runSvr;
