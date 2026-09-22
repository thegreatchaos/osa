#!/bin/bash
set -x
M_PATH=/home/chaos/prjs/models/Qwen3.8-27B
Q_TYPE="fp8"
if [ $# -eq 0 ]; then
    echo "a session name required"
    exit 1;
fi
UNITRACE_SESSION=${UNITRACE_SERVING_MODEL}$1
export VLLM_MQ_MAX_CHUNKS=4096
runSvr(){
    dirRoot="/home/chaos/tmp/${UNITRACE_SESSION}"
    mkdir -p ${dirRoot}

    #--mamba-ssm-cache-dtype float16 \  removed
    #unitrace -d -v -h -s \
    unitrace -v \
	--chrome-kernel-logging \
	--session "${UNITRACE_SESSION}" \
	--start-paused \
	--teardown-on-signal 15 \
	--follow-child-process 1 \
	--output-dir-path ${dirRoot} \
	--output-dir-path ${dirRoot} \
	vllm serve \
	--port "${UNITRACE_PORT}" --host 127.0.0.1 \
	--gpu-memory-utilization 0.9 \
	--max-num-batched-tokens 2048 \
	--max-model-len 2000 \
	--max-num-seqs 32 \
	--block-size 64 \
	--dtype float16 \
	--model "${M_PATH}" \
	--served-model-name "${M_PATH}" \
	--tensor-parallel-size 2 \
	--quantization "${Q_TYPE}" \
	--enforce-eager \
	--trust-remote-code 2>&1 | tee /home/chaos/tmp/${UNITRACE_SESSION}/vllm_serve.log
}
runSvrWithoutUnitrace(){
    dirRoot="/home/chaos/tmp/${UNITRACE_SESSION}"
    mkdir -p ${dirRoot}
    vllm serve \
	--port "${UNITRACE_PORT}" --host 127.0.0.1 \
	--gpu-memory-utilization 0.9 \
	--max-num-batched-tokens 2048 \
	--max-model-len 2000 \
	--max-num-seqs 32 \
	--block-size 64 \
	--dtype float16 \
	--model "${M_PATH}" \
	--served-model-name "${M_PATH}" \
	--tensor-parallel-size 2 \
	--quantization "${Q_TYPE}" \
	--enforce-eager \
	--trust-remote-code 2>&1 | tee /home/chaos/tmp/${UNITRACE_SESSION}/vllm_serve.log

}
runSvr;
#runSvrWithoutUnitrace;
