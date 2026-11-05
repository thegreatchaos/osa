source ~/prjs/osa/vllm/env.sh
set -x 

Q_TYPE="compressed-tensors"
cacheType=""
source csVars.sh
if [ "${KVCACHE_TYPE}" = "fp8" ]; then
    cacheType="--kv-cache-dtype fp8"
    echo -e "\033[41mFP8\033[0m"
else
    echo -e "\033[41mFP16\033[0m"
fi

vllm serve \
    --port "${VLLM_SERVER_PORT}" --host 127.0.0.1 \
    --gpu-memory-utilization 0.9 \
    --no-enable-prefix-caching \
    --max-model-len ${MAX_MODEL_LEN} \
    --dtype half \
    --model "${M_PATH}" \
    --served-model-name "${M_PATH}" \
    --tensor-parallel-size 1 \
    ${cacheType} \
    --quantization "${Q_TYPE}" \
    --enforce-eager \
    --trust-remote-code 2>&1 | tee ${DIRROOT}/vllm_serve_${KVCACHE_TYPE}.log
