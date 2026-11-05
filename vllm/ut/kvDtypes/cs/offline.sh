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
ITER=10
for i in 512 102400; do
    for b in 1 2 3 4 5 6 7 8; do
	vllm bench latency \
	    --gpu-memory-utilization 0.9 \
	    --no-enable-prefix-caching \
	    --tensor-parallel-size 1 \
	    --model "${M_PATH}" \
	    --input-len ${i} \
	    --output-len 1000 \
	    --batch-size ${b} \
	    --num-iters ${ITER} \
	    --dtype half \
	    --max-model-len ${MAX_MODEL_LEN} \
	    --quantization "${Q_TYPE}" \
	    --enforce-eager \
	    --trust-remote-code \
	    ${cacheType} \
	    2>&1 | tee ${DIRROOT}/vllm_latency_${KVCACHE_TYPE}_bs${BATCH_SIZE}_input${i}.log
    done
done
