source ~/prjs/osa/vllm/env.sh
source csVars.sh
set -x

batchSize=(1 2 3 4 5 6 7 8)
inputLen=(51200) # 153600 204800 256000 307200 358400 409600)

export OPENAI_API_KEY="EMPTY"
unset http_proxy
unset https_proxy
for i in ${inputLen[@]}; do
    for b in ${batchSize[@]}; do
	O_LEN=2000 #$(expr $MAX_MODEL_LEN - $i - 1000)
	echo "warmp for i= ${i} b = $b, o= ${O_LEN}"
	vllm bench serve \
	    --model "${M_PATH}" \
	    --dataset-name random \
	    --random-input-len="${i}" \
	    --random-output-len="${O_LEN}" \
	    --ignore-eos \
	    --num-prompts 2 \
	    --trust_remote_code \
	    --request-rate inf \
	    --backend vllm \
	    --host 127.0.0.1 \
	    --port "${VLLM_SERVER_PORT}" > /dev/null 2>&1

	vllm bench serve \
	    --model "${M_PATH}" \
	    --dataset-name random \
	    --random-input-len="${i}" \
	    --random-output-len="${O_LEN}" \
	    --ignore-eos \
	    --num-prompts ${b} \
	    --trust_remote_code \
	    --request-rate inf \
	    --backend vllm \
	    --host 127.0.0.1 \
	    --port "${VLLM_SERVER_PORT}" 2>&1 | tee ${DIRROOT}/bench_${KVCACHE_TYPE}_il=${i}_ol=${O_LEN}_b=${b}.log
    done
done


