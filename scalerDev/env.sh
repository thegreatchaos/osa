export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/vtune/vtune-vars.sh --force
source ${ROOT}/.env/bin/activate
#export ZE_AFFINITY_MASK=1 #0 for B60, 1 for B70.................. NOT working, nrigther this and CUDA_VISIBILE...
#export CUDA_VISIBLE_DEVICES="1"
export VLLM_TARGET_DEVICE=xpu
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT="0"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
export PS1='\033[32msd>\033[0m'
