export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source /opt/intel/oneapi/setvars.sh --force
source ${ROOT}/.env/bin/activate
export VLLM_TARGET_DEVICE=xpu
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT="0"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="1"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRITON_CPU_BACKEND=0
export PS1='\033[32mvlm>\033[0m'
