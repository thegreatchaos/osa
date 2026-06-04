export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/vtune/vtune-vars.sh --force
source ${ROOT}/.env/bin/activate
export VLLM_TARGET_DEVICE=xpu
export VLLM_XPU_ENABLE_XPU_GRAPH=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PS1='\033[32mvlm>\033[0m'
