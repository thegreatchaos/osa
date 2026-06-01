export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source /opt/intel/oneapi/setvars.sh
source ${ROOT}/.env/bin/activate
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TARGET_DEVICE=xpu
export PS1='>'
