export ROOT="$(cd $(dirname "$1") && pwd)"
source /opt/intel/oneapi/setvars.sh
source .env/bin/activate
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_TARGET_DEVICE=xpu
export PS1='>'
