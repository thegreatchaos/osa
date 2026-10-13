export ROOT="$(cd $(dirname "$1") && pwd)"
source /opt/intel/oneapi/setvars.sh
export LD_LIBRARY_PATH=${ROOT}/install/lib:${LD_LIBRARY_PATH}   #顺序别错, 不然会加载OneAPI里面的libdnnl*.so
export PATH=${PATH}:${ROOT}/install
export ONEDNN_VERBOSE=1
export PS1="\033[32m1DNN>\033[0m"
