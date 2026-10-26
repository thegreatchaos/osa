isRelease="Release"
if [ $# -ne 0 ]; then
    isRelease="Debug"
fi
echo ${isRelease}
export ROOT="$(cd $(dirname "$1") && pwd)"
source /opt/intel/oneapi/setvars.sh
export LD_LIBRARY_PATH=${ROOT}/install_${isRelease}/lib:${LD_LIBRARY_PATH}   #顺序别错, 不然会加载OneAPI里面的libdnnl*.so
export PATH=${PATH}:${ROOT}/install_${isRelease}
#export ONEDNN_VERBOSE=1
if [ ${isRelease} == "Release" ]; then
    export PS1="\033[32m1dnn>\033[0m"
else
    export PS1="\033[32m1dnn\033[31mDbg>\033[0m"
fi
