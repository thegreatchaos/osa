source env.sh
set -x
cd ${ROOT};
isPerf=0;
if [ $# -ne 0 ]; then
    isPerf=1
fi
TS=`date +%Y%m%d%H%M%S`
export VLLM_ENABLE_V1_MULTIPROCESSING=0
qwen38(){
    echo -e "\033[31mStart profiing\033[0m"
    common;
    #export VLLM_XPU_ENABLE_XPU_GRAPH=1
    reportRoot="/home/chaos/tmp/torchProfile_qwen38-27b"
    mkdir -p ${reportRoot}
    python qwen38-27B-torchProfile.py 2>&1 | tee ${reportRoot}/torchProfiler.log
    cp qwen38-27B-torchProfile.py ${reportRoot}/
    cp 38TorchProfiler.sh ${reportRoot}/
}
qwen38;
