source env.sh
set -x
cd ${ROOT};
isPerf=0;
if [ $# -ne 0 ]; then
    isPerf=1
fi
TS=`date +%Y%m%d%H%M%S`
export VLLM_ENABLE_V1_MULTIPROCESSING=0
common(){
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches; #清楚page caches
    sudo sh -c "echo 0 > /proc/sys/kernel/yama/ptrace_scope"
    sudo echo "0" | sudo tee /proc/sys/dev/xe/observation_paranoid
    sudo echo "0" | sudo tee /proc/sys/kernel/perf_event_paranoid
    sudo echo "0" | sudo tee /proc/sys/kernel/kptr_restrict
    sudo echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
}

awqPerf(){
    if [ ${isPerf} -eq 0 ]; then
	echo -e "\033[31mRun without perf\033[0m"
	sudo sh -c "sync && echo 3 | tee /proc/sys/vm/drop_caches" && python qwen3827bAwqInt4.py 2>&1 | tee ~/tmp/xpu-kernels.log
    else
	echo -e "\033[31mStart profiing\033[0m"
	common;
	#export VLLM_XPU_ENABLE_XPU_GRAPH=1
	reportRoot="/tmp/qwen3827bAwqInt4GpuVtune_${TS}"

	mkdir -p ${reportRoot}
	vtune -r ${reportRoot} -data-limit=0 -collect gpu-hotspots -start-paused -- python qwen3827bAwqInt4.py 2>&1 | tee ${reportRoot}/vtune_collection.log
	#vtune -r ${reportRoot} -data-limit=0 -collect uarch-exploration -start-paused -- python qwen3827bAwqInt4.py 2>&1 | tee ${reportRoot}/vtune_collection.log
	cp qwen3827bAwqInt4.py ${reportRoot}
    fi
}
awqPerf;
