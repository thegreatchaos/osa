source env.sh
set -x
cd ${ROOT};
isPerf=0;
if [ $# -ne 0 ]; then
    isPerf=1
fi
TS=`date +%Y%m%d%H%M%S`

common(){
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches; #清楚page caches
    sudo sh -c "echo 0 > /proc/sys/kernel/yama/ptrace_scope"
    sudo echo "0" | sudo tee /proc/sys/dev/xe/observation_paranoid
    sudo echo "0" | sudo tee /proc/sys/kernel/perf_event_paranoid
    sudo echo "0" | sudo tee /proc/sys/kernel/kptr_restrict
    sudo echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
}

qwen36(){
    if [ ${isPerf} -eq 0 ]; then
	echo -e "\033[31mRun without perf\033[0m"
	sudo sh -c "sync && echo 3 | tee /proc/sys/vm/drop_caches" && python qwen36-35b-a3b.py
    else
	echo -e "\033[31mStart profiing\033[0m"
	common;
	export VLLM_XPU_ENABLE_XPU_GRAPH=1
	vtune -r /tmp/qwen36a3b_uarch_${TS} -data-limit=0 -collect uarch-exploration -start-paused -- python qwen36-35b-a3b.py
	vtune -r /tmp/qwen36a3b_gpu_${TS} -data-limit=0 -collect gpu-hotspots -start-paused -- python qwen36-35b-a3b.py
	cp qwen36-35b-a3b.py /tmp/qwen36a3b_uarch_${TS}/
	cp qwen36-35b-a3b.py /tmp/qwen36a3b_gpu_${TS}/
    fi
}
qwen36;
