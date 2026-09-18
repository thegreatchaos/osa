#source env.sh
set -x
set -o pipefail
: "${ROOT:?run 'source env.sh' first}"
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
    echo "0" | sudo tee /proc/sys/dev/xe/observation_paranoid
    echo "0" | sudo tee /proc/sys/kernel/perf_event_paranoid
    echo "0" | sudo tee /proc/sys/kernel/kptr_restrict
    echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo 
}

qwen38(){
    if [ ${isPerf} -eq 0 ]; then
	echo -e "\033[31mRun without perf\033[0m"
	dirRoot=/home/chaos/tmp/qwen38_27B_base_${TS}
	mkdir -p ${dirRoot};
	common;
	python qwen38-27B.py 2>&1 | tee ${dirRoot}/run_${TS}.log
    else
	echo -e "\033[31mStart profiing\033[0m"
	common;
	dirRoot=/home/chaos/tmp/qwen38_27B_perf_${TS}
	mkdir -p ${dirRoot};
	unitrace    -d -h -s -v \
		    --chrome-call-logging \
		    --chrome-kernel-logging \
		    --chrome-device-logging \
		    --chrome-dnn-logging \
		    --chrome-itt-logging \
		    --start-paused \
		    --follow-child-process 1 \
		    --teardown-on-signal 15 \
		    --output-dir-path ${dirRoot} \
		    python qwen38-27B.py 2>&1 | tee ${dirRoot}/unitrace_${TS}.log
    fi
}
qwen38;
