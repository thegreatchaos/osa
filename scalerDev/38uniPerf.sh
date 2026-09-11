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

#证据留存: 脚本/版本/commit/环境, 事后比对用
meta(){
    dst=$1
    cp qwen38-27B.py env.sh $0 ${dst}/
    {
	echo "date_utc=`date -u +%Y-%m-%dT%H:%M:%SZ`"
	echo "hostname=`hostname`"
	echo "working_directory=${PWD}"
	echo "no_turbo=`cat /sys/devices/system/cpu/intel_pstate/no_turbo`"
	echo "scaler_commit=`git rev-parse HEAD`"
	echo "vllm_commit=`git -C ${ROOT}/ws/vBase rev-parse HEAD 2>/dev/null`"
	echo "unitrace_version=`unitrace --version 2>&1 | head -1`"
    } > ${dst}/metadata.env
    env | LC_ALL=C sort | grep -E '^(ZE_|ONEAPI|SYCL|VLLM|PYTHONPATH=|LD_LIBRARY_PATH=|OMP_|MKL_|CCL_|CUDA_)' > ${dst}/environment.txt
}

validate(){
    dst=$1; log=$2; rc=0
    if ! ls ${dst}/*.json >/dev/null 2>&1; then
	echo -e "\033[31m[FAIL] 没有生成chrome trace json\033[0m"; rc=1
    elif find ${dst} -maxdepth 1 -name '*.json' -size -4k | grep -q .; then
	echo -e "\033[31m[FAIL] trace json为空或被截断\033[0m"; rc=1
    elif ! grep -q -m1 "Infer-" ${dst}/*.json; then
	echo -e "\033[31m[FAIL] trace里没有ITT marker Infer-*, 采集很可能一直处于paused\033[0m"; rc=1
    fi
    grep -q "=== Device Timing Summary ===" ${log} || { echo -e "\033[31m[FAIL] 日志里没有device timing汇总\033[0m"; rc=1; }
    if [ ${rc} -eq 0 ]; then
	echo -e "\033[32m[OK] trace可用: ${dst}\033[0m"
    fi
    return ${rc}
}

qwen38(){
    if [ ${isPerf} -eq 0 ]; then
	echo -e "\033[31mRun without perf\033[0m"
	dirRoot=/home/chaos/tmp/qwen38_27B_base_${TS}
	mkdir -p ${dirRoot};
	common;
	python qwen38-27B.py 2>&1 | tee ${dirRoot}/run_${TS}.log
	meta ${dirRoot};
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
	meta ${dirRoot};
	validate ${dirRoot} ${dirRoot}/unitrace_${TS}.log;
    fi
}
qwen38;
