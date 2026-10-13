source ../env.sh

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
set -x
perf(){
    if [ ${isPerf} -eq 1 ]; then
	common;
	vtune	-r perf_${TS} \
		-data-limit=0 \
		-collect gpu-hotspots \
		-- \
		benchdnn --matmul --mode=P --engine=gpu \
		     --dt=f16:u4:f16 \
		     --stag=ab \
		     --wtag=ba \
		     --dtag=ab \
		     --attr-scales=wei:per_ocic:f16:32x1 \
		     --attr-zero-points=wei:per_ocic:u4:32x1 \
		     --attr-fpmath=f16:true \
		     1x5120:5120x16384
	exit;
	unitrace -d -h -s -v \
		--chrome-call-logging \
		--chrome-kernel-logging \
		--chrome-device-logging \
		--chrome-dnn-logging \
		--chrome-itt-logging \
		--follow-child-process 1 \
		--teardown-on-signal 15 \
		--output-dir-path ${dirRoot} \
		python -m pytest tests/test_int4_gemm_w4a16_trace.py -v -s
    else
	#ONEDNN_JIT_DUMP=1 \
	    benchdnn --matmul --mode=P --engine=gpu \
		     --fix-times-per-prb=1500 \
		     --dt=f16:u4:f16 \
		     --stag=ab \
		     --wtag=ba \
		     --dtag=ab \
		     --attr-scales=wei:per_ocic:f16:32x1 \
		     --attr-zero-points=wei:per_ocic:u4:32x1 \
		     --attr-fpmath=f16:true \
		     1x5120:5120x16384
    fi
}
perf;
