source env.sh
set -x
cd ${ROOT};

TS=`date +%Y%m%d%H%M%S`

common(){
    source env.sh
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches; #清楚page caches
    sudo sh -c "echo 0 > /proc/sys/kernel/yama/ptrace_scope"
    sudo echo "0" | sudo tee /proc/sys/dev/xe/observation_paranoid
    sudo echo "0" | sudo tee /proc/sys/kernel/perf_event_paranoid
    sudo echo "0" | sudo tee /proc/sys/kernel/kptr_restrict
    sudo echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
}

gemma4(){
    common;
    export VLLM_XPU_ENABLE_XPU_GRAPH=1
    vtune -r /tmp/gemma4_${TS} -data-limit=0 -collect gpu-hotspots -start-paused -- python gemma4.py
}

qwen36(){
    common;
    export VLLM_XPU_ENABLE_XPU_GRAPH=1
    vtune -r /tmp/qwen36a3b_gpu_${TS} -data-limit=0 -collect gpu-hotspots -start-paused -- python qwen36-35b-a3b.py
    vtune -r /tmp/qwen36a3b_uarch_${TS} -data-limit=0 -collect uarch-exploration -start-paused -- python qwen36-35b-a3b.py #for exploding the deadlocks
}




#gemma4;
qwen36;
