source env.sh
set -x
cd ${ROOT};

TS=`date +%Y%m%d%H%M%S`


gemma4(){
    source env.sh
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches; #清楚page caches

    sudo echo "0" | sudo tee /proc/sys/dev/xe/observation_paranoid
    sudo echo "0" | sudo tee /proc/sys/kernel/perf_event_paranoid
    sudo echo "0" | sudo tee /proc/sys/kernel/kptr_restrict
    sudo echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

    export VLLM_XPU_ENABLE_XPU_GRAPH=1
    vtune -r /tmp/gemma4_${TS} -data-limit=0 -collect gpu-hotspots -start-paused -- python gemma4.py
}

qwen36(){
    source env.sh
    sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches; #清楚page caches

    sudo echo "0" | sudo tee /proc/sys/dev/xe/observation_paranoid
    sudo echo "0" | sudo tee /proc/sys/kernel/perf_event_paranoid
    sudo echo "0" | sudo tee /proc/sys/kernel/kptr_restrict
    sudo echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

    export VLLM_XPU_ENABLE_XPU_GRAPH=1
    vtune -r /tmp/qwen36a3b_${TS} -data-limit=0 -collect gpu-hotspots -start-paused -- python qwen36-35b-a3b.py
}




#gemma4;
qwen36;
