## VLLM 

# Targets
- To enable Qwen3.5/6 with MoE on PTL for 40 TPS, TTFT as small as possible

# Steps:

- Setup envs on bare metal for profiling to figure out the hotspot / gaps
```
bash setupEnv.sh 2>&1 | tee /tmp/setEnv.log # for setting up the envs on a bare metal
source env.sh #to init envs
```
- Further optimization
```
bash perf.sh #for profiling with vtune
```
