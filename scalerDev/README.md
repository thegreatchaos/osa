## 简介
baremental env setup
```
bash setupDevEnv.sh
```
Docker环境
```
1. 先pull或者从其他机器拿到这几个image:
    intel/llm-scaler-vllm:0.26.0-b2      
    intel/omix:0.1.0-devel-ubuntu24.04           
    ubuntu:22.04 
2. docker build -f ws/vllm/docker/Dockerfile.dev
3. bash buildUnitraceImage.sh
4. run:
    bash runDocker.sh
    进入image之后: cd /model, bash perf or bash perf jfkdaljfkdlas
```
