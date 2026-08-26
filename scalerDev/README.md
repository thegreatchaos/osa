## 简介
以向特定版本的vLLM, xpu-kernels打patch的方式实现优化/bugfix/feature/..., 这些patch及相应的dockerfile/scripts/...放在[llm-scaler](https://github.com/intel/llm-scaler/)中


流程概要:
    1. 从[vLLM](https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu), [xpu-kernels](https://github.com/vllm-project/vllm-xpu-kernels)fork [vBase](https://github.com/intel-sandbox/llm-scaler-vllm-xpu), [kBase](https://github.com/analytics-zoo/vllm-xpu-kernels). 
    2. 在vBase & kBase上以独立的分支开发
    3. 开发后将改动以patch形式放到llm-scaler repo中
如果改动只涉及kBase/vBase我们只需要关心第二步. ESIMD开发需要将ESIMD kernels提交到llm-scaler, 并修改必要的vBase

