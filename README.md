## Open Source Arena

# 1 directory for 1 project

|目录|解释|
|:----|:----|
|vllm-xpu-kernels|xpu kernel enable/optimization for vllm|
|oneDNN|1 lib for all DNN|

目录结构
```
dirName
    |__src # The reporsitory added into this project as a submodule
    |__tmpBuild # for building the src
    |__install #Into which the files installed after building
    |__env.sh
```
