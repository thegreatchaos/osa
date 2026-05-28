set -x
gs(){ #get source
    if [ ! -d "src" ]; then
	git submodule add --force https://github.com/vllm-project/vllm-xpu-kernels.git src
	cd src;
	git checkout v0.1.7
    fi
    if [ ! -d "app.src" ]; then
	git submodule add --force https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu app.src
	cd app.src
	git checkout 58625a191b2367dba56386065a35e9599a56c20f
    fi
}
pei(){ #python env init
    if [ ! -d ".env" ]; then
	python3.12 -m venv .env
    fi;
    source env.sh
    if [ ! -d ".env" ]; then
	pip install --upgrade pip
    fi
}
xpuDeps(){
    pip install -r ${ROOT}/src/requirements.txt
}

## for xpu kernels
kerns(){
    gs;
    pei;
    xpuDeps;
    export VLLM_TARGET_DEVICE="xpu"
    export CMAKE_BUILD_TYPE="Release" #在ptl 64GB + 即便128GB swap上也无法build debug信息的
    export BUILD_SYCL_TLA_KERNELS="ON"
    export VLLM_ENABLE_XE2="ON"
    export VLLM_ENABLE_XE_DEFAULT="ON"
    export VERBOSE="1"
    cd ${ROOT}/src
    pip install --no-build-isolation -e . -v
}
#XpuKerns;


## for https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu
app(){
    gs;
    cd ${ROOT}/app.src
    pip install -r requirements/xpu.txt
    VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
    #pip uninstall vllm-xpu-kernels
    pip install triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
}




pei;
app;
kerns;
