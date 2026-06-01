export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
set -x
gs(){ #get source
    cd ${ROOT}
    if [ ! -d "src" ]; then
	git submodule add --force https://github.com/vllm-project/vllm-xpu-kernels.git src
    fi
    if [ ! -d "app.src" ]; then
	git submodule add --force https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu app.src
    fi
}
pei(){ #python env init
    cd ${ROOT}
    if [ ! -d ".env" ]; then
	python3.12 -m venv .env
	source env.sh
	pip install --upgrade pip
    fi;
    source env.sh
}
## for xpu kernels
kerns(){
    pip install -r ${ROOT}/src/requirements.txt
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
    cd ${ROOT}/app.src
    pip install -r requirements/xpu.txt
    pip install triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu #or else torch::nms* does not exist
    VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e . -v
}



gs;
pei;
app;
kerns;
