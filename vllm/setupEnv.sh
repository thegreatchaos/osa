#!/bin/bash 
### 参考::: https://github.com/intel/llm-scaler/blob/main/vllm/docker/Dockerfile
### bash setupEnv.sh 2>&1 | tee /tmp/setupEnv.log
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -x

oneapis(){ #安装特定版本的OneAPI & Vtune(or else vtune无法显示gpu hotspots details)
    mkdir -p ${ROOT}/deps;
    cd ${ROOT}/deps;
    if [ ! -f "multi-arc-bmg-offline-installer-26.18.8.2-combo.tar.xz" ]; then
	wget http://multi-arc-serving.intel.com/offline/bmg/26.18.8.2/multi-arc-bmg-offline-installer-26.18.8.2-combo.tar.xz
    fi
    if [ ! -f "Intel_VTune_Profiler_2026.0.0_internal.tar.gz" ]; then
	wget https://ubit-artifactory-ba.intel.com/artifactory/analyzerengineering-ba-local/Products/vtune/archive/2026.0.0/631955/linux/release/build/Intel_VTune_Profiler_2026.0.0_internal.tar.gz
    fi
    cd ${ROOT}/deps;
    tar -xf multi-arc-bmg-offline-installer-26.18.8.2-combo.tar.xz
    cd  multi-arc-bmg-offline-installer-26.18.8.2-combo/ubuntu-24.04.3-server/oneapi/main && sudo dpkg -i *.deb
    cd ${ROOT}/deps;
    tar -xf Intel_VTune_Profiler_2026.0.0_internal.tar.gz
    sudo mv Intel_VTune_Profiler_2026.0.0_internal /opt/intel/oneapi/vtune
}

pvi(){ # python venv init
    if [ ! -f "/usr/bin/python3.12" ]; then
	echo -e "\033[41mpython 3.12 required\033[0m";
	exit;
    fi

    python3.12 -m venv .env

    source ${ROOT}/env.sh

    pip install --upgrade pip
    pip install bigdl-core==2.4.0b2
    #pip install accelerate hf_transfer 'modelscope!=1.15.0'
    pip install librosa soundfile decord
    pip install transformers==5.8.0
    pip install ijson
}

gsp(){ # get source and patch

    source ${ROOT}/env.sh

    if [ ! -d "scaler" ]; then
	git clone https://github.com/intel/llm-scaler.git scaler
    fi
    if [ ! -d "vllm" ]; then
	git clone -b v0.14.0 https://github.com/vllm-project/vllm.git vllm
    fi

    if [ ! -d "MinerU" ]; then
	git clone -b release-2.6.2 https://github.com/opendatalab/MinerU.git;
    fi

    if [ ! -d "vllm-xpu-kernels" ]; then
	git clone https://github.com/vllm-project/vllm-xpu-kernels.git vllm-xpu-kernels;
    fi

    cd ${ROOT}/vllm-xpu-kernels;
    git checkout c968ba9
    git apply ${ROOT}/scaler/patches/vllm_xpu_kernels.patch;
    sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# --extra-index-url=https://download.pytorch.org/whl/xpu|' requirements.txt
    sed -i '/^torch==/s/^/# /' requirements.txt
    sed -i 's|^triton-xpu|# triton-xpu|' requirements.txt
    sed -i 's|^transformers|# transformers|' requirements.txt



    cd ${ROOT}/vllm && git apply ${ROOT}/scaler/patches/vllm_for_multi_arc.patch 

    cd ${ROOT}/MinerU;
    sed -i 's/kwargs.get("max_concurrency", 100)/kwargs.get("max_concurrency", 200)/' mineru/backend/vlm/vlm_analyze.py;
    sed -i 's/kwargs.get("http_timeout", 600)/kwargs.get("http_timeout", 1200)/' mineru/backend/vlm/vlm_analyze.py

}
doBuild(){
    source ${ROOT}/env.sh
    cd ${ROOT}/vllm;
    pip install -r requirements/xpu.txt;
    pip install arctic-inference==0.1.1;
    export CPATH=/opt/intel/oneapi/dpcpp-ct/2025.2/include/:${CPATH}
    pip install --no-build-isolation -e . -v

    cd ${ROOT}/scaler/vllm/custom-esimd-kernels-vllm;
    TORCH_XPU_ARCH_LIST=bmg-g21 MAX_JOBS=1 python setup.py bdist_wheel && pip install dist/*.whl --no-deps 

    cd ${ROOT}/MinerU;
    pip install -e .[core]


    cd ${ROOT}/vllm-xpu-kernels;
    pip install -r requirements.txt;
    pip install --no-build-isolation -e . -v
    pip uninstall triton triton-xpu -y;
    pip install triton-xpu==3.6.0 --extra-index-url=https://download.pytorch.org/whl/test/xpu
}

#oneapis;
#pvi;
#gsp;
#doBuild;
### TODO: install ubuntu 24.04.4 server for this PTL 




###
