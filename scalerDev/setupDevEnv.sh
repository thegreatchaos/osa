export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export SROOT=${ROOT}/src
export WS=${ROOT}/ws #workspace for xpu-kernels & vllm
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_XPU_KERNELS_COMMIT=3cab97a
export TORCH_XPU_ARCH_LIST=${TORCH_XPU_ARCH_LIST:-bmg,ptl}
export sBaseVer="vllm-0.26.0-b1"
export vBaseVer="v0.26.0"
export kBaseCommitId="a6929869587fa6c40dbe393eedc96eeab076bfbb"
set -x
die(){ # $1: 死因
    echo -e "\033[41m$1\033[0m";
    exit;
}
pvi(){ # python venv init
    cd ${ROOT}
    if [ ! -f "/usr/bin/python3.12" ]; then
	die "python3.12 ONLY."
    fi

    if [ ! -d ".env" ]; then
	python3.12 -m venv .env
	source ${ROOT}/env.sh
	pip install --upgrade pip
	pip install ittapi
    fi;
    source ${ROOT}/env.sh
}

gsb(){ # Get source and apply patches
    mkdir -p ${WS}

    ####不需要再向vBase, kBase中打patch了, 因为这两个不是从官方而是从自己的repo拿的
    cd ${WS}
    if [ ! -d "sBase" ]; then
	git clone -b ${sBaseVer} https://github.com/intel/llm-scaler.git sBase
    fi
    cd ${WS}
    if [ ! -d "vBase" ]; then
	git clone -b ${vBaseVer} https://github.com/vllm-project/vllm.git vBase
	cd ${WS}/vBase
	git apply --reject ${WS}/sBase/vllm/patches/vllm_for_multi_arc.patch
    fi
    cd ${WS}
    if [ ! -d "kBase" ]; then
	git clone https://github.com/analytics-zoo/vllm-xpu-kernels kBase
	cd ${WS}/kBase
	git checkout ${kBaseCommitId}
	git apply --reject ${WS}/sBase/vllm/patches/vllm_xpu_kernels.patch
    fi
}

bv(){ # build vllm
    cd ${WS}/vBase
    pip install -r requirements/xpu.txt
    pip uninstall -y triton triton-xpu
    pip install --force-reinstall triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
    pip install --no-build-isolation -e . --extra-index-url https://download.pytorch.org/whl/xpu

    pip uninstall triton -y
    pip install --force-reinstall triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
}
bk(){ #build xpu-kernels
    cd ${WS}/kBase

    pip install -r requirements.txt
    echo -e "\033[41mStart building xpu-kernels\033[0m"
    time pip install --no-build-isolation -e . -v
    echo -e "\033[41mBuilt xpu-kernels\033[0m"
}
bc(){ #build custom-esimd
    cd ${WS}/sBase/vllm/custom-esimd-kernels-vllm
    pip install -q wheel
    echo -e "\033[41mStart building esimd-kernels (AOT: ${TORCH_XPU_ARCH_LIST})\033[0m"
    time python setup.py bdist_wheel || die "esimd-kernels build failed"
    pip install --force-reinstall --no-deps "$(ls -t dist/*.whl | head -1)"
    echo -e "\033[41mBuilt esimd-kernels\033[0m"
}

pvi;
gsb;
bv
bk
bc

