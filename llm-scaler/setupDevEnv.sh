export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export SROOT=${ROOT}/src
export WS=${ROOT}/ws #workspace for xpu-kernels & vllm
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_XPU_KERNELS_COMMIT=3cab97a
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

gsa(){ # Get source and apply patch
    mkdir -p ${WS}
    cd ${WS}
    if [ ! -d "vllm" ]; then
	git clone --depth 1 -b v0.21.0 https://github.com/vllm-project/vllm.git vllm
	cd vllm;
	git apply ${ROOT}/src/vllm/patches/vllm_for_multi_arc.patch;
    fi
    cd ${WS}/vllm
    pip install -r requirements/xpu.txt
    time pip install --no-build-isolation -e . --extra-index-url https://download.pytorch.org/whl/xpu

    cd ${WS}
    if [ ! -d "kerns" ]; then
	git clone https://github.com/vllm-project/vllm-xpu-kernels.git kerns;
	cd kerns;
	git checkout ${VLLM_XPU_KERNELS_COMMIT};
	git apply ${ROOT}/src/vllm/patches/vllm_xpu_kernels.patch
    fi
    cd ${WS}/kerns
    sed -i 's|^--extra-index-url=https://download.pytorch.org/whl/xpu|# --extra-index-url=https://download.pytorch.org/whl/xpu|' requirements.txt 
    sed -i '/^torch==/s/^/# /' requirements.txt 
    sed -i 's|^triton-xpu|# triton-xpu|' requirements.txt 
    sed -i 's|^transformers|# transformers|' requirements.txt 

    pip install -r requirements.txt
    echo -e "\033[41mStart building xpu-kernels\033[0m"
    time pip wheel --no-build-isolation --no-deps . -w /tmp/
    echo -e "\033[41mBuilt xpu-kernels\033[0m"
    pip install --no-deps --force-reinstall /tmp/vllm_xpu_kernels-*.whl
}

bck(){ # build custom esimd kernels
    cd ${ROOT}/src/vllm/custom-esimd-kernels-vllm
    TORCH_XPU_ARCH_LIST=ptl MAX_JOBS=1 python3.12 setup.py bdist_wheel
    pip install --no-deps --force-reinstall dist/*.whl
}

pvi;
gsa;
bck;

