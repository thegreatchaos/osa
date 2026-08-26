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

gsb(){ # Get source and build developement install
    mkdir -p ${WS}
    cd ${WS}
    if [ ! -d "vBase" ]; then
	git submodule add https://github.com/intel-sandbox/llm-scaler-vllm-xpu vBase
    fi
    cd ${WS}/vBase
    pip install -r requirements/xpu.txt
    pip uninstall -y triton triton-xpu
    pip install --force-reinstall triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
    pip install --no-build-isolation -e . --extra-index-url https://download.pytorch.org/whl/xpu

    pip uninstall triton -y
    pip install --force-reinstall triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu

    cd ${WS}
    if [ ! -d "kBase" ]; then
	git submodule add https://github.com/analytics-zoo/vllm-xpu-kernels kBase
    fi
    cd ${WS}/kBase

    pip install -r requirements.txt
    echo -e "\033[41mStart building xpu-kernels\033[0m"
    time pip install --no-build-isolation -e . -v
    echo -e "\033[41mBuilt xpu-kernels\033[0m"
}

pvi;
gsb;

