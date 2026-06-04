#!/bin/bash 
### bash setupEnv.sh 2>&1 | tee /tmp/setupEnv.log
export ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export KROOT=${ROOT}/kerns
export AROOT=${ROOT}/app
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -x

die(){ # $1: 死因
    echo -e "\033[41m$1\033[0m";
    exit;
}
oneapis(){ #安装特定版本的OneAPI & Vtune(or else vtune无法显示gpu hotspots details)
    mkdir -p ${ROOT}/deps;
    cd ${ROOT}/deps;
    if [ ! -f "multi-arc-bmg-offline-installer-26.18.8.2-combo.tar.xz" ]; then
	wget --no-check-certificate http://multi-arc-serving.intel.com/offline/bmg/26.18.8.2/multi-arc-bmg-offline-installer-26.18.8.2-combo.tar.xz
	tar -xf multi-arc-bmg-offline-installer-26.18.8.2-combo.tar.xz
	cd  multi-arc-bmg-offline-installer-26.18.8.2-combo && sudo bash installer.sh;
    fi
    cd ${ROOT}/deps;
    if [ ! -f "Intel_VTune_Profiler_2026.0.0_internal.tar.gz" ]; then
	wget --no-check-certificate  https://ubit-artifactory-ba.intel.com/artifactory/analyzerengineering-ba-local/Products/vtune/archive/2026.0.0/631955/linux/release/build/Intel_VTune_Profiler_2026.0.0_internal.tar.gz
	tar -xf Intel_VTune_Profiler_2026.0.0_internal.tar.gz
	sudo mv Intel_VTune_Profiler_2026.0.0_internal /opt/intel/oneapi/vtune
    fi
    echo -e "\033[41mReboot is required\033[0m"
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

gsp(){ # get source and patch

    cd ${ROOT}
    source ${ROOT}/env.sh

    if [ ! -d "kerns" ]; then
	git submodule add -f https://github.com/thegreatchaos/vllm-xpu-kernels.git kerns
    fi
    if [ ! -d "app" ]; then
	git submodule add -f https://github.com/intel-innersource/applications.ai.gpu.vllm-xpu app
    fi
}
doBuild(){
    source ${ROOT}/env.sh

    if [ "$1" = "app" ]; then
	#step 1, build the app first;
	# ref: https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#supported-features
	cd ${AROOT};
	pip install -v -r requirements/xpu.txt
	pip uninstall -y triton triton-xpu
	pip install triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
	pip install --no-build-isolation -e . -v
	echo -e "\033[41m须手动测试vllm能否正常运行\033[0m"

	#问题与解决办法:
	#1.  module 'triton' has no attribute 'next_power_of_2'
	#pip uninstall triton -y
	#pip install --force-reinstall triton-xpu==3.7.0 --extra-index-url https://download.pytorch.org/whl/xpu
	#2. subprocess.CalledProcessError: Command '['/opt/intel/oneapi/compiler/2025.3/bin/icpx', '/tmp/tmpckbr17d1/main.cpp', '-O3', '-shared', '-Wno-psabi', '-fPIC', '-lsycl', '-lze_loader', '-L/opt/intel/oneapi/compiler/2025.3/lib', '-L/home/chaos/prjs/osa/vllm/.env/lib/python3.12/site-packages/triton/backends/intel/lib', '-I/usr/local/include', '-I/opt/intel/oneapi/compiler/2025.3/include', '-I/opt/intel/oneapi/compiler/2025.3/include/sycl', '-I/home/chaos/prjs/osa/vllm/.env/lib/python3.12/site-packages/triton/backends/intel/include', '-I/tmp/tmpckbr17d1', '-I/usr/include/python3.12', '-o', '/tmp/tmpckbr17d1/spirv_utils.cpython-312-x86_64-linux-gnu.so', '-Wl,-rpath,/opt/intel/oneapi/compiler/2025.3/lib', '-fsycl']' returned non-zero exit status 1.
	#   -> 'Python.h' not found. root cause
	#sudo apt-get install -y --no-install-recommends python3.12-dev
	#3. 加载完模型后做infer/?时候gpu mem不够...
	#-> 另开一个窗口. 持续的sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches, 知道开始infer
	#4. DEVICE_LOST
    fi;
    
    if [ "$1" = "kerns" ]; then
	cd ${KROOT};
	pip install --no-build-isolation -e . -v
    fi

    #Step 2, build the kerns source.
}

oneapis;
pvi;
gsp;
doBuild "app";
doBuild "kerns"
