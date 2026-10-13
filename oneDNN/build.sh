set -x
source env.sh
export BD=${ROOT}/build
export ID=${ROOT}/install


if [ ! -d "src" ]; then
    git submodule add https://github.com/thegreatchaos/oneDNN.git src
fi
if [ ! -d "${ID}" ]; then
    mkdir ${ID}
fi
if [ ! -d "${BD}" ]; then
    mkdir ${BD} && cd ${BD}

    export CC=icx
    export CXX=icpx
    cmake   -DCMAKE_BUILD_TYPE=Release \
	    -DDNNL_GPU_RUNTIME=SYCL \
	    -DCMAKE_INSTALL_PREFIX=${ID} \
	    ../src
fi

cd ${BD}
make -j$(( $(nproc) - 1 ))
make install
