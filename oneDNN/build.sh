set -x
source env.sh
isRelease="Release"
if [ $# -ne 0 ]; then
    isRelease="Debug"
fi
export BD=${ROOT}/build_${isRelease}
export ID=${ROOT}/install_${isRelease}


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
    cmake   -DCMAKE_BUILD_TYPE=${isRelease} \
	    -DDNNL_GPU_RUNTIME=SYCL \
	    -DCMAKE_INSTALL_PREFIX=${ID} \
	    ../src
fi

cd ${BD}
make -j$(( $(nproc) - 1 ))
make install
cmake --build . --target benchdnn -j$(nproc)
cp tests/benchdnn/benchdnn ${ID}/
