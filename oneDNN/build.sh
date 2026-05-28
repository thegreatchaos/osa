set -x
source env.sh
export BD=${ROOT}/tmpBuild
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
    cmake   -DCMAKE_BUILD_TYPE=Debug \
	    -DCMAKE_INSTALL_PREFIX=install \
	    -G Ninja
	    ../src
fi

cd ${BD}
make -j$(( $(nproc) - 1 ))
