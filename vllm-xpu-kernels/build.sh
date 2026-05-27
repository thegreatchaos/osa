source env.sh
set -x

cfg1(){
    mkdir -p ${ROOT}/tmpBuild && cd ${ROOT}/tmpBuild

    cmake   -DVLLM_PYTHON_EXECUTABLE=${ROOT}/.env/bin/python3.12\
	    -DVLLM_PYTHON_PATH=/usr/lib/python3.12:/usr/lib/python3.12/lib-dynload:${ROOT}/.env/lib/python3.12/site-packages \
	    -DCMAKE_TOOLCHAIN_FILE=${ROOT}/src/cmake/toolchain.cmake\
	    -DCMAKE_C_COMPILER=${CMPLR_ROOT}/bin/icx \
	    -DCMAKE_CXX_COMPILER=${CMPLR_ROOT}/bin/icpx \
	    -DVLLM_TARGET_DEVICE=xpu \
	    -DBUILD_SYCL_TLA_KERNELS=OFF \
	    -DVLLM_ENABLE_XE2=ON\
	    -DVLLM_ENABLE_XE_DEFAULT=ON\
	    -G Ninja \
	    -DCMAKE_BUILD_TYPE=Debug \
	    ${ROOT}/src
	    #--trace-expand \
}
#编译Debug版本在ptl上64GMem, 32GB swap会崩
cfg2(){
    ##ENVS
    export VLLM_TARGET_DEVICE="xpu"
    export CMAKE_BUILD_TYPE="Release"
    export BUILD_SYCL_TLA_KERNELS="ON"
    export VLLM_ENABLE_XE2="ON"
    export VLLM_ENABLE_XE_DEFAULT="ON"
    export VERBOSE="1"
    cd ${ROOT}/src
    pip install --no-build-isolation -e . -v
}

cfg2;
