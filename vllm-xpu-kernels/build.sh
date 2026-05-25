source env.sh
set -x
mkdir -p ${ROOT}/tmpBuild && cd ${ROOT}/tmpBuild

cmake	--trace-expand \
	-DVLLM_PYTHON_EXECUTABLE=${ROOT}/.env/bin/python3.12\
	-DVLLM_PYTHON_PATH=/usr/lib/python3.12:/usr/lib/python3.12/lib-dynload:${ROOT}/.env/lib/python3.12/site-packages \
	-DCMAKE_TOOLCHAIN_FILE=${ROOT}/src/cmake/toolchain.cmake\
	-DCMAKE_C_COMPILER=${CMPLR_ROOT}/bin/icx \
	-DCMAKE_CXX_COMPILER=${CMPLR_ROOT}/bin/icpx \
	-DVLLM_TARGET_DEVICE=xpu \
	-G "Unix Makefiles" \
	-DCMAKE_BUILD_TYPE=Debug \
	${ROOT}/src
#make -j
