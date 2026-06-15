OFLG=-O0 -g
OFLG=-O3

INTEL_CFLG=-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DDPCPP_SYCL_TARGET=ptl -fsycl-targets=spir64_gen -fsycl -Xs "-device ptl" -v -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier
INTEL_LFLG=-lsycl

CFLG=${OFLG} -I${ROOT}/src/include -I${ROOT}/src/tools/util/include/ ${INTEL_CFLG} -Isrc
LFLG=${INTEL_LFLG}

bin/test:src/test.cpp
	mkdir -p bin;
	icpx ${CFLG} $^ -o $@ ${LFLG}

clean:
	-rm -rf bin
run:bin/test
	./bin/test

