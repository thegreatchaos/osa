ifeq "${v}" "rel"
    OFLG=-O3
    DBG_INSTRUMENTS=-fno-sycl-instrument-device-code #是否开启调试用的instruments. 如果要则保留此行, 否则下一行
else
    OFLG=-O0 -g
    DBG_INSTRUMENTS=
endif

TS:=$(shell date +%Y%m%d%H%M%S)

INTEL_CFLG=-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED
INTEL_CFLG+=-std=c++17 -fPIE -fsycl-targets=spir64_gen -fsycl 
INTEL_CFLG+=${DBG_INSTRUMENTS}

INTEL_LFLG=-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc" #此行如果缺失-fsycl-targets=spir64_gen 则会导致No kernel named...
INTEL_LFLG+=-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate 
INTEL_LFLG+=${DBG_INSTRUMENTS} -lsycl -lm -ldl -lpthread

CFLG=${OFLG} -I${ROOT}/src/include -I${ROOT}/src/tools/util/include/ ${INTEL_CFLG} -Isrc
LFLG=${INTEL_LFLG}

bin/test:bin/test.cpp.o
	mkdir -p bin;
	icpx -v $^ -o $@ ${LFLG}

bin/test.cpp.o:src/test.cpp
	mkdir -p bin;
	icpx -v -c ${CFLG} $^ -o $@

clean:
	-rm -rf bin
run:bin/test
	./bin/test

perf:bin/test
	vtune -r /tmp/test_${TS} -data-limit=0 -collect gpu-hotspot ./bin/test
	cp src/* bin/test /tmp/test_${TS}/
