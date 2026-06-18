ifeq "${v}" "debug"
    OFLG=-O0 -g  #Debug始终无法build
    DBG_INSTRUMENTS=
else
    OFLG=-O3
    DBG_INSTRUMENTS=-fno-sycl-instrument-device-code #是否开启调试用的instruments. 如果要则保留此行, 否则下一行
endif

TS:=$(shell date +%Y%m%d%H%M%S)

INTEL_CFLG=-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED
INTEL_CFLG+=-std=c++17 -fPIE -fsycl-targets=spir64_gen -fsycl 
INTEL_CFLG+=${DBG_INSTRUMENTS}

INTEL_LFLG=-fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device ptl" #此行如果缺失-fsycl-targets=spir64_gen 则会导致No kernel named...
INTEL_LFLG+=-Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate 
INTEL_LFLG+=${DBG_INSTRUMENTS} -lsycl -lm -ldl -lpthread

### 参数
#ACTION=-DPREFILL
ACTION=-DDECODE
HEAD_DIM=-DHEAD_DIM=192

PARAMS=${ACTION} ${HEAD_DIM}

CFLG=${OFLG} -I${ROOT}/src/include -I${ROOT}/src/applications -I${ROOT}/src/tools/util/include/ ${INTEL_CFLG} -Isrc ${PARAMS}
LFLG=${INTEL_LFLG}

bin/test:bin/test.cpp.o
	mkdir -p bin;
	icpx $^ -o $@ ${LFLG}

bin/test.cpp.o:src/test.cpp
	mkdir -p bin;
	icpx -c ${CFLG} $^ -o $@
	#icpx -Xclang -ast-print -fsyntax-only ${CFLG} $^ > ~/tmp/instancilized_cpp/inst.cpp ##将模板实例化生成的代码约237MB, 编译器崩了, 可能有些截断. 以此来分析
clean:
	-rm -rf bin
run:bin/test
	./bin/test

perf:bin/test
	vtune -r /tmp/test_${TS} -data-limit=0 -collect gpu-hotspot ./bin/test
	cp src/* bin/test /tmp/test_${TS}/

