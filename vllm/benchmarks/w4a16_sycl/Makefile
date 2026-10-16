TLA_ROOT ?= $(abspath ../../../sycl-tla)
CXX = /opt/intel/oneapi/compiler/latest/bin/icpx
DEVICE ?= pvc
CPPFLAGS += -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED
CPPFLAGS += -I$(TLA_ROOT)/src/include -I$(TLA_ROOT)/src/tools/util/include
CXXFLAGS += -O3 -std=c++17 -fsycl -fsycl-targets=spir64_gen -fno-sycl-instrument-device-code
LDFLAGS += -Xsycl-target-backend=spir64_gen "-device $(DEVICE)"
LDFLAGS += -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate
LDFLAGS += -Xs "-options \"-igc_opts 'allowDecompose2DBlockFuncs=0'\""

.PHONY: all run
all: w4a16_bench
w4a16_bench: w4a16_bench.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
run: w4a16_bench
	./w4a16_bench $(ARGS)
