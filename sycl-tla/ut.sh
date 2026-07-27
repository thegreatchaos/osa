cd ${ROOT};
mkdir -p build 
cd build
CC=icx CXX=icpx cmake ${ROOT}/src -G Ninja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET="intel_gpu_ptl_u" 
ninja test_unit -j7
