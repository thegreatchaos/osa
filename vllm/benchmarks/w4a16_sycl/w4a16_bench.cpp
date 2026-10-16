/***************************************************************************************************
 * Copyright (C) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// W4A16 benchmark using sycl-tla's fused dequantization mainloop.
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>
using namespace cute;

void check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess)
    throw std::runtime_error(cutlassGetStatusString(status));
}

int main(int argc, char** argv) {
  try {
    int m=1, n=16384, k=5120, iterations=1500, warmup=10;
    bool verify=true;
    for (int i=1; i<argc; ++i) {
      std::string arg=argv[i];
      if (arg=="--help") {
        std::cout << "Usage: w4a16_bench [--m M] [--n N] [--k K] "
                     "[--iterations I] [--warmup W] [--skip-check]\n"
                     "Also accepts --name=value. K must be a multiple of 32; N of 8.\n";
        return 0;
      }
      if (arg=="--skip-check") { verify=false; continue; }
      auto eq=arg.find('=');
      std::string name=arg.substr(0,eq), value;
      if (eq!=std::string::npos) value=arg.substr(eq+1);
      else if (++i<argc) value=argv[i];
      else throw std::runtime_error("Missing value for " + name);
      size_t used=0;
      int v=std::stoi(value,&used);
      if (used!=value.size()) throw std::runtime_error("Invalid integer: " + value);
      if (name=="--m") m=v;
      else if (name=="--n") n=v;
      else if (name=="--k") k=v;
      else if (name=="--iterations") iterations=v;
      else if (name=="--warmup") warmup=v;
      else throw std::runtime_error("Unknown option: " + name);
    }
    if (m<=0 || n<=0 || k<=0 || k%32 || n%8 || iterations<=0 || warmup<0)
      throw std::runtime_error("Require M,N,K>0, K%32=0, N%8=0, iterations>0, warmup>=0");
    const auto device = compat::get_default_queue().get_device();
    std::cout << "SYCL device: " << device.get_info<sycl::info::device::name>()
              << "\nVendor: " << device.get_info<sycl::info::device::vendor>()
              << "\nPlatform: " << device.get_platform().get_info<sycl::info::platform::name>()
              << "\nDriver: " << device.get_info<sycl::info::device::driver_version>()
              << "\nCompute units: " << device.get_info<sycl::info::device::max_compute_units>()
              << "\nGlobal memory: "
              << device.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024)
              << " MiB\n" << std::flush;
  using ElementAccumulator = float;      // <- data type of accumulator
  using ElementComputeEpilogue = float;  // <- data type of epilogue operations
  using ElementInputA = uint4_t;         // <- data type of elements in input matrix A
  using ElementInputB = half_t;          // <- data type of elements in input matrix B
  using ElementOutput = half_t;          // <- data type of elements in output matrix D

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using ElementZero = half_t;
  using ElementScale = half_t;

  using StrideScale = cute::Stride<_1, int64_t, int64_t>;
  using StrideZero = StrideScale;

  using GmemTiledCopyA = XE_2D_U4x32x16_LD_T;
  using GmemTiledCopyB = XE_2D_U16x16x32_LD_N;

  // Workgroup-level tile
  using TileShape = Shape<_16, _64, _32>;

  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_8x16x16_F32F16F16F32_TT>, Layout<TileShape>,
                                    Layout<Shape<_1, _2, _1>, Stride<_2, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 3;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16MixedPrecision<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementAccumulator, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD>,
          FusionCallBacks,
          XE_2D_U32x8x16_LD_N,
          void, void,
          XE_2D_U16x8x16_ST_N,
          void, void>;


    using Mainloop = cutlass::gemm::collective::CollectiveMma<
        GEMMDispatchPolicy, TileShape, half_t,
        cutlass::gemm::TagToStrideA_t<LayoutA>,
        cute::tuple<uint4_t, ElementScale, StrideScale, ElementZero, StrideZero>,
        cutlass::gemm::TagToStrideB_t<LayoutB>, TiledMma,
        GmemTiledCopyB, void, void, cute::identity,
        GmemTiledCopyA, void, void, cute::identity>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<
        cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Mainloop, CollectiveEpilogue>>;
    const int groups=k/32, padded_groups=groups;
    const size_t mk=size_t(m)*k, nk=size_t(n)*k, mn=size_t(m)*n;
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> uniform(0.f,1.f);
    std::uniform_int_distribution<int> byte(0,255);
    std::vector<half_t> a(mk), scales(size_t(padded_groups)*n,half_t(0));
    std::vector<uint8_t> weights(nk/2), trace_zeros(size_t(groups)*n/2);
    std::vector<half_t> zeros(size_t(groups)*n);
    for (auto& v:a) v=half_t(uniform(rng));
    for (auto& v:weights) v=byte(rng);
    for (int g=0; g<groups; ++g)
      for (int j=0; j<n; ++j) scales[size_t(g)*n+j]=half_t(uniform(rng));
    for (auto& v:trace_zeros) v=byte(rng);
    auto nibble=[](const std::vector<uint8_t>& data,size_t index) {
      return (data[index/2] >> (4*(index%2))) & 15;
    };
    // Unpack unsigned trace zero points to FP16 for the TLA zero-point loads.
    for (int g=0; g<groups; ++g) for (int j=0; j<n; ++j) {
      zeros[size_t(g)*n+j]=half_t(float(nibble(trace_zeros,size_t(g)*n+j)));
    }
    cutlass::DeviceAllocation<half_t> da(mk), ds(scales.size()), dd(mn);
    cutlass::DeviceAllocation<float> dc(mn);
    cutlass::DeviceAllocation<uint4_t> db(nk);
    cutlass::DeviceAllocation<half_t> dz(zeros.size());
    da.copy_from_host(a.data()); ds.copy_from_host(scales.data());
    cutlass::device_memory::copy_to_device(reinterpret_cast<uint8_t*>(db.get()),weights.data(),weights.size());
    dz.copy_from_host(zeros.data());
    auto sa=cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{},make_shape(m,k,1));
    auto sb=cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{},make_shape(n,k,1));
    auto sc=cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{},make_shape(m,n,1));
    auto sd=cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideD{},make_shape(m,n,1));
    StrideScale ss=make_stride(_1{},int64_t(n),int64_t(n)*padded_groups);
    StrideZero sz=ss;
    typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k,1},
      {da.get(),sa,db.get(),sb,ds.get(),ss,dz.get(),sz,32},
      {{1.f,0.f},dc.get(),sc,dd.get(),sd}};
    Gemm gemm;
    cutlass::device_memory::allocation<uint8_t> workspace(Gemm::get_workspace_size(args));
    check(gemm.can_implement(args)); check(gemm.initialize(args,workspace.get()));
    check(gemm.run()); compat::get_default_queue().wait_and_throw();
    if (verify) {
      // Full independent CPU reference with FP16 dequantization and FP32 accumulation.
      std::vector<half_t> result(mn);
      dd.copy_to_host(result.data()); compat::get_default_queue().wait_and_throw();
      double max_error=0;
      for (int i=0; i<m; ++i) for (int j=0; j<n; ++j) {
        float sum=0;
        for (int t=0; t<k; ++t) {
          int w=nibble(weights,size_t(j)*k+t);
          int z=nibble(trace_zeros,size_t(t/32)*n+j);
          half_t dq(float(w-z)*float(scales[size_t(t/32)*n+j]));
          sum += float(a[size_t(i)*k+t])*float(dq);
        }
        float ref=float(half_t(sum)), actual=float(result[size_t(i)*n+j]);
        double error=std::abs(actual-ref);
        max_error=std::max(max_error,error);
        if (!std::isfinite(actual) || !std::isfinite(ref) || error>0.01+0.01*std::abs(ref))
          throw std::runtime_error("Verification failed at ("+std::to_string(i)+","+std::to_string(j)+
              "): actual="+std::to_string(actual)+", reference="+std::to_string(ref));
      }
      std::cout << "Verification passed (atol=rtol=0.01), max abs error=" << max_error << "\n";
    }
    for (int i=0; i<warmup; ++i) check(gemm.run());
    compat::get_default_queue().wait_and_throw();
    auto start=std::chrono::steady_clock::now();
    for (int i=0; i<iterations; ++i) check(gemm.run());
    compat::get_default_queue().wait_and_throw();
    double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
    std::cout << "W4A16 ["<<m<<","<<k<<"] x ["<<k<<","<<n<<"] group_size=32: "
              <<us<<" us/call, "<<(2.0*m*n*k/us/1e6)<<" TFLOP/s ("
              <<iterations<<" calls, synchronized wall time)\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
