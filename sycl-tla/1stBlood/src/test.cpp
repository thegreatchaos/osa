/*! \file
    \brief Flash Attention V2 Prefill for Intel BMG
    从sycl-tla/example/06_bmg_flash_attention修改而来
    To run this example:
      $ ./bin/test --seq_len_qo=512 --seq_len_kv=512 --head_size_vo=128 --head_size_qk=128
*/
#include "xe_fmha_fwd_runner.hpp"
int main(int argc, const char **argv) {
  Options options;
  options.parse(argc, argv);
  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }
  // Define the work-group tile shape depending on the head-size of the second matmul
#ifdef PREFILL
#pragma message("\033[41mPREFILL\033[0m")
#if HEAD_DIM == 16
  /* Tiny config for testing */
  using ShapeQK = Shape<_1, _16, _16>;       // (q,k,d)
  using ShapePV = Shape<_1, _16, _16>;       // (q,v,k)
  using ShapeOut = Shape<_1, _16>;           // (q,v)
  using SubgroupLayoutQK = Layout<Shape<_1, _1, _1>>;
#elif HEAD_DIM == 64
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _64>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
#elif HEAD_DIM == 96
  using ShapeQK = Shape<_128, _64, _32>;
  using ShapePV = Shape<_128, _32, _64>;
  using ShapeOut = Shape<_128, _96>;
  using SubgroupLayoutQK = Layout<Shape<_8, _1, _1>>;
#elif HEAD_DIM == 128
  using ShapeQK = Shape<_256, _32, _32>;
  using ShapePV = Shape<_256, _32, _32>;
  using ShapeOut = Shape<_256, _128>;
  using SubgroupLayoutQK = Layout<Shape<_16, _1, _1>>;
#elif HEAD_DIM == 192
  using ShapeQK = Shape<_256, _64, _32>;
  using ShapePV = Shape<_256, _32, _64>;
  using ShapeOut = Shape<_256, _192>;
  using SubgroupLayoutQK = Layout<Shape<_32, _1, _1>>;
#endif
#elif defined(DECODE)
#pragma message("\033[42mDECODE\033[0m")
#if PERSISTENT
#define NUM_SG _16
#define KV_TILE_SIZE _256
#else
#define NUM_SG _8
#define KV_TILE_SIZE _512
#endif
#if HEAD_DIM == 16
  /* Tiny config for testing */
  using ShapeQK = Shape<_1, _16, _16>;       // (q,k,d)
  using ShapePV = Shape<_1, _16, _16>;       // (q,v,k)
  using ShapeOut = Shape<_1, _16>;           // (q,v)
  using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#elif HEAD_DIM == 64
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _64>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#elif HEAD_DIM == 96
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _96>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#elif HEAD_DIM == 128
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _128>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#elif HEAD_DIM == 192
    using ShapeQK = Shape<_1, KV_TILE_SIZE, _64>;
    using ShapePV = Shape<_1, _32, KV_TILE_SIZE>;
    using ShapeOut = Shape<_1, _192>;
    using SubgroupLayoutQK = Layout<Shape<_1, NUM_SG, _1>>;
#endif
#else
#error Either DECODE or PREFILL should be defined.
#endif
#ifdef DECODE
  constexpr int PipelineStages = 1;
#else
  constexpr int PipelineStages = 2;
#endif
#ifdef IS_FLOAT_E5M2
  using ElementQ = cutlass::float_e5m2_t;
  using ElementK = cutlass::float_e5m2_t;
  using ElementV = cutlass::float_e5m2_t;
#elif defined(IS_FLOAT_E4M3)
  using ElementQ = cutlass::float_e4m3_t;
  using ElementK = cutlass::float_e4m3_t;
  using ElementV = cutlass::float_e4m3_t;
#else
  using ElementQ = bfloat16_t;
  using ElementK = bfloat16_t;
  using ElementV = bfloat16_t;
#endif
#if PERSISTENT
  return FMHAConfig<false/*isCausal*/, 
		    ShapeQK/*TileShapeQK*/, 
		    ShapePV/*TileShapePV*/, 
		    ShapeOut/*TileShapeOutput*/, 
		    SubgroupLayoutQK/*SgLayoutQK*/, 
		    void,/*SgLayoutPV*/ 
		    PipelineStages, 
		    true/*isPersistent*/, 
		    ElementQ/*Datatype Q*/, 
		    ElementK/*DataType K*/, 
		    ElementV/*DataType V*/>::run(options);
#else
  return options.is_causal ? FMHAConfig<true, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,  /*persistent=*/false, ElementQ, ElementK, ElementV>::run(options)
  : FMHAConfig<false, ShapeQK, ShapePV, ShapeOut, SubgroupLayoutQK, void, PipelineStages,  /*persistent=*/false, ElementQ, ElementK, ElementV>::run(options);
#endif
}
