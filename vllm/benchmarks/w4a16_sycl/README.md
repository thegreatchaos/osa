# SYCL W4A16 benchmark

参照根目录 `test_int4_gemm_w4a16_trace.py`，复用 `sycl-tla` 的混合精度
GEMM mainloop，计算 FP16 `[m,k]` × UINT4 `[k,n]` → FP16 `[m,n]`，
使用 FP32 累加、group size 32、无 bias。

```bash
source /opt/intel/oneapi/setvars.sh
make -C benchmarks/w4a16_sycl
./benchmarks/w4a16_sycl/w4a16_bench --m 1 --n 16384 --k 5120
./benchmarks/w4a16_sycl/w4a16_bench --m=16 --n=96 --k=5120 --iterations=1500
```

默认 `m=1,n=16384,k=5120`，预热 10 次，计时 1500 次。支持
`--warmup`、`--iterations`、`--skip-check` 和 `--help`。
M/N/K 必须为正整数，K 必须是 32 的倍数，N 必须是 8 的倍数；
其他设备/内核对齐限制由 `can_implement` 检查。

默认 sibling 路径为 `../../../sycl-tla`（相对本目录），可通过
`make TLA_ROOT=/path/to/sycl-tla DEVICE=ptl CXX=/path/to/icpx`
指定依赖、GPU 编译目标和编译器。默认 GPU 目标为 `pvc`，与依赖的
示例构建日志一致。构建时保留 IGC `allowDecompose2DBlockFuncs=0` workaround。

权重沿 K 打包，等价于 trace 中 `weight.stride()=(1,k/8)`；scales 为
`[k/32,n]`。零点先按 trace 的 UINT4 `[k/32,n]`（沿 N 打包）生成，
初始化时展开成 FP16，以使用 TLA 的零点加载布局，支持所有 UINT4 零点
和不足八个 group 的情况。展开与数据传输均不计入 benchmark。
因此零点读流量与原始 vLLM 内核不同。

默认使用独立 CPU reference 校验全部输出：先按 `(w-z)*scale` 反量化到
FP16，再以 FP32 累加，最终转换为 FP16，容差 `atol=rtol=0.01`。
大矩阵的 CPU 校验可能较慢，可用 `--skip-check` 跳过。

计时使用 `steady_clock`，在预热后及计时循环末尾同步 SYCL 队列，输出
平均 us/call 和 TFLOP/s。时间包含 host 提交开销，与 trace 的 XPU Event
计时口径不同；重复使用同一组输入以保持 trace 的缓存行为。
