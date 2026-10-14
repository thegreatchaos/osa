# SPDX-License-Identifier: Apache-2.0
import statistics

import pytest
import torch

from tests.register_ops import int4_gemm_w4a16
from vllm_xpu_kernels.quantization._quantize_convert import dequantize


@pytest.mark.parametrize("m,k,n", [
    #(1, 5120, 14336),
    (1, 5120, 16384),
    #(1, 5120, 96),
    #(1, 6144, 5120),
    #(1, 5120, 34816),
    #(1, 17408, 5120),
])
def test_int4_gemm_w4a16_trace_shape(m, k, n):
    torch.manual_seed(1234)

    group_size = 32
    packed_width = k // 8
    groups = k // group_size
    input = torch.rand((m, k), device="xpu", dtype=torch.float16)
    packed = torch.randint(-128, 128, (k * n // 2,),
                           device="xpu").to(torch.int8).view(torch.int32)
    weight = packed.reshape(n, packed_width).t()
    scales = torch.rand((groups, n), device="xpu", dtype=torch.float16)
    zero_points = torch.randint(-128, 128, (groups * n // 2,),
                                device="xpu").to(torch.int8).view(
                                    torch.int32).reshape(groups, n // 8)

    assert input.shape == (m, k) and input.stride() == (k, 1)
    assert weight.shape == (packed_width, n)
    assert weight.stride() == (1, packed_width)
    assert scales.shape == (groups, n) and scales.stride() == (n, 1)
    assert zero_points.shape == (groups, n // 8)
    assert zero_points.stride() == (n // 8, 1)

    output_ref = torch.matmul(input.cpu(),
                              dequantize(weight, scales, zero_points,
                                         group_size, None).cpu())
    output = int4_gemm_w4a16(input, weight, None, scales, zero_points,
                             group_size, None)
    torch.testing.assert_close(output.cpu().float(),
                               output_ref.float(),
                               atol=1e-2,
                               rtol=1e-2)

    for _ in range(10):
        int4_gemm_w4a16(input, weight, None, scales, zero_points, group_size,
                         None)
    torch.xpu.synchronize()

    LOOPS=1500
    start = torch.xpu.Event(enable_timing=True)
    end = torch.xpu.Event(enable_timing=True)
    start.record()
    for i in range(LOOPS):
        int4_gemm_w4a16(input, weight, None, scales, zero_points, group_size, None)
    end.record()
    torch.xpu.synchronize()
    avg = start.elapsed_time(end) * 1000 / LOOPS

    print(f"int4_gemm_w4a16 [{m},{k}] x [{packed_width},{n}]: "
          f"{avg:.2f} us/call "
          "(XPU Event, %d calls)" % LOOPS)
