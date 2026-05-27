# SPDX-License-Identifier: Apache-2.0
import gc
import os
import random
import sys
import time
import torch
import vllm_xpu_kernels._C  # noqa: F401
import vllm_xpu_kernels._moe_C  # noqa: F401
import vllm_xpu_kernels._xpu_C  # noqa: F401
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe
DEVICE = "xpu"

def lstSOs():
    print(vllm_xpu_kernels._C.__file__)
    print(vllm_xpu_kernels._moe_C.__file__)
    print(vllm_xpu_kernels._xpu_C.__file__)
    print(vllm_xpu_kernels.fused_moe_interface.__file__)
def main():
    lstSOs();
if __name__ == '__main__':
    main()
