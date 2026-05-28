import gc
import os
import random
import sys
import time
import torch
import vllm
import vllm_xpu_kernels._C  
import vllm_xpu_kernels._moe_C 
import vllm_xpu_kernels._xpu_C 
from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe
DEVICE = "xpu"

def lstSOs():
    print(vllm_xpu_kernels._C.__file__)
    print(vllm_xpu_kernels._moe_C.__file__)
    print(vllm_xpu_kernels._xpu_C.__file__)
    print(vllm_xpu_kernels.fused_moe_interface.__file__)
    print("vllm ver: ", vllm.__version__)
    print("torch ver: ", torch.__version__)
def main():
    lstSOs();
if __name__ == '__main__':
    main()
