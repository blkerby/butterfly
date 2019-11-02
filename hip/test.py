import torch
import torch.autograd
import torch.nn
import torch.utils.cpp_extension
import math
import os

# TODO: Use setup.py instead of JIT loading
hip_test = torch.utils.cpp_extension.load(
    name="hip_test", 
    sources=["test.cpp", "hip.cu"],
    extra_include_paths=["/opt/rocm/hip/include"],
)

device = torch.device('cuda')
dtype = torch.float64
data_in = torch.arange(32, dtype=dtype, device=device)
data_out = torch.zeros([32], dtype=dtype, device=device)
hip_test.square(data_in, data_out)
print(data_in)
print(data_out)
