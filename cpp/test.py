import torch
import torch.autograd
import torch.nn
import torch.utils.cpp_extension
import math
import os

# TODO: Possibly use setup.py instead of JIT loading
butterfly_cpp = torch.utils.cpp_extension.load(
    name="butterfly_cpp", 
    sources=["cpp/butterfly.cpp"],
    extra_include_paths=["/opt/rocm/hip/include", "cpp/MIPP/src"],
)

device = torch.device('cpu')
dtype = torch.float32
data = torch.rand(size=[16, 1024], dtype=dtype, device=device)
data = torch.zeros(size=[16, 1024], dtype=dtype, device=device)
data[0, 0] = 1.0
data_in = data.clone()
angles = torch.rand(size=[8, 8], dtype=dtype, device=device)
# angles = torch.zeros(size=[8, 8], dtype=dtype, device=device)
col_indices = torch.arange(16, dtype=torch.int, device=device)
butterfly_cpp.butterfly_forward(data, angles, col_indices, -1)
print(torch.sum(data_in ** 2))
print(torch.sum(data ** 2))
data[:, 0]
