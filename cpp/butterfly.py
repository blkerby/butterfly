import torch
import torch.autograd
import torch.nn
import torch.utils.cpp_extension
import math

# TODO: Possibly use setup.py instead of JIT loading
butterfly_cpp = torch.utils.cpp_extension.load(
    name="butterfly_cpp",
    sources=["cpp/butterfly.cpp"],
    extra_include_paths=["/opt/rocm/hip/include", "cpp/MIPP/src"],
    extra_cflags=['-O3', '-funroll-loops']
)

COL_BLOCK_WIDTH = 16
NUM_BLOCK_LAYERS = 8

class ButterflyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, angles, indices_in, idx_out):
        butterfly_cpp.butterfly_forward(data, angles, indices_in, idx_out)
        ctx.indices_in = indices_in
        ctx.idx_out = idx_out
        ctx.save_for_backward(data, angles)
        return data

    @staticmethod
    def backward(ctx, grad_data):
        data, angles = ctx.saved_tensors
        grad_angles = torch.zeros_like(angles)
        butterfly_cpp.butterfly_backward(data, grad_data, angles, grad_angles, ctx.indices_in, ctx.idx_out)
        return grad_data, grad_angles, None, None


class ButterflyModule(torch.nn.Module):
    def __init__(self, indices_in, idx_out, dtype=torch.float32, device=None):
        super().__init__()
        self.angles = torch.nn.Parameter((torch.rand(size=[NUM_BLOCK_LAYERS, len(indices_in) // 2], dtype=dtype, device=device) * 2 - 1) * math.pi)
        self.indices_in = indices_in
        self.idx_out = idx_out

    def forward(self, data):
        return ButterflyFunction().apply(data, self.angles, self.indices_in, self.idx_out)


class ButterflyNetwork(torch.nn.Module):
    def __init__(self, indices_in_list, idx_out_list, dtype=torch.float32, device=None):
        super().__init__()
        layers = []
        assert len(indices_in_list) == len(idx_out_list)
        for i in range(len(indices_in_list)):
            layers.append(ButterflyModule(indices_in_list[i], idx_out_list[i]))
        self.mods = torch.nn.ModuleList(layers)

    def forward(self, data):
        for m in self.mods:
            data = m(data)
        return data
