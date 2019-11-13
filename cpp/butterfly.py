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
    def forward(ctx, data, angles, indices_in, idx_out, num_input_layers, num_output_layers):
        butterfly_cpp.butterfly_forward(data, angles, indices_in, idx_out, num_input_layers, num_output_layers)
        ctx.indices_in = indices_in
        ctx.idx_out = idx_out
        ctx.num_input_layers = num_input_layers
        ctx.num_output_layers = num_output_layers
        ctx.save_for_backward(data, angles)
        return data

    @staticmethod
    def backward(ctx, grad_data):
        data, angles = ctx.saved_tensors
        grad_angles = torch.zeros_like(angles)
        butterfly_cpp.butterfly_backward(data, grad_data, angles, grad_angles, ctx.indices_in, ctx.idx_out, ctx.num_input_layers, ctx.num_output_layers)
        return grad_data, grad_angles, None, None, None, None


class ButterflyModule(torch.nn.Module):
    def __init__(self, indices_in, idx_out, num_input_layers, num_output_layers, dtype=torch.float32, device=None):
        super().__init__()
        num_layers = num_input_layers + num_output_layers
        self.angles = torch.nn.Parameter((torch.rand(size=[num_layers, len(indices_in) // 2], dtype=dtype, device=device) * 2 - 1) * math.pi)
        self.indices_in = indices_in
        self.idx_out = idx_out
        self.num_input_layers = num_input_layers
        self.num_output_layers = num_output_layers

    def forward(self, data):
        return ButterflyFunction().apply(data, self.angles, self.indices_in, self.idx_out, self.num_input_layers, self.num_output_layers)


class ButterflyNetwork(torch.nn.Module):
    def __init__(self, indices_in_list, idx_out_list, num_input_layers, num_output_layers, dtype=torch.float32, device=None):
        super().__init__()
        layers = []
        assert len(indices_in_list) == len(idx_out_list)
        for i in range(len(indices_in_list)):
            layers.append(ButterflyModule(indices_in_list[i], idx_out_list[i], num_input_layers, num_output_layers, dtype=dtype, device=device))
        self.mods = torch.nn.ModuleList(layers)

    def forward(self, data):
        for m in self.mods:
            data = m(data)
        return data
