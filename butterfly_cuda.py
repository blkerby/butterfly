import torch
import torch.autograd
import torch.nn
import torch.utils.cpp_extension
import math


# TODO: Use setup.py instead of JIT loading
cuda_butterfly = torch.utils.cpp_extension.load(
    name="cuda_butterfly", 
    sources=["butterfly.cpp", "butterfly.cu"],
    extra_cuda_cflags=['-Xcompiler', '-O3', '-Xptxas', '-O3', '--gpu-code=compute_30', 
    '--gpu-architecture=compute_30', '-Xptxas', '--warn-on-spills', '-Xptxas', '--warn-on-local-memory-usage'])


class OrthogonalButterflyLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data_in, angles):
        output = torch.empty_like(data_in)
        cuda_butterfly.butterfly_forward_slow(data_in, angles, output)
        ctx.save_for_backward(output, angles)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, angles = ctx.saved_tensors
        grad_input = torch.empty_like(grad_output)
        grad_angles_accum = torch.zeros_like(angles)
        cuda_butterfly.butterfly_backward_slow(output, angles, grad_output, grad_input, grad_angles_accum)
        return grad_input, grad_angles_accum
        

# TODO: combine CPU and CUDA versions into one class (also implement optimized CPU version in C++)
class OrthogonalButterfly(torch.nn.Module):
    def __init__(self, width_pow, depth, l2_interact, dtype=torch.float, device=None):
        super().__init__()
        self.dtype = dtype
        self.width_pow = width_pow
        self.width = 2**width_pow
        self.depth = depth
        self.l2_interact = l2_interact
        initial_angles = torch.rand(depth, self.width // 2, dtype=dtype, device=device) * math.pi * 2
        self.angles = torch.nn.Parameter(initial_angles)

    def forward(self, X):
        assert X.dtype == self.dtype
        X = torch.cat([X, torch.zeros([self.width - X.shape[0], X.shape[1]], dtype=X.dtype, device=X.device)], dim=0)
        for i in range(self.depth):
            X = OrthogonalButterflyLayerFunction.apply(X, self.angles[i, :])
        return X

    def penalty(self):
        return torch.sum(self.l2_interact * torch.sin(2*self.angles)**2)


