import torch
import torch.nn
import torch.utils.cpp_extension
import math

# TODO: Use setup.py instead of JIT loading
cuda_butterfly = torch.utils.cpp_extension.load(
    name="butterfly", 
    sources=["butterfly.cpp", "butterfly.cu"],
    extra_cuda_cflags=['-Xcompiler', '-O3', '-Xptxas', '-O3', '--gpu-code=compute_30', 
    '--gpu-architecture=compute_30', '-Xptxas', '--warn-on-spills', '-Xptxas', '--warn-on-local-memory-usage'])


# TODO: combine CPU and CUDA versions into one class (also implement optimized CPU version in C++)
class OrthogonalButterflyLayerCUDA(torch.nn.Module):
    def __init__(self, width_pow, l2_interact, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.width_pow = width_pow
        self.width = 2**width_pow
        self.half_width = 2**(width_pow - 1)
        self.depth = depth
        self.l2_interact = l2_interact
        initial_params = torch.rand(self.half_width, depth, dtype=dtype) * math.pi * 2
        self.params = torch.nn.Parameter(initial_params)
        self.perm = torch.zeros([self.width], dtype=torch.long)
        for i in range(self.width):
            if i % 2 == 0:
                self.perm[i] = i // 2
            else:
                self.perm[i] = i // 2 + self.half_width

    def forward(self, X):
        assert X.dtype == self.dtype
        input_width = X.shape[1]
        X = torch.cat([X, torch.zeros([X.shape[0], self.width - X.shape[1]], dtype=X.dtype)], dim=1)
        out = torch.empty_like(X)
        cuda_butterfly.butterfly_forward_slow(X, out, self.params.data)
        return out


class OrthogonalButterfly(torch.nn.Module):
    def __init__(self, width_pow, depth, l2_interact, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.width_pow = width_pow
        self.width = 2**width_pow
        self.half_width = 2**(width_pow - 1)
        self.depth = depth
        self.l2_interact = l2_interact
        initial_params = torch.rand(self.half_width, depth, dtype=dtype) * math.pi * 2
        self.params = torch.nn.Parameter(initial_params)
        self.perm = torch.zeros([self.width], dtype=torch.long)
        for i in range(self.width):
            if i % 2 == 0:
                self.perm[i] = i // 2
            else:
                self.perm[i] = i // 2 + self.half_width

    def forward(self, X):
        assert X.dtype == self.dtype
        input_width = X.shape[1]
        X = torch.cat([X, torch.zeros([X.shape[0], self.width - X.shape[1]], dtype=X.dtype)], dim=1)
        for i in range(self.depth):
            theta = self.params[:, i]
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            X0 = X[:, :self.half_width]
            X1 = X[:, self.half_width:]
            new_X0 = X0 * cos_theta + X1 * sin_theta
            new_X1 = X0 * -sin_theta + X1 * cos_theta
            X = torch.cat([new_X0, new_X1], dim=1)
            X = X[:, self.perm]
        return X[:, :input_width]

    def penalty(self):
        return torch.sum(self.l2_interact * torch.sin(2*self.params)**2)


class SmoothBendActivation(torch.nn.Module):
    def __init__(self, width, penalty_bias, penalty_slope, penalty_curvature, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.penalty_bias = penalty_bias
        self.penalty_slope = penalty_slope
        self.penalty_curvature = penalty_curvature
        eps = 0.01
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - 1)
        self.slope_1 = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - 1)
        self.slope_2 = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - 1)
        self.curvature = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - 1)

    def forward(self, X):
        assert X.dtype == self.dtype
        m1 = self.slope_1
        m2 = self.slope_2
        # m1 = self.slope_1 + 1
        # m2 = self.slope_2 + 1
        a = (m1 + m2) / 2
        c = (m2 - m1) / (2 * a)
        # c = (m2 - m1) / 2
        b = self.bias
        k = self.curvature
        # b = self.bias
        # k = self.curvature ** 2 + 1
        u = X * a - b
        return u + c * torch.sqrt(u**2 + 1 / k**2)

    def penalty(self):
        return torch.sum(self.penalty_bias * self.bias**2) + \
            torch.sum(self.penalty_slope * ((self.slope_1 - 1)**2 + (self.slope_2 - 1)**2)) + \
            torch.sum(self.penalty_curvature * self.curvature**2)


class SmoothSquashActivation(torch.nn.Module):
    def __init__(self, width, l2_slope, l2_scale, l2_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        eps = 0.1
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.slope = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        self.scale = torch.nn.Parameter(torch.rand([width], dtype=dtype))

    def forward(self, X):
        assert X.dtype == self.dtype
        u = X * self.slope - self.bias
        return self.scale * u / torch.sqrt(u**2 + self.scale ** 2)

    def penalty(self):
        return torch.sum(self.l2_slope * (self.slope - 1) ** 2) + \
                torch.sum(self.l2_scale * self.scale ** 2) + \
                torch.sum(self.l2_bias * self.bias ** 2)



class CubicSquashActivation(torch.nn.Module):
    def __init__(self, width, l2_slope, l2_scale, l2_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        eps = 0.001
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.slope = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        self.scale = torch.nn.Parameter(torch.rand([width], dtype=dtype))

    def forward(self, X):
        assert X.dtype == self.dtype
        u = torch.clamp((X * self.slope - self.bias) / self.scale, -1, 1)
        return self.scale * (1.5 * u - 0.5 * u ** 3)

    def penalty(self):
        return torch.sum(self.l2_slope * (self.slope - 1) ** 2) + \
                torch.sum(self.l2_scale * self.scale ** 2) + \
                torch.sum(self.l2_bias * self.bias ** 2)



class QuinticSquashActivation(torch.nn.Module):
    def __init__(self, width, l2_slope, l2_scale, l2_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        eps = 1 / width
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.slope = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        self.scale = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)

    def forward(self, X):
        assert X.dtype == self.dtype
        u = torch.clamp((X * self.slope - self.bias) / self.scale, -1, 1)
        return self.scale * (1.875 * u - 1.25 * u ** 3 + 0.375 * u ** 5)

    def penalty(self):
        return torch.sum(self.l2_slope * (self.slope - 1) ** 2) + \
                torch.sum(self.l2_scale * self.scale ** 2) + \
                torch.sum(self.l2_bias * self.bias ** 2)


class CustomNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, width_pow, depth, butterfly_depth,
                 l2_slope, l2_scale, l2_bias, l2_interact, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.width_pow = width_pow
        self.depth = depth
        self.butterfly_depth = butterfly_depth
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        self.l2_interact = l2_interact
        self.width = 2 ** width_pow
        self.butterfly_layers = []
        self.activation_layers = []
        self.all_layers = []
        # self.output_bias = torch.nn.Parameter(torch.zeros([num_outputs], dtype=dtype))
        for i in range(depth):
            butterfly = OrthogonalButterfly(width_pow, butterfly_depth, l2_interact, dtype=dtype)
            activation = QuinticSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = CubicSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = SmoothSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = SmoothBendActivation(self.width, l2_bias, l2_slope, l2_scale, dtype=torch.float)
            self.butterfly_layers.append(butterfly)
            self.all_layers.append(butterfly)
            if i != depth - 1:
                self.activation_layers.append(activation)
                self.all_layers.append(activation)
        self.sequential = torch.nn.Sequential(*self.all_layers)

    def forward(self, X):
        assert X.dtype == self.dtype
        assert X.shape[1] == self.num_inputs
        X_in = torch.cat([X, torch.zeros([X.shape[0], self.width - self.num_inputs], dtype=self.dtype)], axis=1)
        return self.sequential.forward(X_in)[:, :self.num_outputs] #+ self.output_bias

    def penalty(self):
        total = 0
        for layer in self.activation_layers:
            layer.l2_slope = self.l2_slope
            layer.l2_scale = self.l2_scale
            layer.l2_bias = self.l2_bias
            total += layer.penalty()
        for layer in self.butterfly_layers:
            layer.l2_interact = self.l2_interact
            total += layer.penalty()
        return total


class L2Linear(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, l2):
        super().__init__()
        self.l2 = l2
        self.lin = torch.nn.Linear(num_inputs, num_outputs, bias=False)

    def forward(self, X):
        return self.lin(X)

    def penalty(self):
        return self.l2 * torch.sum(self.lin.weight ** 2)


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, width_pow, depth,
                 l2_slope, l2_scale, l2_bias, l2_lin, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.width_pow = width_pow
        self.depth = depth
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        self.l2_lin = l2_lin
        self.width = 2 ** width_pow
        self.lin_layers = []
        self.activation_layers = []
        self.all_layers = []
        for i in range(depth):
            lin = L2Linear(self.width, self.width, l2_lin)
            activation = QuinticSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = CubicSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = SmoothSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = SmoothBendActivation(self.width, l2_bias, l2_slope, l2_scale, dtype=torch.float)
            self.lin_layers.append(lin)
            self.all_layers.append(lin)
            if i != depth - 1:
                self.activation_layers.append(activation)
                self.all_layers.append(activation)
        self.sequential = torch.nn.Sequential(*self.all_layers)

    def forward(self, X):
        assert X.dtype == self.dtype
        assert X.shape[1] == self.num_inputs
        X_in = torch.cat([X, torch.zeros([X.shape[0], self.width - self.num_inputs], dtype=self.dtype)], axis=1)
        return self.sequential.forward(X_in)[:, :self.num_outputs]

    def penalty(self):
        total = 0
        for layer in self.activation_layers:
            layer.l2_slope = self.l2_slope
            layer.l2_scale = self.l2_scale
            layer.l2_bias = self.l2_bias
            total += layer.penalty()
        for layer in self.lin_layers:
            layer.l2 = self.l2_lin
            total += layer.penalty()
        return total
