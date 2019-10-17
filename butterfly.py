from abc import abstractmethod
import numpy as np
import torch
import torch.optim
import torch.optim.lbfgs
import torch.autograd
import torch.nn
import math
import matplotlib.pyplot as plt
from sr1_optimizer import SR1Optimizer
from agd_optimizer import AGDOptimizer
from spline import SplineFamily

def compute_loss(pY, Y):
    err = pY - Y
    return (err ** 2).mean()

def compute_objective(model, loss):
    return loss + model.penalty()

# def gen_data_perm(N, n, perm):
#     X = torch.rand([N, n], dtype=torch.float) * 2 - 1
#     Y = X[:, perm]
#     return X, Y

def gen_data(N, scale, noise, dtype=torch.float):
    X = (torch.rand([N, 1], dtype=torch.float) - 0.5) * scale
    # Y = torch.cos(X)
    # Y_true = X * torch.sin(1 / X)
    # Y_true = 0.1 * (torch.sin(X) + 3 * torch.cos(2*X) + 4*torch.sin(3*X) + 5*torch.cos(3*X) + torch.cos(0.7*X))
    Y_true = torch.round(0.15 * (torch.sin(X) + 3 * torch.cos(2*X) + 4*torch.sin(3*X) + 5*torch.cos(3*X) + torch.cos(0.7*X))) * 1.5
    # Y_true = torch.where(X > 0.2, torch.full_like(X, 1.0), torch.full_like(X, -1.0))
    Y = Y_true + noise * torch.randn([N, 1], dtype=torch.float)
    return torch.tensor(X, dtype=dtype), torch.tensor(Y_true, dtype=dtype), torch.tensor(Y, dtype=dtype)


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
        self.slope_1 = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        self.slope_2 = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        self.curvature = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)

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
        self.scale = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        # self.scale = 1.0

    def forward(self, X):
        assert X.dtype == self.dtype
        u = (X * self.slope - self.bias) / self.scale
        return self.scale * u / torch.sqrt(u**2 + self.scale ** 2)
        # a = torch.exp(2*u)
        # return self.scale * (a - 1) / (a + 1)

    def penalty(self):
        return torch.sum(self.l2_slope * (self.slope - 1) ** 2) + \
                torch.sum(self.l2_scale * (self.scale) ** 2) + \
                torch.sum(self.l2_bias * self.bias ** 2)

class KernelActivation(torch.nn.Module):
    def __init__(self, width, l2_slope, l2_scale, dimension, range, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.dimension = dimension
        self.range = range
        self.centers = range * (torch.arange(dimension, dtype=dtype) / (dimension - 1) * 2 - 1)
        eps = 0.5
        self.weights = torch.nn.Parameter(torch.rand([width, dimension], dtype=dtype) * 2 * eps - eps)
        self.scale = self.centers[1] - self.centers[0]
        print(self.centers)

    @abstractmethod
    def base_activation(self, X):
        pass

    def forward(self, X):
        assert X.dtype == self.dtype
        total = torch.zeros_like(X)
        for i in range(self.dimension):
            total += self.base_activation((X - self.centers[i]) / self.scale) * self.weights[:, i]
        return total

    def penalty(self):
        return self.l2_slope * (torch.sum(self.weights[:, 0] ** 2 + self.weights[:, -1] ** 2) + torch.sum((self.weights[:, 1:] - self.weights[:, :(self.dimension - 1)]) ** 2)) + \
            self.l2_scale * torch.sum(self.weights ** 2)

class GaussianKernelActivation(KernelActivation):
    def base_activation(self, X):
        return torch.exp(-X ** 2)


class SplineActivation(torch.nn.Module):
    def __init__(self, width, knots, degree, penalty_coefs, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.spline_family = SplineFamily(knots, degree)
        self.penalty_coefs = penalty_coefs
        eps = 0.1
        self.coefs = torch.nn.Parameter(torch.rand([width, self.spline_family.dimension], dtype=dtype) * 2 * eps - eps)

    def forward(self, X):
        assert X.shape[1] == self.coefs.shape[0]
        out = torch.empty_like(X)
        for i in range(X.shape[1]):
            spline = self.spline_family.bspline(self.coefs[i, :])
            out[:, i] = spline.eval(X[:, i])
        return out

    def penalty(self):
        total = 0.0
        for i in range(self.coefs.shape[0]):
            spline = self.spline_family.bspline(self.coefs[i, :])
            total += spline.penalty(self.penalty_coefs)
        return total


class CubicSquashActivation(torch.nn.Module):
    def __init__(self, width, l2_slope, l2_scale, l2_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        eps = 0.01
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.slope = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        # self.scale = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        self.scale = 1.0

    def forward(self, X):
        assert X.dtype == self.dtype
        u = torch.clamp((X * self.slope - self.bias) / self.scale, -1, 1)
        return self.scale * (1.5 * u - 0.5 * u ** 3)
        # return self.scale * u

    def penalty(self):
        return torch.sum(self.l2_slope * (self.slope - 1) ** 2) + \
               torch.sum(self.l2_bias * self.bias ** 2)
# torch.sum(self.l2_scale * (self.scale ) ** 2) + \

class ReLU(torch.nn.Module):
    def __init__(self, width, l2_slope, l2_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.l2_slope = l2_slope
        self.l2_bias = l2_bias
        eps = 0.01
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.slope = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)

    def forward(self, X):
        return torch.clamp(X * self.slope - self.bias, 0.0)

    def penalty(self):
        return 0.0

class QuinticSquashActivation(torch.nn.Module):
    def __init__(self, width, l2_slope, l2_scale, l2_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        eps = 0.01
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.slope = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        # self.scale = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        self.scale = torch.tensor(1.0)

    def forward(self, X):
        assert X.dtype == self.dtype
        u = torch.clamp((X * self.slope - self.bias) / self.scale, -1, 1)
        return self.scale * (1.875 * u - 1.25 * u ** 3 + 0.375 * u ** 5)
        # return self.scale * u

    def penalty(self):
        return torch.sum(self.l2_slope * (self.slope - 1) ** 2) + \
               torch.sum(self.l2_scale * (self.scale) ** 2) + \
               torch.sum(self.l2_bias * self.bias ** 2)


class ThreeParameterFamily(torch.nn.Module):
    def __init__(self, width, l2_slope, l2_scale, l2_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        eps = 1.0
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.slope = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.scale = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        # self.scale = torch.tensor(1.0)
        # self.slope = torch.tensor(1.0)

    def forward(self, X):
        assert X.dtype == self.dtype
        u = (X * self.slope - self.bias) / self.scale
        return self.scale * self.base_activation(u)

    @abstractmethod
    def base_activation(self, x):
        pass

    def penalty(self):
        return torch.sum(self.l2_slope * (self.slope - 1) ** 2) + \
               torch.sum(self.l2_bias * self.bias ** 2) + \
               torch.sum(self.l2_scale * (self.scale) ** 2)


class QuadraticSquashActivation(ThreeParameterFamily):
    def base_activation(self, x):
        u = torch.clamp(x, -1, 1)
        return 2 * u - u * torch.abs(u)

class AtanSquashActivation(ThreeParameterFamily):
    def base_activation(self, x):
        return torch.atan(x)

class SigmoidActivation(ThreeParameterFamily):
    def base_activation(self, x):
        return torch.sigmoid(x)

class SquashFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        y = torch.clamp(x / scale, -0.95, 0.95)
        u = torch.exp(-4 * y / (1 - y ** 2))
        ctx.save_for_backward(y, u)
        return scale * (1 - u) / (1 + u)

    @staticmethod
    def backward(ctx, g):
        y, u = ctx.saved_tensors
        y2 = y ** 2
        return (8 * u / (1 + u) ** 2 * (1 + y2) / (1 - y2) ** 2 * g, None)


class BumpFamily(torch.nn.Module):
    def __init__(self, width, l2_slope, l2_scale, l2_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        eps = 0.1
        self.bias_1 = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.bias_2 = torch.nn.Parameter(self.bias_1 + torch.rand([width], dtype=dtype) * 2 * eps - eps)
        # self.slope = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.slope_1 = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        self.slope_2 = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        # self.scale = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.scale_1 = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        self.scale_2 = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        # self.scale_1 = torch.full_like(self.bias_1.data, 1.0)
        # self.scale_2 = torch.full_like(self.bias_1.data, 1.0)

    def forward(self, X):
        assert X.dtype == self.dtype
        # u1 = X * self.slope + self.bias_1
        # u2 = X * self.slope + self.bias_2
        u1 = X * self.slope_1 + self.bias_1
        u2 = X * self.slope_2 + self.bias_2
        return self.scale_1 * self.base_activation(u1 / self.scale_1) - self.scale_2 * self.base_activation(u2 / self.scale_2)
        # return self.scale * self.base_activation(u1 / self.scale) - self.scale * self.base_activation(u2 / self.scale)

    @abstractmethod
    def base_activation(self, x):
        pass

    def penalty(self):
        return torch.sum(self.l2_slope * (self.slope_1 ** 2 + self.slope_2 ** 2)) + \
               torch.sum(self.l2_bias * (self.bias_1 ** 2 + self.bias_2 ** 2)) + \
               torch.sum(self.l2_scale * (self.scale_1 ** 2 + self.scale_2 ** 2))
# return torch.sum(self.l2_slope * self.slope ** 2) + \
# torch.sum(self.l2_scale * (self.scale ** 2))


class QuadraticBumpActivation(BumpFamily):
    def base_activation(self, x):
        u = torch.clamp(x, -1, 1)
        return u - 0.5 * (u * torch.abs(u) - 1)
        # return u * (2 - torch.abs(u))


class CompactSmoothSquashActivation(torch.nn.Module):
    def __init__(self, width, l2_slope, l2_scale, l2_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        eps = 0.01
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps)
        self.slope = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        self.scale = torch.nn.Parameter(torch.rand([width], dtype=dtype) * 2 * eps - eps + 1)
        # self.scale = 1.0

    def forward(self, X):
        assert X.dtype == self.dtype
        # u = torch.clamp((X * self.slope - self.bias) / self.scale, -0.95, 0.95)
        # # a = torch.exp(-2 / (1 + u))
        # # b = torch.exp(-2 / (1 - u))
        # # out = self.scale * (a - b) / (a + b)
        #
        # # a = torch.exp(2 * u / (1 - u ** 2))
        # # b = 1 / a
        # # out = (a - b) / (a + b)
        #
        # a = torch.exp(-4 * u / (1 - u ** 2))
        # out = (1 - a) / (1 + a)
        #
        # # print("X={}, out={}".format(X, out))
        # return out
        return SquashFunction.apply(X * self.slope - self.bias, self.scale)

    def penalty(self):
        return torch.sum(self.l2_slope * (self.slope - 1) ** 2) + \
                torch.sum(self.l2_bias * self.bias ** 2) + \
                torch.sum(self.l2_scale * (self.scale) ** 2)


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
        # self.bias = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))
        # self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=dtype))
        for i in range(depth):
            butterfly = OrthogonalButterfly(width_pow, butterfly_depth, l2_interact, dtype=dtype)
            # activation = CompactSmoothSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = QuinticSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = CubicSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            activation = QuadraticSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = SmoothSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = AtanSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = SigmoidActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = AsinhActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = KernelActivation(self.width, l2_slope, dimension=32, range=2.0, dtype=dtype)
            # activation = SmoothBendActivation(self.width, l2_bias, l2_slope, l2_scale, dtype=torch.float)
            # activation = ReLU(self.width, l2_bias, l2_slope, dtype=torch.float)
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
        return self.sequential.forward(X_in)[:, :self.num_outputs] #* self.scale + self.bias

    def penalty(self):
        # total = self.l2_slope * (self.scale - 1) ** 2
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
            activation = QuadraticSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
            # activation = QuinticSquashActivation(self.width, l2_slope, l2_scale, l2_bias, dtype=dtype)
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


class DynamicRotation(torch.nn.Module):
    def __init__(self, inner, width, l2_scale):
        super().__init__()
        self.inner = inner
        self.width = width
        self.groups = width // 3
        self.l2_scale = l2_scale
        eps = 0.1
        # self.bias = torch.nn.Parameter(torch.rand([self.groups], dtype=dtype) * 2 * math.pi)
        # self.scale = torch.nn.Parameter(torch.rand([self.groups], dtype=dtype) * eps)

    def forward(self, X):
        X0 = X[:, :self.groups]
        X1 = X[:, self.groups:(self.groups * 2)]
        X2 = X[:, (self.groups * 2):(self.groups * 3)]
        Xr = X[:, (self.groups * 3):]
        angle = self.inner(X0)
        c = torch.cos(angle)
        s = torch.sin(angle)
        return torch.cat([X0, c * X1 + s * X2, -s * X1 + c * X2, Xr], dim=1)

    def penalty(self):
        return self.inner.penalty()
        # return self.l2_scale * torch.sum(self.scale ** 2)


class ReversibleActivation(torch.nn.Module):
    def __init__(self, inner, width):
        super().__init__()
        self.inner = inner
        self.half_width = width // 2
        # self.l2_weight = l2_weight
        # eps = 0.01
        # self.weight = torch.nn.Parameter(eps * (torch.rand([width], dtype=dtype) * 2 - 1))

    def forward(self, X):
        assert X.shape[1] == self.half_width * 2
        X0 = X[:, :self.half_width]
        X1 = X[:, self.half_width:]
        activations = self.inner(X1)
        self.input = X1
        return torch.cat([X0 + activations, X1], dim=1)

    def penalty(self):
        return self.inner.penalty() #+ self.l2_weight * torch.sum(self.weight ** 2)


class ReversibleDoubleActivation(torch.nn.Module):
    def __init__(self, inner_1, inner_2, width):
        super().__init__()
        self.inner_1 = inner_1
        self.inner_2 = inner_2
        self.half_width = width // 2
        # self.l2_weight = l2_weight
        eps = 0.01
        # self.weight = torch.nn.Parameter(eps * (torch.rand([width], dtype=dtype) * 2 - 1))

    def forward(self, X):
        assert X.shape[1] == self.half_width * 2
        X0 = X[:, :self.half_width]
        X1 = X[:, self.half_width:]
        Y0 = X0 + self.inner_1(X1)
        Y1 = X1 + self.inner_2(Y0)
        return torch.cat([Y0, Y1], dim=1)

    def penalty(self):
        return self.inner_1.penalty() + self.inner_2.penalty() #+ self.l2_weight * torch.sum(self.weight ** 2)



class DoubleReLUQuadratic(torch.nn.Module):
    def __init__(self, width, curvature, dtype=torch.float):
        super().__init__()
        self.width = width
        self.bias = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))
        self.a = curvature / 2
        self.c = 1 / (4 * self.a)

    def forward(self, X):
        u1 = torch.clamp_min(X, -self.c)
        y1 = torch.where(X > self.c, X, self.a * (X + self.c) ** 2)
        u2 = torch.clamp_max(X, self.c)
        y2 = torch.where(X < -self.c, X, self.a * (X - self.c) ** 2)
        return torch.cat([y1, y2], dim=1)


class DoubleSigmoidQuadratic(torch.nn.Module):
    def __init__(self, width, curvature, dtype=torch.float):
        super().__init__()
        self.width = width
        self.bias = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))
        self.a = curvature / 2
        self.c = 1 / (4 * self.a)

    def forward(self, X):
        u1 = torch.clamp(X, -self.c)
        y1 = torch.where(X > self.c, X, self.a * (X + self.c) ** 2)
        u2 = torch.clamp_max(X, self.c)
        y2 = torch.where(X < -self.c, X, self.a * (X - self.c) ** 2)
        return torch.cat([y1, y2], dim=1)



class ReversibleNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, width_pow, depth, butterfly_depth,
                 l2_slope, l2_scale, l2_bias, l2_interact, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.width_pow = width_pow
        self.width = 2 ** width_pow
        self.depth = depth
        self.butterfly_depth = butterfly_depth
        self.l2_slope = l2_slope
        self.l2_scale = l2_scale
        self.l2_bias = l2_bias
        self.l2_interact = l2_interact
        self.butterfly_layers = []
        self.activation_layers = []
        self.all_layers = []
        # self.bias = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))
        # self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=dtype))
        for i in range(depth):
            butterfly = OrthogonalButterfly(width_pow, butterfly_depth, l2_interact, dtype=dtype)
            # activation = ReversibleActivation(QuadraticSquashActivation(self.width // 2, l2_slope, None, 0.0, dtype=dtype), self.width, l2_weight)
            # inner_activation = QuinticSquashActivation(self.width // 2, l2_slope, l2_scale, 0.0, dtype=dtype)
            # inner_activation = SigmoidActivation(self.width // 2, l2_slope, l2_scale, 0.0, dtype=dtype)
            inner_activation_1 = SplineActivation(
                self.width // 2,
                knots=torch.tensor([-256.0, -128.0, -64.0, -32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0], dtype=dtype) / 32.0,
                degree=2,
                penalty_coefs=[0, 1e-3, 0],
                dtype=dtype
            )
            # inner_activation_1 = QuadraticSquashActivation(self.width // 2, l2_slope, l2_scale, 0.0, dtype=dtype)
            # inner_activation_2 = QuadraticSquashActivation(self.width // 2, l2_slope, l2_scale, 0.0, dtype=dtype)
            # inner_activation_1 = GaussianKernelActivation(self.width // 2, l2_slope, l2_scale, 8, 8, dtype=dtype)
            # inner_activation_2 = KernelActivation(self.width // 2, l2_slope, l2_scale, 16, 4, dtype=dtype)
            # inner_activation = QuadraticSquashActivation(self.width // 3, l2_slope, l2_scale, 0.0, dtype=dtype)
            # inner_activation = CompactSmoothSquashActivation(self.width // 2, l2_slope, l2_scale, 0.0, dtype=dtype)
            # activation = QuadraticSquashActivation(self.width, l2_slope, None, 0.0, dtype=dtype)
            # inner_activation = QuadraticBumpActivation(self.width // 2, l2_slope, l2_scale, l2_bias, dtype)
            activation = ReversibleActivation(inner_activation_1, self.width)
            # activation = ReversibleDoubleActivation(inner_activation_1, inner_activation_2, self.width)
            # activation = DynamicRotation(inner_activation, self.width, l2_scale)
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
        return self.sequential.forward(X_in)[:, :self.num_outputs] #* self.scale + self.bias

    def penalty(self):
        # total = self.l2_slope * (self.scale - 1) ** 2
        # total = self.l2_scale * self.scale ** 2
        total = 0
        for layer in self.activation_layers:
            total += layer.penalty()
        for layer in self.butterfly_layers:
            total += layer.penalty()
        return total


N = 200
# scale = 25
scale = 5
seed = 0
# dtype = torch.double
dtype = torch.float

torch.random.manual_seed(seed)

# Generate the data
X, Y_true, Y = gen_data(N, scale, noise=0.0, dtype=dtype)
X_test, _, Y_test = gen_data(5000, scale, 0, dtype)

# Construct/initialize the model (0.023523079231381416)
# model = CustomNetwork(
#     num_inputs=1,
#     num_outputs=1,
#     width_pow=4,
#     depth=4,
#     butterfly_depth=4,
#     l2_slope=1e-5,#1e-5,#1e-7,#1e-6,#2e-2, #1e-3,#1e-8, #0.0000005, #0.0001,
#     # l2_slope=0.000001, #0.0001,
#     l2_scale=1e-5,#1e-3,#2e-3, #1e-2, #1e-2, #1e-5, #1e-4, #2e-4, #0.0000001, # 0.0000001,#0.00001,
#     l2_bias=0, #1e-6,
#     l2_interact=0,#1e-5,#1e-4, #1e-5, #0.0001, #0.0001,
#     dtype=dtype
# )

model = ReversibleNetwork(
    num_inputs=1,
    num_outputs=1,
    width_pow=5,
    depth=4,
    butterfly_depth=5,
    l2_slope=1e-6,#e-5,#, 1e-3,#1e-4,
    l2_scale=1e-6,#5e-4,#1e-3,#1e-3,
    l2_bias=0,
    l2_interact=0,
    dtype=dtype)


# model = KernelActivation(width=1, l2_slope=1e-2, dimension=32, range=2.0, dtype=dtype)

# (0.03714136406779289)
# model = FullyConnectedNetwork(
#     num_inputs=1,
#     num_outputs=1,
#     width_pow=5,
#     depth=3,
#     l2_slope=1e-7, #0.0000005, #0.0001,
#     # l2_slope=0.000001, #0.0001,
#     l2_scale=1e-7, #1e-5, #1e-4, #2e-4, #0.0000001, # 0.0000001,#0.00001,
#     l2_bias=0.0,
#     l2_lin=1e-7, #1e-4, #1e-5, #0.0001, #0.0001,
#     dtype=dtype
# )


# optimizer = AGDOptimizer(model.parameters())
optimizer = SR1Optimizer(model.parameters(), memory=2000)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=10, history_size=1000, tolerance_grad=0, tolerance_change=0,
#                               line_search_fn='strong_wolfe')
# optimizer =torch.optim.SGD(model.parameters(), lr=0.02, nesterov=True, momentum=0.1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

fig = plt.gcf()
fig.show()
fig.canvas.draw()

last_loss = float("Inf")
last_gn = float("Inf")
for i in range(100000):
    eval_cnt = 0

    def closure():
        global eval_cnt
        eval_cnt += 1
        optimizer.zero_grad()
        model.zero_grad()
        pY = model(X)
        loss = compute_loss(pY, Y)
        obj = compute_objective(model, loss)
        obj.backward()
        return obj

    optimizer.step(closure)
    # optimizer.step(closure)

    if i % 5 == 0:
        model.zero_grad()
        pY = model(X)
        loss = compute_loss(pY, Y)
        obj = compute_objective(model, loss)
        obj.backward()

        with torch.no_grad():
            # print(list(model.parameters()))
            gn = torch.sqrt(sum(torch.sum(X.grad**2) for X in model.parameters()))
            pY = model(X)
            train_loss = compute_loss(pY, Y)
            obj = compute_objective(model, train_loss)

            with torch.no_grad():
                pY_test = model(X_test)
                test_loss = compute_loss(pY_test, Y_test)
                ind = torch.argsort(X_test[:, 0])
                fig.clear()
                plt.plot(X_test[ind, 0], Y_test[ind, 0], color='blue')
                plt.plot(X_test[ind, 0], pY_test[ind, 0], color='black')
                plt.scatter(X[:, 0], Y[:, 0], color='black', marker='.', alpha=0.3)
                fig.canvas.draw()

            print("seed={}, iteration={}: obj={:.7f}, train={:.7f}, true={:.7f}, obj grad norm={:.7g}, tr_radius={}, eig5={}".format(
                seed, i, float(obj), float(loss), float(test_loss), gn, optimizer.state['tr_radius'], optimizer.state['eig5']))
            # if gn < 1e-7 or (last_loss == loss and last_gn == gn):
            #     print("Perturbing")
            #     for p in model.parameters():
            #         p += 1e-4 * (torch.rand_like(p) * 2 - 1)
            #     #
            #     # # if loss < 1e-7:
            #     # print("seed {}, iteration {}: obj {}, train {}, true {}, obj grad norm {}".format(
            #     #     seed, i, float(obj), float(loss), float(test_loss), gn))
            #     # break
            last_loss = loss
            last_gn = gn

            inputs = []
            for layer in model.activation_layers:
                inputs.append(layer.input.detach())
            all_inputs = torch.cat(inputs)
            # print("Activation quantiles: {}".format(np.quantile(all_inputs, np.linspace(0, 1, 8))))


# model.weight.data = torch.round(model.weight)


print(compute_loss(pY_test, Y_test))
#
# model.penalty_coef = 0.0005
# optimizer.param_groups[0]['lr'] = 0.0001