from abc import abstractmethod
from typing import List
import torch
import math
import torch.nn.functional


class DoubleReLUGeneric(torch.nn.Module):
    def __init__(self, width, l2_bias=0.0, curvature=5.0, dtype=torch.float, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * 2 - 1)
        self.curvature = curvature

    @abstractmethod
    def base_activation(self, X):
        pass

    def forward(self, X):
        assert X.shape[1] >= self.width
        X_unused = X[:, :(-self.width)]
        X_used = X[:, (-self.width):] + self.bias
        y1 = self.base_activation(X_used / self.curvature) * self.curvature
        y2 = self.base_activation(-X_used / self.curvature) * self.curvature
        return torch.cat([X_unused, y1, y2], dim=1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2)


class DoubleReLUSqrt(DoubleReLUGeneric):
    def base_activation(self, X):
        u = torch.clamp_min(X, -1.0)
        c = math.sqrt(2.0) / 3 * (math.sqrt(8.0) - 1)
        y = torch.where(u > 1, u - 1 + c, math.sqrt(2.0) / 3 * ((1 + u) ** 1.5 - 1))
        return y

class DoubleReLUSqrt2(DoubleReLUGeneric):
    def base_activation(self, X):
        v = torch.clamp_min(X, -1.0)
        f1 = 1 / math.sqrt(2) * (v + v * v / 2)
        c = 1 / math.sqrt(2) * (math.pi / 4 + 1 / 2)
        u = 1 / math.sqrt(2) * (1 - v)
        u0 = torch.clamp(u, 0.0, 1 / math.sqrt(2))  # Workaround Pytorch weird backprop behavior with NaNs in torch.where
        f2 = c - 1 / math.sqrt(2) * (torch.asin(u0) + u0 * torch.sqrt(1 - u0 ** 2))
        return torch.where(v > 1, v - 1 + c, torch.where(v > 0, f2, f1))


class DoubleReLUQuadraticDynamicCurvature(torch.nn.Module):
    def __init__(self, width, l2_bias=0.0, l2_curvature=0.0, dtype=torch.float, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.l2_curvature = l2_curvature
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * 2 - 1)
        self.curvature = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device))

    def base_activation(self, X):
        u = torch.clamp_min(X, -1)
        return torch.where(u > 1, u, 0.25 * (u + 1) ** 2)

    def forward(self, X):
        assert X.shape[1] >= self.width
        X_unused = X[:, :(-self.width)]
        X_used = X[:, (-self.width):] + self.bias
        y1 = self.base_activation(X_used / self.curvature) * self.curvature
        y2 = self.base_activation(-X_used / self.curvature) * self.curvature
        return torch.cat([X_unused, y1, y2], dim=1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2) + self.l2_curvature * torch.sum((self.curvature - 1) ** 2)





class DoubleReLUQuadratic(torch.nn.Module):
    def __init__(self, width, l2_bias=0.0, curvature=1.0, dtype=torch.float, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * 2 - 1)
        self.curvature = curvature

    def base_activation(self, X):
        u = torch.clamp_min(X, -1)
        return torch.where(u > 1, u, 0.25 * (u + 1) ** 2)

    def forward(self, X):
        assert X.shape[1] >= self.width
        X_unused = X[:, :(-self.width)]
        X_used = X[:, (-self.width):] + self.bias
        y1 = self.base_activation(X_used * self.curvature) / self.curvature
        y2 = self.base_activation(-X_used * self.curvature) / self.curvature
        return torch.cat([X_unused, y1, y2], dim=1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2)



class DoubleReLUSmooth(torch.nn.Module):
    def __init__(self, width, l2_bias=0.0, curvature=1.0, dtype=torch.float, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * 2 - 1)
        self.curvature = curvature

    def base_activation(self, X):
        # u = torch.clamp_min(X, -1)
        # return torch.where(u > 1, u, 0.25 * (u + 1) ** 2)
        return 0.5 * (X - 1 + torch.sqrt(X * 2 + 1))

    def forward(self, X):
        assert X.shape[1] >= self.width
        X_unused = X[:, :(-self.width)]
        X_used = X[:, (-self.width):] + self.bias
        y1 = self.base_activation(X_used / self.curvature) * self.curvature
        y2 = self.base_activation(-X_used / self.curvature) * self.curvature
        return torch.cat([X_unused, y1, y2], dim=1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2)


class DoubleReLUCubic(torch.nn.Module):
    def __init__(self, width, l2_bias=0.0, curvature=1.0, dtype=torch.float, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * 2 - 1)
        self.curvature = curvature

    def base_activation(self, X):
        u = torch.clamp_min(X, -1.0)
        return torch.where(u > 1, u - 3 / 16, 0.5 * u + 3/8 * u**2 - 1/16 * u**4)

    def forward(self, X):
        assert X.shape[1] >= self.width
        X_unused = X[:, :(-self.width)]
        X_used = X[:, (-self.width):] + self.bias
        y1 = self.base_activation(X_used / self.curvature) * self.curvature
        y2 = self.base_activation(-X_used / self.curvature) * self.curvature
        return torch.cat([X_unused, y1, y2], dim=1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2)



class DoubleReLU(torch.nn.Module):
    def __init__(self, width, l2_bias=0.0, dtype=torch.float, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.bias = torch.nn.Parameter(torch.full([width], 0.0, dtype=dtype, device=device))

    def forward(self, X):
        assert X.shape[1] >= self.width
        X_unused = X[:, :(-self.width)]
        X_used = X[:, (-self.width):] + self.bias
        y1 = torch.clamp_min(X_used, 0.0)
        y2 = torch.clamp_max(X_used, 0.0)
        return torch.cat([X_unused, y1, y2], dim=1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2)


class DoubleReLUFlexible(torch.nn.Module):
    def __init__(self, width, l2_bias=0.0, curvature=5.0, dtype=torch.float, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.bias = torch.nn.Parameter(torch.full([width], 0.0, dtype=dtype, device=device))
        self.angles_plus = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * math.pi * 2)
        self.angles_minus = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * math.pi * 2)
        # self.curvature = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * 0.1 + 0.95)
        self.curvature = curvature

    @abstractmethod
    def base_activation(self, X):
        pass

    def forward(self, X):
        assert X.shape[1] >= self.width
        X_unused = X[:, :(-self.width)]
        X_used = X[:, (-self.width):] + self.bias
        u1 = self.base_activation(X_used * self.curvature) / self.curvature
        u2 = self.base_activation(-X_used * self.curvature) / self.curvature
        y1 = u1 * torch.cos(self.angles_plus) + u2 * torch.cos(self.angles_minus)
        y2 = u1 * torch.sin(self.angles_plus) + u2 * torch.sin(self.angles_minus)
        return torch.cat([X_unused, y1, y2], dim=1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2) #+ self.l2_curvature * torch.sum((1 / self.curvature) ** 2)

class DoubleReLUFlexibleLinear(DoubleReLUFlexible):
    def base_activation(self, X):
        return torch.clamp_min(X, 0.0)


class DoubleReLUFlexibleQuadratic(DoubleReLUFlexible):
    def base_activation(self, X):
        u = torch.clamp_min(X, -1)
        return torch.where(u > 1, u, 0.25 * (u + 1) ** 2)

class DoubleReLUFlexibleSmooth(DoubleReLUFlexible):
    def base_activation(self, X):
        return 0.5 * (X - 1 + torch.sqrt(X ** 2 + 1))


class DoubleReLUFlexibleSoftPlus(DoubleReLUFlexible):
    def base_activation(self, X):
        return torch.nn.functional.softplus(X)

class DoubleReLUFlexibleSqrt(DoubleReLUFlexible):
    def base_activation(self, X):
        u = torch.clamp_min(X, -1.0)
        c = math.sqrt(2.0) / 3 * (math.sqrt(8.0) - 1)
        y = torch.where(u > 1, u - 1 + c, math.sqrt(2.0) / 3 * ((1 + u) ** 1.5 - 1))
        return y


class DoubleSigmoid(torch.nn.Module):
    def __init__(self, width, l2_bias=0.0, l2_clamp=0.0, dtype=torch.float, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.l2_clamp = l2_clamp
        self.bias = torch.nn.Parameter(torch.full([width], 0.0, dtype=dtype, device=device))
        self.clamp = torch.nn.Parameter(torch.full([width], 1.0, dtype=dtype, device=device))
        # self.clamp = torch.full([width], 1.0, dtype=dtype, device=device)

    @abstractmethod
    def base_activation(self, X):
        pass

    def forward(self, X):
        assert X.shape[1] >= self.width
        X_unused = X[:, :(-self.width)]
        X_used = X[:, (-self.width):] + self.bias
        y1, y2 = self.base_activation(X_used / self.clamp)
        y1 *= self.clamp
        y2 *= self.clamp
        return torch.cat([X_unused, y1, y2], dim=1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2) + self.l2_clamp * torch.sum(self.clamp ** 2)



class DoubleSigmoidQuadratic(DoubleSigmoid):
    def base_activation(self, X):
        u = torch.clamp(X, -1.0, 1.0)
        y1 = u * (1 - 1/2 * torch.abs(u))
        y2 = X - y1
        return y1, y2


class DoubleSigmoidCubic(DoubleSigmoid):
    def base_activation(self, X):
        u = torch.clamp(X, -1.0, 1.0)
        y1 = u - 1/3 * u**3
        y2 = X - y1
        return y1, y2


class DoubleSigmoidQuintic(DoubleSigmoid):
    def base_activation(self, X):
        u = torch.clamp(X, -1.0, 1.0)
        # y1 = u - 1/3 * u**3
        y1 = 1 / 1.875 * (1.875 * u - 1.25 * u ** 3 + 0.375 * u ** 5)
        y2 = X - y1
        return y1, y2


class DoubleSigmoidCos(DoubleSigmoid):
    def base_activation(self, X):
        u = torch.clamp(X, -math.pi / 2, math.pi / 2)
        y1 = torch.sin(u)
        y2 = torch.where(u >= 0, 1 - torch.cos(u), torch.cos(u) - 1) + torch.sign(X) * torch.clamp_min(
            torch.abs(X) - math.pi / 2, 0.0)
        return y1, y2


# class DoubleSigmoidQuadratic(torch.nn.Module):
#     def __init__(self, width, l2_bias=0.0, l2_clamp=0.0, dtype=torch.float, device=None):
#         super().__init__()
#         self.width = width
#         self.l2_bias = l2_bias
#         self.l2_clamp = l2_clamp
#         self.bias = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * 2 - 1)
#         # self.clamp = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * 2 - 1)
#         self.clamp = torch.full([width], 1.0, dtype=dtype, device=device)
#
#     def forward(self, X):
#         assert X.shape[1] >= self.width
#         X_unused = X[:, :(-self.width)]
#         X_used = (X[:, (-self.width):] + self.bias)
#         u = torch.clamp(X_used / self.clamp, -1.0, 1.0)
#         y1 = self.clamp * u * (1 - 1/2 * torch.abs(u))
#         y2 = X_used - y1
#         return torch.cat([X_unused, y1, y2], dim=1)
#
#     def penalty(self):
#         return self.l2_bias * torch.sum(self.bias ** 2) + self.l2_clamp * torch.sum(self.clamp ** 2)


class DoubleSigmoidLinear(torch.nn.Module):
    def __init__(self, width, l2_bias=0.0, l2_clamp=0.0, dtype=torch.float, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.l2_clamp = l2_clamp
        self.bias = torch.nn.Parameter(torch.full([width], 0.0, dtype=dtype, device=device))
        self.scale = torch.nn.Parameter(torch.rand([width], dtype=dtype, device=device) * 2 - 1)

    def forward(self, X):
        assert X.shape[1] >= self.width
        X_unused = X[:, :(-self.width)]
        X_used = X[:, (-self.width):] + self.bias
        y1 = self.scale * torch.clamp(X_used / self.scale, -1.0, 1.0)
        y2 = X_used - y1
        return torch.cat([X_unused, y1, y2], dim=1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2) + self.l2_clamp * torch.sum(self.scale ** 2)


class OrthogonalExchange(torch.nn.Module):
    def __init__(self, width, working_width, depth, l2_load=0.0, dtype=torch.float, device=None):
        super().__init__()
        while working_width * 2 ** depth >= width * 2:
            depth -= 1
        self.depth = depth
        self.working_width = working_width
        self.width = width
        self.l2_load = l2_load
        self.angles = torch.nn.ParameterList([
            torch.nn.Parameter(torch.rand(min(width - working_width * 2**i, working_width * 2**i),
                                          dtype=dtype, device=device) * 2 * math.pi)
            for i in range(depth - 1, -1, -1)
        ])

    def forward(self, X):
        assert self.width == X.shape[1]
        size = self.working_width * 2 ** (self.depth - 1)
        for i in range(self.depth):
            angles = self.angles[i]
            cos_angles = torch.cos(angles)
            sin_angles = torch.sin(angles)
            X0 = X[:, (-2*size):(-size)]
            X1 = X[:, (-size):]
            X1_unused = X1[:, :(-len(angles))]
            X1_used = X1[:, (-len(angles)):]
            new_X0 = X0 * cos_angles + X1_used * sin_angles
            new_X1 = X0 * -sin_angles + X1_used * cos_angles
            new_X = torch.cat([X[:, :(-2*size)], new_X0, X1_unused, new_X1], dim=1)
            assert new_X.shape == X.shape
            X = new_X
            size //= 2
        return X

    def penalty(self):
        return self.l2_load * sum(torch.sum(torch.sin(angles)**2) for angles in self.angles)


class OrthogonalButterfly(torch.nn.Module):
    def __init__(self, width, depth, l2_interact=0.0, dtype=torch.float, device=None):
        super().__init__()
        self.dtype = dtype
        self.width = width
        self.half_width = width // 2
        self.depth = depth
        self.l2_interact = l2_interact
        initial_angles = torch.rand(self.half_width, depth, dtype=dtype, device=device) * math.pi * 2
        self.angles = torch.nn.Parameter(initial_angles)
        self.perm = torch.zeros([self.width], dtype=torch.long, device=device)
        for i in range(self.width):
            if i % 2 == 0:
                self.perm[i] = i // 2
            else:
                self.perm[i] = i // 2 + self.half_width

    def forward(self, X):
        assert X.dtype == self.dtype
        assert X.shape[1] >= self.width
        X_unused = X[:, :(-self.width)]
        X_used = X[:, (-self.width):]
        for i in range(self.depth):
            angles = self.angles[:, i]
            cos_angles = torch.cos(angles)
            sin_angles = torch.sin(angles)
            X0 = X_used[:, :self.half_width]
            X1 = X_used[:, self.half_width:]
            new_X0 = X0 * cos_angles + X1 * sin_angles
            new_X1 = X0 * -sin_angles + X1 * cos_angles
            X_used = torch.cat([new_X0, new_X1], dim=1)
            X_used = X_used[:, self.perm]
        return torch.cat([X_unused, X_used], dim=1)

    def penalty(self):
        return torch.sum(self.l2_interact * torch.sin(2*self.angles)**2)


class ZeroPadding(torch.nn.Module):
    def __init__(self, zero_padding):
        super().__init__()
        self.zero_padding = zero_padding

    def forward(self, X):
        return torch.cat([X, torch.zeros([X.shape[0], self.zero_padding], dtype=X.dtype, device=X.device)], dim=1)

    def penalty(self):
        return 0.0

class TameNetwork(torch.nn.Module):
    def __init__(self,
                 input_width: int,
                 output_width: int,
                 working_width: int,
                 zero_padding: int,
                 exchange_depths: List[int],
                 butterfly_depth: int,
                 l2_scale=0.0,
                 l2_load=0.0,
                 l2_interact=0.0,
                 l2_bias=0.0,
                 l2_clamp=0.0,
                 curvature=5.0,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.working_width = working_width
        self.zero_padding = zero_padding
        self.exchange_depths = exchange_depths
        self.l2_scale = l2_scale
        self.l2_load = l2_load
        self.l2_interact = l2_interact
        self.l2_bias = l2_bias
        self.scales = torch.nn.Parameter(torch.full([input_width], 1.0, dtype=dtype, device=device))
        # self.scales = torch.full([input_width], 80.0)
        self.layers = []
        self.dtype = dtype
        width = input_width
        for i, exchange_depth in enumerate(exchange_depths):
            padding = max(zero_padding, working_width - width)
            self.layers.append(ZeroPadding(padding))
            width += padding
            self.layers.append(OrthogonalExchange(width, working_width, exchange_depth, l2_load=l2_load, dtype=dtype, device=device))
            self.layers.append(OrthogonalButterfly(working_width, butterfly_depth, l2_interact=l2_interact, dtype=dtype, device=device))
            if i == len(exchange_depths) - 1:
                break
            self.layers.append(DoubleReLU(working_width, l2_bias=l2_bias, dtype=dtype, device=device))

            # self.layers.append(DoubleReLUQuadratic(working_width, l2_bias=l2_bias, curvature=curvature, dtype=dtype, device=device))
            # self.layers.append(
            #     DoubleReLUCubic(working_width, l2_bias=l2_bias, curvature=curvature, dtype=dtype, device=device))

            # self.layers.append(DoubleReLUSqrt2(working_width, l2_bias=l2_bias, curvature=curvature, dtype=dtype, device=device))

            # self.layers.append(
            #     DoubleSigmoidLinear(working_width, l2_bias=l2_bias, dtype=dtype, device=device))
            # self.layers.append(
            #     DoubleSigmoidQuadratic(working_width, l2_bias=l2_bias, l2_clamp=l2_clamp, dtype=dtype, device=device))
            # self.layers.append(
            #     DoubleSigmoidCubic(working_width, l2_bias=l2_bias, l2_clamp=l2_clamp, dtype=dtype, device=device))
            # self.layers.append(
            #     DoubleSigmoidQuintic(working_width, l2_bias=l2_bias, l2_clamp=l2_clamp, dtype=dtype, device=device))
            # self.layers.append(
            #     DoubleSigmoidCos(working_width, l2_bias=l2_bias, l2_clamp=l2_clamp, dtype=dtype, device=device))

            # self.layers.append(DoubleReLUSqrt(working_width, l2_bias=l2_bias, curvature=curvature, dtype=dtype, device=device))
            # self.layers.append(
            #     DoubleReLUFlexibleSqrt(working_width, l2_bias=l2_bias, curvature=curvature, dtype=dtype, device=device))
            # self.layers.append(DoubleReLUSmooth(working_width, l2_bias=l2_bias, dtype=dtype, device=device))
            # self.layers.append(DoubleReLUFlexibleLinear(working_width, l2_bias=l2_bias, dtype=dtype, device=device))
            # self.layers.append(DoubleReLUFlexibleQuadratic(working_width, l2_bias=l2_bias, curvature=curvature, dtype=dtype, device=device))
            # self.layers.append(DoubleReLUFlexibleSmooth(working_width, l2_bias=l2_bias, curvature=curvature, dtype=dtype, device=device))
            # self.layers.append(DoubleReLUFlexibleSoftPlus(working_width, l2_bias=l2_bias, curvature=curvature, dtype=dtype, device=device))
            width += working_width
        self.sequential = torch.nn.Sequential(*self.layers)
        assert width >= output_width

    def forward(self, X):
        assert X.dtype == self.dtype
        assert X.shape[1] == self.input_width
        X_in = X * self.scales
        return self.sequential.forward(X_in)[:, (-self.output_width):]

    def penalty(self):
        for layer in self.layers:
            layer.l2_load = self.l2_load
            layer.l2_interact = self.l2_interact
            layer.l2_bias = self.l2_bias
        return sum(layer.penalty() for layer in self.layers) + self.l2_scale * torch.sum(self.scales ** 2)

# x = torch.arange(1, dtype=torch.float32).view(1, -1)
#
# network = TameNetwork(
#     input_width=1,
#     output_width=1,
#     working_width=2,
#     zero_padding=1,
#     exchange_depths=[3, 2, 2, 1, 3],
#     butterfly_depth=1,
#     l2_scale=0.0,
#     l2_load=0.0,
#     l2_interact=0.0,
#     l2_bias=0.0,
#     curvature=1.0,
#     dtype=torch.float32,
#     device=None
# )
# y = network.forward(x)
#
# butterfly = OrthogonalButterfly(8, 3)
# # exchange = OrthogonalExchange(width=10, working_width=2, depth=3)
# # activation = DoubleReLUQuadratic(2, 1.0)
# # exchange.angles[0][0] = 0.0
# # exchange.angles[0][1] = 0.0
# # exchange.angles[1][0] = 0.0
# # exchange.angles[1][1] = 0.0
# # exchange.angles[1][2] = 0.0
# # exchange.angles[1][3] = 0.0
# # exchange.angles[2][0] = 0.0
# # exchange.angles[2][1] = math.pi / 2
# # y = exchange.forward(x)
# butterfly.angles[0, 0] = 0.0
# butterfly.angles[1, 0] = 0.0
# butterfly.angles[2, 0] = 0.0
# butterfly.angles[3, 0] = 0.0
# butterfly.angles[0, 1] = 0.0
# butterfly.angles[1, 1] = 0.0
# butterfly.angles[2, 1] = 0.0
# butterfly.angles[3, 1] = 0.0
# butterfly.angles[0, 2] = 0.0
# butterfly.angles[1, 2] = 0.0
# butterfly.angles[2, 2] = 0.0
# butterfly.angles[3, 2] = 0
# y = butterfly.forward(x)
# print(y)
# # y = activation.forward(x)
# # print(y)
# # y = exchange.forward(x)
