import torch
import math

def l1_rotation(x, y, c):
    r = torch.abs(x) + torch.abs(y)
    c0 = c % 1
    c1 = torch.where(y >= 0,
                     torch.where(x >= 0, torch.full_like(x, 0), torch.full_like(x, 1)),
                     torch.where(x >= 0, torch.full_like(x, 3), torch.full_like(x, 2)))
    x1 = torch.where(c1 == 0, x,
                     torch.where(c1 == 1, y,
                                 torch.where(c1 == 2, -x, -y)))
    y1 = torch.where(c1 == 0, y,
                     torch.where(c1 == 1, -x,
                                 torch.where(c1 == 2, -y, x)))
    x2 = x1 - c0 * r
    y2 = y1 + c0 * r
    y3 = torch.where(y2 >= r, 2 * r - y2, y2)
    c2 = torch.floor(c1 + c) % 4

    x_out = torch.where(c2 == 0, x2,
                        torch.where(c2 == 1, -y3,
                                    torch.where(c2 == 2, -x2, y3)))
    y_out = torch.where(c2 == 0, y3,
                        torch.where(c2 == 1, x2,
                                    torch.where(c2 == 2, -y3, -x2)))
    return x_out, y_out



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


class L1Butterfly(torch.nn.Module):
    def __init__(self, width_pow, depth, l2_bias, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.width_pow = width_pow
        self.width = 2**width_pow
        self.half_width = 2**(width_pow - 1)
        self.depth = depth
        self.l2_bias = l2_bias
        initial_params = torch.rand(self.half_width, depth, dtype=dtype) * 4
        self.params = torch.nn.Parameter(initial_params)
        # self.bias = torch.nn.Parameter(torch.zeros([self.width, depth], dtype=dtype))
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
            c = self.params[:, i]
            X0 = X[:, :self.half_width] #+ self.bias[:self.half_width, i]
            X1 = X[:, self.half_width:] #+ self.bias[self.half_width:, i]
            new_X0, new_X1 = l1_rotation(X0, X1, c)
            X = torch.cat([new_X0, new_X1], dim=1)
            X = X[:, self.perm]
        return X[:, :input_width]

    def penalty(self):
        return torch.sum(self.l2_bias * self.bias ** 2)

# import matplotlib.pyplot as plt
# import time
#
# xs = []
# ys = []
#
# plt.ion()
# for i in range(100):
#     x = torch.tensor([3.0])
#     y = torch.tensor([-2.0])
#     x1, y1 = l1_rotation(x, y, 4 + i / 20)
#     xs.append(x1)
#     ys.append(y1)
#     xv = torch.cat(xs)
#     yv = torch.cat(ys)
#     plt.scatter(xv, yv)
#     plt.draw()
#     plt.pause(0.1)
#     plt.clf()

