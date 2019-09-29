import torch
import torch.optim
import torch.optim.lbfgs
import torch.nn
import math

def compute_loss(pY, Y):
    err = pY - Y
    return (err ** 2).mean()

def compute_objective(model, loss):
    return loss + model.penalty()

# def gen_data_perm(N, n, perm):
#     X = torch.rand([N, n], dtype=torch.float) * 2 - 1
#     Y = X[:, perm]
#     return X, Y

def gen_data(N, scale):
    X = (torch.rand([N, 1], dtype=torch.double) - 0.5) * scale
    # Y = torch.cos(X)
    Y = X * torch.sin(1 / X)
    return X, Y


class OrthogonalButterfly(torch.nn.Module):
    def __init__(self, width_pow, depth, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.width_pow = width_pow
        self.width = 2**width_pow
        self.half_width = 2**(width_pow - 1)
        self.depth = depth
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
        return 0


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
        m1 = torch.exp(self.slope_1)
        m2 = torch.exp(self.slope_2)
        # m1 = self.slope_1 + 1
        # m2 = self.slope_2 + 1
        a = (m1 + m2) / 2
        c = (m2 - m1) / (2 * a)
        # c = (m2 - m1) / 2
        b = torch.sinh(self.bias)
        k = torch.exp(self.curvature)
        # b = self.bias
        # k = self.curvature ** 2 + 1
        u = X * a - b
        return u + c * torch.sqrt(u**2 + 1 / k)

    def penalty(self):
        return torch.sum(self.penalty_bias * self.bias**2) + \
            torch.sum(self.penalty_slope * (self.slope_1**2 + self.slope_2**2)) + \
            torch.sum(self.penalty_curvature * self.curvature**2)


class CustomNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs, width_pow, depth, penalty_bias, penalty_slope, penalty_curvature, dtype=torch.float):
        super().__init__()
        self.dtype = dtype
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.width_pow = width_pow
        self.depth = depth
        self.penalty_bias = penalty_bias
        self.penalty_slope = penalty_slope
        self.penalty_curvature = penalty_curvature
        self.width = 2 ** width_pow
        self.butterfly_layers = []
        self.activation_layers = []
        self.all_layers = []
        for i in range(depth):
            butterfly = OrthogonalButterfly(width_pow, 1, dtype=dtype)
            activation = SmoothBendActivation(self.width, penalty_bias, penalty_slope, penalty_curvature, dtype=dtype)
            self.butterfly_layers.append(butterfly)
            self.activation_layers.append(activation)
            self.all_layers.append(butterfly)
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
            layer.penalty_bias = self.penalty_bias
            layer.penalty_slope = self.penalty_slope
            layer.penalty_curvature = self.penalty_curvature
            total += layer.penalty()
        return total

N = 1000
scale = 2
seed = 0
torch.random.manual_seed(seed)
model = CustomNetwork(
    num_inputs=1,
    num_outputs=1,
    width_pow=3,
    depth=16,
    # penalty_bias=0.001,
    # penalty_slope=0.004,
    # penalty_curvature=0.0001
    penalty_bias=0.0, #0.00001,
    penalty_slope=0.00001,
    penalty_curvature=0.0, #001,
    dtype=torch.double
)
X, Y = gen_data(N, scale)

model.zero_grad()
pY = model(X)
loss = compute_loss(pY, Y)
obj = compute_objective(model, loss)
obj.backward()
gn = torch.sqrt(sum(torch.sum(X.grad ** 2) for X in model.parameters()))

optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=100, max_eval=500, history_size=20, tolerance_grad=0, tolerance_change=0,
                              line_search_fn='strong_wolfe')
# optimizer =torch.optim.SGD(model.parameters(), lr=0.00001)


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

    model.zero_grad()
    pY = model(X)
    loss = compute_loss(pY, Y)
    obj = compute_objective(model, loss)
    obj.backward()

    with torch.no_grad():
        # print(list(model.parameters()))
        gn = torch.sqrt(sum(torch.sum(X.grad**2) for X in model.parameters()))
        pY = model(X)
        loss = compute_loss(pY, Y)
        obj = compute_objective(model, loss)

        # if i % 100 == 0:
        print("seed {}, iteration {}: obj {}, loss {}, grad norm {}, eval_cnt {}, lam {}".format(seed, i, float(obj), float(loss), gn, eval_cnt, lam))
        if gn < 1e-7 or (last_loss == loss and last_gn == gn):
        # if loss < 1e-7:
            print("seed {}, iteration {}: obj {}, loss {}, grad norm {}, eval_cnt {}, lam {}".format(seed, i, float(obj), float(loss), gn, eval_cnt, lam))
            break
        last_loss = loss
        last_gn = gn



# model.weight.data = torch.round(model.weight)
X_test, Y_test = gen_data(2000, scale)
with torch.no_grad():
    pY_test = model(X_test)


print(compute_loss(pY_test, Y_test))

import matplotlib.pyplot as plt

ind = torch.argsort(X_test[:, 0])
plt.plot(X_test[ind, 0], Y_test[ind, 0])
plt.plot(X_test[ind, 0], pY_test[ind, 0], color='black')
plt.show()
#
# model.penalty_curvature = 0.1
# optimizer.param_groups[0]['lr'] = 0.0001