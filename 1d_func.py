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
from tame import TameNetwork

def compute_loss(pY, Y):
    err = pY - Y
    return (err ** 2).mean()

def compute_objective(model, loss):
    return loss + model.penalty()

def gen_data(N, scale, noise, dtype=torch.float):
    X = (torch.rand([N, 1], dtype=torch.float) - 0.5) * scale
    # Y = torch.cos(X)
    # Y_true = X * torch.sin(1 / X)
    # Y_true = 0.1 * (torch.sin(X) + 3 * torch.cos(2*X) + 4*torch.sin(3*X) + 5*torch.cos(3*X) + torch.cos(0.7*X))
    Y_true = torch.round(0.15 * (torch.sin(X) + 3 * torch.cos(2*X) + 4*torch.sin(3*X) + 5*torch.cos(3*X) + torch.cos(0.7*X))) * 1.5
    # Y_true = torch.where(X > 0.2, torch.full_like(X, 1.0), torch.full_like(X, -1.0))
    Y = Y_true + noise * torch.randn([N, 1], dtype=torch.float)
    return torch.tensor(X, dtype=dtype), torch.tensor(Y_true, dtype=dtype), torch.tensor(Y, dtype=dtype)


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

model = TameNetwork(
    input_width=1,
    output_width=1,
    working_width=2,
    zero_padding=1,
    exchange_depths=[2, 2, 2, 2, 2, 2, 2, 2, 5],
    butterfly_depth=1,
    l2_scale=1e-8,
    l2_load=0.0,
    l2_interact=0.0,
    l2_bias=0.0,
    curvature=1.0,
    dtype=torch.float32,
    device=None
)


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

            print("seed={}, iteration={}: obj={:.7f}, train={:.7f}, true={:.7f}, obj grad norm={:.7g}, tr_radius={}, eig5={}, scale={}".format(
                seed, i, float(obj), float(loss), float(test_loss), gn, optimizer.state['tr_radius'], optimizer.state['eig5'], model.scales[0]))
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

            # inputs = []
            # for layer in model.activation_layers:
            #     inputs.append(layer.input.detach())
            # all_inputs = torch.cat(inputs)
            # # print("Activation quantiles: {}".format(np.quantile(all_inputs, np.linspace(0, 1, 8))))
