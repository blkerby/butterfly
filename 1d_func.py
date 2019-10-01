import torch
import butterfly
import torch.optim
import torch.optim.lbfgs
import matplotlib.pyplot as plt
from butterfly import CustomNetwork, FullyConnectedNetwork


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
    X = (torch.rand([N, 1], dtype=dtype) - 0.5) * scale
    # Y = torch.cos(X)
    # Y = X * torch.sin(1 / X)
    Y_true = 0.1 * (torch.sin(X) + 3 * torch.cos(2*X) + 4*torch.sin(3*X) + 5*torch.cos(3*X) + torch.cos(0.7*X))
    Y = Y_true + noise * torch.randn([N, 1], dtype=dtype)
    return X, Y_true, Y


N = 200
scale = 20
seed = 0
# dtype = torch.double
dtype = torch.float

torch.random.manual_seed(seed)

# Generate the data
X, Y_true, Y = gen_data(N, scale, noise=0.25, dtype=dtype)
X_test, _, Y_test = gen_data(5000, scale, 0, dtype)

# Construct/initialize the model (0.023523079231381416)
# model = CustomNetwork(
#     num_inputs=1,
#     num_outputs=1,
#     width_pow=6,
#     depth=3,
#     butterfly_depth=6,
#     l2_slope=0, # 0.00003, #0.0000005, #0.0001,
#     # l2_slope=0.000001, #0.0001,
#     l2_scale=0, #1e-7, #1e-5, #1e-4, #2e-4, #0.0000001, # 0.0000001,#0.00001,
#     l2_bias=0.0,
#     l2_interact=0.0,
#     dtype=dtype
# )

# (0.03714136406779289)
model = FullyConnectedNetwork(
    num_inputs=1,
    num_outputs=1,
    width_pow=6,
    depth=3,
    l2_slope=0, #0.00006, #0.0000005, #0.0001,
    # l2_slope=0.000001, #0.0001,
    l2_scale=0, #1e-7, #1e-5, #1e-4, #2e-4, #0.0000001, # 0.0000001,#0.00001,
    l2_bias=0.0,
    l2_lin=0, #0.00006, #1e-4, #1e-5, #0.0001, #0.0001,
    dtype=dtype
)



optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=5, max_eval=20, history_size=500, tolerance_grad=0, tolerance_change=0,
                              line_search_fn='strong_wolfe')
# optimizer =torch.optim.SGD(model.parameters(), lr=0.1)

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

    if i % 1 == 0:
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
                plt.scatter(X[:, 0], Y[:, 0], color='black', marker='.')
                fig.canvas.draw()

            # if i % 100 == 0:
            print("seed {}, iteration {}: obj {}, train {}, true {}, obj grad norm {}".format(
                seed, i, float(obj), float(loss), float(test_loss), gn))
            if gn < 1e-7 or (last_loss == loss and last_gn == gn):
            # if loss < 1e-7:
                print("seed {}, iteration {}: obj {}, train {}, true {}, obj grad norm {}".format(
                    seed, i, float(obj), float(loss), float(test_loss), gn))
                break
            last_loss = loss
            last_gn = gn


plt.show()
# model.weight.data = torch.round(model.weight)


print(compute_loss(pY_test, Y_test))
#
# model.penalty_coef = 0.0005
# optimizer.param_groups[0]['lr'] = 0.0001