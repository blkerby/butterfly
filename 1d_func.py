import torch
import torch.optim
import torch.optim.lbfgs
import torch.autograd
import torch.nn
import math
import matplotlib.pyplot as plt
from sr1_optimizer import SR1Optimizer
from tame import TameNetwork
from sponge import Sponge

def compute_loss(pY, Y):
    err = pY - Y
    return (err ** 2).mean()

def compute_objective(model, loss):
    return loss + model.penalty()

def gen_data(N, scale, noise, dtype=torch.float):
    X = (torch.rand([N, 1], dtype=torch.float) - 0.5) * scale
    # Y = torch.cos(X)
    # Y_true = X * torch.sin(1 / X)
    Y_true = 0.1 * (torch.sin(X) + 3 * torch.cos(2*X) + 4*torch.sin(3*X) + 5*torch.cos(3*X) + torch.cos(0.7*X))
    # Y_true = torch.round(0.15 * (torch.sin(X) + 3 * torch.cos(2*X) + 4*torch.sin(3*X) + 5*torch.cos(3*X) + torch.cos(0.7*X))) * 1.5
    # Y_true = torch.where(X > 0.2, torch.full_like(X, 1.0), torch.full_like(X, -1.0))
    Y = Y_true + noise * torch.randn([N, 1], dtype=torch.float)
    return torch.tensor(X, dtype=dtype), torch.tensor(Y_true, dtype=dtype), torch.tensor(Y, dtype=dtype)


N = 200
# scale = 25
scale = 5
seed = 18
# dtype = torch.double
dtype = torch.float

torch.random.manual_seed(seed)


def add_noise(X, num_noise_inputs, scale):
    noise = (torch.rand([X.shape[0], num_noise_inputs], dtype=X.dtype, device=X.device) - 0.5) * scale
    return torch.cat([X, noise], dim=1)

# Generate the data
X, Y_true, Y = gen_data(N, scale, noise=0.0, dtype=dtype)
X_test, _, Y_test = gen_data(5000, scale, 0, dtype)

num_noise_inputs = 20
X = add_noise(X, num_noise_inputs, scale)
X_test = add_noise(X_test, num_noise_inputs, scale)

# model = TameNetwork(
#     input_width=1 + num_noise_inputs,
#     output_width=1,
#     working_width=4,
#     zero_padding=0,
#     exchange_depths=[6,1,2,1,3,1,2,1,4,1,2,1,3,1,2,1,5,1,2,1,3,1,2,1,4,1,2,1,3,1,2,1,6],
#     butterfly_depth=2,
#     l2_scale=1e-3,
#     l2_load=0.0,
#     l2_interact=0.0,
#     l2_bias=1e-7,
#     l2_activation=0.0,
#     # l2_clamp=0.0,
#     curvature=5.0,
#     l2_curvature=1e-7,
#     # l2_clamp=1e-4,
#     dtype=dtype,
#     device=None
# )


model = Sponge(
    input_size=1 + num_noise_inputs,
    output_size=1,
    sponge_size=4,
    activation_size=1,
    recall_size=0,
    depth=8,
    butterfly_depth=2,
    neutral_curvature=1.0,
    l2_scale=0.0,
    l2_interact=0.0,#1e-4,
    l2_curvature=1e-7,
    l2_bias=0.0,
    # activation_function=sine_activation(5.0), #relu_activation,
    dtype=dtype,
    device=None
)

print("Number of parameters: {}".format(sum(len(x.view(-1)) for x in model.parameters())))


# optimizer = AGDOptimizer(model.parameters())
optimizer = SR1Optimizer(model.parameters(), memory=2000)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=10, history_size=1000, tolerance_grad=0, tolerance_change=0,
#                               line_search_fn='strong_wolfe')
# optimizer =torch.optim.SGD(model.parameters(), lr=0.02, nesterov=True, momentum=0.1)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
batch_size = X.shape[0]

fig = plt.gcf()
fig.show()
fig.canvas.draw()

last_loss = float("Inf")
last_gn = float("Inf")
for i in range(100000):
    eval_cnt = 0
    # model.l2_interact *= 0.999
    # model.l2_scale *= 0.999

    def closure():
        global eval_cnt
        eval_cnt += 1
        optimizer.zero_grad()
        model.zero_grad()

        # ind = torch.randint(0, X.shape[0], (X.shape[0],))
        # X_batch = X[ind, :]
        # Y_batch = Y[ind]

        pY = model(X)
        loss = compute_loss(pY, Y)
        # pY = model(X_batch)
        # loss = compute_loss(pY, Y_batch)
        obj = compute_objective(model, loss)
        obj.backward()
        return obj

    optimizer.step(closure)
    # optimizer.step(closure)

    if i % 5 == 0:
        # for a in model.activations:
        #     print(a.curvature)
        # print(model.bias)
        # for j, s in enumerate(model.sponge_steps):
        #     print("Layer {}: {}".format(j, torch.mean(torch.sum(s ** 2, dim=1))))

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
                plt.plot(X_test[ind, 0], pY_test[ind, 0], color='red', linewidth=2.0, zorder=1)
                plt.scatter(X[:, 0], Y[:, 0], color='black', marker='.', alpha=0.3, zorder=2)
                plt.plot(X_test[ind, 0], Y_test[ind, 0], color='blue', zorder=3)
                fig.canvas.draw()

            print("seed={}, iteration={}: obj={:.7f}, train={:.7f}, true={:.7f}, obj grad norm={:.7g}, tr_radius={}, eig5={}, scale={:.4f}, wrong_scale={:.4f}".format(
                seed, i, float(obj), float(loss), float(test_loss), gn, optimizer.state['tr_radius'], optimizer.state['eig5'], model.scales[0], torch.norm(model.scales[1:])))
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
