import torch
import torch.autograd
import torch.nn
import torch.utils.cpp_extension
import math
import os
import logging
from scipy.stats import special_ortho_group

import cpp.butterfly
from sr1_optimizer import SR1Optimizer

import time

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel('INFO')


device = torch.device('cpu')
dtype = torch.float64
# num_activations = 0
# idx_out = 16
# num_input_layers = 16
# num_output_layers = 16
# num_layers = num_input_layers + num_output_layers
# indices_in = torch.arange(16, dtype=torch.int, device=device)
# model = cpp.butterfly.ButterflyModule(indices_in, 16, num_input_layers, num_output_layers, num_activations, curvature=100.0,
#                                       l2_interact=0, l2_bias=0, biases_initial_std=1.0, dtype=dtype, device=device)

model = cpp.butterfly.ButterflyNetwork(
    input_width=64,
    output_width=64,
    zero_inputs=0,
    network_depth=16,
    initial_scale=1.0,
    l2_scale=0.0,
    butterfly_in_depth=4,
    butterfly_out_depth=4,
    activations_per_block=0,
    blocks_per_layer=4,
    curvature=1.0,
    l2_interact=0.0,
    l2_bias=0.0,
    dtype=dtype,
    device=device)


# # #
# torch.random.manual_seed(0)
# f = cpp.butterfly.ButterflyFunction()
# data = torch.rand([16 + num_activations, 512], dtype=dtype, device=device)
# angles = torch.rand([num_layers, 8], dtype=dtype, device=device)
# biases = torch.rand([num_activations], dtype=dtype, device=device)
# curvature = 1.0
# indices_in = torch.arange(16, dtype=torch.int, device=device)
# # data[:, 0] = 10.0
# # data[0, 0] = 3.0
# # data[1, 0] = 7.0
# # delta = torch.rand_like(data[:, 0]) * 1e-2
# # data[:, 0] += delta
# # delta = torch.rand_like(biases) * 1e-2
# # biases += delta
# delta = torch.rand_like(angles) * 1e-5
# angles += delta
# data.requires_grad = True
# biases.requires_grad = True
# angles.requires_grad = True
# #
# # # biases[0] = 0.0
# print(data[:,0])
# out = f.apply(data, angles, biases, curvature, indices_in, idx_out, num_input_layers, num_output_layers, num_activations)
# L = out[0, 0]
# print(data[:,0])
# # print(torch.sum(data[:16, 0] ** 2))
# #
# L.backward()
# print(data[:, 0])
#
# print(data.grad[:, 0])
# # # torch.dot(data.grad[:, 0], delta)
# # torch.dot(biases.grad, delta)
# torch.sum(angles.grad * delta)*10000


# #
# # print(biases.grad)
# # # print(data[:, 0])
# # #
# # # print(out[:, 0])
#
#
# # num_input_layers = 8
# # num_output_layers = 0
# # model = cpp.butterfly.ButterflyNetwork([indices_in, indices_in], [-1, -1], num_input_layers, num_output_layers, dtype=dtype, device=device)

ortho_dim = 64
ortho = torch.tensor(special_ortho_group.rvs(ortho_dim), dtype=dtype, device=device)

X_train = torch.rand(size=[ortho_dim, 1 << 9], dtype=dtype, device=device)
X_test = torch.rand(size=[ortho_dim, 1 << 9], dtype=dtype, device=device)
Y_train = ortho.matmul(X_train[:ortho_dim, :])
Y_test = ortho.matmul(X_test[:ortho_dim, :])

def compute_loss(pY, Y):
    err = Y - pY
    return torch.mean(err ** 2)

def compute_objective(model, loss):
    return loss

optimizer = SR1Optimizer(model.parameters(), memory=2000)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

for i in range(100000):
    eval_cnt = 0

    def closure():
        global obj
        global train_loss

        optimizer.zero_grad()
        model.zero_grad()

        pY = model(X_train)
        train_loss = compute_loss(pY, Y_train)
        obj = compute_objective(model, train_loss)
        obj.backward()
        if isinstance(optimizer, SR1Optimizer):
            return obj, train_loss
        else:
            return obj

    optimizer.step(closure)

    if i % 10 == 0:
        with torch.no_grad():
            if isinstance(optimizer, SR1Optimizer):
                gn = torch.sqrt(torch.sum(optimizer.state['grad'] ** 2))
                train_loss = float(optimizer.state['loss'])
                obj = float(optimizer.state['f'])
                tr = float(optimizer.state['tr_radius'])
            else:
                gn = torch.sqrt(sum(torch.sum(x.grad ** 2) for x in model.parameters()))
                tr = 0
            # interact = torch.mean(torch.sin(2 * model.angles) ** 2)

            pY_test = model(X_test.clone())
            test_loss = compute_loss(pY_test, Y_test)

                # logging.info("iter={}: obj={:.8f}, train={:.8f}, test={:.8f}, grad={:.7g}".format(
                #     i, float(obj), float(train_loss), float(test_loss), gn))

            logging.info("iter={}: obj={:.7g}, train={:.7g}, test={:.7g}, grad={:.7g}, tr_radius={:.3g}".format(
                i, obj, train_loss, float(test_loss), gn, tr))
        if gn < 1e-15:
            break
#
#
# # data = torch.rand(size=[16, 1024], dtype=dtype, device=device)
# data = torch.zeros(size=[16, 1024], dtype=dtype, device=device)
# data[0, 0] = 1.0
# data_in = data.clone()
# angles = torch.rand(size=[8, 8], dtype=dtype, device=device, requires_grad=True)
# # angles = torch.zeros(size=[8, 8], dtype=dtype, device=device)
# indices_in = torch.arange(16, dtype=torch.int, device=device)
# f = cpp.butterfly.ButterflyFunction()
# data_out = f.apply(data, angles, indices_in, -1)
#
# grad_data = torch.zeros_like(data)
# L = data_out[0, 0]
# L.backward()
# angles.grad
#

# # TODO: Possibly use setup.py instead of JIT loading
# butterfly_cpp = torch.utils.cpp_extension.load(
#     name="butterfly_cpp",
#     sources=["cpp/butterfly.cpp"],
#     extra_include_paths=["/opt/rocm/hip/include", "cpp/MIPP/src"],
# )
#
# device = torch.device('cpu')
# dtype = torch.float32
# data = torch.rand(size=[16, 1024], dtype=dtype, device=device)
# data = torch.zeros(size=[16, 1024], dtype=dtype, device=device)
# data[0, 0] = 1.0
# data_in = data.clone()
# angles = torch.rand(size=[8, 8], dtype=dtype, device=device)
# # angles = torch.zeros(size=[8, 8], dtype=dtype, device=device)
# col_indices = torch.arange(16, dtype=torch.int, device=device)
# butterfly_cpp.butterfly_forward(data, angles, col_indices, -1)
# data_out = data.clone()
#
# grad_data = torch.zeros_like(data)
# grad_data[0, 0] = 1.0
# grad_angles = torch.zeros_like(angles)
# butterfly_cpp.butterfly_backward(data, grad_data, angles, grad_angles, col_indices, -1)
#
# print(torch.sum(data_in ** 2))
# print(torch.sum(data_out ** 2))
# print(torch.sum(data ** 2))
# data[:, 1]
