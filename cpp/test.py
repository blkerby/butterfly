import torch
import torch.autograd
import torch.nn
import torch.utils.cpp_extension
import math
import os
import logging

import cpp.butterfly
from sr1_optimizer import SR1Optimizer



logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel('INFO')


device = torch.device('cpu')
dtype = torch.float32
indices_in = torch.arange(16, dtype=torch.int, device=device)
model = cpp.butterfly.ButterflyModule(indices_in, -1)

perm = torch.randperm(16)
perm = perm[perm]

X_train = torch.rand(size=[16, 1024], dtype=dtype, device=device)
X_test = torch.rand(size=[16, 1024], dtype=dtype, device=device)
Y_train = X_train[perm, :]
Y_test = X_test[perm, :]

def compute_loss(pY, Y):
    err = Y - pY
    return torch.mean(err ** 2)

def compute_objective(model, loss):
    return loss

optimizer = SR1Optimizer(model.parameters(), memory=2000)

for i in range(100000):
    eval_cnt = 0

    def closure():
        global eval_cnt
        global train_loss
        global obj
        eval_cnt += 1
        optimizer.zero_grad()
        model.zero_grad()

        pY = model(X_train.clone())
        train_loss = compute_loss(pY, Y_train)
        obj = compute_objective(model, train_loss)
        obj.backward()
        return obj

    optimizer.step(closure)

    if i % 1 == 0:
        with torch.no_grad():
            gn = torch.sqrt(sum(torch.sum(x.grad**2) for x in model.parameters()))
            interact = torch.mean(torch.sin(2 * model.angles) ** 2)

            with torch.no_grad():
                pY_test = model(X_test.clone())
                test_loss = compute_loss(pY_test, Y_test)

            if optimizer.state['f'] != obj:
                logging.info("iter={}: tr_radius={:.3g} (rejected)".format(i, optimizer.state['tr_radius']))
            else:
                logging.info("iter={}: obj={:.8f}, train={:.8f}, test={:.8f}, grad={:.7g}, interact={:.7f}, tr_radius={:.3g}".format(
                    i, float(optimizer.state['f']), float(train_loss), float(test_loss), gn, interact, optimizer.state['tr_radius']))


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
