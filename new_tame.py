import torch.nn
import math
from sr1_optimizer import SR1Optimizer
import sklearn.model_selection
import sklearn.metrics
import logging
import numpy as np


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')
logging.getLogger().setLevel('INFO')


class TameLinear(torch.nn.Module):
    def __init__(self,
                 input_width: int,
                 output_width: int,
                 l2_linear: float,
                 initial_scale: float = 1.0,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        assert output_width <= input_width
        self.input_width = input_width
        self.output_width = output_width
        self.l2_linear = l2_linear
        self.initial_scale = initial_scale
        self.A = torch.nn.Parameter(torch.randn([output_width, input_width], dtype=dtype, device=device) * initial_scale)

    def forward(self, X):
        V, theta, W = torch.svd(self.A)
        cos_theta_minus_1 = torch.cos(theta) - 1
        sin_theta = torch.sin(theta)
        WX = torch.matmul(W.t(), X)
        out = torch.matmul(V, WX * sin_theta.view(-1, 1))
        new_X = X + torch.matmul(W, WX * cos_theta_minus_1.view(-1, 1))
        return new_X, out

    def penalty(self):
        # return self.l2_linear * torch.sum(self.A ** 2)
        return self.l2_linear * torch.sum(torch.abs(self.A))


class DoubleReLUSmooth(torch.nn.Module):
    def __init__(self, width, l2_bias=0.0, curvature=1.0, bias_initial_scale=1.0, dtype=torch.float, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.initial_bias_scale = bias_initial_scale
        self.curvature = curvature
        self.bias = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device) * bias_initial_scale)

    def base_activation(self, X):
        return 0.5 * (X - 1 + torch.sqrt(X ** 2 + 1))

    def forward(self, X):
        assert X.shape[0] == self.width
        y1 = self.base_activation(X / self.curvature) * self.curvature
        y2 = self.base_activation(-X / self.curvature) * self.curvature
        return torch.cat([y1, y2], dim=0)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2)


class TameNetwork(torch.nn.Module):
    def __init__(self,
                 input_width,
                 output_width,
                 depth,
                 num_activations,
                 l2_input=1.0,
                 l2_linear=1.0,
                 l2_bias=0.0,
                 curvature=1.0,
                 input_initial_scale=1.0,
                 linear_initial_scale=1.0,
                 bias_initial_scale=1.0,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.depth = depth
        self.num_activations = num_activations
        self.l2_input = l2_input
        self.l2_linear = l2_linear
        self.l2_bias = l2_bias
        self.curvature = curvature
        self.input_initial_scale = input_initial_scale
        self.linear_initial_scale = linear_initial_scale
        self.bias_initial_scale = bias_initial_scale
        self.input_scales = torch.nn.Parameter(torch.randn([input_width], dtype=dtype, device=device) * input_initial_scale)
        self.linear_layers = []
        self.activation_layers = []
        width = input_width

        for i in range(depth + 1):
            l2_linear = self._get_from_list_or_scalar(self.l2_linear, i, depth + 1)

            if i == depth:
                out_width = output_width
            else:
                out_width = self._get_from_list_or_scalar(self.num_activations, i, depth)
            layer = TameLinear(
                input_width=width,
                output_width=out_width,
                l2_linear=l2_linear,
                initial_scale=linear_initial_scale,
                dtype=dtype,
                device=device)
            self.linear_layers.append(layer)
            width += out_width

            if i < depth:
                l2_bias = self._get_from_list_or_scalar(self.l2_bias, i, depth)
                curvature = self._get_from_list_or_scalar(self.curvature, i, depth)
                layer = DoubleReLUSmooth(
                    width=out_width,
                    l2_bias=l2_bias,
                    curvature=curvature,
                    bias_initial_scale=bias_initial_scale,
                    dtype=dtype,
                    device=device)
                self.activation_layers.append(layer)
                width += out_width

        self.linear_layers = torch.nn.ModuleList(self.linear_layers)
        self.activation_layers = torch.nn.ModuleList(self.activation_layers)

    @staticmethod
    def _get_from_list_or_scalar(x, i, n):
        if isinstance(x, (int, float)):
            return x
        else:
            assert len(x) == n
            return x[i]

    def forward(self, X):
        X = X * self.input_scales.view(-1, 1)
        for i in range(self.depth + 1):
            X, out = self.linear_layers[i](X)
            if i == self.depth:
                return out
            else:
                out = self.activation_layers[i](out)
                X = torch.cat([X, out], dim=0)

    def penalty(self):
        # return self.l2_input * torch.sum(self.input_scales ** 2) + \
        return self.l2_input * torch.sum(torch.abs(self.input_scales)) + \
               sum(layer.penalty() for layer in self.linear_layers) + \
               sum(layer.penalty() for layer in self.activation_layers)

#
#
# def compute_loss(pY, Y):
#     # return torch.nn.functional.binary_cross_entropy_with_logits(pY[:, 0], Y)
#     return torch.nn.functional.binary_cross_entropy_with_logits(pY[0, :], Y)
#
# def compute_objective(model, loss, n):
#     return loss + model.penalty() / n
#
#
#
#
# dtype = torch.float64
# seed = 3
# torch.random.manual_seed(seed)
# N = 1 << 12
# M = 256
# coef = torch.randn([M], dtype=dtype) ** 2 + 0.5
# coef[:(M // 2)] = 0.0
#
# def gen_data(N):
#     X = torch.randn([M, N], dtype=dtype)
#     logit = torch.sum(coef.view(-1, 1) * X, dim=0)
#     y = (torch.rand_like(logit) <= 1.0 / (1 + torch.exp(-logit))).to(dtype)
#     return X, y
#
# X_train, y_train = gen_data(N)
# X_test, y_test = gen_data(N)
#
#
# dtype = torch.float64
# model = TameNetwork(
#     input_width=M,
#     output_width=1,
#     depth=0,
#     num_activations=0,
#     l2_input=0.1,
#     l2_linear=0.075,
#     # l2_input=0.1,
#     # l2_linear=0.075,
#     dtype=dtype)  # 0.14946657
# # X = torch.rand([2, 1], dtype=dtype)
#
# # model.layer.A[:, :] = 0.0
# # model.layer.A[0, 1] = math.pi
# # model.update()
# # Y = model(X)
# # print(X)
# # print(Y)
#
#
# # optimizer = AGDOptimizer(model.parameters())
# optimizer = SR1Optimizer(model.parameters(), memory=2000)
# # optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1, max_eval=10, history_size=1000, tolerance_grad=0, tolerance_change=0,
# #                               line_search_fn='strong_wolfe')
# # optimizer =torch.optim.SGD(model.parameters(), lr=0.02, nesterov=True, momentum=0.1)
# # optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# batch_size = X_train.shape[0]
#
# logging.info("Starting training")
# for i in range(100000):
#     eval_cnt = 0
#     # model.l2_interact *= 0.999
#     # model.l2_scale *= 0.999
#
#     def closure():
#         global eval_cnt
#         global train_loss
#         global obj
#         eval_cnt += 1
#         optimizer.zero_grad()
#         model.zero_grad()
#
#         # ind = torch.randint(0, X.shape[0], (X.shape[0],))
#         # X_batch = X[ind, :]
#         # Y_batch = Y[ind]
#
#         pY = model(X_train)
#         train_loss = compute_loss(pY, y_train)
#         # pY = model(X_batch)
#         # loss = compute_loss(pY, Y_batch)
#         obj = compute_objective(model, train_loss, X_train.shape[1])
#         obj.backward()
#         return obj
#
#     optimizer.step(closure)
#
#     if i % 1 == 0:
#         with torch.no_grad():
#             gn = torch.sqrt(sum(torch.sum(x.grad**2) for x in model.parameters()))
#             # interact = sum(torch.sum(torch.sin(2 * layer.angles) ** 2) for layer in model.layers)
#
#             with torch.no_grad():
#                 pY_test = model(X_test)
#                 test_loss = compute_loss(pY_test, y_test)
#                 # ind = torch.argsort(X_test[:, 0])
#                 # auc = sklearn.metrics.roc_auc_score(y_test.numpy().astype(np.bool), pY_test[:, 0])
#                 ind = torch.argsort(X_test[0, :])
#                 auc = sklearn.metrics.roc_auc_score(y_test.numpy().astype(np.bool), pY_test[0, :])
#
#             if optimizer.state['f'] != obj:
#                 logging.info("iter={}: tr_radius={:.3g} (rejected)".format(i, optimizer.state['tr_radius']))
#             else:
#                 logging.info("iter={}: obj={:.8g}, train={:.8g}, test={:.8f}, auc={:.8f}, grad={:.7g}, scale={:.4f}, tr_radius={:.3g}".format(
#                     i, float(optimizer.state['f']), float(train_loss), float(test_loss), float(auc), gn, torch.norm(model.input_scales), optimizer.state['tr_radius']))
#     #
#     # if i % 5 == 0:
#     #     pickle.dump(model, open('uk/model19.pkl', 'wb'))
