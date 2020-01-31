import torch
import math
import linalg
from trust_region import trust_region_solve
import logging


logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.INFO)

class SFOCore:
    def __init__(self, N, M, L, K, tr_min=1e-6, tr_max=1.0, tr_down=0.5, tr_up=1.1, sr1_clamp=1e5, Hscalar=0.1, dtype=torch.float32, device=None):
        """
        :param N (int): Number of subfunctions
        :param M (int): Number of parameters
        :param L (int): Minimum number of history items to keep, when collapsing subspace
        :param K (int): Maximum dimension of subspace
        :param downscale (float): Factor by which to scale down Hessian diagonal per step (should be <1.0)
        :param delta (float): Maximum ratio between the curvature in the step direction and the spectral radius of the approximate Hessian (should be >1.0)
        """
        self.N = N
        self.M = M
        self.L = L
        self.K = K
        self.tr = tr_max
        self.tr_min = tr_min
        self.tr_max = tr_max
        self.tr_down = tr_down
        self.tr_up = tr_up
        self.sr1_clamp = sr1_clamp
        self.Q = torch.empty([M, K], dtype=dtype, device=device)  # Orthogonal basis of shared subspace
        self.Qdim = 0  # Current dimension of the shared subspace
        self.H = torch.zeros([N, K, K], dtype=dtype, device=device)  # Low-rank part of the Hessian approximation (with respect to Q basis) for each subfunction
        self.Hscalar = Hscalar  # Scalar diagonal part of the Hessian approximation for each subfunction
        self.active = torch.zeros([N], dtype=dtype, device=device)  # Mask of active subfunctions
        self.f = torch.zeros([N], dtype=dtype, device=device)  # Latest evaluated value for each subfunction
        self.x = torch.zeros([N, K], dtype=dtype, device=device)  # Latest evaluated position for each subfunction
        self.g = torch.zeros([N, K], dtype=dtype, device=device)  # Latest evaluated gradient for each subfunction
        self.x_list = [[] for _ in range(N)]  # For each subfunction, list of past positions (with respect to Q basis) -- used when collapsing the subspace
        self.g_list = [[] for _ in range(N)]  # For each subfunction, list of past gradients (with respect to Q basis) -- used when collapsing the subspace
        self.total_f = None

    def update_subfunction(self, i, position, f, grad):
        """
        :param i (int): Index of subfunction
        :param x (1D tensor of size M): Position
        :param g (1D tensor of size M): Gradient
        """
        self.active[i] = True
        self.f[i] = f

        # Expand Q to include `s` in the subspace
        x = linalg.expand_Q(self.Q, position, self.Qdim)
        if x.shape[0] > self.Qdim:
            self.H[i, x.shape[0] - 1, x.shape[0] - 1] = self.Hscalar
            self.Qdim = x.shape[0]
        x = torch.cat([x, torch.zeros([self.K - x.shape[0]], dtype=x.dtype, device=x.device)])  # Pad x with zeros
        self.x[i, :] = x
        self.x_list[i].append(x)

        # Expand Q to include `y` in the subspace
        g = linalg.expand_Q(self.Q, grad, self.Qdim)
        if g.shape[0] > self.Qdim:
            self.H[i, g.shape[0] - 1, g.shape[0] - 1] = self.Hscalar
            self.Qdim = g.shape[0]
        g = torch.cat([g, torch.zeros([self.K - g.shape[0]], dtype=g.dtype, device=g.device)])  # Pad g with zeros
        self.g[i, :] = g
        self.g_list[i].append(g)

        if len(self.x_list[i]) >= 2:
            s = x[:self.Qdim] - self.x_list[i][-2][:self.Qdim]
            y = g[:self.Qdim] - self.g_list[i][-2][:self.Qdim]
            # ys = torch.dot(y, s)
            # H = self.H[i, :self.Qdim, :self.Qdim]
            # Hs = torch.mv(H, s)
            # sHs = torch.dot(s, Hs)
            # if sHs > 0:
            #     self.H[i, :self.Qdim, :self.Qdim] = self.H[i, :self.Qdim, :self.Qdim] + (1 / ys) * torch.ger(y, y) - (1 / sHs) * torch.ger(Hs, Hs)
            H = self.H[i, :self.Qdim, :self.Qdim]
            r = y - torch.mv(H, s)
            rs = torch.dot(r, s)
            rr = torch.dot(r, r)
            # if rs != 0.0:
            #     if rr > self.sr1_clamp * abs(rs):
            #         r = r / (rr / (self.delta * abs(rs)))
            #         rs = torch.dot(r, s)
            #         rr = torch.dot(r, r)
            #         logging.info("Clamping SR1 update: {}".format(rr / rs))
            #     self.H[i, :self.Qdim, :self.Qdim] = self.H[i, :self.Qdim, :self.Qdim] + (1 / rs) * torch.ger(r, r)
            if rr < self.sr1_clamp * abs(rs):
                self.H[i, :self.Qdim, :self.Qdim] = self.H[i, :self.Qdim, :self.Qdim] + (1 / rs) * torch.ger(r, r)
            else:
                logging.info("Skipping SR1 update")
        if self.Qdim >= self.K - 1:
            self.collapse_subspace()

    def eval(self, position):
        """Evaluate the full quadratic model at `position`.
        :returns: quadratic function value, gradient (in Q-basis), and Hessian (in Q-basis)"""
        # TODO: Support evaluating even without all subfunctions active
        assert torch.all(self.active == 1.0)

        x = torch.mv(self.Q[:, :self.Qdim].T, position)
        s = x.view(1, -1) - self.x[:, :self.Qdim]
        Hs = torch.einsum('ijk,ik->ij', self.H[:, :self.Qdim, :self.Qdim], s)
        g0 = torch.sum(self.g[:, :self.Qdim], dim=0)
        f0 = torch.sum(self.f)
        g = g0 + torch.sum(Hs, dim=0)
        f = f0 + torch.einsum('ij,ij->', self.g[:, :self.Qdim], s) + 0.5 * torch.einsum('ij,ij->', Hs, s)
        H = torch.sum(self.H[:, :self.Qdim, :self.Qdim], dim=0)
        return f, g, H

    def collapse_subspace(self):
        xs = [x[:self.Qdim] for ls in self.x_list for x in ls[-self.L:]]
        gs = [g[:self.Qdim] for ls in self.g_list for g in ls[-self.L:]]
        vs = torch.stack(xs + gs, dim=1)
        P, _ = torch.qr(vs)
        ndim = P.shape[1]
        logging.info("Collapsing SFO subspace (dim {} -> {})".format(self.Qdim, ndim))
        Q1 = torch.mm(self.Q[:, :self.Qdim], P)
        x1 = torch.mm(self.x[:, :self.Qdim], P)
        g1 = torch.mm(self.g[:, :self.Qdim], P)
        H1 = torch.einsum('ijk,jm,kn', self.H[:, :self.Qdim, :self.Qdim], P, P)
        # self.Q[:, :ndim] = Q1
        self.x[:, :ndim] = x1
        self.x[:, ndim:] = 0.0
        self.g[:, :ndim] = g1
        self.g[:, ndim:] = 0.0
        self.H[:, ndim:, :] = 0.0
        self.H[:, :, ndim:] = 0.0
        self.H[:, :ndim, :ndim] = H1
        z = torch.zeros([self.K - ndim], dtype=P.dtype, device=P.device)
        self.x_list = [[torch.cat([torch.mv(P.t(), x[:self.Qdim]), z]) for x in ls[-self.L:]] for ls in self.x_list]
        self.g_list = [[torch.cat([torch.mv(P.t(), g[:self.Qdim]), z]) for g in ls[-self.L:]] for ls in self.g_list]
        self.Qdim = ndim

        # To prevent `Q` from drifting from orthonormal, we project it back using its polar decomposition (computed using SVD).
        U, S, V = torch.svd(Q1)
        Q2 = torch.mm(U, V.t())
        self.Q[:, :ndim] = Q2

    def trust_region_step(self, x0):
        f, g, H = self.eval(x0)
        if self.total_f is not None:
            if f < self.total_f:
                self.tr = min(self.tr_max, self.tr * self.tr_up)
            else:
                self.tr = max(self.tr_min, self.tr * self.tr_down)
        self.total_f = f
        eig, U = torch.symeig(H, eigenvectors=True)
        self.Hscalar = torch.median(eig)
        Ug = torch.mv(U.t(), g)
        s = torch.mv(U, trust_region_solve(eig, Ug, self.tr))
        return x0 + torch.mv(self.Q[:, :self.Qdim], s)


class SFO(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 num_subfunctions,
                 min_history,
                 subspace_dim,
                 tr_min=1e-6,
                 tr_max=1.0,
                 tr_down=0.5,
                 tr_up=1.1,
                 sr1_clamp=1e5,
                 Hscalar=0.1,
                 dtype=torch.float32, device=None):
        super().__init__(params, {})
        self.num_params = sum(p.numel() for group in self.param_groups for p in group['params'])
        self.core = SFOCore(
            N=num_subfunctions,
            M=self.num_params,
            L=min_history,
            K=subspace_dim,
            tr_min=tr_min,
            tr_max=tr_max,
            tr_down=tr_down,
            tr_up=tr_up,
            sr1_clamp=sr1_clamp,
            Hscalar=Hscalar,
            dtype=dtype,
            device=device)
        self.next_subfunction = 0

    def step(self, closure):
        f = closure(self.next_subfunction)
        grad = self._gather_grad()
        x0 = self._gather_params()
        self.core.update_subfunction(self.next_subfunction, x0, f, grad)
        x1 = self.core.trust_region_step(x0)
        self._update_params(x1)
        self.next_subfunction = (self.next_subfunction + 1) % self.core.N

    def full_pass(self, closure):
        for i in range(self.core.N):
            f = closure(self.next_subfunction)
            grad = self._gather_grad()
            x0 = self._gather_params()
            self.core.update_subfunction(self.next_subfunction, x0, f, grad)
            self.next_subfunction = (self.next_subfunction + 1) % self.core.N

    def _gather_params(self):
        tensors = []
        for group in self.param_groups:
            for p in group['params']:
                tensors.append(p.data.view(-1))
        x = torch.cat(tuple(tensors))
        assert x.numel() == self.num_params  # Make sure that the number of parameters is unchanged since initialization
        return x

    def _gather_grad(self):
        tensors = []
        for group in self.param_groups:
            for p in group['params']:
                tensors.append(p.grad.data.view(-1))
        g = torch.cat(tuple(tensors))
        assert g.numel() == self.num_params  # Make sure that the number of parameters is unchanged since initialization
        return g

    def _update_params(self, x):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.copy_(x[offset:(offset + numel)].view_as(p.data))
                offset += numel
        assert offset == self.num_params

#
# p = 10
# rho = 0.1
# delta = 2.0
# s = torch.rand([p])
# y = torch.rand([p])
#
# ss = torch.dot(s, s)
# ys = torch.dot(s, y)
# yy = torch.dot(y, y)
# print(rho, ys, yy, yy - ys * delta * rho)
# if ys < 0 or yy > ys * delta * rho:
#     a = (delta - 1) * ss
#     b = delta * rho * ss + (delta - 2) * ys
#     c = delta * rho * ys - yy
#     alpha = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
#     y = y + alpha * s
#     rho += alpha
#     ss = torch.dot(s, s)
#     ys = torch.dot(s, y)
#     yy = torch.dot(y, y)
#     print(rho, ys, yy, yy - ys * delta * rho)
#
#
#
def compute_loss(pY, y):
    return torch.nn.functional.cross_entropy(pY, y) * y.shape[0]

def compute_objective(model, loss, n):
    return loss + model.penalty() * n

import sklearn.datasets
import numpy as np
import torch
import math
import sklearn.preprocessing
np_X_train, np_y_train = sklearn.datasets.load_svmlight_file('data/protein.patch', dtype=np.float32)
np_X_test, np_y_test = sklearn.datasets.load_svmlight_file('data/protein.t.patch', dtype=np.float32)
np_X_train = np_X_train.todense()
np_X_test = np_X_test.todense()

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(np_X_train)
pre_X_train = scaler.transform(np_X_train)
pre_X_test = scaler.transform(np_X_test)

dtype = torch.float32
device = torch.device('cpu')

X_train = torch.tensor(pre_X_train, dtype=dtype, device=device)
X_test = torch.tensor(pre_X_test, dtype=dtype, device=device)
y_train = torch.tensor(np_y_train, dtype=torch.long, device=device)
y_test = torch.tensor(np_y_test, dtype=torch.long, device=device)


# X_train = X_train[:, :5]
# X_test = X_test[:, :5]


class Linear(torch.nn.Module):
    def __init__(self, input_width, output_width, l2_weights, dtype=torch.float32, device=None):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.l2_weights = l2_weights
        self.weights = torch.nn.Parameter(torch.randn([input_width, output_width], dtype=dtype, device=device))
        self.bias = torch.nn.Parameter(torch.randn([output_width], dtype=dtype, device=device))

    def forward(self, X):
        return torch.mm(X, self.weights) + self.bias

    def penalty(self):
        return self.l2_weights * torch.sum(self.weights ** 2)

model = Linear(X_train.shape[1], int(torch.max(y_train) + 1), l2_weights=0.015, dtype=dtype, device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer.param_groups[0]['lr'] = 0.005
num_rows_train = X_train.shape[0]
num_rows_test = X_test.shape[0]

# 2186
num_batches = 3
min_history = 4

sfo = SFO(
    params=model.parameters(),
    num_subfunctions=num_batches,
    min_history=min_history,
    tr_min=1e-5,
    tr_max=5.0 / num_batches,
    tr_down=0.1 ** (1 / num_batches),
    tr_up=1.1 ** (1 / num_batches),
    sr1_clamp=1e10,
    Hscalar=1.0,
    subspace_dim=(min_history * 2) * num_batches * 2,
    dtype=dtype,
    device=device)


def closure(batch_num):
    start_row = batch_num * num_rows_train // num_batches
    end_row = (batch_num + 1) * num_rows_train // num_batches
    X_batch = X_train[start_row:end_row, :]
    y_batch = y_train[start_row:end_row]
    model.zero_grad()
    pY_batch = model(X_batch)
    train_loss = compute_loss(pY_batch, y_batch)
    train_obj = compute_objective(model, train_loss, pY_batch.shape[0])
    train_obj.backward()
    return train_obj
    # msfo.update_subfunction(batch_num,
    #                         model.weights.view(-1).detach(),
    #                         train_obj.detach(),
    #                         model.weights.grad.view(-1).detach())
    # total_train_loss += train_loss.detach()

sfo.full_pass(closure)
sfo.tr_radius = 0.00001
sfo.core.delta = 1e6
for it in range(100000):
    # sfo.tr_radius = 100.0 / (it + 10) ** 1.5
    sfo.step(closure)
    with torch.no_grad():
        # gn = math.sqrt(sum(torch.sum(p.grad ** 2) for p in model.parameters())) / num_rows_train
        pY_test = model(X_test)
        test_loss = compute_loss(pY_test, y_test)
        gn = torch.norm(torch.sum(sfo.core.g, dim=0)) / num_rows_train
        print("it={}, obj={}, grad={}, test={}, tr={}".format(
            it, sfo.core.total_f / num_rows_train, gn, test_loss / num_rows_test, sfo.core.tr))
    # print("epoch={}, obj={}, train={}, test={}, grad={}".format(
    #     epoch, train_obj / num_rows_train, total_train_loss / num_rows_train, test_loss / num_rows_test, gn))