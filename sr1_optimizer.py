import torch
from linalg import spectral_update, stable_norm, eps
from trust_region import trust_region_solve

class SR1Optimizer(torch.optim.Optimizer):
    def __init__(self, params, lam0=1.0, tr_radius=1.0, tr_growth=4.0, tr_factor=10.0, memory=10, f_tol=None):
        super().__init__(params, {})
        self._dtype = self.param_groups[0]['params'][0].data.dtype
        self._numel = sum(p.numel() for group in self.param_groups for p in group['params'])

        self.state['x'] = None
        self.state['f'] = None
        self.state['grad'] = None
        self.state['lam0'] = torch.tensor(lam0, dtype=self._dtype)
        self.state['tr_radius'] = torch.tensor(tr_radius, dtype=self._dtype)
        self.state['tr_growth'] = torch.tensor(tr_growth, dtype=self._dtype)
        self.state['tr_factor'] = torch.tensor(tr_factor, dtype=self._dtype)
        self.state['Q_buf'] = torch.zeros([self._numel, memory], dtype=self._dtype)
        self.state['M_buf'] = torch.zeros([memory, memory], dtype=self._dtype)
        self.state['k'] = 0
        self.state['memory'] = memory
        if f_tol is not None:
            self.state['f_tol'] = f_tol
        elif self._dtype == torch.float64:
            self.state['f_tol'] = 1e-14
        else:
            self.state['f_tol'] = 1e-6

    def _gather_params(self):
        tensors = []
        for group in self.param_groups:
            for p in group['params']:
                tensors.append(p.data.view(-1))
        x = torch.cat(tuple(tensors))
        assert x.numel() == self._numel  # Make sure that the number of parameters is unchanged since initialization
        return x

    def _gather_grad(self):
        tensors = []
        for group in self.param_groups:
            for p in group['params']:
                tensors.append(p.grad.data.view(-1))
        g = torch.cat(tuple(tensors))
        assert g.numel() == self._numel  # Make sure that the number of parameters is unchanged since initialization
        return g

    def _update_params(self, x):
        offset = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.copy_(x[offset:(offset + numel)].view_as(p.data))
                offset += numel
        assert offset == self._numel

    def _eval(self, x, closure):
        self._update_params(x)
        f = closure()
        grad = self._gather_grad()
        return f, grad

    def step(self, closure):
        Q_buf = self.state['Q_buf']
        M_buf = self.state['M_buf']
        k = self.state['k']
        Q = Q_buf[:, :k]
        M = M_buf[:k, :k]
        f_tol = self.state['f_tol']
        lam0 = self.state['lam0']
        tr_radius = self.state['tr_radius']
        tr_growth = self.state['tr_growth']
        tr_factor = self.state['tr_factor']
        if self.state['x'] is not None:
            x0 = self.state['x']
            f0 = self.state['f']
            grad0 = self.state['grad']
        else:
            x0 = self._gather_params()
            f0, grad0 = self._eval(x0, closure)

        # To determine where we should evaluate the function next, we use the trust-region method to find the minimum
        # of the current quadratic model of the function restricted to a ball of radius `tr_radius` centered at
        # the current point `x0`.
        eig, U = torch.symeig(M, eigenvectors=True)
        if eig.shape[0] > 0:
            # First update the quadratic model on the unexplored subspace
            lam0 = (lam0 + torch.median(eig)) / 2
        Qtg = torch.mv(Q.t(), grad0)
        UQtg = torch.mv(U.t(), Qtg)
        g0 = grad0 - torch.mv(Q, Qtg)
        g0_norm = stable_norm(g0)
        ug0 = g0 / g0_norm
        td = torch.empty([len(eig) + 1], dtype=eig.dtype)
        td[0] = lam0
        td[1:] = eig
        tg = torch.empty([len(eig) + 1], dtype=eig.dtype)
        tg[0] = g0_norm
        tg[1:] = UQtg
        ts = trust_region_solve(td, tg, tr_radius)
        s = ts[0] * ug0 + torch.mv(Q, torch.mv(U, ts[1:]))
        x1 = x0 + s

        # Evaluate the function and its gradient at the new point
        f1, grad1 = self._eval(x1, closure)

        # Update our quadratic model of the function, by an SR1 update.
        y = grad1 - grad0
        Qts = Q.t().mv(s)
        ps = s - Q.mv(Qts)
        Bs = Q.mv(M.mv(Qts)) + lam0 * ps
        u = y - Bs
        us = torch.dot(u, s)
        uu = torch.dot(u, u)
        ss = torch.dot(s, s)
        if us * us > 1e-12 * uu * ss:
            c = 1 / us
            k = spectral_update(Q_buf, M_buf, lam0, u, c, k)
        else:
            print("Skipping SR1 update")

        # Update the trust-radius
        g0s = torch.dot(grad0, s)
        g1s = torch.dot(grad1, s)
        tr_radius = torch.norm(s) * torch.clamp(-g0s / (g1s - g0s) * tr_growth, 1 / tr_factor, tr_factor)

        if f1 > f0 + f_tol:
            # The new point is worse than the old one, so we reject it.
            tr_radius /= tr_factor
            x1 = x0
            f1 = f0
            grad1 = grad0
            self._update_params(x0)
        else:
            tr_radius *= tr_growth

        self.state['x'] = x1
        self.state['f'] = f1
        self.state['grad'] = grad1
        self.state['k'] = k
        self.state['lam0'] = lam0
        self.state['tr_radius'] = tr_radius

# n = 8
#
# dtype = torch.double
# A = 2 * torch.eye(n, dtype=dtype) + torch.rand([n, n], dtype=dtype)
# A = A + A.t()
# g0 = torch.rand([n], dtype=dtype)
#
# class TestModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.x = torch.nn.Parameter(torch.rand([n], dtype=dtype))
#
# model = TestModule()
# optimizer = SR1Optimizer(model.parameters())
#
# def closure():
#     model.zero_grad()
#     Ax = torch.mv(A, model.x)
#     obj = torch.dot(0.5 * Ax + g0, model.x)
#     obj.backward()
#     return obj
#
# print("obj={}, x={}, grad={}".format(obj, model.x, model.x.grad))
# optimizer.step(closure)
# obj = closure()
# print("obj={}, x={}, grad={}, tr_radius={}".format(obj, model.x, model.x.grad, optimizer.state['tr_radius']))
#
