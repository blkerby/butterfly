import torch
# from sr1_optimizer_core import SR1OptimizerCore
from linalg import spectral_update

def project(u, Q):
    """
    Project vector u onto the orthogonal complement of the column space of Q (which must have orthonormal columns).
    """
    if Q.shape[1] == 0:
        # Special case here, to work around some PyTorch bugs/limitations in dealing with empty dimensions
        return u
    u = u - Q.mv(Q.t().mv(u))
    u = u - Q.mv(Q.t().mv(u))   # This second iteration greatly improves the numerical stability.
    return u


def obj_subgrad(x, f_grad, penalty):
    """Returns the minimum-norm subgradient of the objective function. The negative of this is the steepest-descent
    direction. """
    g = f_grad + penalty * torch.sign(x)
    xz = (x == 0)
    gz = g[xz]
    g[xz] = torch.sign(gz) * torch.clamp(torch.abs(gz) - penalty[xz], 0)
    return g


def l1_trust_region(x0, f_grad0, Q, M, lam0, penalty, relax0, max_iter=100, tol=1e-8, obj_tol=1e-12, max_condition_num=None, relax_multiplier=4.0):
    obj0 = torch.sum(penalty * torch.abs(x0))

    x1 = x0
    f_grad1 = f_grad0
    obj1 = obj0
    obj_subgrad1 = obj_subgrad(x1, f_grad1, penalty)
    relax1 = relax0

    def f_grad_and_obj_fn(x):
        dx = x - x0
        Qdx = torch.mv(Q.t(), dx)
        odx = project(dx, Q)
        if M.shape[0] == 0:
            # Special case here, to work around some PyTorch bugs/limitations in dealing with empty dimensions
            QMQdx = torch.zeros([Q.shape[0]], dtype=Q.dtype)
        else:
            QMQdx = torch.mv(Q, torch.mv(M, Qdx))
        Hdx = QMQdx + lam0 * odx
        f_grad = f_grad0 + Hdx
        obj = torch.dot(dx, f_grad0) + 0.5 * torch.dot(dx, Hdx) + torch.sum(penalty * torch.abs(x))
        return f_grad, obj

    for i in range(max_iter):
        # Determine the current active set of variables
        act = (x1 != 0.0) | (obj_subgrad1 != 0.0)
        sgn_act = torch.sign(x1[act])
        sz = x1[act] == 0.0
        sgn_act[sz] = -torch.sign(obj_subgrad1[act][sz])

        # Determine the spectral decomposition of the Hessian projected to the active set
        Q_act0 = Q[act, :]
        if Q_act0.shape[1] == 0:
            # Special case here, to work around some PyTorch bugs/limitations in dealing with empty dimensions
            Q_act = torch.zeros_like(Q_act0)
            # R = torch.zeros([0, 0], dtype=x0.dtype)
            M1 = torch.zeros([0, 0], dtype=x0.dtype)
        else:
            Q_act, R = torch.qr(Q_act0)
            M1 = torch.mm(R, torch.mm(M - lam0 * torch.eye(M.shape[0], dtype=M.dtype), R.t())) + lam0 * torch.eye(R.shape[0], dtype=R.dtype)
        eig, _ = M1.symeig()
        if eig.shape[0] > 0:
            relax_min = torch.clamp((eig[-1] - max_condition_num * eig[0]) / (max_condition_num - 1), 0.0)
            relax1 = torch.clamp(relax1, max(relax_min, 1 / max_condition_num))

        M2 = M1 + relax1 * torch.eye(M1.shape[0], dtype=M1.dtype)
        obj_subgrad1_act = obj_subgrad1[act]
        orth = project(-obj_subgrad1_act, Q_act)
        if M2.shape[0] == 0:
            # Special case here, to work around some PyTorch bugs/limitations in dealing with empty dimensions
            sol = torch.zeros([0], dtype=M2.dtype)
        else:
            Qsd = torch.mv(Q_act.t(), -obj_subgrad1_act)
            sol = torch.gesv(Qsd.view(-1, 1), M2)[0].view(-1)
        s_act = torch.mv(Q_act, sol) + 1 / (lam0 + relax1) * orth
        expected_change = torch.dot(s_act, obj_subgrad1_act + 0.5 * (torch.mv(Q_act, torch.mv(M1, sol)) + lam0 / (lam0 + relax1) * orth))
        x2_act = x1[act] + s_act
        # print("Number projected: {}, orth: {}, s_act: {}, del: {}".format(
        #     torch.sum(torch.sign(x2_act) == -sgn_act), orth.norm(2), s_act.norm(2), x2_act - x1[act]))
        num_proj = torch.sum(torch.sign(x2_act) == -sgn_act)

        x2_act[torch.sign(x2_act) == -sgn_act] = 0.0
        x2 = torch.zeros_like(x1)
        x2[act] = x2_act

        f_grad2, obj2 = f_grad_and_obj_fn(x2)
        obj_subgrad2 = obj_subgrad(x2, f_grad2, penalty)

        # Update the relaxation parameter (for the next relaxed-Newton step) based on how good the current step was:
        if (obj2 - obj1 >= 0.75 * expected_change):
            relax1 = relax1 * relax_multiplier
        elif (obj2 - obj1 <= 0.95 * expected_change):
            relax1 = relax1 / relax_multiplier
        # else:
        #     relax2 = relax1 + torch.dot(obj_subgrad2[act], s_act) / torch.dot(s_act, s_act) #* relax_ratio
        #     relax1 = relax2.clamp(max(relax_min, relax1 / relax_multiplier), relax1 * relax_multiplier)

        # print("Expected: {}, Actual: {}, Num projected={}".format(expected_change, obj2 - obj1, num_proj))
        if obj2 - obj1 < obj_tol:
            x1 = x2
            obj1 = obj2
            obj_subgrad1 = obj_subgrad2

        nm = obj_subgrad1.norm(2)
        # print("Iteration={}, obj={}, subgrad={}, active={}, x1={}, relax={}".format(
        #     i, obj1.item(), nm.item(), torch.sum(x1 != 0.0).item(), x1.norm(2).item(), relax1))
        if nm < tol:
            return x1, obj1 - obj0
    else:
        raise RuntimeError("l1_trust_region failed to converge")

# # for k in range(100):
# dtype = torch.float64
# n = 100
# m = 10
# Q = torch.randn([n, m], dtype=dtype).qr()[0]
# relax = 100000
# M = (torch.diag(torch.tensor(range(m), dtype=dtype) + 1) / m) ** 3 + relax * torch.eye(M.shape[0], dtype=dtype)
#
# # M = (torch.diag(torch.tensor(range(m), dtype=dtype) + 1) / m)
# # f_grad0 = torch.randn(n, dtype=dtype)
# f_grad0 = torch.randn(n, dtype=dtype) ** 2
# # x0 = torch.zeros_like(f_grad0)
# x0 = torch.randn(n, dtype=dtype)
# penalty = torch.full_like(x0, 0.5)
# lam0 = 1.0
#
# x1, ec = l1_trust_region(x0, f_grad0, Q, M, lam0 + relax, penalty, relax0=torch.tensor(0.1, dtype=torch.float64), max_iter=10000, relax_multiplier=4.0, tol=1e-8, max_condition_num=1e8)
# # print((x1 - x0).norm(2))

class SR1Optimizer(torch.optim.Optimizer):
    def __init__(self, params, penalty=0.0, lam0=0.0, relax=1.0, relax_multiplier=5, memory=10, obj_tol=None, max_condition_num=None):
        defaults = {'penalty': penalty }
        super().__init__(params, defaults)
        self._dtype = self.param_groups[0]['params'][0].data.dtype
        self._numel = sum(p.numel() for group in self.param_groups for p in group['params'])

        self.state['x'] = None
        self.state['f'] = None
        self.state['f_grad'] = None
        self.state['obj'] = None
        self.state['obj_subgrad'] = None
        self.state['lam0'] = torch.tensor(lam0, dtype=self._dtype)
        self.state['relax'] = torch.tensor(relax, dtype=self._dtype)
        self.state['relax_multiplier'] = torch.tensor(relax_multiplier, dtype=self._dtype)
        self.state['Q'] = torch.zeros([self._numel, 0], dtype=self._dtype)
        self.state['M'] = torch.zeros([0, 0], dtype=self._dtype)
        self.state['memory'] = memory
        if obj_tol is not None:
            self.state['obj_tol'] = obj_tol
        elif self._dtype == torch.float64:
            self.state['obj_tol'] = 1e-14
        else:
            self.state['obj_tol'] = 1e-6
        if max_condition_num is None:
            if self._dtype == torch.float64:
                max_condition_num = 1e12
            else:
                max_condition_num = 1e4
        self.state['max_condition_num'] = max_condition_num

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

    def _gather_penalty(self):
        tensors = []
        for group in self.param_groups:
            for p in group['params']:
                tensors.append(torch.full_like(p.data.view(-1), group['penalty']))
        x = torch.cat(tuple(tensors))
        assert x.numel() == self._numel  # Make sure that the number of parameters is unchanged since initialization
        return x

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
        penalty = self._gather_penalty()
        obj = f + torch.sum(penalty * torch.abs(x))
        f_grad = self._gather_grad()
        obj_subgrad0 = obj_subgrad(x, f_grad, penalty)
        return f, obj, f_grad, obj_subgrad0

    def step(self, closure):
        Q = self.state['Q']
        M = self.state['M']
        lam0 = self.state['lam0']
        obj_tol = self.state['obj_tol']
        relax = self.state['relax']
        relax_multiplier = self.state['relax_multiplier']
        max_condition_num = self.state['max_condition_num']
        if self.state['x'] is not None:
            x0 = self.state['x']
            f0 = self.state['f']
            obj0 = self.state['obj']
            f_grad0 = self.state['f_grad']
            obj_subgrad0 = self.state['obj_subgrad']
        else:
            x0 = self._gather_params()
            f0, obj0, f_grad0, obj_subgrad0 = self._eval(x0, closure)

        # # To determine where we should evaluate the function next, use the trust-region method to find the minimum of
        # # the current quadratic model of the function restricted to a ball of radius `self.trust_radius` centered at
        # # the current point `self.x0`.
        eig, U = torch.symeig(M, eigenvectors=True)
        if eig.shape[0] > 0:
            # lam0 = (lam0 + torch.median(eig)) / 2
            lam0 = (lam0 + torch.median(eig)) / 2
        penalty = self._gather_penalty()

        if eig.shape[0] > 0:
            relax_min = torch.clamp((eig[-1] - max_condition_num * eig[0]) / (max_condition_num - 1), 0.0)
        else:
            relax_min = 0.0
            # relax = torch.clamp(relax, max(relax_min, 1 / max_condition_num))

        M1 = M + (relax + relax_min) * torch.eye(M.shape[0], dtype=M.dtype)
        # print("symeig: ",M1.symeig()[0])
        x1, expected_change = l1_trust_region(x0, f_grad0, Q, M1, lam0 + relax_min + relax, penalty, relax, max_condition_num=max_condition_num, tol=1e-8, max_iter=10000)
        s = x1 - x0

        # Evaluate the function and its gradient at the new point
        f1, obj1, f_grad1, obj_subgrad1 = self._eval(x1, closure)

        # logging.info("x0: {}, x1: {}, x1-x0: {}, s: {}, lam0: {}, M1: {}".format(x0.norm(2), x1.norm(2), (x1 - x0).norm(2), s.norm(2), lam0, M1))


        # Update our quadratic model of the function, by an SR1 update.
        y = f_grad1 - f_grad0
        Qts = Q.t().mv(s)
        ps = s - Q.mv(Qts)
        Bs = Q.mv(M.mv(Qts)) + lam0 * ps
        u = y - Bs
        us = torch.dot(u, s)
        uu = torch.dot(u, u)
        ss = torch.dot(s, s)
        if us * us > 1e-16 * uu * ss:
            c = 1. / torch.dot(u, s)
            Q, M = spectral_update(Q, M, lam0, u, c)
        else:
            print("Skipping SR1 update")
        # print("Step size={} out of {}".format(s.norm(2), self.trust_radius))

        # Update the relaxation parameter (for the next relaxed-Newton step) based on how good the current step was:

        if (obj1 - obj0 >= 0.75 * expected_change):
            relax = relax * relax_multiplier
        elif (obj1 - obj0 <= 0.95 * expected_change):
            relax = relax / relax_multiplier
        # logging.info("Expected: {}, Actual: {}".format(expected_change, obj1 - obj0))
        if obj1 > obj0 + obj_tol:
            # The new point is worse than the old one, so we reject it.
            x1 = x0
            f1 = f0
            f_grad1 = f_grad0
            obj1 = obj0
            obj_subgrad1 = obj_subgrad0
            self._update_params(x0)

        self.state['x'] = x1
        self.state['f'] = f1
        self.state['f_grad'] = f_grad1
        self.state['obj'] = obj1
        self.state['obj_subgrad'] = obj_subgrad1
        self.state['Q'] = Q
        self.state['M'] = M
        self.state['lam0'] = lam0
        self.state['relax'] = relax
