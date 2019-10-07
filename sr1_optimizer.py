import torch


def stable_norm(u):
    """Compute the Euclidean norm of a vector `u` in a numerically stable way. We want a non-zero vector to always have
    a non-zero norm, which, due to the possibility of underflow, is not ensured by the PyTorch function torch.norm() on
    its own (at least not for the torch.float64 datatype; for torch.float32 this might already be the case, at least in
    one configuration, but this is undocumented and we do not want to rely on it.)"""
    m = u.abs().max()
    if m == 0.0:
        return m
    u = u / m
    return u.norm(2) * m


def eps(dtype):
    if dtype == torch.float64:
        return 1e-15
    elif dtype == torch.float32:
        return 1e-6
    else:
        raise RuntimeError("Unexpected dtype: {}".format(dtype))


def expand_Q(Q, u, k, max_iter=3):
    """Given a matrix `Q`, considered as only defined on its first k columns (the remaining columns being assumed to
    be unallocated memory), which are assumed to be orthonormal, modify Q by possibly adding a new column (in place,
    by replacing its (k+1)st column) with a new unit-length column orthogonal to the first `k` columns, determining a
    vector `r` (of length `k` or `k+1`) such that after this operation `Q*r = u`. The situation where `r` has length `k`
    would be when `u` is already numerically equal to a linear combination of the columns of `Q` (in particular this
    happens if `Q` is square.).
    """
    Q0 = Q[:, :k]
    nm0 = stable_norm(u)

    # Subtract away from `u` its projection onto the columns of `Q`; the new value of `u` will then be approximately
    # orthogonal to the columns of `Q`.
    r = Q0.t().mv(u)
    if Q0.shape[0] == k:
        # `Q` is already square, so we already have the required `r` without needing to add a column to `Q` (and
        # in any case it would be impossible to add a column while retaining orthonormality)
        return r
    u = u - Q0.mv(r)

    nm1 = nm0
    for i in range(max_iter):
        nm2 = stable_norm(u)
        if nm2 >= 0.5 * nm1:
            break
        elif nm2 <= eps(u.dtype) * nm0:
            # The original `u` was numerically already equal to a linear combination of the columns of `Q`, so there is no
            # need to add another column (and adding `u1` as a column would risk destroying the orthonormality of `Q`).
            return r
        # For numerical stability we subtract away the projection of `u` onto the columns `Q` again. This is based
        # on a similar idea to the Modified Gram-Schmidt method, except this way is faster and more accurate. This is
        # important to do, since otherwise in certain cases the orthonormality of `Q` could be completely ruined
        # (e.g., see https://fgiesen.wordpress.com/2013/06/02/modified-gram-schmidt-orthogonalization/)
        r1 = Q0.t().mv(u)
        u -= Q0.mv(r1)
        r += r1
        nm1 = nm2
    else:
        raise RuntimeError("expand_Q failed to converge")

    u /= nm2
    Q[:, k] = u
    r = torch.cat([r, nm2.view(-1)])
    return r

def spectral_update(Q, M, lam0, u, c, k):
    """
    Given
    - a matrix `Q`, considered as only defined on its first k columns (the remaining columns being assumed to
    be unallocated memory), which are assumed to be orthonormal
    - a square matrix `M`, considered as only defined on its first k rows and columns
    - a vector `u` having the same number of entries as `Q` has rows
    - scalars `lam0` and `c`
    determines matrices `Q1` and `M1` such that
        Q1*M1*Q1^T + lam0*(I - Q1*Q1^T) = Q*M*Q^T + lam0*(I - Q*Q^T) + c*u*u^T
    `Q` are `M` are modified in place so that they are replaced with `Q1` and `M1` respectively.
    """
    r = expand_Q(Q, u, k)
    m = r.shape[0]
    if m > k:
        M[k, :] = 0.0
        M[:, k] = 0.0
        M[k, k] = lam0
    M[:m, :m].add_(c, torch.ger(r, r))
    return m

class SR1Optimizer(torch.optim.Optimizer):
    def __init__(self, params, lam0=1.0, relax=1.0, relax_multiplier=5, memory=10, f_tol=None, max_condition_num=None):
        super().__init__(params, {})
        self._dtype = self.param_groups[0]['params'][0].data.dtype
        self._numel = sum(p.numel() for group in self.param_groups for p in group['params'])

        self.state['x'] = None
        self.state['f'] = None
        self.state['grad'] = None
        self.state['lam0'] = torch.tensor(lam0, dtype=self._dtype)
        self.state['relax'] = torch.tensor(relax, dtype=self._dtype)
        self.state['relax_multiplier'] = torch.tensor(relax_multiplier, dtype=self._dtype)
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
        relax = self.state['relax']
        relax_multiplier = self.state['relax_multiplier']
        max_condition_num = self.state['max_condition_num']
        if self.state['x'] is not None:
            x0 = self.state['x']
            f0 = self.state['f']
            grad0 = self.state['grad']
        else:
            x0 = self._gather_params()
            f0, grad0 = self._eval(x0, closure)

        # To determine where we should evaluate the function next, we use the relaxed-Newton method to find the minimum
        # of the current quadratic model of the function restricted to a ball of some radius (depending on the
        # parameter `relax`) of the the current point `x0`.
        eig, U = torch.symeig(M, eigenvectors=True)
        if eig.shape[0] > 0:
            lam0 = (lam0 + torch.median(eig)) / 2
            max_eig = max(lam0, eig[-1])
            min_eig = min(lam0, eig[0])
            relax_min = torch.clamp((max_eig - max_condition_num * min_eig) / (max_condition_num - 1), 0.0)
        else:
            relax_min = 0.0

        r = max(relax, relax_min)
        # r = 0.0  # TODO: remove this
        eig_r = eig + r
        Qtg = torch.mv(Q.t(), grad0)
        UQtg = torch.mv(U.t(), Qtg)
        IUQtg = UQtg / eig_r
        pg = grad0 - torch.mv(Q, Qtg)
        cf = 1 / (lam0 + r)
        s = -torch.mv(Q, torch.mv(U, IUQtg)) - cf * pg
        expected_change = -torch.dot(UQtg, (eig / 2 + r) / eig_r ** 2 * UQtg) - (lam0 / 2 + r) / (lam0 + r) ** 2 * torch.sum(pg ** 2)
        x1 = x0 + s

        # Evaluate the function and its gradient at the new point
        f1, grad1 = self._eval(x1, closure)

        print("change={}, expected={}".format(f1 - f0, expected_change))

        # Update our quadratic model of the function, by an SR1 update.
        y = grad1 - grad0
        Qts = Q.t().mv(s)
        ps = s - Q.mv(Qts)
        Bs = Q.mv(M.mv(Qts)) + lam0 * ps
        u = y - Bs
        us = torch.dot(u, s)
        uu = torch.dot(u, u)
        ss = torch.dot(s, s)
        if us * us > eps(self._dtype) * uu * ss:
            c = 1 / us
            k = spectral_update(Q_buf, M_buf, lam0, u, c, k)
        else:
            print("Skipping SR1 update")

        # Update the relaxation parameter (for the next relaxed-Newton step) based on how good the current step was:
        if (f1 - f0 >= 0.75 * expected_change):
            relax = relax * relax_multiplier
        elif (f1 - f0 <= 0.95 * expected_change and relax != relax_min):
            relax = relax / relax_multiplier
        if f1 > f0 + f_tol:
            # The new point is worse than the old one, so we reject it.
            print("Rejecting step")
            x1 = x0
            f1 = f0
            grad1 = grad0
            self._update_params(x0)

        self.state['x'] = x1
        self.state['f'] = f1
        self.state['grad'] = grad1
        self.state['k'] = k
        self.state['lam0'] = lam0
        self.state['relax'] = relax


# n = 4
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
# obj = closure()
# print("obj={}, x={}, grad={}".format(obj, model.x, model.x.grad))
# optimizer.step(closure)
# print("obj={}, x={}, grad={}, relax={}".format(obj, model.x, model.x.grad, optimizer.state['relax']))
#
#
# Q = optimizer.state['Q_buf'][:, :4]
# M = optimizer.state['M_buf'][:4, :4]
