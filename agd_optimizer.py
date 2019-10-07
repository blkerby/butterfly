import torch
from linalg import spectral_update, stable_norm, eps
from trust_region import trust_region_solve

class AGDOptimizer(torch.optim.Optimizer):
    """Gradient descent with an adaptive learning rate"""
    def __init__(self, params, rate=0.01, rate_factor=0.5, growth_clamp=2.0, mu=0.01, f_tol=None):
        super().__init__(params, {})
        self._dtype = self.param_groups[0]['params'][0].data.dtype
        self._numel = sum(p.numel() for group in self.param_groups for p in group['params'])

        self.state['x'] = None
        self.state['f'] = None
        self.state['grad'] = None
        self.state['rate'] = rate
        self.state['rate_factor'] = rate_factor
        self.state['growth_clamp'] = growth_clamp
        self.state['mu'] = mu
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
        f_tol = self.state['f_tol']
        rate = self.state['rate']
        rate_factor = self.state['rate_factor']
        mu = self.state['mu']
        growth_clamp = self.state['growth_clamp']
        if self.state['x'] is not None:
            x0 = self.state['x']
            f0 = self.state['f']
            grad0 = self.state['grad']
        else:
            x0 = self._gather_params()
            f0, grad0 = self._eval(x0, closure)

        # Determine new point using gradient descent at the current rate
        s = -grad0 * rate
        x1 = x0 + s

        # Evaluate the function and its gradient at the new point
        f1, grad1 = self._eval(x1, closure)

        # Adapt the rate
        sg0 = torch.dot(s, grad0)
        sg1 = torch.dot(s, grad1)
        print("rate={}, f0={}, f1={}, sg0={}, sg1={}".format(rate, f0, f1, sg0, sg1))
        rate1 = rate * torch.clamp(rate_factor * -sg0 / (sg1 - sg0), 1 / growth_clamp, growth_clamp)
        # rate1 = rate_factor * -sg0 / (sg1 - sg0)
        rate = mu * rate1 + (1 - mu) * rate
        # rate = rate1

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
        self.state['rate'] = rate

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
