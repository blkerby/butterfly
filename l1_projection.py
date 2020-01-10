import torch

def simplex_projection(x):
    xv, xi = x.sort(descending=True, dim=1)
    cs = torch.cumsum(xv, dim=1)
    ts = (cs - 1) / torch.arange(1, x.shape[1] + 1, dtype=x.dtype).view(1, -1)
    d = ts - xv
    d1 = torch.where(d > 0, torch.full_like(d, -float('inf')), d)
    js = torch.argmax(d1, dim=1)
    outs = torch.clamp_min(xv - ts[torch.arange(ts.shape[0]), js], 0.0)
    out = torch.empty_like(outs)
    out.scatter_(dim=1, index=xi, src=outs)
    return out

def l1_sphere_projection(x):
    sgn = torch.sign(x)
    return sgn * simplex_projection(torch.abs(x))

def simplex_tangent_projection(xz, g):
    gv, gi = g.sort(descending=True, dim=1)
    xs = torch.gather(xz, dim=1, index=gi)
    zero = torch.zeros_like(gv)
    gz = torch.where(xs, gv, zero)
    gnz = torch.where(xs, zero, gv)
    snz = torch.sum(gnz, dim=1)
    nz1 = torch.sum(~xz, dim=1)
    cs = torch.cumsum(gz, dim=1)
    c1 = torch.cumsum(xs, dim=1) + nz1.view(-1, 1)
    ts = (snz.view(-1, 1) + cs) / c1.type(g.dtype)
    d = ts - gv
    d1 = torch.where(d > 0, torch.full_like(d, -float('inf')), d)
    js = torch.argmax(d1, dim=1)
    gd = gv - ts[torch.arange(ts.shape[0]), js].view(-1, 1)
    outs = torch.where(xs, torch.clamp_min(gd, 0.0), gd)
    out = torch.empty_like(outs)
    out.scatter_(dim=1, index=gi, src=outs)
    return out

def l1_sphere_tangent_projection(x, g):
    xz = x == 0
    sgn = torch.where(xz, torch.sign(g), torch.sign(x))
    return simplex_tangent_projection(xz, g * sgn) * sgn


import abc

class ParameterManifold(torch.nn.Parameter):
    @abc.abstractmethod
    def randomize(self):
        pass

    @abc.abstractmethod
    def retract(self, tangent_vec):
        pass

    @abc.abstractmethod
    def project_gradient(self, penalty_fn):
        pass


class L1Sphere(ParameterManifold):
    def randomize(self):
        """Randomly initialize the points using a uniform distribution"""
        x = torch.log(torch.rand_like(self.data))
        x = x / torch.sum(x, dim=1).view(-1, 1)
        x = x * (torch.randint(2, x.shape, dtype=x.dtype) * 2 - 1)
        self.data = x

    def retract(self, tangent_vec):
        """From the current point, take a step in the direction `tangent_vec`, stopping each component at any
        zero-crossing, and projecting the result back onto the manifold."""
        x = self.data + tangent_vec
        sgn = torch.where(self.data == 0, torch.sign(self.data.grad), torch.sign(self.data))
        stopped = torch.where(torch.sign(x) == -sgn, torch.zeros_like(x), x)
        self.data = l1_sphere_projection(stopped)

    def project_neg_gradient(self, penalty_fn) -> torch.Tensor:
        """Project the negative gradient of the objective onto the space of tangent vectors to the manifold at the
        current point. Here the objective is defined as a loss + penalty, it assumed that the gradient of the
        loss is already stored in `self.data.grad`, and the penalty is defined as the sum of the evaluation of
        `penalty_fn` at the absolute value of `self.data` (it is assumed that `penalty_fn` is smooth -- in particular
        it should be differentiable at 0)."""
        ng = -self.data.grad
        d1 = torch.tensor(torch.abs(self.data), requires_grad=True)
        pen = torch.sum(penalty_fn(d1))
        pen.backward()
        dz = self.data == 0
        sgn = torch.sign(ng)
        abs_ng = torch.abs(ng)
        obj_ng = torch.where(dz, sgn * torch.clamp_min(abs_ng - d1.grad, 0.0), ng - d1.grad)
        return l1_sphere_tangent_projection(self.data, obj_ng)

s = L1Sphere(torch.zeros([3, 4], dtype=torch.float32))
s.randomize()
print(torch.sum(torch.abs(s.data), dim=1))


b = torch.randn([8, 16]) < 0.0
A = torch.randn([8, 16])

# out = simplex_tangent_projection(
#     torch.tensor([[False, True, False, False]]),
#     torch.tensor([[1.5, 0.7, 0.6, 0.2]]))
out = simplex_tangent_projection(b, A)
print(torch.sum(out, dim=1))
#
# out = simplex_projection(torch.tensor([[1.1, 0.1, 1.2, 1.0],
#                                        [0.8, -0.2, 0.5, 0.3]]))
# print(out)
#
# out = l1_sphere_projection(torch.tensor([[1.1, 0.1, 1.2, 1.0],
#                                        [-0.8, -0.2, -0.5, 0.3]]))
# print(out)