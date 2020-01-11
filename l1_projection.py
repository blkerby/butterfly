import torch
import abc

def simplex_projection(x):
    xv, xi = x.sort(descending=True, dim=1)
    cs = torch.cumsum(xv, dim=1)
    ts = (cs - 1) / torch.arange(1, x.shape[1] + 1, dtype=x.dtype).view(1, -1)
    d = ts - xv
    d1 = torch.where(d > 0, torch.full_like(d, -float('inf')), d)
    js = torch.argmax(d1, dim=1)
    outs = torch.clamp_min(xv - ts[torch.arange(ts.shape[0]), js].view(-1, 1), 0.0)
    out = torch.empty_like(outs)
    out.scatter_(dim=1, index=xi, src=outs)
    return out

def l1_sphere_projection(x):
    sgn = torch.where(x >= 0, torch.full_like(x, 1.0), torch.full_like(x, -1.0))
    return sgn * simplex_projection(torch.abs(x))

def l1_ball_projection(x):
    l1_norm = torch.sum(torch.abs(x), dim=1)
    return torch.where(l1_norm <= 1.0, x, l1_sphere_projection(x))


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


class ParameterManifold(torch.nn.Parameter):
    @abc.abstractmethod
    def project(self):
        pass


class L1Sphere(ParameterManifold):
    def randomize(self):
        """Randomly initialize the points using a uniform distribution"""
        x = torch.log(torch.rand_like(self.data))
        x = x / torch.sum(x, dim=1).view(-1, 1)
        x = x * (torch.randint(2, x.shape, dtype=x.dtype) * 2 - 1)
        self.data = x

    def project(self):
        self.data = l1_sphere_projection(self.data)



class L1Ball(ParameterManifold):
    def randomize(self):
        """Randomly initialize the points using a uniform distribution on the boundary"""
        x = torch.log(torch.rand_like(self.data))
        x = x / torch.sum(x, dim=1).view(-1, 1)
        x = x * (torch.randint(2, x.shape, dtype=x.dtype) * 2 - 1)
        self.data = x

    def project(self):
        self.data = l1_ball_projection(self.data)

    def project_neg_grad(self):
        ng = -self.grad
        d = l1_sphere_tangent_projection(self.data, ng)
        nm = torch.sum(torch.abs(self.data), dim=1)
        return torch.where(nm > 0.9999, d, ng)

class LpBall(ParameterManifold):
    def __new__(cls, data, p):
        return super().__new__(cls, data)

    def __init__(self, data, p):
        self.p = p

    def randomize(self):
        x = torch.log(torch.rand_like(self.data))
        x = x / torch.sum(x, dim=1).view(-1, 1)
        x = x * (torch.randint(2, x.shape, dtype=x.dtype) * 2 - 1)
        self.data = x
        self.project()

    def project(self):
        norm = torch.sum(torch.abs(self.data) ** self.p, dim=1) ** (1 / self.p)
        self.data = torch.where(norm >= 1.0, self.data / norm, self.data)


class Box(ParameterManifold):
    def __new__(cls, data, min_val, max_val):
        return super().__new__(cls, data)

    def __init__(self, data, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def project(self):
        self.data = torch.clamp(self.data, self.min_val, self.max_val)


class TestModule(torch.nn.Module):
    def __init__(self, k, p, pen_scale=0.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.pen_scale = pen_scale
        # self.scale = Box(torch.tensor(0.0), -1.0, 1.0)
        # self.scale = 5.0
        self.w = L1Ball(torch.zeros([1, k], dtype=torch.float32))
        # self.w = LpBall(torch.zeros([1, k], dtype=torch.float32), p=p)
        self.w.randomize()

    def forward(self, X):
        return torch.matmul(self.w, X) * self.scale

    def penalty(self):
        return torch.abs(self.scale) * self.pen_scale


def compute_loss(py, y):
    err = py - y
    return torch.mean(err ** 2)

def compute_objective(loss, model, n):
    return loss + model.penalty() / n

k = 200
knz = 3
N_train = 100
N_test = 100
p = 0.8
torch.manual_seed(3)
w0 = torch.randn([k])
w0[knz:] = 0.0
X_train = torch.randn([k, N_train])
X_test = torch.randn([k, N_test])
y_train = torch.matmul(w0, X_train) + torch.randn([N_train]) * 0.1
y_test = torch.matmul(w0, X_test)
model = TestModule(k, p, pen_scale=2.0)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

model.pen_scale = 4.5
# model.w.p = 1.0
for i in range(10000000):
    model.zero_grad()
    py_train = model(X_train)
    loss = compute_loss(py_train, y_train)
    obj = compute_objective(loss, model, N_train)
    obj.backward()
    # optimizer.step()
    for param in model.parameters():
        # param.data = param.data - param.grad * 0.00001
        if isinstance(param, ParameterManifold):
            param.data = param.data - param.grad * 0.001
            # param.data = param.data + param.project_neg_grad() * 0.001
            param.project()
        else:
            param.data = param.data - param.grad * 0.001

    if i % 100 == 0:
        with torch.no_grad():
            py_test = model(X_test)
            test_loss = compute_loss(py_test, y_test)
            print("iter={}, obj={}, train={}, test={}, scale={}".format(i, obj, loss, test_loss, model.scale))

#
# s = L1Sphere(torch.zeros([3, 4], dtype=torch.float32))
# s.randomize()
# print(torch.sum(torch.abs(s.data), dim=1))
#
#
#
# b = torch.randn([8, 16]) < 0.0
# A = torch.randn([8, 16])
#
# # out = simplex_tangent_projection(
# #     torch.tensor([[False, True, False, False]]),
# #     torch.tensor([[1.5, 0.7, 0.6, 0.2]]))
# out = simplex_tangent_projection(b, A)
# print(torch.sum(out, dim=1))
# #
# out = simplex_projection(torch.tensor([[1.1, 0.1, 1.2, 1.0],
#                                        [0.8, -0.2, 0.5, 0.3]]))
# print(out)
# #
# # out = l1_sphere_projection(torch.tensor([[1.1, 0.1, 1.2, 1.0],
# #                                        [-0.8, -0.2, -0.5, 0.3]]))
# # print(out)
#
# for i in range(100):
#     A = torch.randn([2, 4])
#     A[0,0] = 0.0
#     out = l1_sphere_projection(A)
#     # out = simplex_projection()
#     print(torch.sum(torch.abs(out), dim=1))
