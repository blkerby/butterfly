import torch
import abc
import matplotlib.pyplot as plt

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
    l1_norm = torch.sum(torch.abs(x), dim=1).view(-1, 1)
    out = torch.where(l1_norm <= 1.0, x, l1_sphere_projection(x))
    return out


class ParameterManifold(torch.nn.Parameter):
    @abc.abstractmethod
    def project(self):
        pass


class L1Ball(ParameterManifold):
    def randomize(self):
        """Randomly initialize the points using a uniform distribution on the boundary"""
        x = torch.log(torch.rand_like(self.data))
        x = x / torch.sum(x, dim=1).view(-1, 1)
        x = x * (torch.randint(2, x.shape, dtype=x.dtype) * 2 - 1)
        self.data = x

    def project(self):
        self.data = l1_ball_projection(self.data)


class Box(ParameterManifold):
    def __new__(cls, data, min_val, max_val):
        return super().__new__(cls, data)

    def __init__(self, data, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def project(self):
        self.data = torch.clamp(self.data, self.min_val, self.max_val)


class L1Linear(torch.nn.Module):
    def __init__(self, input_width, output_width, dtype=torch.float32, device=None):
        super().__init__()
        self.weights = L1Ball(torch.zeros([output_width, input_width], dtype=dtype, device=device))
        self.weights.randomize()

    def forward(self, X):
        return torch.matmul(self.weights, X)


class LeakyReLU(torch.nn.Module):
    def __init__(self, width, dtype=torch.float32, device=None):
        super().__init__()
        self.width = width
        self.slope_left = Box(torch.rand([width], dtype=dtype, device=device) * 2 - 1, min_val=-1.0, max_val=1.0)
        self.slope_right = Box(torch.rand([width], dtype=dtype, device=device) * 2 - 1, min_val=-1.0, max_val=1.0)
        self.knot_position = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))

    def forward(self, X):
        X0 = X - self.knot_position.view(-1, 1)
        out = torch.where(X0 >= 0, self.slope_right.view(-1, 1) * X0, self.slope_left.view(-1, 1) * X0)
        return out

class Clamp(torch.nn.Module):
        def __init__(self, width, dtype=torch.float32, device=None):
            super().__init__()
            self.width = width
            self.clamp_width = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
            self.clamp_offset = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))

        def forward(self, X):
            w = torch.abs(self.clamp_width).view(-1, 1)
            c_low = self.clamp_offset.view(-1, 1) - w
            c_high = self.clamp_offset.view(-1, 1) + w
            return torch.max(torch.min(X, c_high), c_low)


class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, width, dtype=torch.float32, device=None):
        super().__init__()
        self.width = width
        self.slope_left = Box(torch.rand([width], dtype=dtype, device=device) * 2 - 1, min_val=-1.0, max_val=1.0)
        self.slope_right = Box(torch.rand([width], dtype=dtype, device=device) * 2 - 1, min_val=-1.0, max_val=1.0)
        self.knot_position = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        # self.rounding = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        self.rounding = torch.tensor(10.0)

    def forward(self, X):
        X0 = X - self.knot_position.view(-1, 1)
        r = self.rounding.view(-1, 1)
        right = 0.5 * (X0 + torch.sqrt(X0 ** 2 + r ** 2) - torch.abs(r))
        left = 0.5 * (X0 - torch.sqrt(X0 ** 2 + r ** 2) - torch.abs(r))
        out = self.slope_left.view(-1, 1) * left + self.slope_right.view(-1, 1) * right
        # out = torch.where(X0 >= 0, self.slope_right.view(-1, 1) * X0, self.slope_left.view(-1, 1) * X0)
        return out


# model = SmoothLeakyReLU(width=1)
# model.slope_left[0]=1.0
# model.slope_right[0]=-1.0
# model.knot_position[0] = 10.0
# with torch.no_grad():
#     xs = 0.1 * (torch.arange(100, dtype=dtype) * 2 - 1)
#     ys = model(xs).view(-1)
#     plt.clf()
#     plt.plot(xs, ys)
#     plt.show()

class L1Network(torch.nn.Module):
    def __init__(self, widths, penalty_scale, dtype=torch.float32, device=None):
        super().__init__()
        self.widths = widths
        self.depth = len(widths) - 1
        self.input_width = widths[0]
        self.output_width = widths[-1]
        self.scale = torch.nn.Parameter(torch.full([self.output_width], 1.0, dtype=dtype, device=device))
        # self.scale = torch.tensor(1.0)
        self.bias = torch.nn.Parameter(torch.zeros([self.output_width], dtype=dtype, device=device))
        self.penalty_scale = torch.tensor(penalty_scale)
        self.layers = []
        for i in range(self.depth):
            self.layers.append(L1Linear(widths[i], widths[i + 1], dtype=dtype, device=device))
            if i != self.depth - 1:
                self.layers.append(LeakyReLU(widths[i + 1]))
                # self.layers.append(SmoothLeakyReLU(widths[i + 1]))
                # self.layers.append(Clamp(widths[i + 1]))
        self.seq = torch.nn.Sequential(*self.layers)

    def forward(self, X):
        return self.seq(X) * self.scale.view(-1, 1) + self.bias.view(-1, 1)

    def penalty(self):
        return torch.sum(torch.abs(self.scale) * self.penalty_scale)


def gen_data(N, scale, dtype=torch.float):
    X = (torch.rand([1, N], dtype=torch.float) - 0.5) * scale
    Y = 0.1 * (torch.sin(X) + 3 * torch.cos(2*X) + 4*torch.sin(3*X) + 5*torch.cos(3*X) + torch.cos(0.7*X))
    return torch.tensor(X, dtype=dtype), torch.tensor(Y, dtype=dtype)

def compute_loss(py, y):
    err = py - y
    return torch.mean(err ** 2)

def compute_objective(loss, model, n):
    return loss + model.penalty() / n

N_train = 100
N_test = 100
dtype = torch.float32
device = torch.device('cpu')
torch.manual_seed(0)

X_train, y_train = gen_data(N_train, 5) #+ torch.randn([N_train, 1], dtype=dtype)
X_test, y_test = gen_data(N_test, 5)

model = L1Network(
    widths=[1, 64, 1],
    penalty_scale=[5.0],
    dtype=dtype,
    device=device)

fig = plt.gcf()
fig.show()
fig.canvas.draw()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# model.pen_scale = 4.5
# model.w.p = 1.0
lr = 0.01
model.penalty_scale = 0.0


for i in range(1000000):
    model.zero_grad()
    py_train = model(X_train)
    loss = compute_loss(py_train, y_train)
    obj = compute_objective(loss, model, N_train)
    obj.backward()
    # optimizer.step()
    for param in model.parameters():
        # param.data = param.data - param.grad * 0.00001
        if isinstance(param, ParameterManifold):
            param.data = param.data - param.grad * lr
            # param.data = param.data + param.project_neg_grad() * 0.001
            param.project()
        else:
            param.data = param.data - param.grad * lr

    if i % 50 == 0:
        with torch.no_grad():
            py_test = model(X_test)
            test_loss = compute_loss(py_test, y_test)
            print("iter={}, obj={}, train={}, test={}, scale={}".format(i, obj, loss, test_loss, model.scale.tolist()))

            ind = torch.argsort(X_test[0, :])
            fig.clear()
            plt.plot(X_test[0, ind], py_test[0, ind], color='red', linewidth=2.0, zorder=1)
            plt.scatter(X_train[0, :], y_train[0, :], color='black', marker='.', alpha=0.3, zorder=2)
            plt.plot(X_test[0, ind], y_test[0, ind], color='blue', zorder=3)
            fig.canvas.draw()


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
