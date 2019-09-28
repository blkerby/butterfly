import torch
import torch.optim
import torch.optim.lbfgs
import torch.nn
import math

def compute_loss(pY, Y):
    err = pY - Y
    return (err ** 2).mean()

def compute_objective(params, loss, lam):
    # return loss + lam*torch.mean(params * (1-params))
    # return loss + lam*torch.mean((params**2 + (1-params)**2))
    return loss + lam*torch.mean(torch.sin(2*params)**2)

def gen_data_perm(N, n, perm):
    X = torch.rand([N, n], dtype=torch.float) * 2 - 1
    Y = X[:, perm]
    return X, Y

def gen_data_cos(N, n, scale):
    X = torch.zeros([N, n], dtype=torch.float)
    X[:, 0] = torch.rand([N], dtype=torch.float) * scale
    Y = torch.cos(X[:, 0])
    return X, Y


class OrthogonalButterfly(torch.nn.Module):
    def __init__(self, width_pow, depth, dtype=torch.float):
        super().__init__()
        self.width_pow = width_pow
        self.width = 2**width_pow
        self.half_width = 2**(width_pow - 1)
        self.depth = depth
        initial_params = torch.rand(self.half_width, depth, dtype=dtype) * math.pi * 2
        self.params = torch.nn.Parameter(initial_params)
        self.perm = torch.zeros([self.width], dtype=torch.long)
        for i in range(self.width):
            if i % 2 == 0:
                self.perm[i] = i // 2
            else:
                self.perm[i] = i // 2 + self.half_width

    def forward(self, X):
        input_width = X.shape[1]
        X = torch.cat([X, torch.zeros([X.shape[0], self.width - X.shape[1]], dtype=X.dtype)], dim=1)
        for i in range(self.depth):
            theta = self.params[:, i]
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            X0 = X[:, :self.half_width]
            X1 = X[:, self.half_width:]
            new_X0 = X0 * cos_theta + X1 * sin_theta
            new_X1 = X0 * -sin_theta + X1 * cos_theta
            X = torch.cat([new_X0, new_X1], dim=1)
            X = X[:, self.perm]
        return X[:, :input_width]


class SmoothBendActivation(torch.nn.Module):
    def __init__(self, width, dtype=torch.float):
        self.bias = torch.nn.Parameter(torch.zeros([width], dtype=dtype))
        self.slope_1 = torch.nn.Parameter(torch.zeros([width], dtype=dtype))
        self.slope_2 = torch.nn.Parameter(torch.zeros([width], dtype=dtype))
        self.curvature = torch.nn.Parameter(torch.zeros([width], dtype=dtype))

    def forward(self, X):
        m1 = torch.exp(self.slope_1)
        m2 = torch.exp(self.slope_2)
        a = (m1 + m2) / 2
        c = (m2 - m1) / (2 * a)
        b = torch.sinh(self.bias)
        k = torch.exp(self.curvature)
        u = a * X - b
        return u + c * torch.sqrt(u**2 + k)



# model = DoublyStochasticButterfly(1, 1)
# model(torch.tensor([[100, 200]], dtype=torch.float))

for seed in range(0, 1000):
    n = 32
    N = 16
    # seed = 0
    # lam = 0.02
    # lam = 0.005
    lam = 0
    torch.random.manual_seed(seed)
    perm = torch.randperm(n)
    perm = perm[perm]
    X, Y = gen_data(N, n, perm)
    model = OrthogonalButterfly(6, 19)



    optimizer = torch.optim.LBFGS([model.params], tolerance_grad=0, tolerance_change=0,
                                  line_search_fn='strong_wolfe')
    # a = torch.zeros([1, 16])
    # a[0, 0] = 1
    # print(model(a))


    last_loss = float("Inf")
    last_gn = float("Inf")
    for i in range(1000000):
        # lam *= 1 - 1e-3
        eval_cnt = 0

        def closure():
            global eval_cnt
            eval_cnt += 1
            optimizer.zero_grad()
            model.zero_grad()
            pY = model(X)
            loss = compute_loss(pY, Y)
            obj = compute_objective(model.params, loss, lam)
            obj.backward()
            return obj

        optimizer.step(closure)


        with torch.no_grad():
            g = model.params.grad
            pY = model(X)
            loss = compute_loss(pY, Y)
            obj = compute_objective(model.params, loss, lam)

            gn = torch.sqrt(torch.sum(g ** 2))
            # if i % 100 == 0:
            print("seed {}, iteration {}: obj {}, loss {}, grad norm {}, eval_cnt {}, lam {}".format(seed, i, float(obj), float(loss), gn, eval_cnt, lam))
            if gn < 1e-6 or (last_loss == loss and last_gn == gn):
            # if loss < 1e-7:
                # print("seed {}, iteration {}: loss {}, grad norm {}, lam {}".format(seed, i, float(loss), gn, lam))
                break
            last_loss = loss
            last_gn = gn

    # model.weight.data = torch.round(model.weight)
    X_test, Y_test = gen_data(N*100, n, perm)
    print(compute_loss(model(X_test), Y_test))

#
# w0 = model.params.clone().detach()
# # g0 = model.params.grad.clone().detach()
# g0 = g
with torch.no_grad():
    for t in range(100):
        model.params.data = torch.clamp(w0 + 1.0*(1.2**t) * g0, 0.0, 1.0)
        # model.params.data = w0
        pY = model(X)
        loss = compute_loss(pY, Y)
        print(t, float(loss))

a = torch.tensor([[0,0,0,0,0,1,0,0]], dtype=torch.float)
print(model(a))
