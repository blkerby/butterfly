import torch
import torch.optim
import torch.optim.lbfgs
import torch.nn

def loss(pY, Y):
    err = pY - Y
    return (err ** 2).mean()

def penalty(lam, W):
    return lam * torch.sum(W * W)

def objective(W, pY, Y, lam):
    return loss(pY, Y) + penalty(lam, W)

def gen_data(N, n, perm):
    X = torch.randn([N, n])
    Y = X[:, perm]
    return X, Y

class DoublyStochasticButterfly(torch.nn.Module):
    def __init__(self, width_pow, depth):
        super().__init__()
        self.width_pow = width_pow
        self.width = 2**width_pow
        self.half_width = 2**(width_pow - 1)
        self.depth = depth
        rand_sgn = (torch.randint(2, [self.half_width, depth]) * 2 - 1).type(torch.float)
        initial_params = rand_sgn * (1 - torch.rand(self.half_width, depth) / depth) / 2 + 0.5
        self.params = torch.nn.Parameter(initial_params)

    def forward(self, X):
        for i in range(self.depth):
            W = self.params[:, i]
            X0 = X[:, :self.half_width]
            X1 = X[:, self.half_width:]
            X0W = torch.matmul(X0, W)
            X1W = torch.matmul(X1, W)
            new_X0 = X0 - X0W + X1W
            new_X1 = X1 - X1W + X0W
            X = torch.cat([new_X0, new_X1], dim=1)
        return X

# model = DoublyStochasticButterfly(1, 1)
# model(torch.tensor([[100, 200]], dtype=torch.float))


n = 128
N = n
seed = 0
lam = 0 # 1e-5
torch.random.manual_seed(seed)
perm = torch.randperm(n)
X, Y = gen_data(N, n, perm)
model = torch.nn.Linear(n, n, bias=False)
W = model.weight

optimizer = torch.optim.LBFGS([W], lr=1.0, max_iter=20, tolerance_grad=1e-15, tolerance_change=1e-15, history_size=100, line_search_fn='strong_wolfe')
# optimizer = torch.optim.SGD([W], lr=10.0)
last_obj = float("Inf")
last_gn = float("Inf")
for i in range(1000000):
    def closure():
        optimizer.zero_grad()
        pY = model(X)
        obj = objective(W, pY, Y, lam)
        obj.backward()
        return obj

    optimizer.step(closure)
    # with torch.no_grad():
    pY = model(X)
    g = model.weight.grad
    gn = torch.sqrt(torch.sum(g ** 2))
    obj = objective(model.weight, pY, Y, lam)
    l = loss(pY, Y)
    print("iteration {}: obj {}, loss {}, grad norm {}".format(i, float(obj), float(l), gn))
    if gn < 1e-6 or (last_obj == obj and last_gn == gn):
        break
    last_obj = obj
    last_gn = gn

# model.weight.data = torch.round(model.weight)
X_test, Y_test = gen_data(N*100, n, perm)
print(loss(model(X_test), Y_test))
