import torch
import torch.optim
import torch.optim.lbfgs
import torch.nn

def compute_loss(pY, Y):
    err = pY - Y
    return (err ** 2).mean()

def compute_objective(params, loss, lam):
    # return loss + lam*torch.mean(params * (1-params))
    return loss + lam*torch.mean((params**2 + (1-params)**2))

def gen_data(N, n, perm):
    X = torch.rand([N, n])
    # X = torch.randn([N, n])
    # X = torch.empty([N, n])
    # for i in range(N):
    #     X[i, :] = torch.randperm(n, dtype=torch.float) / (n - 1)
    Y = X[:, perm]
    return X, Y

class DoublyStochasticButterfly(torch.nn.Module):
    def __init__(self, width_pow, depth):
        super().__init__()
        self.width_pow = width_pow
        self.width = 2**width_pow
        self.half_width = 2**(width_pow - 1)
        self.depth = depth
        # rand_sgn = (torch.randint(2, [self.half_width, depth]) * 2 - 1).type(torch.float)
        # initial_params = rand_sgn * (1 - torch.rand(self.half_width, depth) / depth) / 2 + 0.5
        initial_params = torch.rand(self.half_width, depth)
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
            W = self.params[:, i]
            X0 = X[:, :self.half_width]
            X1 = X[:, self.half_width:]
            # X0W = X0 * W
            # X1W = X1 * W
            # new_X0 = X0 - X0W + X1W
            # # new_X1 = X1 - X1W + X0W
            # new_X1 = X1 - X1W - X0W
            new_X0 = X0 * (1 - W) + X1 * W
            new_X1 = X0 * W + X1 * (1 - W)
            X = torch.cat([new_X0, new_X1], dim=1)[:, self.perm]
        return X[:, :input_width]


# model = DoublyStochasticButterfly(1, 1)
# model(torch.tensor([[100, 200]], dtype=torch.float))


n = 64
N = 64
# N = 8
seed = 0
lam = 1e-3
# lam = 0.0
lr = 5.0
torch.random.manual_seed(seed)
perm = torch.randperm(n)
X, Y = gen_data(N, n, perm)
model = DoublyStochasticButterfly(7, 22)

last_loss = float("Inf")
last_gn = float("Inf")
for i in range(1000000):
    # lam *= 1 - 1e-3
    model.zero_grad()
    pY = model(X)
    loss = compute_loss(pY, Y)
    obj = compute_objective(model.params, loss, lam)
    obj.backward()

    with torch.no_grad():
        raw_g = model.params.grad
        g = torch.where(((model.params == 0.0) & (raw_g > 0)) | ((model.params == 1.0) & (raw_g < 0)),
                        torch.zeros_like(raw_g), raw_g)
        # print(torch.stack([model.params[:, 0], raw_g[:, 0], g[:, 0]], dim=1))
        # print(X, pY, Y)
        model.params.data = torch.clamp(model.params.data - lr * g, 0.0, 1.0)
        gn = torch.sqrt(torch.sum(g ** 2))
        print("iteration {}: obj {}, loss {}, grad norm {}, lam {}".format(i, float(obj), float(loss), gn, lam))
        if lam < 1e-5 and (gn < 1e-6 or (last_loss == loss and last_gn == gn)):
            break
        last_loss = loss
        last_gn = gn

# model.weight.data = torch.round(model.weight)
X_test, Y_test = gen_data(N*100, n, perm)
print(compute_loss(model(X_test), Y_test))
