import torch
import torch.optim
import torch.optim.lbfgs
import torch.nn

def compute_loss(pY, Y):
    err = pY - Y
    return (err ** 2).mean()

def compute_objective(params, loss, lam):
    # return loss + lam*torch.mean(params * (1-params))
    # return loss + lam*torch.mean((params**2 + (1-params)**2))
    return loss + lam*torch.mean(params**2)

def gen_data(N, n, perm):
    X = torch.rand([N, n]) * 2 - 1
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
        # rand_sgn = (torch.randint(2, [self.width, depth]) * 2 - 1).type(torch.float)
        # initial_params = rand_sgn * (1 - torch.rand(self.width, depth) / depth)
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
            theta = self.params[:, i]
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            # W1 = self.params[self.half_width:, i]
            X0 = X[:, :self.half_width]
            X1 = X[:, self.half_width:]
            # X0W = X0 * W
            # X1W = X1 * W
            # new_X0 = X0 - X0W + X1W
            # # new_X1 = X1 - X1W + X0W
            # new_X1 = X1 - X1W - X0W
            new_X0 = X0 * cos_theta + X1 * sin_theta
            new_X1 = X0 * -sin_theta + X1 * cos_theta
            # print((U.abs() + V.abs()))
            X = torch.cat([new_X0, new_X1], dim=1)[:, self.perm]
        return X[:, :input_width]


# model = DoublyStochasticButterfly(1, 1)
# model(torch.tensor([[100, 200]], dtype=torch.float))


for seed in range(1000):
    n = 64
    N = 64
    # seed = 0
    # lam = 0.1
    lam = 0.0
    lr = 20.0
    torch.random.manual_seed(seed)
    perm = torch.randperm(n)
    # perm = perm[perm]
    X, Y = gen_data(N, n, perm)
    model = DoublyStochasticButterfly(7, 15)


    # a = torch.zeros([1, 16])
    # a[0, 0] = 1
    # print(model(a))


    last_loss = float("Inf")
    last_gn = float("Inf")
    for i in range(1000000):
        lam *= 1 - 1e-3
        model.zero_grad()
        pY = model(X)
        loss = compute_loss(pY, Y)
        obj = compute_objective(model.params, loss, lam)
        obj.backward()

        with torch.no_grad():
            raw_g = model.params.grad
            g = raw_g
            # g = torch.where(((model.params == 0.0) & (raw_g > 0)) | ((model.params == 1.0) & (raw_g < 0)),
            #                 torch.zeros_like(raw_g), raw_g)
            # print(torch.stack([model.params[:, 0], raw_g[:, 0], g[:, 0]], dim=1))
            # print(X, pY, Y)
            # model.params.data = torch.clamp(model.params.data - lr * g, 0.0, 1.0)
            model.params.data = model.params.data - lr * g

            gn = torch.sqrt(torch.sum(g ** 2))
            if i % 100 == 0:
                print("seed {}, iteration {}: obj {}, loss {}, grad norm {}, lam {}".format(seed, i, float(obj), float(loss), gn, lam))
            # if lam < 1e-5 and (gn < 1e-6 or (last_loss == loss and last_gn == gn)):
            if loss < 1e-5:
                print("seed {}, iteration {}: obj {}, loss {}, grad norm {}, lam {}".format(seed, i, float(obj),
                                                                                            float(loss), gn, lam))
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
