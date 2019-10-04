import torch
import torch.autograd
from butterfly_cuda import OrthogonalButterfly, OrthogonalButterflyLayerFunction

dtype = torch.float64
n = 16
device = torch.device('cuda')

X = torch.rand(n, 256, dtype=dtype, device=device)
perm = torch.randperm(n)
perm = perm[perm]
target = X[perm, :]

def loss(pred, target):
    return torch.mean((pred[:len(target)] - target)**2)

# torch.autograd.gradcheck(OrthogonalButterflyLayerFunction.apply,
#     (X, angles), eps=1e-6, atol=1e-4)

model = OrthogonalButterfly(
    width_pow=5, 
    depth=8,
    l2_interact=0.0, 
    dtype=dtype,
    device=device
)

optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, tolerance_grad=1e-15, tolerance_change=1e-15, history_size=100, line_search_fn='strong_wolfe')
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
last_obj = float("Inf")
last_gn = float("Inf")
for i in range(1000000):
    def closure():
        global l
        optimizer.zero_grad()
        model.zero_grad()
        pY = model(X)
        l = loss(pY, target)
        l.backward()
        return l

    optimizer.step(closure)
    with torch.no_grad():
        gn = torch.sqrt(sum(torch.sum(X.grad**2) for X in model.parameters()))
    print("iteration {}: obj {}, grad norm {}".format(i, float(l), float(gn)))
    if gn < 1e-6 or (last_obj == l and last_gn == gn):
        break
    last_obj = l
    last_gn = gn

