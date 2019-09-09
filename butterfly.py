import torch
import torch.sparse

def butterfly_indices(n, stride):
    segments = n // (stride * 2)
    I1 = (torch.arange(stride) + stride * 2 * torch.arange(segments).view([-1, 1])).view([-1])
    I2 = I1 + stride
    return I1, I2

def benes_indices(n):
    stride = 1
    i = 0
    out = []
    while stride < n:
        out.append(butterfly_indices(n, stride))
        i += 1
        stride *= 2
    stride //= 4
    while stride >= 1:
        out.append(butterfly_indices(n, stride))
        i += 1
        stride //= 2
    return out

def benes_transform(indices, A, B, X):
    for i in range(len(indices)):
        I1 = indices[i][0]
        I2 = indices[i][1]
        X1 = torch.zeros_like(X)
        X1[:, I1] = A[i, :] * X[:, I1] + B[i, :] * X[:, I2]
        X1[:, I2] = -B[i, :] * X[:, I1] + A[i, :] * X[:, I2]
        X = X1
    return X

def normalize(A, B):
    nm = (A ** 2 + B ** 2).sqrt()
    return A / nm, B / nm

def loss(indices, A, B, X, Y):
    n = Y.shape[1]
    pred = benes_transform(indices, A, B, X)[:, :n]
    err = pred - Y
    return (err ** 2).mean()

def penalty(A, B, lam):
    A2 = A ** 2
    B2 = B ** 2
    return lam * (A2 * (1 - A2) + B2 * (1 - B2)).sum()

def objective(indices, A, B, X, Y, lam):
    L = loss(indices, A, B, X, Y)
    pen = penalty(A, B, lam)
    return L + pen

def project(gA, gB, A, B):
    u = gA * A + gB * B
    return gA - u * A, gB - u * B

# A = torch.arange(1, 21, dtype=torch.float).view(5, 4)
# B = torch.arange(101, 121, dtype=torch.float).view(5, 4)

def gen_data(N, n):
    X = torch.randn([N, n])
    return torch.cat([X, X], dim=1)

n = 32
seed = 7
lr = 5.0

torch.random.manual_seed(seed)

indices = benes_indices(n * 2)
perm = torch.randperm(n)
N = n*10
X = gen_data(N, n)
Y = X[:, perm]

# A = (A**2).round()
# B = (B**2).round()

A = torch.randn([len(indices), n])
B = torch.randn([len(indices), n])
A, B = normalize(A, B)
#


# lam = 1e-3
lam = 0

losses = []
grad_norms = []

for i in range(1000000):
    A.requires_grad = True
    B.requires_grad = True
    obj = objective(indices, A, B, X, Y, lam)

    obj.backward()
    A.requires_grad = False
    B.requires_grad = False
    dA, dB = project(A.grad, B.grad, A, B)

    A = A - lr * dA
    B = B - lr * dB
    #
    # rA = torch.randn([len(indices), n // 2])
    # rB = torch.randn([len(indices), n // 2])
    # A = A + rA * 1e-2
    # B = B + rB * 1e-2

    A, B = normalize(A, B)
    L = loss(indices, A, B, X, Y)
    gn = float((dA**2 + dB**2).sum().sqrt())

    losses.append(float(L))
    grad_norms.append(gn)
    print("iteration {}: obj {}, loss {}, grad norm {}".format(i, float(obj), float(L), gn))
    if L < 1e-12:
        break

X_test = gen_data(N, n)
print(loss(indices, A, B, X_test, X_test[:, perm]))

import pandas as pd
df = pd.DataFrame({
    'iteration': list(range(len(losses))),
    'loss': losses,
    'grad_norm': grad_norms,
})
df.to_feather("n{}s{}lr{}.feather".format(n, seed, lr))


#
# for i in range(1000):
#     rA = torch.randn([len(indices), n // 2])
#     rB = torch.randn([len(indices), n // 2])
#     A1 = A + rA * 1e-3
#     B1 = B + rB * 1e-3
#     A1, B1 = normalize(A1, B1)
#     if loss(indices, A1, B1, X, Y) < L:
#         print(loss(indices, A1, B1, X, Y) - L)
