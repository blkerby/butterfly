import torch
import torch.sparse

def butterfly_matrix(a, b, stride):
    segments = len(a) // stride
    idx1 = (torch.arange(stride) + stride * 2 * torch.arange(segments).view([-1, 1])).view([-1])
    idx2 = idx1 + stride
    i = torch.stack([torch.cat([idx1, idx1, idx2, idx2]),
                     torch.cat([idx1, idx2, idx1, idx2])])
    v = torch.cat([a, b, -b, a])
    A = torch.sparse.FloatTensor(i, v)
    return A

def benes_matrices(A, B):
    n = A.shape[1]
    stride = 1
    i = 0
    out = []
    while stride <= n:
        out.append(butterfly_matrix(A[i, :], B[i, :], stride))
        i += 1
        stride *= 2
    stride = n // 2
    while stride >= 1:
        out.append(butterfly_matrix(A[i, :], B[i, :], stride))
        i += 1
        stride //= 2
    return out

def multimatrix_prod(M, X):
    for S in reversed(M):
        X = S.mm(X)
    return X

def normalize(A, B):
    nm = (A ** 2 + B ** 2).sqrt()
    return A / nm, B / nm

def loss(A, B, X, Y):
    M = benes_matrices(A, B)
    pred = multimatrix_prod(M, X)
    err = pred - Y
    return (err ** 2).sum()

def project(gA, gB, A, B):
    u = gA * A + gB * B
    return gA - u * A, gB - u * B

# A = torch.arange(1, 21, dtype=torch.float).view(5, 4)
# B = torch.arange(101, 121, dtype=torch.float).view(5, 4)
A = torch.randn([5, 4], requires_grad=True)
B = torch.randn([5, 4], requires_grad=True)
A, B = normalize(A, B)

perm = torch.randperm(8)
n = 2
X = torch.randn([8, n])
Y = X[perm, :]

L = loss(A, B, X, Y)

print(loss(A, B, X, Y))

for i in range(len(M)):
    print(M[i].to_dense())
