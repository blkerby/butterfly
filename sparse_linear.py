import torch



class SparseLinear(torch.nn.Module):
    def __init__(self, input_width, output_width, degree, dtype=torch.float32, device=None):
        super().__init__()
        self.input_width = input_width
        self.output_width = output_width
        self.degree = degree

        num_blocks = (input_width + degree - 1) // degree
        first_block_start = input_width * torch.arange(degree) // degree
        first_block_end = input_width * (torch.arange(degree) + 1) // degree
        self.input_indices = torch.empty([num_blocks, degree], dtype=torch.long)
        for i in range(degree):
            num_elements = first_block_end[i] - first_block_start[i]
            self.input_indices[:num_elements, i] = torch.arange(first_block_start[i], first_block_end[i])
            if num_elements < num_blocks:
                self.input_indices[num_elements, i] = first_block_end[i] - 1

        self.out_degree = (output_width + num_blocks - 1) // num_blocks
        self.weights = torch.nn.Parameter(torch.randn([num_blocks, degree, self.out_degree],
                                                      dtype=dtype, device=device))

    def forward(self, X):
        out = torch.einsum('ijk,ijl->ilk', X[self.input_indices], self.weights)
        return out.reshape(out.shape[0] * out.shape[1], out.shape[2])[:self.output_width, :]


class SparseNetwork(torch.nn.Module):
    def __init__(self, width, depth, degree):
        super().__init__()
        self.seq = torch.nn.Sequential(*[SparseLinear(width, width, degree) for _ in range(depth)])

    def forward(self, X):
        return self.seq(X)


width = 16
model = SparseNetwork(
    width=width,
    depth=1,
    degree=16
)


perm = torch.randperm(width)
perm = perm[perm]

N = 100
dtype = torch.float32
device = torch.device('cpu')
X_train = torch.rand(size=[width, N], dtype=dtype, device=device)
X_test = torch.rand(size=[width, N], dtype=dtype, device=device)
Y_train = X_train[perm, :]
Y_test = X_test[perm, :]


def compute_loss(pY, Y):
    err = Y - pY
    return torch.mean(err ** 2)

def compute_objective(loss, model, n):
    return loss

for i in range(10000000):
    model.zero_grad()
    pY_train = model(X_train)
    loss = compute_loss(pY_train, Y_train)
    obj = compute_objective(loss, model, Y_train.shape[1])
    obj.backward()
    # optimizer.step()
    for param in model.parameters():
        # param.data = param.data - param.grad * 0.00001
        # if isinstance(param, ParameterManifold):
        #     param.data = param.data - param.grad * 0.01
        #     # param.data = param.data + param.project_neg_grad() * 0.001
        #     param.project()
        # else:
        param.data = param.data - param.grad * 0.1

    if i % 100 == 0:
        with torch.no_grad():
            py_test = model(X_test)
            test_loss = compute_loss(py_test, Y_test)
            print("iter={}, obj={}, train={}, test={}".format(i, obj, loss, test_loss))
