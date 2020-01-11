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

model = SparseLinear(
    input_width=13,
    output_width=42,
    degree=3
)

X = torch.zeros([model.input_width, 4])
out = model(X)
print(out.shape)