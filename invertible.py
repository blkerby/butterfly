from abc import abstractmethod
import torch.autograd
import torch

# class ConsumableTensor:
#     def __init__(self, tensor: torch.Tensor):
#         self.tensor = tensor
#
#     def take(self):
#         t = self.tensor
#         if t is None:
#             raise RuntimeError("ConsumableTensor is empty")
#         self.tensor = None
#         return t
#
#     def put(self, tensor: torch.Tensor):
#         if self.tensor is not None:
#             raise RuntimeError("ConsumableTensor is already non-empty")
#         self.tensor = tensor
#
#
# class InvertibleFunction:
#     @abstractmethod
#     def forward(self, ):
#         pass
#


class DoubleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        X.mul_(2)
        ctx.save_for_backward(X)
        return X

    @staticmethod
    def backward(ctx, dX):
        X, = ctx.saved_variables
        X.div_(2)
        dX.mul_(2)
        return dX

X = torch.arange(10, dtype=torch.float, requires_grad=True)
f = DoubleFunction()
y = f.apply(X)
L = y[2]
L.backward()

print(X.grad)

