from typing import Union
import torch
import math
import random
from tame import OrthogonalButterfly

class ReLUActivation(torch.nn.Module):
    def __init__(self, width, l2_bias,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.width = width
        self.l2_bias = l2_bias
        self.bias = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))

    def base_activation(self, X):
        return torch.clamp_min(X, 0.0)

    def forward(self, X):
        X1 = X + self.bias
        return self.base_activation(X1), self.base_activation(-X1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2)



class FlexibleActivation(torch.nn.Module):
    def __init__(self, width, neutral_curvature,
                 l2_curvature, l2_bias,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.width = width
        self.neutral_curvature = neutral_curvature
        self.l2_curvature = l2_curvature
        self.l2_bias = l2_bias
        self.bias = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        # self.curvature = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        self.curvature = torch.tensor(0.0, dtype=dtype)

    def base_activation(self, X):
        # Transform `curvature` onto a scale between 0 and infinity
        c = 0.5 * self.neutral_curvature * (self.curvature + torch.sqrt(self.curvature ** 2 + 1))

        # Compute the positive threshold beyond which the activation becomes linear
        t = math.pi / 4 / c

        # Compute the value and slope of the line beyond the positive threshold
        y0 = 1 / c / math.sqrt(2)

        # Clamp at the negative threshold
        X1 = torch.max(X, -t)

        return torch.where(X > t, X - t + y0,
                           1 / c * (1 / math.sqrt(2) - torch.cos(math.pi / 4 + c * X1)))

    def forward(self, X):
        X1 = X + self.bias
        return self.base_activation(X1), self.base_activation(-X1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2) + \
                self.l2_curvature * torch.sum(self.curvature ** 2)


class FlexibleQuadraticActivation(torch.nn.Module):
    def __init__(self, width, neutral_curvature,
                 l2_curvature, l2_bias,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.width = width
        self.neutral_curvature = neutral_curvature
        self.l2_curvature = l2_curvature
        self.l2_bias = l2_bias
        self.bias = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        # self.curvature = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        self.curvature = torch.tensor(0.0, dtype=dtype)

    def base_activation(self, X):
        # Transform `curvature` onto a scale between 0 and infinity
        c = 0.5 * self.neutral_curvature * (self.curvature + torch.sqrt(self.curvature ** 2 + 1))

        # Compute the positive threshold beyond which the activation becomes linear
        t = 0.5 / c

        u = torch.max(X, -t)
        return torch.where(u > t, u, 0.25 / t * (u + t) ** 2) #- 0.25 * t

    def forward(self, X):
        X1 = X + self.bias
        return self.base_activation(X1), self.base_activation(-X1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2) + \
                self.l2_curvature * torch.sum(self.curvature ** 2)


class SmoothActivation(torch.nn.Module):
    def __init__(self, width, neutral_curvature,
                 l2_curvature, l2_bias,
                 dtype=torch.float32, device=None):
        super().__init__()
        self.width = width
        self.neutral_curvature = neutral_curvature
        self.l2_curvature = l2_curvature
        self.l2_bias = l2_bias
        self.bias = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        # self.curvature = torch.nn.Parameter(torch.randn([width], dtype=dtype, device=device))
        self.curvature = torch.tensor(0.0, dtype=dtype)

    def base_activation(self, X):
        # Transform `curvature` onto a scale between 0 and infinity
        c = 1.0 / (0.5 * self.neutral_curvature * (self.curvature + torch.sqrt(self.curvature ** 2 + 1)))
        return 0.5 * (X + torch.sqrt(X ** 2 + c ** 2) - c)

    def forward(self, X):
        X1 = X + self.bias
        return self.base_activation(X1), self.base_activation(-X1)

    def penalty(self):
        return self.l2_bias * torch.sum(self.bias ** 2) + \
                self.l2_curvature * torch.sum(self.curvature ** 2)


class Sponge(torch.nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 sponge_size: int,
                 activation_size: int,
                 recall_size: int,
                 depth: int,
                 butterfly_depth: int,
                 neutral_curvature: float,
                 l2_scale: Union[float, torch.Tensor],
                 l2_interact: float,
                 l2_curvature: float,
                 l2_bias: float,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        self.store_size = activation_size + recall_size
        assert recall_size <= sponge_size
        assert activation_size <= sponge_size
        self.input_size = input_size
        self.output_size = output_size
        self.sponge_size = sponge_size
        self.activation_size = activation_size
        self.recall_size = recall_size
        self.depth = depth
        self.butterfly_depth = butterfly_depth
        self.neutral_curvature = neutral_curvature
        self.activation_position = self.store_size
        self.l2_scale = l2_scale
        self.l2_interact = l2_interact
        self.l2_bias = l2_bias
        self.dtype = dtype
        self.angles = torch.nn.Parameter(torch.rand([depth, butterfly_depth, sponge_size // 2], dtype=dtype, device=device) * 2 * math.pi)
        self.scales = torch.nn.Parameter(torch.full([input_size], 1.0 / math.sqrt(input_size), dtype=dtype, device=device))
        self.activations = torch.nn.ModuleList([SmoothActivation(
            width=activation_size,
            neutral_curvature=neutral_curvature,
            l2_curvature=l2_curvature,
            l2_bias=l2_bias,
            dtype=dtype,
            device=device,
        ) for _ in range(depth)])
        # self.activations = torch.nn.ModuleList([FlexibleQuadraticActivation(
        #     width=activation_size,
        #     neutral_curvature=neutral_curvature,
        #     l2_curvature=l2_curvature,
        #     l2_bias=l2_bias,
        #     dtype=dtype,
        #     device=device,
        # ) for _ in range(depth)])
        # self.activations = torch.nn.ModuleList([FlexibleActivation(
        #     width=activation_size,
        #     neutral_curvature=neutral_curvature,
        #     l2_curvature=l2_curvature,
        #     l2_bias=l2_bias,
        #     dtype=dtype,
        #     device=device,
        # ) for _ in range(depth)])
        # self.activations = torch.nn.ModuleList([ReLUActivation(
        #     width=activation_size,
        #     l2_bias=l2_bias,
        #     dtype=dtype,
        #     device=device,
        # ) for _ in range(depth)])

        # Generate the permutation to use for the perfect shuffle (aka Faro shuffle)
        self.shuffle_perm = torch.zeros([sponge_size], dtype=torch.long)
        half_sponge_size = (sponge_size + 1) // 2
        for i in range(sponge_size):
            self.shuffle_perm[i] = (i // 2) + (i % 2) * half_sponge_size

        # Generate the sequence of recall locations
        self.start_size = max(0, input_size - sponge_size)
        unused_set = set(range(self.start_size))
        self.recall_elements = []
        for i in range(depth):
            unused_set.update(set(range(self.start_size + i * self.store_size, self.start_size + (i + 1) * self.store_size)))
            elements = random.sample(unused_set, recall_size)
            self.recall_elements.append(elements)
            unused_set.difference_update(elements)

        # Set up the butterfly for the input
        bf_size = input_size // 2 * 2
        bf_depth = int(math.ceil(math.log2(bf_size)))
        self.input_butterfly = OrthogonalButterfly(bf_size, bf_depth, l2_interact=l2_interact, dtype=dtype,
                                                    device=device)

        # Set up the butterfly for the output
        bf_size = (len(unused_set) + sponge_size) // 2 * 2
        bf_depth = int(math.ceil(math.log2(bf_size)))
        self.output_memory_list = sorted(unused_set)

        self.output_butterfly = OrthogonalButterfly(bf_size, bf_depth, l2_interact=l2_interact, dtype=dtype,
                                                    device=device)
        # self.output_memory_list = sorted(unused_set)

    def fetch_memory(self, memory, j):
        """Fetch the j-th column from the memory list of tensors, as if `memory` were concatenated
         into a single tensor (We maintain the memory as a list like this because it allows us to
         efficiently accumulate onto it layer-by-layer; if we kept it concatenated as one tensor then
         we would have to rebuild the whole thing on each layer, because Pytorch backprop would not
         allow us to modify it in place.)"""
        if j < self.start_size:
            return memory[0][:, j]
        else:
            k = (j - self.start_size) // self.store_size
            l = (j - self.start_size) % self.store_size
            return memory[k + 1][:, l]

    def forward(self, X):
        assert X.dtype == self.dtype
        assert X.device == self.angles.device
        assert X.shape[1] == self.input_size

        X = X * self.scales

        X = self.input_butterfly(X)

        # Initialize the sponge using the first part of the input (as much as will fit in the sponge), padding with zero
        # if the input size is less than the sponge size.
        sponge = torch.cat([torch.zeros([X.shape[0], max(0, self.sponge_size - self.input_size)], dtype=X.dtype, device=X.device),
                            X[:, :self.sponge_size]], dim=1)

        # Memory will be accumulated as a Python list of tensors. Initially the memory consists of the remaining input
        # which did not fit in the sponge.
        memory = [X[:, self.sponge_size:]]

        # Keep track of sponge state at end of each stage, for diagnostics
        self.sponge_steps = []
        for i in range(self.depth):
            # print("sponge {}: {}".format(i, sponge))
            for j in range(self.butterfly_depth):
                # Faro shuffle
                sponge = sponge[:, self.shuffle_perm]

                # Perform orthogonal exchange
                exch_size = self.sponge_size // 2
                x1 = sponge[:, :exch_size]
                x2 = sponge[:, exch_size:(2 * exch_size)]
                cos_angles = torch.cos(self.angles[i, j, :])
                sin_angles = torch.sin(self.angles[i, j, :])
                y1 = cos_angles * x1 + sin_angles * x2
                y2 = -sin_angles * x1 + cos_angles * x2
                extra = sponge[:, (2 * exch_size):]  # At most one column which does not participate in the exchange (if sponge_size is odd)
                sponge = torch.cat([y1, y2, extra], dim=1)

            # Compute activations
            act_in = sponge[:, self.activation_position:(self.activation_position + self.activation_size)]
            act_plus, act_minus = self.activations[i](act_in)
            act_out = torch.stack([act_plus, act_minus], dim=2).view(act_plus.shape[0], act_plus.shape[1] * 2)
            self.sponge_steps.append(sponge)

            # Store off data which will leave the sponge
            memory.append(sponge[:, :self.store_size])

            # Recall stored data which will enter the sponge
            recall = [self.fetch_memory(memory, j).view(-1, 1) for j in self.recall_elements[i]]

            # Create the next iteration of the sponge
            sponge = torch.cat([
                *recall,
                act_out,
                sponge[:, self.store_size:self.activation_position],
                sponge[:, (self.activation_position + self.activation_size):],
            ], dim=1)

        pre_output = torch.cat([self.fetch_memory(memory, j).view(-1, 1) for j in self.output_memory_list] + [sponge], dim=1)
        output = self.output_butterfly(pre_output)
        return output[:, :self.output_size]
        # return pre_output[:, -self.output_size:]

    def penalty(self):
        return self.l2_scale * torch.sum(self.scales ** 2) + \
            self.l2_interact * torch.sum(torch.sin(self.angles * 2) ** 2) + \
            sum(a.penalty() for a in self.activations)

    def __repr__(self):
        return """Sponge(
            input_size={input_size},
            output_size={output_size},
            sponge_size={sponge_size},
            activation_size={activation_size},
            recall_size={recall_size},
            depth={depth},
            butterfly_depth={butterfly_depth},
            l2_scale={l2_scale},
            l2_interact={l2_interact},
            l2_bias={l2_bias})""".format(
            input_size=self.input_size,
            output_size=self.output_size,
            sponge_size=self.sponge_size,
            activation_size=self.activation_size,
            recall_size=self.recall_size,
            depth=self.depth,
            butterfly_depth=self.butterfly_depth,
            neutral_curvature=self.neutral_curvature,
            l2_scale=self.l2_scale,
            l2_interact=self.l2_interact,
            # l2_curvature=self.l2_curvature,
            l2_bias=self.l2_bias)