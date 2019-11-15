import torch
import torch.autograd
import torch.nn
import torch.utils.cpp_extension
import math

# TODO: Possibly use setup.py instead of JIT loading
butterfly_cpp = torch.utils.cpp_extension.load(
    name="butterfly_cpp",
    sources=["cpp/butterfly.cpp"],
    extra_include_paths=["/opt/rocm/hip/include", "cpp/MIPP/src"],
    extra_cflags=['-O3', '-funroll-loops'],
    # extra_cflags=['-g'],  # For debugging
)

COL_BLOCK_WIDTH = 16

class ButterflyFunction(torch.autograd.Function):
    """Caution: `forward` and `backward` both modify the input data and gradients in place."""
    @staticmethod
    def forward(ctx, data, angles, biases, curvature, indices_in, idx_out, num_input_layers, num_output_layers, num_activations):
        assert data.dtype == angles.dtype and data.dtype == biases.dtype
        assert torch.min(indices_in) >= 0 and torch.max(indices_in) <= data.shape[0]
        # TODO: Add more needed validations, to prevent the possibility of crashing in the C++ code
        butterfly_cpp.butterfly_forward(data, angles, biases, indices_in, idx_out, curvature, num_input_layers, num_output_layers, num_activations)
        ctx.indices_in = indices_in
        ctx.idx_out = idx_out
        ctx.num_input_layers = num_input_layers
        ctx.num_output_layers = num_output_layers
        ctx.num_activations = num_activations
        ctx.curvature = curvature
        ctx.save_for_backward(data, angles, biases)
        return data

    @staticmethod
    def backward(ctx, grad_data):
        data, angles, biases = ctx.saved_tensors
        grad_angles = torch.zeros_like(angles)
        grad_biases = torch.zeros_like(biases)
        butterfly_cpp.butterfly_backward(data, grad_data, angles, grad_angles, biases, grad_biases, ctx.indices_in, ctx.idx_out, ctx.curvature, ctx.num_input_layers, ctx.num_output_layers, ctx.num_activations)
        return grad_data, grad_angles, grad_biases, None, None, None, None, None, None


class ButterflyModule(torch.nn.Module):
    """Caution: The forward and backward passes both modify the input data and gradients in place."""
    def __init__(self,
                 indices_in,
                 idx_out,
                 num_input_layers,
                 num_output_layers,
                 num_activations,
                 curvature,
                 l2_interact,
                 l2_bias,
                 biases_initial_std,
                 dtype=torch.float32,
                 device=None):
        super().__init__()
        assert indices_in.dim() == 1
        assert indices_in.dtype == torch.int
        num_layers = num_input_layers + num_output_layers
        num_col_blocks = indices_in.shape[0] // COL_BLOCK_WIDTH
        self.angles = torch.nn.Parameter((torch.rand(size=[num_layers, indices_in.shape[0] // 2], dtype=dtype, device=device) * 2 - 1) * math.pi)
        self.biases = torch.nn.Parameter(torch.randn(size=[num_activations * num_col_blocks], dtype=dtype, device=device) * biases_initial_std)
        self.indices_in = indices_in
        self.idx_out = idx_out
        self.num_input_layers = num_input_layers
        self.num_output_layers = num_output_layers
        self.num_activations = num_activations
        self.curvature = curvature
        self.l2_interact = l2_interact
        self.l2_bias = l2_bias

    def forward(self, data):
        return ButterflyFunction().apply(data, self.angles, self.biases, self.curvature, self.indices_in, self.idx_out,
                                         self.num_input_layers, self.num_output_layers, self.num_activations)

    def penalty(self):
        return torch.sum(self.l2_interact * torch.sin(2 * self.angles) ** 2) + \
               torch.sum(self.l2_bias * self.biases ** 2)


class ButterflyNetwork(torch.nn.Module):
    def __init__(self,
                 input_width,
                 output_width,
                 zero_inputs,
                 network_depth,
                 initial_scale,
                 l2_scale,
                 butterfly_in_depth,
                 butterfly_out_depth,
                 activations_per_block,
                 blocks_per_layer,
                 curvature,
                 l2_interact,
                 l2_bias,
                 dtype=torch.float32,
                 device=None):
        super().__init__()

        # Network-level hyperparameters:
        self.input_width = input_width
        self.output_width = output_width
        self.zero_inputs = zero_inputs
        self.network_depth = network_depth
        self.initial_scale = initial_scale
        self.l2_scale = l2_scale

        # Hyperparameters specific to the individual ButterflyModule instances:
        self.butterfly_in_depth = butterfly_in_depth
        self.butterfly_out_depth = butterfly_out_depth
        self.activations_per_block = activations_per_block
        self.blocks_per_layer = blocks_per_layer
        self.curvature = curvature
        self.l2_interact = l2_interact
        self.l2_bias = l2_bias

        # Network-level parameters (The rest of the parameters are inside the ButterflyModules created below):
        self.scales = torch.nn.Parameter(torch.full([input_width], initial_scale, dtype=dtype, device=device))

        layers = []
        self.initial_width = input_width + zero_inputs
        self.input_locations = torch.linspace(0, self.initial_width - 1, steps=input_width).to(torch.long)
        total_width = self.initial_width
        for i in range(network_depth):
            layer_butterfly_in_depth = self._get_from_list_or_scalar(butterfly_in_depth, i)
            layer_butterfly_out_depth = self._get_from_list_or_scalar(butterfly_out_depth, i)
            layer_activations_per_block = self._get_from_list_or_scalar(activations_per_block, i)
            layer_blocks_per_layer = self._get_from_list_or_scalar(blocks_per_layer, i)
            layer_curvature = self._get_from_list_or_scalar(curvature, i)
            layer_l2_interact = self._get_from_list_or_scalar(l2_interact, i)
            layer_l2_bias = self._get_from_list_or_scalar(l2_bias, i)

            num_layer_inputs = layer_blocks_per_layer * COL_BLOCK_WIDTH
            assert num_layer_inputs <= total_width
            indices_in = torch.randperm(total_width, dtype=torch.int)[:num_layer_inputs]

            layers.append(ButterflyModule(
                indices_in,
                total_width,
                layer_butterfly_in_depth,
                layer_butterfly_out_depth,
                layer_activations_per_block,
                layer_curvature,
                layer_l2_interact,
                layer_l2_bias,
                biases_initial_std=initial_scale,
                dtype=dtype,
                device=device))

            total_width += layer_blocks_per_layer * activations_per_block

        self.total_width = total_width
        self.sequential = torch.nn.Sequential(*layers)

    @staticmethod
    def _get_from_list_or_scalar(x, i):
        if isinstance(x, (int, float)):
            return x
        else:
            return x[i]

    def forward(self, input_data):
        assert input_data.dim() == 2
        assert input_data.shape[0] == self.input_width
        data = torch.zeros([self.total_width, input_data.shape[1]], dtype=self.scales.dtype, device=self.scales.device)
        data[self.input_locations, :] = self.scales.view(-1, 1) * input_data
        out = self.sequential(data)
        return out[(self.total_width - self.output_width):, ]

    def penalty(self):
        return torch.sum(self.l2_scale * self.scales ** 2) + \
               sum(layer.penalty() for layer in self.layers)
