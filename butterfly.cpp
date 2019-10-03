#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void cuda_butterfly_forward_slow(at::Tensor data_in, at::Tensor angles, at::Tensor data_out);

void cuda_butterfly_backward_slow(
    at::Tensor data_in,
    at::Tensor angles, 
    at::Tensor grad_in,
    at::Tensor grad_out,
    at::Tensor grad_angles_accum
);

void butterfly_forward_slow(at::Tensor data_in, at::Tensor angles, at::Tensor data_out) {
  CHECK_INPUT(data_in);
  CHECK_INPUT(angles);
  CHECK_INPUT(data_out);
  cuda_butterfly_forward_slow(data_in, angles, data_out);
}

void butterfly_backward_slow(
    at::Tensor data_in,
    at::Tensor angles, 
    at::Tensor grad_in,
    at::Tensor grad_out,
    at::Tensor grad_angles_accum
) {
  CHECK_INPUT(data_in);
  CHECK_INPUT(angles);
  CHECK_INPUT(grad_in);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(grad_angles_accum);
  cuda_butterfly_backward_slow(data_in, angles, grad_in, grad_out, grad_angles_accum);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("butterfly_forward_slow", &butterfly_forward_slow, "slow version of butterfly layer forward pass");
    m.def("butterfly_backward_slow", &butterfly_backward_slow, "slow version of butterfly layer backward pass");
}
