#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void hip_square(at::Tensor data_in, at::Tensor data_out);

void square(at::Tensor data_in, at::Tensor data_out) {
  CHECK_INPUT(data_in);
  CHECK_INPUT(data_out);
  hip_square(data_in, data_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", &square, "test square vector");
}
