#include <torch/extension.h>
#include <vector>

template <typename scalar_t>
torch::Tensor mknn_cuda(
    const torch::Tensor x,
    const torch::Tensor y,
    torch::optional<torch::Tensor> ptr_x,
    torch::optional<torch::Tensor> ptr_y,
    const int64_t k,
    const scalar_t speed_of_light = 1,
    const scalar_t space_time_diff = 1);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mknn_cuda", &mknn_cuda<float>, "MKNN CUDA (float)");
  m.def("mknn_cuda", &mknn_cuda<double>, "MKNN CUDA (double)");
  }