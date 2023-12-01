ps#ifdef WITH_PYTHON
#include <Python.h>
#endif
#include <torch/script.h>

#include "cpu/mknn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/mknn_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_PYTHON
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__knn_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__knn_cpu(void) { return NULL; }
#endif
#endif
#endif

template<typename scalar_t> CLUSTER_API torch::Tensor mknn(
        torch::Tensor x, torch::Tensor y,
        torch::optional<torch::Tensor> ptr_x,
        torch::optional<torch::Tensor> ptr_y,
        int64_t k,
        int64_t num_workers,
        scalar_t speed_of_light,
        scalar_t space_time_diff
        ) {
    if (x.device().is_cuda()) {
        #ifdef WITH_CUDA
        return mknn_cuda(x, y, ptr_x, ptr_y, k, speed_of_light, space_time_diff);
        #else
        AT_ERROR("Not compiled with CUDA support");
        #endif
    } else {
        AT_ERROR("Minkowski knn only defined cuda. (I need at least 2 functioning braincells to make a cpu implementation (: )")
        return knn_cpu(x, y, ptr_x, ptr_y, k, num_workers);
    }
}

static auto registry = torch::RegisterOperators().op("graphnet::mknn", &mknn);