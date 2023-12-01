#include <torch/torch.h>

template <typename scalar_t>
void knn_kernel(const scalar_t* x, const scalar_t* y,
                const int64_t* ptr_x, const int64_t* ptr_y,
                int64_t* row, int64_t* col,
                const int64_t k, const int64_t n, const int64_t m,
                const int64_t dim, const int64_t num_examples,
                const scalar_t speed_of_light, const scalar_t space_time_diff) {

    for (int64_t n_y = 0; n_y < m; ++n_y) {
        const int64_t example_idx = get_example_idx(n_y, ptr_y, num_examples);

        scalar_t best_dist[100];
        int64_t best_idx[100];

        for (int e = 0; e < k; e++) {
            best_dist[e] = 1e10;
            best_idx[e] = -1;
        }

        for (int64_t n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {
            scalar_t tmp_dist = 0;
            for (int64_t d = 0; d < dim - 1; d++) {
                tmp_dist += (x[n_x * dim + d] - y[n_y * dim + d]) *
                            (x[n_x * dim + d] - y[n_y * dim + d]);
                d = dim - 1;
                tmp_dist -=
                        speed_of_light *
                        speed_of_light *
                        (x[n_x * dim + d] - y[n_y * dim + d]) *
                        (x[n_x * dim + d] - y[n_y * dim + d]);
            }
            tmp_dist = (tmp_dist < 0) ? (0 - space_time_diff * tmp_dist) : tmp_dist;  // Aim for "light-like separation".

            for (int64_t e1 = 0; e1 < k; e1++) {
                if (best_dist[e1] > tmp_dist) {
                    for (int64_t e2 = k - 1; e2 > e1; e2--) {
                        best_dist[e2] = best_dist[e2 - 1];
                        best_idx[e2] = best_idx[e2 - 1];
                    }
                    best_dist[e1] = tmp_dist;
                    best_idx[e1] = n_x;
                    break;
                }
            }
        }

        for (int64_t e = 0; e < k; e++) {
            row[n_y * k + e] = n_y;
            col[n_y * k + e] = best_idx[e];
        }
    }
}

torch::Tensor knn_cpu(const torch::Tensor& x, const torch::Tensor& y,
                      const torch::optional<torch::Tensor>& ptr_x,
                      const torch::optional<torch::Tensor>& ptr_y,
                      const int64_t k, const scalar_t speed_of_light = 1,
                      const scalar_t space_time_diff = 1) {

    torch::TensorOptions options = ptr_y.has_value() ? ptr_y.value().options() : x.options();

    AT_ASSERTM(k <= 100, "`k` needs to be smaller than or equal to 100");

    const int64_t num_examples = ptr_x.has_value() ? (ptr_x.value().size(0) - 1) : (x.size(0));

    torch::Tensor row = torch::empty({y.size(0) * k}, options);
    torch::Tensor col = torch::full({y.size(0) * k}, -1, options);

    const int64_t dim = x.size(1);
    const int64_t n = x.size(0);
    const int64_t m = y.size(0);

    knn_kernel<float>(x.data_ptr<float>(), y.data_ptr<float>(),
                      ptr_x.has_value() ? ptr_x.value().data_ptr<int64_t>() : nullptr,
                      ptr_y.has_value() ? ptr_y.value().data_ptr<int64_t>() : nullptr,
                      row.data_ptr<int64_t>(), col.data_ptr<int64_t>(), k, n, m, dim,
                      num_examples, speed_of_light, space_time_diff);

    torch::Tensor mask = col != -1;
    return torch::stack({row.masked_select(mask), col.masked_select(mask)}, 0);
}