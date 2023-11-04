#include <torch/extension.h>
#include <vector>
#include <stdio.h>
//
//#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
//#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
//#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

int check_stencil(
        const std::vector<std::vector<int>> &cartesian_stencil,
        std::vector<int> &dim_max) {

    int stencil_dimension = -1;

     // Check that the input stencil is valid, and get its max extent
    for (auto &_ : cartesian_stencil) {

        if(stencil_dimension == -1) {
            stencil_dimension = _.size();
            dim_max = _;
        } else {
            TORCH_CHECK(
                stencil_dimension == _.size(),
                "Stencil must contain points with the same dimension"
            );

            for(int i = 0; i < stencil_dimension; i++)
                dim_max[i] = std::max(dim_max[i], _[i]);
        }
    }
    return stencil_dimension;
}

torch::Tensor max_pool_1d(
        const torch::Tensor &input,
        const std::vector<std::vector<int>> &cartesian_stencil,
        std::vector<int> &dim_max) {

    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto output_w = input.size(2) - dim_max[0];

    auto ret_tensor = torch::zeros({batch_size, channels, output_w});
    TORCH_CHECK(output_w > 0, "max_pool_2d, pool stencil must not be bigger than the input")

    for(int b = 0; b < batch_size; b++) {
        for(int c = 0; c < channels; c++) {
            for(int w = 0; w < output_w; w++) {
                double max = -std::numeric_limits<double>::infinity();
                for(auto &_ : cartesian_stencil){
                    auto i = _[0];
                    max = std::max(max, input[b][c][w+i].item<double>());
                }
                ret_tensor[b][c][w] = max;
            }
        }
    }
    return ret_tensor;
}

torch::Tensor max_pool_2d(
        const torch::Tensor &input,
        const std::vector<std::vector<int>> &cartesian_stencil,
        std::vector<int> &dim_max) {

    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto output_h = input.size(2) - dim_max[1];
    auto output_w = input.size(3) - dim_max[0];

    auto ret_tensor = torch::zeros({batch_size, channels, output_h, output_w});
    TORCH_CHECK(output_h > 0, "max_pool_2d, pool stencil must not be bigger than the input");
    TORCH_CHECK(output_w > 0, "max_pool_2d, pool stencil must not be bigger than the input")

    for(int b = 0; b < batch_size; b++) {
        for(int c = 0; c < channels; c++) {
            for(int h = 0; h < output_h; h++) {
                for(int w = 0; w < output_w; w++) {
                    double max = -std::numeric_limits<double>::infinity();
                    for(auto &_ : cartesian_stencil){
                        auto i = _[0], j = _[1];
                        max = std::max(max, input[b][c][h+j][w+i].item<double>());
                    }
                    ret_tensor[b][c][h][w] = max;
                }
            }
        }
    }
    return ret_tensor;
}



torch::Tensor max_pool_3d(
        const torch::Tensor &input,
        const std::vector<std::vector<int>> &cartesian_stencil,
        std::vector<int> &dim_max) {

    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto output_d = input.size(2) - dim_max[2];
    auto output_h = input.size(3) - dim_max[1];
    auto output_w = input.size(4) - dim_max[0];

    auto ret_tensor = torch::zeros({batch_size, channels, output_d, output_h, output_w});
    TORCH_CHECK(output_d > 0, "max_pool_2d, pool stencil must not be bigger than the input");
    TORCH_CHECK(output_h > 0, "max_pool_2d, pool stencil must not be bigger than the input");
    TORCH_CHECK(output_w > 0, "max_pool_2d, pool stencil must not be bigger than the input")

    for(int b = 0; b < batch_size; b++) {
        for(int c = 0; c < channels; c++) {
            for(int d = 0; d < output_d; d++) {
                for(int h = 0; h < output_h; h++) {
                    for(int w = 0; w < output_w; w++) {
                        double max = -std::numeric_limits<double>::infinity();
                        for(auto &_ : cartesian_stencil){
                            auto i = _[0], j = _[1], k = _[2];
                            max = std::max(max, input[b][c][d+k][h+j][w+i].item<double>());
                        }
                        ret_tensor[b][c][d][h][w] = max;
                    }
                }
            }
        }
    }
    return ret_tensor;
}


torch::Tensor max_pool_atrous_forward(
    torch::Tensor input,
    std::vector<std::vector<int>> cartesian_stencil) {

    std::vector<int> dim_max;
    int dimension = check_stencil(cartesian_stencil, dim_max);

    torch::Tensor ret_val;

    TORCH_CHECK(dimension >= 1 && dimension <=3, "MaxPool only implemented for 1<= dim <= 3")
    if(dimension == 1) {
        ret_val = max_pool_1d(input, cartesian_stencil, dim_max);
    } else if(dimension == 2) {
        ret_val = max_pool_2d(input, cartesian_stencil, dim_max);
    } else if(dimension == 3) {
        ret_val = max_pool_3d(input, cartesian_stencil, dim_max);
    }

    return ret_val;
}

torch::Tensor max_unpool_2d(
        const torch::Tensor &grad_y,
        const torch::Tensor &input,
        const std::vector<std::vector<int>> &cartesian_stencil,
        std::vector<int> &dim_max) {

    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto output_h = input.size(2) - dim_max[1];
    auto output_w = input.size(3) - dim_max[0];

    auto ret_tensor = torch::zeros_like(input);
    TORCH_CHECK(output_h > 0, "max_unpool_2d, pool stencil must not be bigger than the input");
    TORCH_CHECK(output_w > 0, "max_unpool_2d, pool stencil must not be bigger than the input")

    for(int b = 0; b < batch_size; b++) {
        for(int c = 0; c < channels; c++) {
            for(int h = 0; h < output_h; h++) {
                for(int w = 0; w < output_w; w++) {
                    double max = -std::numeric_limits<double>::infinity();
                    int max_i = -1, max_j = -1;
                    for(auto &_ : cartesian_stencil){
                        auto i = _[0], j = _[1];
                        auto val = input[b][c][h+j][w+i].item<double>();

                        if (val > max) {
                            max = val;
                            max_i = i, max_j = j;
                        }
                    }
                    ret_tensor[b][c][h+max_j][w + max_i] += grad_y[b][c][h][w].item<double>();
                }
            }
        }
    }
    return ret_tensor;
}

torch::Tensor max_unpool_2d(
        const torch::Tensor &grad_y,
        const torch::Tensor &input,
        const std::vector<std::vector<int>> &cartesian_stencil,
        std::vector<int> &dim_max) {

    auto batch_size = input.size(0);
    auto channels = input.size(1);
    auto output_h = input.size(2) - dim_max[1];
    auto output_w = input.size(3) - dim_max[0];

    auto ret_tensor = torch::zeros_like(input);
    TORCH_CHECK(output_h > 0, "max_unpool_2d, pool stencil must not be bigger than the input");
    TORCH_CHECK(output_w > 0, "max_unpool_2d, pool stencil must not be bigger than the input")

    for(int b = 0; b < batch_size; b++) {
        for(int c = 0; c < channels; c++) {
            for(int h = 0; h < output_h; h++) {
                for(int w = 0; w < output_w; w++) {
                    double max = -std::numeric_limits<double>::infinity();
                    int max_i = -1, max_j = -1;
                    for(auto &_ : cartesian_stencil){
                        auto i = _[0], j = _[1];
                        auto val = input[b][c][h+j][w+i].item<double>();

                        if (val > max) {
                            max = val;
                            max_i = i, max_j = j;
                        }
                    }
                    ret_tensor[b][c][h+max_j][w + max_i] += grad_y[b][c][h][w].item<double>();
                }
            }
        }
    }
    return ret_tensor;
}

torch::Tensor max_pool_atrous_backward(
    torch::Tensor grad_y,
    torch::Tensor input,
    std::vector<std::vector<int>> cartesian_stencil) {


    std::vector<int> dim_max;
    int dimension = check_stencil(cartesian_stencil, dim_max);

    torch::Tensor ret_val;

    TORCH_CHECK(dimension >= 1 && dimension <=3, "MaxUnpool only implemented for 1<= dim <= 3")
    if(dimension == 1) {
        ret_val = max_unpool_1d(input, cartesian_stencil, dim_max);
    } else if(dimension == 2) {
        ret_val = max_unpool_2d(grad_y, input, cartesian_stencil, dim_max);
    } else if(dimension == 3) {
        ret_val = max_unpool_3d(input, cartesian_stencil, dim_max);
    }
    return ret_val;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &max_pool_atrous_forward, "Max pooling with holes forward");
  m.def("backward", &max_pool_atrous_backward, "Max pooling with holes backward");
}