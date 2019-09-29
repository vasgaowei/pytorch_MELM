#include <torch/extension.h>

#include <cmath>
#include <vector>

int ROIRingPoolForwardLaucher(
    const at::Tensor bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const at::Tensor bottom_rois,
    at::Tensor top_data, at::Tensor argmax_data);

int ROIRingPoolBackwardLaucher(const at::Tensor top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int pooled_height,
    const int pooled_width, const at::Tensor bottom_rois,
    at::Tensor bottom_diff, const at::Tensor argmax_data);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int roi_ring_pool_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                               at::Tensor features, at::Tensor rois, at::Tensor output, at::Tensor argmax)
{
    // Grab the input tensor
    CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    CHECK_INPUT(argmax);

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 9)
    {
        return 0;
    }
    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);

    ROIRingPoolForwardLaucher(
        features, spatial_scale, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois,
        output, argmax);

    return 1;
}

int roi_ring_pool_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                                at::Tensor top_grad, at::Tensor rois, at::Tensor bottom_grad, at::Tensor argmax)
{
    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 9)
    {
        return 0;
    }

    // batch size
    int batch_size = bottom_grad.size(0);
    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);
    ROIRingPoolBackwardLaucher(
        top_grad, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois,
        bottom_grad, argmax);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roi_ring_pool_forward_cuda, "Roi_Ring_Pooling forward (CUDA)");
  m.def("backward", &roi_ring_pool_backward_cuda, "Roi_Ring_Pooling backward (CUDA)");
}