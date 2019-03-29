import torch
from torch.autograd import Function
from .._ext import roi_ring_pooling
import copy

import pdb

class RoIRingPoolFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale, scale_inner, scale_outer):
        ctx.pooled_height = pooled_height
        ctx.pooled_width = pooled_width
        ctx.spatial_scale = spatial_scale
        ctx.scale_inner = scale_inner
        ctx.scale_outer = scale_outer
        ctx.feature_size = None
    
    def forward(ctx, features, rois):
        ctx.feature_size = features.size()
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        num_rois = rois.size(0)
        output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()

        ctx.rois = rois
        ctx.processed_rois = features.new(rois.size(0), 9).zero_()
        
        RectangularRing(rois, ctx.processed_rois, ctx.spatial_scale, ctx.scale_inner, ctx.scale_outer)
        #print('rois ', rois[100:101, :])
        #print('preco ', ctx.processed_rois[100:101,:])
        #if not features.is_cuda:
        #    _features = features.permute(0, 2, 3, 1)
        #    roi_ring_pooling.roi_ring_pooling_forward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
        #                                              _features, ctx.processed_rois, output)
        #else:
        #print('3333', rois)
        #print('ctx process roi ', ctx.processed_rois)
        roi_ring_pooling.roi_ring_pooling_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                           features, ctx.processed_rois, output, ctx.argmax)
        return output
    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()

        roi_ring_pooling.roi_ring_pooling_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                        grad_output, ctx.processed_rois, grad_input, ctx.argmax)
        return grad_input, None



def RectangularRing(rois, processed_rois,spatial_scale, scale_inner, scale_outer):
    #widths = rois[:, 3] - rois[:, 1] + 1.0
    #heights = rois[:, 4] - rois[:, 2] + 1.0
    #ctr_x = rois[:, 1] + 0.5 * widths
    #ctr_y = rois[:, 2] + 0.5 * heights

    ctr_x = (rois[:, 1] + rois[:, 3]) / 2
    ctr_y = (rois[:, 2] + rois[:, 4]) / 2
    w_half = (rois[:, 3] - rois[:, 1]) / 2
    h_half = (rois[:, 4] - rois[:, 2]) / 2

    
    #for i in range(rois.size(0)):
    #    processed_rois[i, 0] = 0
    processed_rois[:, 1] = torch.tensor(ctr_x - w_half * scale_outer, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(0.5).floor_()
    processed_rois[:, 2] = torch.tensor(ctr_y - h_half * scale_outer, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(0.5).floor_()
    processed_rois[:, 3] = torch.tensor(ctr_x + w_half * scale_outer, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(-0.5).ceil_()
    processed_rois[:, 4] = torch.tensor(ctr_y + h_half * scale_outer, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(-0.5).ceil_()
    processed_rois[:, 5] = torch.tensor(ctr_x - w_half * scale_inner, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(0.5).floor_()
    processed_rois[:, 6] = torch.tensor(ctr_y - h_half * scale_inner, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(0.5).floor_()
    processed_rois[:, 7] = torch.tensor(ctr_x + w_half * scale_inner, dtype=rois.dtype, device=rois.device)  ##.mul_(spatial_scale).add_(-0.5).ceil_()
    processed_rois[:, 8] = torch.tensor(ctr_y + h_half * scale_inner, dtype=rois.dtype, device=rois.device) 
    
    if scale_inner == 0:
        processed_rois[:, 5:] = 0

    return 1