from torch.nn.modules.module import Module
from ..functions.roi_ring_pool import RoIRingPoolFunction


class _RoIRingPooling(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale,  scale_inner, scale_outer):
        super(_RoIRingPooling, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.scale_inner = scale_inner
        self.scale_outer = scale_outer

    def forward(self, features, rois):
        return RoIRingPoolFunction(self.pooled_height, self.pooled_width, self.spatial_scale, self.scale_inner, self.scale_outer)(features, rois)