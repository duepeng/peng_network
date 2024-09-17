import torch
from torch import nn
from PFNLayerV2 import PFNLayerV2
class DynamicPillarVFE(nn.Module):
    def __init__(self, voxel_size, grid_size, point_cloud_range):
        self.use_norm = True

        num_filters = [8, 32]

        # gen pfn layers
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

    def forward(self, inputs):
        pass




if __name__ == "__main_":
    pass