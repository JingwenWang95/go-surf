import torch
from torch import nn
from smooth_sampler import SmoothSampler


class MultiGrid(nn.Module):
    def __init__(self, sizes, sdf_feature_dim):
        super().__init__()
        volumes = []
        self.sizes = sizes

        for i, size in enumerate(sizes):
            init_val = torch.rand(1, sdf_feature_dim[i], 1, 1, 1) * 2. - 1.
            volume = nn.Parameter(init_val.repeat(1, 1, *size))
            volumes.append(volume)

        self.volumes = nn.ParameterList(volumes)

    def forward(self, grid, align_corners=True, apply_smoothstep=False, concat=True):
        out = []

        for volume in self.volumes:
            out.append(SmoothSampler.apply(volume, grid, "border", align_corners, apply_smoothstep))

        if concat:
            return torch.cat(out, dim=1)

        return out
