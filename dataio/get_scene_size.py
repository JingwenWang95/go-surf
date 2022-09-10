import torch
from dataio.get_scene_bounds import get_scene_bounds


def compute_world_dims(bounds, voxel_sizes, n_levels, margin=0.0, device=torch.device("cpu")):
    coarsest_voxel_dims = ((bounds[:,1] - bounds[:,0] + margin*2) / voxel_sizes[-1])
    coarsest_voxel_dims = torch.ceil(coarsest_voxel_dims) + 1
    coarsest_voxel_dims = coarsest_voxel_dims.int()
    world_dims = (coarsest_voxel_dims - 1) * voxel_sizes[-1]

    # Center the model in within the grid
    volume_origin = bounds[:,0] - (world_dims - bounds[:,1] + bounds[:,0]) / 2

    # Multilevel dimensions
    voxel_dims = (coarsest_voxel_dims.view(1,-1).repeat(n_levels,1) - 1) * (voxel_sizes[-1] / torch.tensor(voxel_sizes).unsqueeze(-1)).int() + 1

    return world_dims.to(device), volume_origin.to(device), voxel_dims.to(device)


if __name__ == "__main__":
    scene = "scene0050_00"
    scene_bound = torch.from_numpy(get_scene_bounds(scene, new_bound=True))
    world_dims_glob, volume_origin_glob, voxel_dims_glob = compute_world_dims(scene_bound, [0.03, 0.06, 0.24, 0.96], 4, margin=0.1)
    print(scene_bound[:, 1] - scene_bound[:, 0])
    print(world_dims_glob)
    print(voxel_dims_glob[0])