import time
from typing import Optional

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
from skimage import measure
from torch.nn import functional as F


def get_scene_mesh(density, voxel_sizes, threshold=30., offset=np.array([0, 0, 0]), colors=None):
    vertices, faces, normals, _ = measure.marching_cubes(density,
                                                         threshold,
                                                         spacing=tuple(voxel_sizes),
                                                         allow_degenerate=False)
    vertices = (np.array(vertices)) + offset[None,:]
    if colors is not None:
        verts_ind = np.floor((vertices - offset[None, :]) / voxel_sizes[0]).astype(np.int32)
        color_vals = colors[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        # color_vals = color_vals.astype(np.int8)
    else:
        color_vals = None
    normals = (np.array(normals))

    return trimesh.Trimesh(vertices, faces, vertex_normals=normals, vertex_colors=color_vals)


def get_scaled_intrinsics(intrinsics: torch.Tensor, scale_x: float, scale_y: Optional[float] = None):
    if scale_y is None:
        scale_y = scale_x
        
    scale_mat = torch.tensor([[scale_x, 0., scale_x],
                              [0., scale_y, scale_y],
                              [0., 0., 1.]], device=intrinsics.device)
    
    return scale_mat * intrinsics

PIXEL_MEAN = torch.Tensor([103.53, 116.28, 123.675]).view(1, -1, 1, 1)  # [B, 3, H, W]
PIXEL_STD = torch.Tensor([1., 1., 1.]).view(1, -1, 1, 1)

def normalize_rgb(x):
    """ Normalizes the RGB images to the input range"""
    return (x - PIXEL_MEAN.type_as(x)) / PIXEL_STD.type_as(x)

def invert_pose(pose):
    rotmat_c2w = pose[:3,:3]
    rotmat_w2c = rotmat_c2w.T
    trans_c2w = pose[:3,-1:]
    trans_w2c = -(rotmat_w2c @ trans_c2w)

    return np.concatenate([rotmat_w2c, trans_w2c], axis=1)

def invert_pose_torch(pose):
    rotmat_c2w = pose[...,:3,:3]
    rotmat_w2c = rotmat_c2w.transpose(-1, -2)
    trans_c2w = pose[...,:3,-1:]
    trans_w2c = -torch.bmm(rotmat_w2c, trans_c2w)

    return torch.cat([torch.cat([rotmat_w2c, trans_w2c], axis=-1),
                      torch.tensor([[[0., 0., 0., 1.]]], device=pose.device).repeat(pose.shape[0], 1, 1)], dim=-2)

# TODO: make align_corners as a parameter?
def compute_world_dims(bounds, voxel_sizes, n_levels, margin=0, device=torch.device("cpu")):
    coarsest_voxel_dims = ((bounds[:,1] - bounds[:,0] + margin*2) / voxel_sizes[-1])
    coarsest_voxel_dims = torch.ceil(coarsest_voxel_dims) + 1
    coarsest_voxel_dims = coarsest_voxel_dims.int()
    world_dims = (coarsest_voxel_dims - 1) * voxel_sizes[-1]
    
    # Center the model in within the grid
    volume_origin = bounds[:,0] - (world_dims - bounds[:,1] + bounds[:,0]) / 2
    
    # Multilevel dimensions
    voxel_dims = (coarsest_voxel_dims.view(1,-1).repeat(n_levels,1) - 1) * (voxel_sizes[-1] / torch.tensor(voxel_sizes).unsqueeze(-1)).int() + 1

    return world_dims.to(device), volume_origin.to(device), voxel_dims.to(device)

# https://math.stackexchange.com/questions/382760/composition-of-two-axis-angle-rotations
def compose_axis_angles(a, b):
    norm_a = a.norm(dim=-1)
    norm_b = b.norm(dim=-1)
    cos_a = torch.cos(norm_a)
    cos_b = torch.cos(norm_b)
    a_hat = a / norm_a
    b_hat = b / norm_b
    a_hat_sin_a = a_hat * torch.sin(norm_a)
    b_hat_sin_b = b_hat * torch.sin(norm_b)
    c = torch.acos(cos_a * cos_b - (a_hat_sin_a * b_hat_sin_b).sum(dim=-1))
    d = cos_a * b_hat_sin_b + cos_b * a_hat_sin_a + torch.cross(a_hat_sin_a, b_hat_sin_b, dim=-1)
    return c * F.normalize(d, dim=-1)

def axis_angle_to_matrix(data):
    batch_dims = data.shape[:-1]

    theta = torch.norm(data, dim=-1, keepdim=True)
    omega = data / theta

    omega1 = omega[...,0:1]
    omega2 = omega[...,1:2]
    omega3 = omega[...,2:3]
    zeros = torch.zeros_like(omega1)

    K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
                      torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
                      torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
    I = torch.eye(3, device=data.device).expand(*batch_dims,3,3)

    return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)

def matrix_to_axis_angle(rot):
    """
    :param rot: [N, 3, 3]
    :return:
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(rot))

def pose6d_to_matrix(batch_poses):
    c2w = torch.eye(4, device=batch_poses.device).unsqueeze(0).repeat(batch_poses.shape[0], 1, 1)
    c2w[:,:3,:3] = axis_angle_to_matrix(batch_poses[:,:,0])
    c2w[:,:3,3] = batch_poses[:,:,1]
    return c2w

def matrix_to_pose6d(batch_matrices):
    return torch.cat([matrix_to_axis_angle(batch_matrices[:,:3,:3]).unsqueeze(-1),
                      batch_matrices[:,:3,3:]], dim=-1)

def get_time():
    """
    :return: get timing statistics
    """
    torch.cuda.synchronize()
    return time.time()

def coordinates(voxel_dim, device: torch.device):
    nx, ny, nz = voxel_dim[0], voxel_dim[1], voxel_dim[2]
    x = torch.arange(0, nx, dtype=torch.long, device=device)
    y = torch.arange(0, ny, dtype=torch.long, device=device)
    z = torch.arange(0, nz, dtype=torch.long, device=device)
    x, y, z = torch.meshgrid(x, y, z, indexing="ij")

    return torch.stack((x.flatten(), y.flatten(), z.flatten()))
