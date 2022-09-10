import os
import open3d as o3d
import numpy as np


def compute_scene_bounds(mesh):
    v = np.asarray(mesh.vertices)  # [N, 3]
    bounds = np.stack([v.min(axis=0), v.max(axis=0)], 1).astype(np.float32)
    bounds[:, 0] -= 0.2
    bounds[:, 1] += 0.2
    return bounds


if __name__ == "__main__":
    data_dir = "/media/jingwen/Data/scannet/scans"
    scene = "scene0002_00"
    mesh_gt = o3d.io.read_triangle_mesh("{}/{}/{}_vh_clean_2.ply".format(data_dir, scene, scene))
    bounds = compute_scene_bounds(mesh_gt)
    print(bounds)