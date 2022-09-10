import numpy as np
import trimesh
import torch
import os
import argparse
import json

from dataio.rgbd_dataset import RGBDDataset
from model.utils import pose6d_to_matrix
from tools.frustum_culling import cull_mesh, get_culling_bound
from tools.mesh_metrics import compute_metrics


def transform_mesh(align_matrix, mesh_filepath, mesh_savepath):
    mesh = trimesh.load(mesh_filepath, force='mesh', process=False)
    # pre-multiply
    mesh.apply_transform(align_matrix)
    mesh.export(mesh_savepath)


if __name__ == '__main__':

    print(os.path.dirname(__file__))
    from config import load_config
    parser = argparse.ArgumentParser(
        description='Arguments to cull the mesh.'
    )

    parser.add_argument('--exp_name', type=str, default="release_test_5e-4_std_0.004")
    parser.add_argument('--scene', type=str, default="morning_apartment")
    parser.add_argument('--n_iters', type=int, default=10000)
    parser.add_argument('--remove_missing_depth', action="store_true")
    args = parser.parse_args()

    n_iter = args.n_iters
    config = load_config(exp_name=args.exp_name, scene=args.scene, use_config_snapshot=True)
    mesh_dir = os.path.join(config["log_dir"], "mesh/{}".format(n_iter))
    chkpt_path = os.path.join(config["checkpoints_dir"], "chkpt_{}".format(n_iter))
    mesh_path = os.path.join(mesh_dir, "{}.ply".format(args.scene))
    save_path_transformed = os.path.join(mesh_dir, "{}_transformed.ply".format(args.scene))
    save_path_culled = os.path.join(mesh_dir, "{}_culled.ply".format(args.scene))

    assert config["dataset_type"] == "rgbd", "We only do mesh culling and evaluation on synthetic sequences!"
    dataset = RGBDDataset(os.path.join(config["datasets_dir"], args.scene), load=False, trainskip=1, device=torch.device("cpu"))

    # re-align mesh
    state = torch.load(chkpt_path, map_location=torch.device("cpu"))
    c2w = pose6d_to_matrix(state["poses"]).detach()
    gt_c2w = torch.tensor(dataset.all_gt_poses)
    align_T = gt_c2w[0].numpy() @ np.linalg.inv(c2w[0].numpy())
    transform_mesh(align_T, mesh_path, save_path_transformed)

    # cull mesh
    cull_mesh(dataset, save_path_transformed, save_path_culled, scene_bounds=get_culling_bound(args.scene),
              remove_missing_depth=args.remove_missing_depth, subdivide=True, max_edge=0.015)

    # evaluate culled mesh
    rst = compute_metrics(save_path_culled, os.path.join(config["datasets_dir"], args.scene, "gt_mesh_culled.ply"))
    print(rst)
    with open(os.path.join(mesh_dir, "{}.json".format(args.scene)), "w") as f:
        json.dump(rst, f)
