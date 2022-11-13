import os
import argparse
import numpy as np
import trimesh
import torch
import torch.nn.functional as F

from dataio.rgbd_dataset import RGBDDataset
from dataio.scannet_dataset import ScannetDataset
from model.sdf_grid_model import SDFGridModel, compute_grads
from config import load_config
import marching_cubes as mcubes


def batchify_vol(fn, chunk, dim=[]):
    if chunk is None:
        return fn
    def ret(model, coords, **kwargs):
        full_val = torch.empty(list(coords.shape[1:4]) + dim, device=coords.device)
        for i in range(0, coords.shape[2], chunk):
            val = fn(model, coords[:,:,i:i+chunk,:].contiguous(), **kwargs)
            full_val[:,i:i+chunk,:] = val
        return full_val
    return ret


def query_sdf(model, coords, concat_qp=False, rgb_feature_dim=[]):
    coords = coords.permute(1, 2, 3, 0).unsqueeze(0)
    mlvl_feats = model.grid(coords[..., [2, 1, 0]], concat=False)
    sdf_feats = list(map(lambda feat_pts, rgb_dim: feat_pts[:,:-rgb_dim,...] if rgb_dim > 0 else feat_pts,
                         mlvl_feats, rgb_feature_dim))
    
    if concat_qp:
        sdf_feats.append(coords.permute(0, 4, 1, 2, 3))
    
    sdf = model.decoder.geometry_net(torch.cat(sdf_feats, dim=1).squeeze(0).permute(1, 2, 3, 0))[..., 0]
    return sdf


def query_rgb(model, coords, volume_origin, world_dim, use_normals=False,
              use_view_dirs=False, concat_qp_sdf=False,
              concat_qp_rgb=False, use_dot_prod=False, rgb_feature_dim=[]):
    with torch.enable_grad():
        coords = coords.float().requires_grad_(True)
        vertices_query = 2. * (coords.view(1, 1, 1, -1, 3) - volume_origin) / world_dim - 1.
        mlvl_feats = model.grid(vertices_query[..., [2, 1, 0]], concat=False)

        sdf_feats = list(map(lambda feat_pts, rgb_dim: feat_pts[:, :-rgb_dim, ...] if rgb_dim > 0 else feat_pts,
                             mlvl_feats, rgb_feature_dim))
        if concat_qp_sdf:
            sdf_feats.append(vertices_query.permute(0, 4, 1, 2, 3))

        sdf = model.decoder.geometry_net(torch.cat(sdf_feats, dim=1).squeeze(0).permute(1, 2, 3, 0))[..., 0]
        grads = F.normalize(compute_grads(sdf, coords), dim=-1).squeeze(0).detach()

    rgb_feats = map(lambda feat_pts, rgb_dim: feat_pts[:, -rgb_dim:, ...] if rgb_dim > 0 else None,
                    mlvl_feats, rgb_feature_dim)
    rgb_feats = list(filter(lambda x: x is not None, rgb_feats))
    rgb_feats = [torch.cat(rgb_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()]

    if use_view_dirs:
        rgb_feats.append(-grads)

    if use_normals:
        rgb_feats.append(grads)

    if concat_qp_rgb:
        rgb_feats.append(vertices_query.squeeze())

    if use_dot_prod:
        rgb_feats.append(-torch.ones_like(grads[..., :1]))

    rgb = torch.sigmoid(model.decoder.radiance_net(torch.cat(rgb_feats, dim=-1)))

    return rgb, grads, coords


def main(args):
    config = load_config(scene=args.scene, exp_name=args.exp_name, use_config_snapshot=True)
    suffix = config["iterations"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mesh_dir = os.path.join(config["log_dir"], "mesh/{}".format(suffix))
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)
    
    if config["dataset_type"] == "scannet":
        dataset = ScannetDataset(os.path.join(config["datasets_dir"], args.scene), load=False, trainskip=config["trainskip"], device=torch.device("cpu"))
    elif config["dataset_type"] == "rgbd":
        dataset = RGBDDataset(os.path.join(config["datasets_dir"], args.scene), load=False, trainskip=config["trainskip"], device=torch.device("cpu"))
    else:
        raise NotImplementedError

    model = SDFGridModel(config, device, dataset.get_bounds())
    state = torch.load(os.path.join(config["checkpoints_dir"], "chkpt_{}".format(suffix)), map_location=device)
    model.load_state_dict(state["model"])

    th = 0.
    with torch.no_grad():
        _, _, nx, ny, nz = model.grid.volumes[0].shape  # [1, C, nx, ny, nz]
        nx = (nx - 1) * config["reconstruct_upsample"] + 1
        ny = (ny - 1) * config["reconstruct_upsample"] + 1
        nz = (nz - 1) * config["reconstruct_upsample"] + 1
        unit_grid = torch.stack(torch.meshgrid(torch.linspace(-1, 1, nx),
                                                torch.linspace(-1, 1, ny),
                                                torch.linspace(-1, 1, nz))).to(device)  # [3, nx, ny, nz]

        sdf = batchify_vol(query_sdf, 1)(model, unit_grid,
                                         concat_qp=config["decoder"]["geometry"]["concat_qp"],
                                         rgb_feature_dim=config["rgb_feature_dim"])
        
        print(sdf.std().cpu().numpy(), sdf.min().cpu().numpy(), sdf.max().cpu().numpy())
        # run marching cubes
        voxel_sizes = model.world_dims.cpu().numpy() / (np.array([nx, ny, nz])-1)
        vertices, faces = mcubes.marching_cubes(sdf.cpu().numpy(), th, truncation=3.0)
        vertices *= voxel_sizes
        vertices += model.volume_origin.cpu().numpy()
        if args.color_mesh:
            colors, normals, _ = query_rgb(model, torch.tensor(vertices, device=device),
                                           model.volume_origin, model.world_dims,
                                           use_normals=config["decoder"]["radiance"]["use_normals"],
                                           use_view_dirs=config["decoder"]["radiance"]["use_view_dirs"],
                                           concat_qp_sdf=config["decoder"]["geometry"]["concat_qp"],
                                           concat_qp_rgb=config["decoder"]["radiance"]["concat_qp"],
                                           use_dot_prod=config["decoder"]["radiance"]["use_dot_prod"],
                                           rgb_feature_dim=config["rgb_feature_dim"])
            colors, normals = colors.cpu().numpy(), normals.cpu().numpy()
            mesh = trimesh.Trimesh(vertices, faces, vertex_colors=colors, vertex_normals=normals)
            filename = "{}_color.ply".format(args.scene)
        else:
            mesh = trimesh.Trimesh(vertices, faces)
            filename = "{}.ply".format(args.scene)

        mesh.export(os.path.join(mesh_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="release_test")
    parser.add_argument("--scene", type=str, default="grey_white_room")
    parser.add_argument("--color_mesh", action="store_true")
    args = parser.parse_args()
    
    main(args)
