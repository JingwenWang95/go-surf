import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from model.sdf_grid_model import SDFGridModel, qp_to_sdf
from config import load_config
from model.utils import matrix_to_pose6d, pose6d_to_matrix
from model.utils import coordinates
from dataio.scannet_dataset import ScannetDataset
from dataio.rgbd_dataset import RGBDDataset


def main(args):
    config = load_config(scene=args.scene, exp_name=args.exp_name)
        
    events_save_dir = os.path.join(config["log_dir"], "events")
    if not os.path.exists(events_save_dir):
        os.makedirs(events_save_dir)
    writer = SummaryWriter(log_dir=events_save_dir)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config["dataset_type"] == "scannet":
        dataset = ScannetDataset(os.path.join(config["datasets_dir"], args.scene), trainskip=config["trainskip"], device=torch.device("cpu"))
    elif config["dataset_type"] == "rgbd":
        dataset = RGBDDataset(os.path.join(config["datasets_dir"], args.scene), trainskip=config["trainskip"], device=torch.device("cpu"))
    else:
        raise NotImplementedError
    
    model = SDFGridModel(config, device, dataset.get_bounds())
    
    ray_indices = torch.randperm(len(dataset) * dataset.H * dataset.W)
    
    # Inverse sigma from NeuS paper
    inv_s = nn.parameter.Parameter(torch.tensor(0.3, device=device))
    optimizer = torch.optim.Adam([{"params": model.decoder.parameters(), "lr": config["lr"]["decoder"]},
                                  {"params": model.grid.parameters(), "lr": config["lr"]["features"]},
                                  {"params": inv_s, "lr": config["lr"]["inv_s"]}])
    
    optimise_poses = config["optimise_poses"]
    poses_mat_init = torch.stack(dataset.c2w_list, dim=0).to(device)
    
    if optimise_poses:
        poses = nn.Parameter(matrix_to_pose6d(poses_mat_init))
        poses_optimizer = torch.optim.Adam([poses], config["lr"]["poses"])

    if args.start_iter > 0:
        state = torch.load(os.path.join(config["checkpoints_dir"], "chkpt_{}".format(args.start_iter)), map_location=device)
        inv_s = state["inv_s"]
        model.load_state_dict(state["model"])
        iteration = state["iteration"]
        optimizer.load_state_dict(state["optimizer"])
        
        if optimise_poses:
            poses = state["poses"]
            poses_optimizer.load_state_dict(state["poses_optimizer"])
    else:
        center = model.world_dims / 2. + model.volume_origin
        radius = model.world_dims.min() / 2.
        
        # Train SDF of a sphere
        for _ in range(500):
            optimizer.zero_grad()
            coords = coordinates(model.voxel_dims[1] - 1, device).float().t()
            pts = (coords + torch.rand_like(coords)) * config["voxel_sizes"][1] + model.volume_origin
            sdf, *_ = qp_to_sdf(pts.unsqueeze(1), model.volume_origin, model.world_dims, model.grid, model.sdf_decoder,
                                concat_qp=config["decoder"]["geometry"]["concat_qp"], rgb_feature_dim=config["rgb_feature_dim"])
            sdf = sdf.squeeze(-1)
            target_sdf = radius - (center - pts).norm(dim=-1)
            loss = torch.nn.functional.mse_loss(sdf, target_sdf)
            
            if loss.item() < 1e-10:
                break
            
            loss.backward()
            optimizer.step()
        
        print("Init loss after geom init (sphere)", loss.item())
    
        # Reset optimizer
        optimizer = torch.optim.Adam([{"params": model.decoder.parameters(), "lr": config["lr"]["decoder"]},
                                      {"params": model.grid.parameters(), "lr": config["lr"]["features"]},
                                      {"params": inv_s, "lr": config["lr"]["inv_s"]}])
        
    img_stride = dataset.H * dataset.W
    n_batches = ray_indices.shape[0] // config["batch_size"]
    for iteration in trange(args.start_iter + 1, config["iterations"] + 1):
        batch_idx = iteration % n_batches
        ray_ids = ray_indices[(batch_idx * config["batch_size"]):((batch_idx + 1) * config["batch_size"])]
        frame_id = ray_ids.div(img_stride, rounding_mode='floor')
        v = (ray_ids % img_stride).div(dataset.W, rounding_mode='floor')
        u = ray_ids % img_stride % dataset.W
        
        depth = dataset.depth_list[frame_id, v, u].to(device, non_blocking=True)
        rgb = dataset.rgb_list[frame_id, :, v, u].to(device, non_blocking=True)
        
        fx, fy = dataset.K_list[frame_id, 0, 0], dataset.K_list[frame_id, 1, 1]
        cx, cy = dataset.K_list[frame_id, 0, 2], dataset.K_list[frame_id, 1, 2]

        if config["dataset_type"] == "scannet":  # OpenCV
            rays_d_cam = torch.stack([(u - cx) / fx, (v - cy) / fy, torch.ones_like(fx)], dim=-1).to(device)
        else:  # OpenGL
            rays_d_cam = torch.stack([(u - cx) / fx, -(v - cy) / fy, -torch.ones_like(fy)], dim=-1).to(device)
        
        if optimise_poses:
            batch_poses = poses[frame_id]
            c2w = pose6d_to_matrix(batch_poses)
        else:
            c2w = poses_mat_init[frame_id]
        
        rays_o = c2w[:,:3,3]
        rays_d = torch.bmm(c2w[:, :3, :3], rays_d_cam[..., None]).squeeze()
        
        ret = model(rays_o, rays_d, rgb, depth, inv_s=torch.exp(10. * inv_s),
                    smoothness_std=config["smoothness_std"], iter=iteration)

        loss = config["rgb_weight"] * ret["rgb_loss"] +\
               config["depth_weight"] * ret["depth_loss"] +\
               config["fs_weight"] * ret["fs_loss"] +\
               config["sdf_weight"] * ret["sdf_loss"] +\
               config["normal_regularisation_weight"] * ret["normal_regularisation_loss"] +\
               config["normal_supervision_weight"] * ret["normal_supervision_loss"] +\
               config["eikonal_weight"] * ret["eikonal_loss"]
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.grid.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.)
        
        optimizer.step()
        optimizer.zero_grad()
        
        if optimise_poses:
            if iteration > 100:
                if iteration % 3 == 0:
                    poses_optimizer.step()
                    poses_optimizer.zero_grad()
            else:
                poses_optimizer.zero_grad()

        writer.add_scalar('depth', ret["depth_loss"].item(), iteration)
        writer.add_scalar('rgb', ret["rgb_loss"].item(), iteration)
        writer.add_scalar('fs', ret["fs_loss"].item(), iteration)
        writer.add_scalar('sdf', ret["sdf_loss"].item(), iteration)
        writer.add_scalar('psnr', ret["psnr"].item(), iteration)
        writer.add_scalar('eikonal', ret["eikonal_loss"].item(), iteration)
        writer.add_scalar('normal regularisation', ret["normal_regularisation_loss"].item(), iteration)

        if iteration % args.i_print == 0:
            tqdm.write("Iter: {}, PSNR: {:6f}, RGB Loss: {:6f}, Depth Loss: {:6f}, SDF Loss: {:6f}, FS Loss: {:6f}, "
                       "Eikonal Loss: {:6f}, Smoothness Loss: {:6f}".format(iteration,
                                                                            ret["psnr"].item(),
                                                                            ret["rgb_loss"].item(),
                                                                            ret["depth_loss"].item(),
                                                                            ret["sdf_loss"].item(),
                                                                            ret["fs_loss"].item(),
                                                                            ret["eikonal_loss"].item(),
                                                                            ret["normal_regularisation_loss"].item()))

        # Save checkpoint
        if iteration % args.i_save == 0:
            state = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'iteration': iteration,
                     'poses': poses if 'poses' in locals() else None,
                     'poses_optimizer': poses_optimizer.state_dict() if 'poses_optimizer' in locals() else None,
                     'inv_s': inv_s}
            torch.save(state, os.path.join(config["checkpoints_dir"], "chkpt_{}".format(iteration)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default="release_test")
    parser.add_argument('--scene', type=str, default="grey_white_room")
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--i_print', type=int, default=20)
    parser.add_argument('--i_save', type=int, default=1000)
    args = parser.parse_args()
    main(args)
