import os
import time
import re

import imageio
import torch
import torchvision
import numpy as np
import cv2
import open3d as o3d
from dataio.get_scene_bounds import get_scene_bounds


def load_poses(posefile):
    file = open(posefile, "r")
    lines = file.readlines()
    file.close()
    poses = []
    valid = []
    lines_per_matrix = 4
    for i in range(0, len(lines), lines_per_matrix):
        if 'nan' in lines[i]:
            valid.append(False)
            poses.append(np.eye(4, 4, dtype=np.float32).tolist())
        else:
            valid.append(True)
            pose_floats = [[float(x) for x in line.split()] for line in lines[i:i+lines_per_matrix]]
            poses.append(pose_floats)

    return poses, valid


def load_focal_length(filepath):
    file = open(filepath, "r")
    return float(file.readline())


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [int(x) if x.isdigit() else x for x in re.split('([0-9]+)', s)]


class RGBDDataset(torch.utils.data.Dataset):
    def __init__(self, basedir, load=True, trainskip=1, downsample_factor=1, crop=0, device=torch.device("cpu")):
        super(RGBDDataset).__init__()

        self.basedir = basedir
        self.device = device
        self.downsample_factor = downsample_factor
        self.crop = crop
        # Get image filenames, poses and intrinsics
        self.img_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'images')), key=alphanum_key) if f.endswith('png')]
        self.depth_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'depth_filtered')), key=alphanum_key) if f.endswith('png')]
        self.gt_depth_files = [f for f in sorted(os.listdir(os.path.join(basedir, 'depth')), key=alphanum_key) if f.endswith('png')]
        self.all_poses, valid_poses = load_poses(os.path.join(basedir, 'trainval_poses.txt'))
        self.all_gt_poses, valid_gt_poses = load_poses(os.path.join(basedir, 'poses.txt'))

        # get transformation between first bundle-fusion pose and first gt pose
        init_pose = np.array(self.all_poses[0]).astype(np.float32)
        init_gt_pose = np.array(self.all_gt_poses[0]).astype(np.float32)
        self.align_matrix = init_gt_pose @ np.linalg.inv(init_pose)

        depth = imageio.imread(os.path.join(self.basedir, 'depth_filtered', self.depth_files[0]))
        self.H, self.W = depth.shape[:2]
        focal = load_focal_length(os.path.join(self.basedir, 'focal.txt'))
        self.K = np.array([[focal, 0., (self.W - 1) / 2],
                          [0., focal, (self.H - 1) / 2],
                          [0., 0., 1.]])
        self.convert_tensor = torchvision.transforms.ToTensor()

        num_frames = len(self.img_files)
        train_frame_ids = list(range(0, num_frames, trainskip))

        self.frame_ids = []
        for id in train_frame_ids:
            if valid_poses[id]:
                self.frame_ids.append(id)
        # self.frame_ids = self.frame_ids[:200]

        self.c2w_gt_list = []
        self.c2w_list = []
        self.rgb_list = []
        self.depth_list = []
        self.depth_gt_list = []
        self.K_list = []
        if load:
            self.get_all_frames()
            
    def get_bounds(self):
        # mesh_gt = o3d.io.read_triangle_mesh(self.basedir + "/gt_mesh_culled.ply")
        # return torch.from_numpy(compute_scene_bounds(mesh_gt))
        return torch.from_numpy(get_scene_bounds(self.basedir.split('/')[-1])).float()

    def get_all_frames(self):
        for i, frame_id in enumerate(self.frame_ids):
            c2w_gt = np.array(self.all_gt_poses[frame_id]).astype(np.float32)
            c2w_gt = torch.from_numpy(c2w_gt).to(self.device)
            c2w = np.array(self.all_poses[frame_id]).astype(np.float32)
            # re-align all the poses to world coordinate
            c2w = self.align_matrix @ c2w
            c2w = torch.from_numpy(c2w).to(self.device)
            rgb = imageio.imread(os.path.join(self.basedir, 'images', self.img_files[frame_id]))
            depth = imageio.imread(os.path.join(self.basedir, 'depth_filtered', self.depth_files[frame_id]))
            H, W = depth.shape[:2]
            focal = load_focal_length(os.path.join(self.basedir, 'focal.txt'))
            rgb = (np.array(rgb) / 255.).astype(np.float32)
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            depth = (np.array(depth) / 1000.0).astype(np.float32)

            # Crop the undistortion artifacts
            if self.crop > 0:
                rgb = rgb[:, self.crop:-self.crop, self.crop:-self.crop, :]
                depth = depth[:, self.crop:-self.crop, self.crop:-self.crop, :]
                H, W = depth.shape[:2]

            if self.downsample_factor > 1:
                H = H // self.downsample_factor
                W = W // self.downsample_factor
                focal = focal // self.downsample_factor
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

            rgb = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)
            depth = torch.from_numpy(depth).to(self.device)
            K = torch.tensor([[focal, 0., (W - 1) / 2],
                              [0., focal, (H - 1) / 2],
                              [0., 0., 1.]], dtype=torch.float32).to(self.device)

            self.c2w_gt_list.append(c2w_gt)
            self.c2w_list.append(c2w)
            self.rgb_list.append(rgb)
            self.depth_list.append(depth)
            self.K_list.append(K)
            
        self.rgb_list = torch.stack(self.rgb_list, dim=0)
        self.depth_list = torch.stack(self.depth_list, dim=0)
        self.K_list = torch.stack(self.K_list, dim=0)

    def get_frame(self, id):
        ret = {
            "frame_id": self.frame_ids[id],
            "sample_id": torch.tensor(id),
            "c2w": self.c2w_list[id],
            "c2w_gt": self.c2w_gt_list[id],
            "rgb": self.rgb_list[id],
            "depth": self.depth_list[id],
            "K": self.K_list[id]
        }

        return ret

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, id):
        return self.get_frame(id)


if __name__ == "__main__":
    import torch
    from tools.vis_cameras import visualize, draw_cuboid
    from model.utils import compute_world_dims
    data_dir = "/media/jingwen/Data/neural_rgbd_data"
    scene = "morning_apartment"
    dataset = RGBDDataset(os.path.join(data_dir, scene))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False, num_workers=4)

    poses = []
    start = time.time()
    depth_max = 0.0
    depth_min = 100.0
    for i, data in enumerate(dataloader):
        if i % 40 != 0:
            continue
        depth = data["depth"]
        if depth.max().item() > depth_max:
            depth_max = depth.max().item()
        if depth.min().item() < depth_min:
            depth_min = depth.min().item()
        c2w = data["c2w"]
        c2w_gt = data["c2w_gt"]
        poses.append(c2w)
        poses.append(c2w_gt)
        print("Loaded frame: {}, depth_max: {}".format(i, depth.max()))
    end = time.time()
    print("Time taken to load th entire dataset: {}, depth_min: {}, depth_max: {}".format(end - start, depth_min, depth_max))
    poses = torch.stack(poses, 0).cpu().numpy()
    mesh_gt = o3d.io.read_triangle_mesh(os.path.join(data_dir, scene, "gt_mesh_culled.ply"))
    mesh_gt.compute_vertex_normals()

    # get cuiboid and sphere
    scene_bound = torch.from_numpy(get_scene_bounds(scene, new_bound=True))
    world_dims_glob, volume_origin_glob, voxel_dims_glob = compute_world_dims(scene_bound, [0.04, 0.16, 0.64], 3,
                                                                              margin=0.1, device=torch.device("cpu"))
    centre = (world_dims_glob.cpu().numpy() / 2.) + volume_origin_glob.cpu().numpy()
    radius = world_dims_glob.min().item() / 2
    print("radius: {}".format(radius))
    volume_bound = np.stack([volume_origin_glob.cpu().numpy(), (volume_origin_glob + world_dims_glob).cpu().numpy()], 1)
    cube = draw_cuboid(volume_bound)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
    sphere.translate(centre)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    visualize(poses, [cube, sphere, mesh_gt])
