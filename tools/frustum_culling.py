import os
import numpy as np
import imageio
import trimesh
from copy import deepcopy
import pyrender


def cull_by_bounds(points, scene_bounds):
    eps = 0.02
    inside_mask = np.all(points >= (scene_bounds[0] - eps), axis=1) & np.all(points <= (scene_bounds[1] + eps), axis=1)
    return inside_mask


def cull_mesh(dataset, mesh_path, save_path, remove_missing_depth=True, remove_occlusion=True, scene_bounds=None, subdivide=True, max_edge=0.015):
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    vertices = mesh.vertices
    triangles = mesh.faces
    print(remove_occlusion)

    if subdivide:
        vertices, triangles = trimesh.remesh.subdivide_to_size(vertices, triangles, max_edge=max_edge, max_iter=10)

    # Cull with the bounding box first
    inside_mask = None
    if scene_bounds is not None:
        inside_mask = cull_by_bounds(vertices, scene_bounds)

    inside_mask = inside_mask[triangles[:, 0]] | inside_mask[triangles[:, 1]] | inside_mask[triangles[:, 2]]
    triangles = triangles[inside_mask, :]

    print("Processed culling by bound")
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    # load dataset
    # dataset = RGBDDataset(os.path.join(data_dir), trainskip=1, load=False, device=torch.device("cpu"))
    H, W, K = dataset.H, dataset.W, dataset.K
    c2w_list = []
    depth_gt_list = []
    step = len(dataset) // 300
    for i, frame_id in enumerate(dataset.frame_ids):
        if i % step != 0:
            continue

        c2w = np.array(dataset.all_gt_poses[frame_id]).astype(np.float32)
        depth_gt = imageio.imread(os.path.join(dataset.basedir, 'depth', dataset.gt_depth_files[frame_id]))
        depth_gt = (np.array(depth_gt) / 1000.0).astype(np.float32)
        c2w_list.append(c2w)
        depth_gt_list.append(depth_gt)
        print("Load frame: {}".format(i))

    del dataset
    rendered_depth_maps = render_depth_maps_doublesided(mesh, c2w_list, H, W, K, far=10.0)
    # rendered_depth_maps = render_depth_maps(mesh, c2w_list, H, W, K, far=10.0)

    # we don't need subdivided mesh to render depth
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.remove_unreferenced_vertices()

    # Cull faces
    points = vertices[:, :3]
    obs_mask, invalid_mask = get_grid_culling_pattern(points, c2w_list, H, W, K,
                                                      rendered_depth_list=rendered_depth_maps,
                                                      depth_gt_list=depth_gt_list,
                                                      remove_missing_depth=remove_missing_depth,
                                                      remove_occlusion=remove_occlusion,
                                                      verbose=True)
    obs1 = obs_mask[triangles[:, 0]]
    obs2 = obs_mask[triangles[:, 1]]
    obs3 = obs_mask[triangles[:, 2]]
    th1 = 3
    obs_mask = (obs1 > th1) | (obs2 > th1) | (obs3 > th1)
    inv1 = invalid_mask[triangles[:, 0]]
    inv2 = invalid_mask[triangles[:, 1]]
    inv3 = invalid_mask[triangles[:, 2]]
    invalid_mask = (inv1 > 0.7 * obs1) & (inv2 > 0.7 * obs2) & (inv3 > 0.7 * obs3)
    valid_mask = obs_mask & (~invalid_mask)
    triangles_in_frustum = triangles[valid_mask, :]

    mesh = trimesh.Trimesh(vertices, triangles_in_frustum, process=False)
    mesh.remove_unreferenced_vertices()

    mesh.export(save_path)


def get_culling_bound(scene):  # culling bound, might be different from scene bound defined in dataio
    if scene == "whiteroom":
        scene_bounds = np.array([[-2.46, -0.1, 0.36],
                                 [3.06, 3.3, 8.2]])
    elif scene == "kitchen":
        scene_bounds = np.array([[-3.12, -0.1, -3.18],
                                 [3.75, 3.3, 5.45]])
    elif scene == "breakfast_room":
        scene_bounds = np.array([[-2.23, -0.5, -1.7],
                                 [1.85, 2.77, 3.0]])
    elif scene == "staircase":
        scene_bounds = np.array([[-4.14, -0.1, -5.25],
                                 [2.52, 3.43, 1.08]])
    elif scene == "complete_kitchen":
        scene_bounds = np.array([[-5.55, 0.0, -6.45],
                                 [3.65, 3.1, 3.5]])
    elif scene == "green_room":
        scene_bounds = np.array([[-2.5, -0.1, 0.4],
                                 [5.4, 2.8, 5.0]])
    elif scene == "grey_white_room":
        scene_bounds = np.array([[-0.55, -0.1, -3.75],
                                 [5.3, 3.0, 0.65]])
    elif scene == "morning_apartment":
        scene_bounds = np.array([[-1.38, -0.1, -2.2],
                                 [2.1, 2.1, 1.75]])
    elif scene == "thin_geometry":
        scene_bounds = np.array([[-2.15, 0.0, 0.0],
                                 [0.77, 0.75, 3.53]])
    elif scene == "icl_living_room":
        scene_bounds = np.array([[-2.5, -0.1, -2.1],
                                 [2.6, 2.7, 3.1]])
    else:
        raise NotImplementedError

    return scene_bounds


def cull_from_one_pose(points, pose, H, W, K, rendered_depth=None, depth_gt=None, remove_missing_depth=True, remove_occlusion=True):
    c2w = deepcopy(pose)
    # to OpenCV
    c2w[:3, 1] *= -1
    c2w[:3, 2] *= -1
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]

    # pts under camera frame
    camera_space = rotation @ points.transpose() + translation[:, None]  # [3, N]
    uvz = (K @ camera_space).transpose()  # [N, 3]
    pz = uvz[:, 2] + 1e-8
    px = uvz[:, 0] / pz
    py = uvz[:, 1] / pz

    # step 1: inside frustum
    in_frustum = (0 <= px) & (px <= W - 1) & (0 <= py) & (py <= H - 1) & (pz > 0)
    u = np.clip(px, 0, W - 1).astype(np.int32)
    v = np.clip(py, 0, H - 1).astype(np.int32)
    eps = 0.02
    obs_mask = in_frustum
    # step 2: not occluded
    if remove_occlusion:
        obs_mask = in_frustum & (pz < (rendered_depth[v, u] + eps))  # & (depth_gt[v, u] > 0.)

    # step 3: valid depth in gt
    if remove_missing_depth:
        invalid_mask = in_frustum & (depth_gt[v, u] <= 0.)
    else:
        invalid_mask = np.zeros_like(obs_mask)

    return obs_mask.astype(np.int32), invalid_mask.astype(np.int32)


def get_grid_culling_pattern(points, poses, H, W, K, rendered_depth_list=None, depth_gt_list=None, remove_missing_depth=True, remove_occlusion=True, verbose=False):

    obs_mask = np.zeros(points.shape[0])
    invalid_mask = np.zeros(points.shape[0])
    for i, pose in enumerate(poses):
        if verbose:
            print('Processing pose ' + str(i + 1) + ' out of ' + str(len(poses)))
        rendered_depth = rendered_depth_list[i] if rendered_depth_list is not None else None
        depth_gt = depth_gt_list[i] if depth_gt_list is not None else None
        obs, invalid = cull_from_one_pose(points, pose, H, W, K, rendered_depth=rendered_depth, depth_gt=depth_gt, remove_missing_depth=remove_missing_depth, remove_occlusion=remove_occlusion)
        obs_mask = obs_mask + obs
        invalid_mask = invalid_mask + invalid

    return obs_mask, invalid_mask


def render_depth_maps(mesh, poses, H, W, K, far=10.0):
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=0.01, zfar=far)
    camera_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera_node)
    renderer = pyrender.OffscreenRenderer(W, H)
    render_flags = pyrender.RenderFlags.OFFSCREEN | pyrender.RenderFlags.DEPTH_ONLY

    depth_maps = []
    for pose in poses:
        scene.set_pose(camera_node, pose)
        depth = renderer.render(scene, render_flags)
        depth_maps.append(depth)

    return depth_maps


# For meshes with backward-facing faces. For some reason the no_culling flag in pyrender doesn't work for depth maps
def render_depth_maps_doublesided(mesh, poses, H, W, K, far=10.0):
    depth_maps_1 = render_depth_maps(mesh, poses, H, W, K, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]
    depth_maps_2 = render_depth_maps(mesh, poses, H, W, K, far=far)
    mesh.faces[:, [1, 2]] = mesh.faces[:, [2, 1]]  # it's a pass by reference, so I restore the original order

    depth_maps = []
    for i in range(len(depth_maps_1)):
        depth_map = np.where(depth_maps_1[i] > 0, depth_maps_1[i], depth_maps_2[i])
        depth_map = np.where((depth_maps_2[i] > 0) & (depth_maps_2[i] < depth_map), depth_maps_2[i], depth_map)
        depth_maps.append(depth_map)

    return depth_maps

