import os
import trimesh
from scipy.spatial import cKDTree
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt


def compute_iou(mesh_pred, mesh_target):
    res = 0.05
    v_pred = mesh_pred.voxelized(pitch=res)
    v_target = mesh_target.voxelized(pitch=res)
    v_target_mesh = v_target.as_boxes()
    v_pred_mesh = v_pred.as_boxes()

    v_pred_filled = set(tuple(np.round(x, 4)) for x in v_pred.points)
    v_target_filled = set(tuple(np.round(x, 4)) for x in v_target.points)
    inter = v_pred_filled.intersection(v_target_filled)
    union = v_pred_filled.union(v_target_filled)
    iou = len(inter) / len(union)
    return iou, v_target_mesh, v_pred_mesh


def get_colored_pcd(pcd, metric):
    cmap = plt.cm.get_cmap("jet")
    color = cmap(metric / 0.10)[..., :3]
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d


def compute_metrics(path_pred, path_target):
    mesh_pred = trimesh.load_mesh(path_pred)
    mesh_target = trimesh.load_mesh(path_target)
    area_pred = int(mesh_pred.area * 1e4)
    area_tgt = int(mesh_target.area * 1e4)
    print("pred: {}, target: {}".format(area_pred, area_tgt))

    # iou, v_gt, v_pred = compute_iou(mesh_pred, mesh_target)

    pointcloud_pred, idx = mesh_pred.sample(area_pred, return_index=True)
    pointcloud_pred = pointcloud_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx]

    pointcloud_tgt, idx = mesh_target.sample(area_tgt, return_index=True)
    pointcloud_tgt = pointcloud_tgt.astype(np.float32)
    normals_tgt = mesh_target.face_normals[idx]

    thresholds = np.array([0.05])

    # for every point in gt compute the min distance to points in pred
    completeness, completeness_normals = distance_p2p(
        pointcloud_tgt, normals_tgt, pointcloud_pred, normals_pred
    )
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness ** 2

    # color gt_point_cloud using completion
    com_mesh = get_colored_pcd(pointcloud_tgt, completeness)

    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()

    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(
        pointcloud_pred, normals_pred, pointcloud_tgt, normals_tgt
    )
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy ** 2

    # color pred_point_cloud using completion
    acc_mesh = get_colored_pcd(pointcloud_pred, accuracy)

    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()

    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
            0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    chamferL1 = 0.5 * (completeness + accuracy)

    # F-Score
    F = [
        2 * precision[i] * recall[i] / (precision[i] + recall[i])
        for i in range(len(precision))
    ]
    rst = {
        "Acc": accuracy,
        "Comp": completeness,
        "C-L1": chamferL1,
        "NC": normals_correctness,
        "F-score": F[0]
    }

    return rst


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    """ Computes minimal distances of each point in points_src to points_tgt.
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    """
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def get_threshold_percentage(dist, thresholds):
    """ Evaluates a point cloud.
    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    """
    in_threshold = [
        (dist <= t).astype(np.float32).mean() for t in thresholds
    ]
    return in_threshold


def save_meshes(meshes, mesh_dir, save_name):
    for key in meshes:
        mesh = meshes[key]
        if isinstance(mesh, o3d.geometry.PointCloud):
            o3d.io.write_point_cloud(os.path.join(mesh_dir, "{}_{}.ply".format(key, save_name)), mesh)
        elif isinstance(mesh, trimesh.Trimesh):
            mesh.export(os.path.join(mesh_dir, "{}_{}.ply".format(key, save_name)))
