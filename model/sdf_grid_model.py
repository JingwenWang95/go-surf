import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.decoder import NeRFDecoder
from model.multi_grid import MultiGrid
from model.utils import compute_world_dims, coordinates


class SDFGridModel(torch.nn.Module):
    def __init__(self,
                 config,
                 device,
                 bounds,
                 margin=0.1,
                 ):
        super(SDFGridModel, self).__init__()
        self.device = device
        # Simple decoder
        
        world_dims, volume_origin, voxel_dims = compute_world_dims(bounds,
                                                                   config["voxel_sizes"],
                                                                   len(config["voxel_sizes"]),
                                                                   margin=margin,
                                                                   device=device)
        
        self.world_dims = world_dims
        self.volume_origin = volume_origin
        self.voxel_dims = voxel_dims
        
        grid_dim = (torch.tensor(config["sdf_feature_dim"]) + torch.tensor(config["rgb_feature_dim"])).tolist()
        self.grid = MultiGrid(voxel_dims, grid_dim).to(device)
        
        self.decoder = NeRFDecoder(config["decoder"]["geometry"],
                                   config["decoder"]["radiance"],
                                   sdf_feat_dim=sum(config["sdf_feature_dim"]),
                                   rgb_feat_dim=sum(config["rgb_feature_dim"])).to(device)
        self.sdf_decoder = batchify(self.decoder.geometry_net, max_chunk=None)
        self.rgb_decoder = batchify(self.decoder.radiance_net, max_chunk=None)
        self.config = config

    def forward(self, rays_o, rays_d, target_rgb_select, target_depth_select, inv_s=None, smoothness_std=0., iter=0):
        rend_dict = render_rays(self.sdf_decoder,
                                self.rgb_decoder,
                                self.grid,
                                self.volume_origin,
                                self.world_dims,
                                self.config["voxel_sizes"],
                                rays_o,
                                rays_d,
                                near=self.config["near"],
                                far=self.config["far"],
                                n_samples=self.config["n_samples"],
                                depth_gt=target_depth_select,
                                n_importance=self.config["n_importance"],
                                truncation=self.config["truncation"],
                                inv_s=inv_s,
                                smoothness_std=smoothness_std,
                                use_view_dirs=self.config["decoder"]["radiance"]["use_view_dirs"],
                                use_normals=self.config["decoder"]["radiance"]["use_normals"],
                                concat_qp_to_rgb=self.config["decoder"]["radiance"]["concat_qp"],
                                concat_qp_to_sdf=self.config["decoder"]["geometry"]["concat_qp"],
                                concat_dot_prod_to_rgb=self.config["decoder"]["radiance"]["use_dot_prod"],
                                rgb_feature_dim=self.config["rgb_feature_dim"],
                                iter=iter)

        rendered_rgb, rendered_depth = rend_dict["rgb"], rend_dict["depth"]
        rgb_loss = compute_loss(rendered_rgb, target_rgb_select, "l2")
        psnr = mse2psnr(rgb_loss)
        valid_depth_mask = target_depth_select > 0.
        depth_loss = F.l1_loss(target_depth_select[valid_depth_mask], rendered_depth[valid_depth_mask])

        ret = {
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "sdf_loss": rend_dict["sdf_loss"],
            "fs_loss": rend_dict["fs_loss"],
            "normal_regularisation_loss": rend_dict["normal_regularisation_loss"],
            "normal_supervision_loss": rend_dict["normal_supervision_loss"],
            "eikonal_loss": rend_dict["eikonal_loss"],
            "psnr": psnr
        }

        return ret


def batchify(fn, max_chunk=1024*128):
    if max_chunk is None:
        return fn
    def ret(feats):
        chunk = max_chunk // (feats.shape[1] * feats.shape[2])
        return torch.cat([fn(feats[i:i+chunk]) for i in range(0, feats.shape[0], chunk)], dim=0)
    return ret


def render_rays(sdf_decoder,
                rgb_decoder,
                feat_volume,  # regualized feature volume [1, feat_dim, Nx, Ny, Nz]
                volume_origin,  # volume origin, Euclidean coords [3,]
                volume_dim,  # volume dimensions, Euclidean coords [3,]
                voxel_sizes,  # length of the voxel side, Euclidean distance
                rays_o,
                rays_d,
                truncation=0.10,
                near=0.01,
                far=3.0,
                n_samples=128,
                n_importance=16,
                depth_gt=None,
                inv_s=20.,
                normals_gt=None,
                smoothness_std=0.0,
                randomize_samples=True,
                use_view_dirs=False,
                use_normals=False,
                concat_qp_to_rgb=False,
                concat_qp_to_sdf=False,
                concat_dot_prod_to_rgb=False,
                iter=0,
                rgb_feature_dim=[],
                ):

    n_rays = rays_o.shape[0]
    z_vals = torch.linspace(near, far, n_samples).to(rays_o)
    z_vals = z_vals[None, :].repeat(n_rays, 1)  # [n_rays, n_samples]
    sample_dist = (far - near) / n_samples
    
    if randomize_samples:
        z_vals += torch.rand_like(z_vals) * sample_dist
        
    n_importance_steps = n_importance // 12
    with torch.no_grad():
        for step in range(n_importance_steps):
            query_points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
            sdf, *_ = qp_to_sdf(query_points, volume_origin, volume_dim, feat_volume, sdf_decoder, concat_qp=concat_qp_to_sdf, rgb_feature_dim=rgb_feature_dim)
            
            prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
            prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
            mid_sdf = (prev_sdf + next_sdf) * 0.5
            cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

            prev_cos_val = torch.cat([torch.zeros([n_rays, 1], device=z_vals.device), cos_val[:, :-1]], dim=-1)
            cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
            cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
            cos_val = cos_val.clip(-1e3, 0.0)
            
            dists = next_z_vals - prev_z_vals
            weights = neus_weights(mid_sdf, dists, torch.tensor(64. * 2 ** step, device=mid_sdf.device), cos_val)
            z_samples = sample_pdf(z_vals, weights, 12, det=True).detach()
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
        
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.ones_like(depth_gt).unsqueeze(-1) * sample_dist], dim=1)
    z_vals_mid = z_vals + dists * 0.5
    view_dirs = F.normalize(rays_d, dim=-1)[:, None, :].repeat(1, n_samples + n_importance, 1)
    query_points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_mid[..., :, None]
    query_points = query_points.requires_grad_(True)
    sdf, rgb_feat, world_bound_mask = qp_to_sdf(query_points, volume_origin, volume_dim, feat_volume,
                                                sdf_decoder, concat_qp=concat_qp_to_sdf, rgb_feature_dim=rgb_feature_dim)
    grads = compute_grads(sdf, query_points)
    
    rgb_feat = [rgb_feat]
    
    if use_view_dirs:
        rgb_feat.append(view_dirs)
    if use_normals:
        rgb_feat.append(grads)
    if concat_qp_to_rgb:
        rgb_feat.append(2. * (query_points - volume_origin) / volume_dim - 1.)
    if concat_dot_prod_to_rgb:
        rgb_feat.append((view_dirs * grads).sum(dim=-1, keepdim=True))
    
    rgb = torch.sigmoid(rgb_decoder(torch.cat(rgb_feat, dim=-1)))
    
    cos_val = (view_dirs * grads).sum(-1)
    # cos_val = -F.relu(-cos_val)
    cos_anneal_ratio = min(iter / 5000., 1.)
    cos_val = -(F.relu(-cos_val * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                F.relu(-cos_val) * cos_anneal_ratio)
    weights = neus_weights(sdf, dists, inv_s, cos_val)
    weights[~world_bound_mask] = 0.
    
    rendered_rgb = torch.sum(weights[..., None] * rgb, dim=-2)
    rendered_depth = torch.sum(weights * z_vals_mid, dim=-1)
    # depth_var = torch.sum(weights * torch.square(z_vals_mid - rendered_depth.unsqueeze(-1)), dim=-1)

    eikonal_weights = sdf[world_bound_mask].detach().abs() + 1e-2 # torch.abs(z_vals - depth_gt.unsqueeze(-1))[world_bound_mask] > truncation
    eikonal_loss = (torch.square(grads.norm(dim=-1)[world_bound_mask] - 1.) * eikonal_weights).sum() / eikonal_weights.sum()
    eikonal_loss = eikonal_loss.mean()

    if depth_gt is not None:
        fs_loss, sdf_loss = get_sdf_loss(z_vals_mid, depth_gt[:, None], sdf, truncation)
    else:
        fs_loss, sdf_loss = torch.tensor(0.0, device=rays_o.device), torch.tensor(0.0, device=rays_o.device)
        
    normal_regularisation_loss = torch.tensor(0., device=z_vals.device)
    if smoothness_std > 0:
        coords = coordinates([feat_volume.volumes[1].shape[2] - 1, feat_volume.volumes[1].shape[3] - 1, feat_volume.volumes[1].shape[4] - 1], z_vals.device).float().t()
        world = ((coords + torch.rand_like(coords)) * voxel_sizes[1] + volume_origin).unsqueeze(0)
        surf = rays_o + rays_d * rendered_depth.unsqueeze(-1)
        surf_mask = ((surf > volume_origin) & (surf < (volume_origin + volume_dim))).all(dim=-1)
        surf = surf[surf_mask,:].unsqueeze(0)
        weight = torch.cat([torch.ones(world.shape[:-1], device=world.device) * 0.1, torch.ones(surf.shape[:-1], device=surf.device)], dim=1)
        world = torch.cat([world, surf], dim=1)
        
        query_points = world.requires_grad_(True)
        sdf, *_ = qp_to_sdf(query_points, volume_origin, volume_dim, feat_volume, sdf_decoder, concat_qp=concat_qp_to_sdf, rgb_feature_dim=rgb_feature_dim)
        mask = sdf.abs() < truncation
        
        if mask.any().item():
            grads = compute_grads(sdf, query_points)[mask]
            
            # Sample points inside unit circle orthogonal to gradient direction
            n = F.normalize(grads, dim=-1)
            u = F.normalize(n[...,[1,0,2]] * torch.tensor([1., -1., 0.], device=n.device), dim=-1)
            v = torch.cross(n, u, dim=-1)
            phi = torch.rand(list(grads.shape[:-1]) + [1], device=grads.device) * 2. * np.pi
            w = torch.cos(phi) * u + torch.sin(phi) * v
            
            world2 = world[mask] + w * smoothness_std
            query_points = world2.requires_grad_(True)
            sdf, *_ = qp_to_sdf(query_points, volume_origin, volume_dim, feat_volume, sdf_decoder, concat_qp=concat_qp_to_sdf, rgb_feature_dim=rgb_feature_dim)
            grads2 = compute_grads(sdf, query_points)
            
            normal_regularisation_loss = ((grads - grads2).norm(dim=-1) * weight[mask]).sum() / weight[mask].sum()
    
    normal_supervision_loss = torch.tensor(0., device=z_vals.device)
    if normals_gt is not None:
        depth_mask = (depth_gt > 0.) & (normals_gt != 0.).any(dim=-1)
        query_points = rays_o[depth_mask] + rays_d[depth_mask] * depth_gt[depth_mask, None]
        query_points = query_points.requires_grad_(True)
        sdf, *_ = qp_to_sdf(query_points, volume_origin, volume_dim, feat_volume, sdf_decoder, concat_qp=concat_qp_to_sdf, rgb_feature_dim=rgb_feature_dim)
        normals = compute_grads(sdf, query_points)
        normal_supervision_loss = F.mse_loss(normals, normals_gt[depth_mask])

    ret = {"rgb": rendered_rgb,
           "depth": rendered_depth,
           "sdf_loss": sdf_loss,
           "fs_loss": fs_loss,
           "sdfs": sdf,
           "weights": weights,
           "normal_regularisation_loss": normal_regularisation_loss,
           "eikonal_loss": eikonal_loss,
           "normal_supervision_loss": normal_supervision_loss,
           }

    return ret


mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(x)


def qp_to_sdf(pts, volume_origin, volume_dim, feat_volume, sdf_decoder, sdf_act=nn.Identity(), concat_qp=False, rgb_feature_dim=[]):
    # Normalize point cooridnates and mask out out-of-bounds points
    pts_norm = 2. * (pts - volume_origin[None, None, :]) / volume_dim[None, None, :] - 1.
    mask = (pts_norm.abs() <= 1.).all(dim=-1)
    pts_norm = pts_norm[mask].unsqueeze(0).unsqueeze(0).unsqueeze(0)
    
    mlvl_feats = feat_volume(pts_norm[...,[2,1,0]], concat=False)
    sdf_feats = list(map(lambda feat_pts, rgb_dim: feat_pts[:,:-rgb_dim,...] if rgb_dim > 0 else feat_pts,
                         mlvl_feats, rgb_feature_dim))
    sdf_feats = torch.cat(sdf_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()
    
    rgb_feats = map(lambda feat_pts, rgb_dim: feat_pts[:,-rgb_dim:,...] if rgb_dim > 0 else None,
                    mlvl_feats, rgb_feature_dim)
    rgb_feats = list(filter(lambda x: x is not None, rgb_feats))
    rgb_feats = torch.cat(rgb_feats, dim=1).squeeze(0).squeeze(1).squeeze(1).t()
    
    rgb_feats_unmasked = torch.zeros(list(mask.shape) + [sum(rgb_feature_dim)], device=pts_norm.device)
    rgb_feats_unmasked[mask] = rgb_feats
    
    if concat_qp:
        sdf_feats.append(pts_norm.permute(0,4,1,2,3))
    
    raw = sdf_decoder(sdf_feats)
    sdf = torch.zeros_like(mask, dtype=pts_norm.dtype)
    sdf[mask] = sdf_act(raw.squeeze(-1))
    
    return sdf, rgb_feats_unmasked, mask


def neus_weights(sdf, dists, inv_s, cos_val, z_vals=None):    
    estimated_next_sdf = sdf + cos_val * dists * 0.5
    estimated_prev_sdf = sdf - cos_val * dists * 0.5
    
    prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
    next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

    p = prev_cdf - next_cdf
    c = prev_cdf

    alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
    weights = alpha * torch.cumprod(torch.cat([torch.ones([sdf.shape[0], 1], device=alpha.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
    
    if z_vals is not None:
        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0., torch.ones_like(signs), torch.zeros_like(signs))
        # This will only return the first zero-crossing
        inds = torch.argmax(mask, dim=1, keepdim=True)
        z_surf = torch.gather(z_vals, 1, inds)
        return weights, z_surf
    
    return weights


def apply_log_transform(tsdf):
    sgn = torch.sign(tsdf)
    out = torch.log(torch.abs(tsdf) + 1)
    out = sgn * out
    return out


def compute_loss(prediction, target, loss_type="l2"):
    if loss_type == "l2":
        return F.mse_loss(prediction, target)
    elif loss_type == "l1":
        return F.l1_loss(prediction, target)
    elif loss_type == "log":
        return F.l1_loss(apply_log_transform(prediction), apply_log_transform(target))
    raise Exception("Unknown loss type")


def compute_grads(predicted_sdf, query_points):
    sdf_grad, = torch.autograd.grad([predicted_sdf], [query_points], [torch.ones_like(predicted_sdf)], create_graph=True)
    return sdf_grad


def get_sdf_loss(z_vals, target_d, predicted_sdf, truncation):
    depth_mask = target_d > 0.
    front_mask = (z_vals < (target_d - truncation))
    # bask_mask = (z_vals > (target_d + truncation)) & depth_mask
    front_mask = (front_mask | ((target_d < 0.) & (z_vals < 3.5)))
    bound = (target_d - z_vals)
    bound[target_d[:,0] < 0., :] = 10. # TODO: maybe use noisy depth for bound?
    sdf_mask = (bound.abs() <= truncation) & depth_mask
    
    sum_of_samples = front_mask.sum(dim=-1) + sdf_mask.sum(dim=-1) + 1e-8
    rays_w_depth = torch.count_nonzero(target_d)
    
    fs_loss = (torch.max(torch.exp(-5. * predicted_sdf) - 1., predicted_sdf - bound).clamp(min=0.) * front_mask)
    fs_loss = (fs_loss.sum(dim=-1) / sum_of_samples).sum() / rays_w_depth
    sdf_loss = ((torch.abs(predicted_sdf - bound) * sdf_mask).sum(dim=-1) / sum_of_samples).sum() / rays_w_depth

    return fs_loss, sdf_loss


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    # device = weights.get_device()
    device = weights.device
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / N_importance, 1. - 0.5 / N_importance, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples