import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic
from math import exp, sqrt


def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb: # [-1,1] --> [0,1]
        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1. ) / 2.

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).cuda())

    return psnr

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

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
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class GeoNeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        """
        Render background
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        density, sampled_color = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)

        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        

        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    intrinsics=None, intrinsics_inv=None, poses=None, images=None):
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        gradients = sdf_network.gradient(pts).squeeze()
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        alpha_depth = alpha
        sampled_depth = mid_z_vals * inside_sphere

        # Render with background
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        weights_depth = alpha_depth * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha_depth + 1e-7], -1), -1)[:, :-1]
        depth_fine = (sampled_depth * weights_depth).sum(dim=1)
        pts_depth = rays_o[:, None, :] + rays_d[:, None, :] * depth_fine[..., None, None]


        # Eikonal loss
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        # Updated by Qingshan
        sdf_d = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf_d[:, :-1], sdf_d[:, 1:]
        sign = prev_sdf * next_sdf
        sign = torch.where(sign <= 0, torch.ones_like(sign), torch.zeros_like(sign))
        idx = reversed(torch.Tensor(range(1, n_samples)).cuda())
        tmp = torch.einsum("ab,b->ab", (sign, idx))
        prev_idx = torch.argmax(tmp, 1, keepdim=True)
        next_idx = prev_idx + 1

        prev_inside_sphere = torch.gather(inside_sphere, 1, prev_idx)
        next_inside_sphere = torch.gather(inside_sphere, 1, next_idx)
        mid_inside_sphere = (0.5 * (prev_inside_sphere + next_inside_sphere) > 0.5).float()
        sdf1 = torch.gather(sdf_d, 1, prev_idx)
        sdf2 = torch.gather(sdf_d, 1, next_idx)
        z_vals1 = torch.gather(mid_z_vals, 1, prev_idx)
        z_vals2 = torch.gather(mid_z_vals, 1, next_idx)
        z_vals_sdf0 = (sdf1 * z_vals2 - sdf2 * z_vals1) / (sdf1 - sdf2 + 1e-10)
        z_vals_sdf0 = torch.where(z_vals_sdf0 < 0, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        max_z_val = torch.max(z_vals)
        z_vals_sdf0 = torch.where(z_vals_sdf0 > max_z_val, torch.zeros_like(z_vals_sdf0), z_vals_sdf0)
        pts_sdf0 = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_sdf0[..., :, None]  # [batch_size, 1, 3]
        gradients_sdf0 = sdf_network.gradient(pts_sdf0.reshape(-1, 3)).squeeze().reshape(batch_size, 1, 3)
        gradients_sdf0 = gradients_sdf0 / torch.linalg.norm(gradients_sdf0, ord=2, dim=-1, keepdim=True)
        gradients_sdf0 = torch.matmul(poses[0, :3, :3].permute(1, 0)[None, ...], gradients_sdf0.permute(0, 2, 1)).permute(0, 2, 1).detach()

        project_xyz = torch.matmul(poses[0, :3, :3].permute(1, 0), pts_sdf0.permute(0, 2, 1))
        t = - torch.matmul(poses[0, :3, :3].permute(1, 0), poses[0, :3, 3, None])
        project_xyz = project_xyz + t
        pts_sdf0_ref = project_xyz
        project_xyz = torch.matmul(intrinsics[0, :3, :3], project_xyz)  # [batch_size, 3, 1]
        depth_sdf = project_xyz[:, 2, 0] * mid_inside_sphere.squeeze(1)
        disp_sdf0 = torch.matmul(gradients_sdf0, pts_sdf0_ref)


        # Compute Homography
        K_ref_inv = intrinsics_inv[0, :3, :3]
        K_src = intrinsics[1:, :3, :3]
        num_src = K_src.shape[0]
        R_ref_inv = poses[0, :3, :3]
        R_src = poses[1:, :3, :3].permute(0, 2, 1)
        C_ref = poses[0, :3, 3]
        C_src = poses[1:, :3, 3]
        R_relative = torch.matmul(R_src, R_ref_inv)
        C_relative = C_ref[None, ...] - C_src
        tmp = torch.matmul(R_src, C_relative[..., None])
        tmp = torch.matmul(tmp[None, ...].expand(batch_size, num_src, 3, 1), gradients_sdf0.expand(batch_size, num_src, 3)[..., None].permute(0, 1, 3, 2))  # [Batch_size, num_src, 3, 1]
        tmp = R_relative[None, ...].expand(batch_size, num_src, 3, 3) + tmp / (disp_sdf0[..., None] + 1e-10)
        tmp = torch.matmul(K_src[None, ...].expand(batch_size, num_src, 3, 3), tmp)
        Hom = torch.matmul(tmp, K_ref_inv[None, None, ...])

        pixels_x = project_xyz[:, 0, 0] / (project_xyz[:, 2, 0] + 1e-8)
        pixels_y = project_xyz[:, 1, 0] / (project_xyz[:, 2, 0] + 1e-8)
        pixels = torch.stack([pixels_x, pixels_y], dim=-1).float()
        patch_size = 5
        total_size = (patch_size * 2 + 1) ** 2
        offsets = self.update_patch_size(patch_size, rays_o.device)  # [1, 121, 2]
        pixels_patch = pixels.view(batch_size, 1, 2) + offsets.float()  # [batch_size, 121, 2]

        ref_image = images[0, :, :]
        src_images = images[1:, :, :]
        h, w = ref_image.shape
        
        grid = self.patch_homography(Hom, pixels_patch)
        grid[:, :, 0] = 2 * grid[:, :, 0] / (w - 1) - 1.0
        grid[:, :, 1] = 2 * grid[:, :, 1] / (h - 1) - 1.0
        sampled_gray_val = F.grid_sample(src_images.unsqueeze(1), grid.view(num_src, -1, 1, 2), align_corners=True)
        sampled_gray_val = sampled_gray_val.view(num_src, batch_size, total_size, 1)  # [nsrc, batch_size, 121, 1]
        pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (w - 1) - 1.0
        pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (h - 1) - 1.0
        grid = pixels_patch.detach()
        ref_gray_val = F.grid_sample(ref_image[None, None, ...], grid.view(1, -1, 1, 2), align_corners=True)
        ref_gray_val = ref_gray_val.view(1, batch_size, total_size, 1)
        ncc = self.compute_LNCC(ref_gray_val, sampled_gray_val)
        ncc = ncc * mid_inside_sphere


        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere,
            'depth_sdf': depth_sdf,
            'ncc_cost': ncc,
            'mid_inside_sphere': mid_inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, intrinsics=None, intrinsics_inv=None, poses=None, images=None):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None

        # Up sample
        if self.n_importance > 0:
            with torch.no_grad():
                rgb_d = z_vals.shape[1]
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                

                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    intrinsics=intrinsics, intrinsics_inv=intrinsics_inv, poses=poses, images=images)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)


        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere'],
            'depth_sdf': ret_fine['depth_sdf'],
            'ncc_cost': ret_fine['ncc_cost'],
            'mid_inside_sphere': ret_fine['mid_inside_sphere']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))

    def patch_homography(self, H, uv):
        # H: [batch_size, nsrc, 3, 3]
        # uv: [batch_size, 121, 2]
        N, Npx = uv.shape[:2]
        H = H.permute(1, 0, 2, 3)
        Nsrc = H.shape[0]
        H = H.view(Nsrc, N, -1, 3, 3)
        ones = torch.ones(uv.shape[:-1], device=uv.device).unsqueeze(-1)
        hom_uv = torch.cat((uv, ones), dim=-1)
 
        tmp = torch.einsum("vprik,pok->vproi", H, hom_uv)
        tmp = tmp.reshape(Nsrc, -1, 3)

        grid = tmp[..., :2] / (tmp[..., 2:] + 1e-8)

        return grid
        

    def update_patch_size(self, h_patch_size, device):
        offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
        return torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)  # nb_pixels_patch * 2


    def compute_LNCC(self, ref_gray, src_grays):
        # ref_gray: [1, batch_size, 121, 1]
        # src_grays: [nsrc, batch_size, 121, 1]
        ref_gray = ref_gray.permute(1, 0, 3, 2)  # [batch_size, 1, 1, 121]
        src_grays = src_grays.permute(1, 0, 3, 2)  # [batch_size, nsrc, 1, 121]

        ref_src = ref_gray * src_grays  # [batch_size, nsrc, 1, npatch]

        bs, nsrc, nc, npatch = src_grays.shape
        patch_size = int(sqrt(npatch))
        ref_gray = ref_gray.view(bs, 1, 1, patch_size, patch_size).view(-1, 1, patch_size, patch_size)
        src_grays = src_grays.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)
        ref_src = ref_src.view(bs, nsrc, 1, patch_size, patch_size).contiguous().view(-1, 1, patch_size, patch_size)

        ref_sq = ref_gray.pow(2)
        src_sq = src_grays.pow(2)

        filters = torch.ones(1, 1, patch_size, patch_size, device=ref_gray.device)
        padding = patch_size // 2

        ref_sum = F.conv2d(ref_gray, filters, stride=1, padding=padding)[:, :, padding, padding]
        src_sum = F.conv2d(src_grays, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
        ref_sq_sum = F.conv2d(ref_sq, filters, stride=1, padding=padding)[:, :, padding, padding]
        src_sq_sum = F.conv2d(src_sq, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)
        ref_src_sum = F.conv2d(ref_src, filters, stride=1, padding=padding)[:, :, padding, padding].view(bs, nsrc)

        u_ref = ref_sum / npatch
        u_src = src_sum / npatch

        cross = ref_src_sum - u_src * ref_sum - u_ref * src_sum + u_ref * u_src * npatch
        ref_var = ref_sq_sum - 2 * u_ref * ref_sum + u_ref * u_ref * npatch
        src_var = src_sq_sum - 2 * u_src * src_sum + u_src * u_src * npatch

        cc = cross * cross / (ref_var * src_var + 1e-5)  # [batch_size, nsrc, 1, npatch]
        ncc = 1 - cc
        ncc = torch.clamp(ncc, 0.0, 2.0)
        ncc, _ = torch.topk(ncc, 4, dim=1, largest=False)
        ncc = torch.mean(ncc, dim=1, keepdim=True)

        return ncc




