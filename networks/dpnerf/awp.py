import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.embedding import get_embedder
from networks.dpnerf.mam import MotionAggregationModule


class AdaptiveWeightProposal(nn.Module):
    # DBK - Weight Proposal Network
    def __init__(self, input_ch, num_motion,
                 D_sam, W_sam, D_mot, W_mot, dir_freq, rgb_freq, depth_freq, ray_dir_freq,
                 view_feature_ch, use_origin=True):
        super(AdaptiveWeightProposal, self).__init__()

        self.input_ch = input_ch
        self.num_motion = num_motion
        self.rgb_freq = rgb_freq
        self.depth_freq = depth_freq
        self.ray_dir_freq = ray_dir_freq
        self.view_feature_ch = view_feature_ch
        self.ccw_fine_scale = 0.05

        self.use_origin = use_origin
        if use_origin:
            self.output_ch = num_motion + 1
        else:
            self.output_ch = num_motion

        self.dropout = nn.Dropout(0.1)
        self.temperature = W_mot ** 0.5

        self.dirs_embed_fn, self.dirs_embed_ch = get_embedder(dir_freq, input_dim=3)
        self.rgb_embed_fn, self.rgb_embed_ch = get_embedder(self.rgb_freq, input_dim=3)
        self.depth_embed_fn, self.depth_embed_ch = get_embedder(self.depth_freq, input_dim=1)
        self.ray_dirs_embed_fn, self.ray_dirs_embed_ch = get_embedder(self.ray_dir_freq, input_dim=3)

        self.sample_feature_embed_layer = nn.ModuleList(
            [nn.Linear(self.input_ch, W_sam)] + [nn.Linear(W_sam, W_sam) for i in range(D_sam - 1)])

        self.motion_feature_embed_layer = nn.ModuleList(
            [nn.Linear((W_sam + self.view_feature_ch + self.ray_dirs_embed_ch), W_mot)]
            + [nn.Linear(W_mot, W_mot) for i in range(D_mot)])

        self.MAM = MotionAggregationModule(in_channels=W_mot, k=3, num_motion=self.num_motion)

        self.w_linear = nn.Linear(W_mot, self.output_ch)

    def feature_integration(self, feat, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            feat: [num_rays, num_motion, num_samples along ray, feature_dim]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            feat_integrated: [num_rays, num_motion, feature_dim]. integrated feature of a ray.
        """

        N_rays, N_motion, N_sample, N_dim = feat.shape
        feat = feat.reshape(-1, N_sample, N_dim)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        feat_density = feat[..., :-1, :]
        alpha = - torch.exp(-feat_density * dists[..., None]) + 1
        alpha = torch.cat([alpha, torch.zeros_like(alpha[:, 0:1])], dim=-2)

        weights = alpha * \
                  torch.cumprod(
                      torch.cat([torch.ones((alpha.shape[0], 1, alpha.shape[-1])), - alpha + (1. + 1e-10)], -2), -1)[:,
                  :-1, :]

        feat_integrated = torch.sum(weights * feat, dim=-2)

        return feat_integrated.reshape(N_rays, N_motion, N_dim)

    def forward(self, depth_feature, z_vals, rays_d, view_feature):
        view_embedded = view_feature

        N_ray, _, _, _ = depth_feature.reshape(-1, self.output_ch, depth_feature.shape[-2],
                                               depth_feature.shape[-1]).shape  # N_ray, N_motion, N_samlpe, fearture_dim
        h = depth_feature

        viewdirs = rays_d.reshape(N_ray, self.output_ch, -1)[:, 0, :]
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        ray_dirs_embed = self.ray_dirs_embed_fn(viewdirs)

        if view_embedded is not None:
            view_embedded = torch.cat([view_embedded, ray_dirs_embed], dim=-1)
        else:
            view_embedded = ray_dirs_embed

        for i, l in enumerate(self.sample_feature_embed_layer):
            h = self.sample_feature_embed_layer[i](h)
            h = F.relu(h)

        h_local = h

        h = self.feature_integration(h.reshape(N_ray, self.output_ch, h.shape[-2], h.shape[-1]), z_vals, rays_d)

        view_embedded = view_embedded.unsqueeze(1).repeat(1, self.output_ch, 1)
        h = torch.cat([h, view_embedded], dim=-1)

        for i, l in enumerate(self.motion_feature_embed_layer):
            h = self.motion_feature_embed_layer[i](h)
            h = F.relu(h)

        h = self.MAM(h, h_local)
        h = F.adaptive_avg_pool1d(h.transpose(1, 2), 1).squeeze(-1)  #

        w = torch.sigmoid(self.w_linear(h))
        out = w / (torch.sum(w, -1, keepdims=True))

        return out
