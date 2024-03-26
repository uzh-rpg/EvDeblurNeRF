import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False,
                 rgb_activate='sigmoid', rgb_add_bias=True, sigma_activate='relu', render_rmnearplane=0,
                 extract_feature="after_linear", composite_feature=True):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.extract_feature = extract_feature
        self.composite_feature = composite_feature
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.render_rmnearplane = render_rmnearplane
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        # Implementation according to the official code release
        # (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        # Implementation according to the paper
        activate = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'exp': torch.exp, 'none': lambda x: x,
                    'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
                    'softplus': lambda x: nn.Softplus()(x - 1)}
        self.rgb_activate = activate[rgb_activate]
        self.sigma_activate = activate[sigma_activate]

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3, bias=rgb_add_bias)
        else:
            assert rgb_add_bias is True
            self.output_linear = nn.Linear(W, output_ch)

    def mlpforward(self, inputs, viewdirs, embed_fn, embeddirs_fn, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
            """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = embed_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # batchify execution
        if netchunk is None:
            outputs_flat, feature_flat = self.eval(embedded)
        else:
            outputs_flat, feature_flat = [], []
            for i in range(0, max(1, embedded.shape[0]), netchunk):  # max(1,.) to handle empty rays
                output, feature = self.eval(embedded[i:i + netchunk])
                outputs_flat.append(output)
                feature_flat.append(feature)

            outputs_flat, feature_flat = torch.cat(outputs_flat, 0), torch.cat(feature_flat, 0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        feature = torch.reshape(feature_flat, list(inputs.shape[:-1]) + [feature_flat.shape[-1]])
        return outputs, feature

    def raw2outputs(self, raw, z_vals, rays_d, feature=None, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        def raw2alpha(raw_, dists_, act_fn):
            alpha_ = - torch.exp(-act_fn(raw_) * dists_) + 1.
            return torch.cat([alpha_, torch.ones_like(alpha_[:, 0:1])], dim=-1)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = self.rgb_activate(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., :-1, 3]) * raw_noise_std
            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.tensor(noise)

        density = self.sigma_activate(raw[..., :-1, 3] + noise)
        if not self.training and self.render_rmnearplane > 0:
            mask = z_vals[:, 1:]
            mask = mask > self.render_rmnearplane / 128
            mask = mask.type_as(density)
            density = mask * density

        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)
        # NOTE: cumprod introduces non-determinism
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        feature_map = torch.sum(weights[..., None] * feature, -2) if feature is not None else None  # [N_rays, 3]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, density, acc_map, weights, depth_map, feature_map

    def eval(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        return_feature = None
        if self.extract_feature == "before_linear":
            return_feature = h

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            if self.extract_feature == "after_linear":
                return_feature = feature

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            assert self.extract_feature != "after_linear"
            outputs = self.output_linear(h)

        return outputs, return_feature

    def forward(self, pts, viewdirs, pts_embed, dirs_embed, z_vals, rays_d, raw_noise_std, white_bkgd, is_train):
        raw, feature = self.mlpforward(pts, viewdirs, pts_embed, dirs_embed)

        if self.composite_feature:
            rgb_map, density_map, acc_map, weights, depth_map, feature_map = self.raw2outputs(
                raw, z_vals, rays_d, feature, raw_noise_std, white_bkgd)
        else:
            rgb_map, density_map, acc_map, weights, depth_map, feature_map = self.raw2outputs(
                raw, z_vals, rays_d, None, raw_noise_std, white_bkgd)
            feature_map = feature

        return rgb_map, depth_map, acc_map, weights, feature_map


