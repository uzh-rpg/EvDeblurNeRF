import torch
from torch import nn as nn
from torch.nn import functional as F


class VoxelNeRFBase(nn.Module):

    def __init__(self,
                 aabb,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 add_bias_color=False,
                 input_ch=3, input_ch_views=3,
                 render_rmnearplane=0, app_dim=32,
                 app_n_comp=[64, 16, 16], n_voxels=134217984,
                 rgb_activate="sigmoid",
                 sigma_activate="relu",
                 extract_feature="after_linear",
                 composite_feature=True,
                 app_actfn='none'):

        super(VoxelNeRFBase, self).__init__()

        activate = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'exp': torch.exp, 'none': lambda x: x,
                    'tanh': torch.tanh, 'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
                    'softplus': lambda x: nn.Softplus()(x - 1)}
        self.rgb_activate = activate[rgb_activate]
        self.sigma_activate = activate[sigma_activate]
        self.app_activate = activate[app_actfn]

        self.extract_feature = extract_feature
        self.composite_feature = composite_feature
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.render_rmnearplane = render_rmnearplane
        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        if extract_feature != "after_sigma":
            print(f"VoxelNeRFBase: this backbone does not distinguish the extract_feature "
                  f"mode ({extract_feature}). Features will always be taken after the sigma_net")

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 self.geo_feat_dim features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=add_bias_color))

        self.color_net = nn.ModuleList(color_net)
        self.app_dim = app_dim  # app_dim
        self.app_n_comp = app_n_comp  # [48,12,12]

        self.aabb = torch.stack(aabb).cuda()
        xyz_min, xyz_max = aabb
        voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
        gridSize = ((xyz_max - xyz_min) / voxel_size).long().tolist()
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0 / self.aabbSize
        self.gridSize = torch.LongTensor(gridSize)  # .to(self.device)
        print('Fine Voxel GridSize', self.gridSize)

        # self.gridSize = [164*4, 167*4, 76*4]
        # self.aabbSize = self.aabb[1] - self.aabb[0]
        # self.invaabbSize = 2.0/self.aabbSize

        self.matMode = [[0, 1], [0, 2], [1, 2]]
        self.vecMode = [2, 1, 0]
        self.comp_w = [1, 1, 1]
        self.reg = TVLoss()

        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False)  # .to(device)

    def init_one_svd(self, n_component, gridSize, scale):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))
            line_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        # return plane_coef, line_coef
        return torch.nn.ParameterList(plane_coef), torch.nn.ParameterList(line_coef)

    def get_optparam_groups(self, lr_init_spatialxyz=0.02, lr_init_network=0.001):
        grad_vars_vol = list(self.app_line) + list(self.app_plane)
        grad_vars_net = list(self.basis_mat.parameters()) + list(self.color_net.parameters()) + list(
            self.sigma_net.parameters())
        return grad_vars_vol, grad_vars_net

    def TV_loss_app(self):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + self.reg(self.app_plane[idx]) * 1e-2 + self.reg(self.app_line[idx]) * 1e-3
        return total

    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]],
                                        xyz_sampled[..., self.matMode[2]])).view(3, -1, 1, 2)
        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line),
                                      dim=-1).view(3, -1, 1, 2)

        plane_coef_point, line_coef_point = [], []
        for idx_plane in range(len(self.app_plane)):
            # NOTE: F.grid_sample introduces non-determinism as backward uses cumsum
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                  align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                                 align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        return self.app_activate(self.basis_mat((plane_coef_point * line_coef_point).T))

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, is_train=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            feature_map: [num_rays, 3]. Estimated feature sum of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # NOTE, small difference: here we assume RGBs are from the second element on,
        # contrary to the NeRF class where they are assumed to be the first elements
        rgb = self.rgb_activate(raw[..., 1:])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., :-1, 0]) * raw_noise_std

        # NOTE, small difference: here we assume sigma is first,
        # contrary to the NeRF class where sigma is assumed last
        density = self.sigma_activate(raw[..., :-1, 0] + noise)
        # print(density.shape, raw.shape)
        if not is_train and self.render_rmnearplane > 0:
            mask = z_vals[:, 1:]
            mask = mask > self.render_rmnearplane / 128
            mask = mask.type_as(density)
            density = mask * density

        # print(density.shape, dists.shape)
        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)

        # NOTE: cumprod introduces non-determinism
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)

        return rgb_map, density, acc_map, weights, depth_map  # , sparsity_loss

    def sample(self, pts):
        if pts.numel() > 0:
            xyz_sampled = (pts.reshape(-1, 3) - self.aabb[0]) * self.invaabbSize - 1
            return self.compute_appfeature(xyz_sampled).reshape(pts.shape[0], pts.shape[1], -1)
        else:
            return pts.new_empty([pts.shape[0], pts.shape[1], self.app_dim])

    def forward(self, pts, viewdirs, fts, pts_embed, dirs_embed, z_vals, rays_d, raw_noise_std, is_train):
        input_locs = torch.reshape(pts, [-1, pts.shape[-1]])
        input_locs = pts_embed(input_locs)

        h = torch.cat([fts.view(pts.shape[0] * pts.shape[1], fts.shape[2]), input_locs], -1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        # Always take the output of the sigma layer as feature
        feature_map = h[..., 1:].reshape(*pts.shape[:2], h.shape[-1]-1)

        if self.composite_feature:
            input_dirs = torch.reshape(viewdirs, [-1, viewdirs.shape[-1]])
            input_dirs = dirs_embed(input_dirs)

            h = h.reshape(pts.shape[0], pts.shape[1], -1)
            # Composite the feature map before returning the value
            feature_map, density_map, acc_map, weights, depth_map = self.raw2outputs(
                h, z_vals, rays_d, raw_noise_std, is_train=is_train)

            # color
            h = torch.cat([feature_map, input_dirs], -1)
            for l in range(self.num_layers_color):
                h = self.color_net[l](h)
                if l != self.num_layers_color - 1:
                    h = F.relu(h, inplace=True)

            color = torch.sigmoid(h)
        else:
            input_dirs = viewdirs[:, None].expand(pts.shape)  # 1024, 128, 3
            input_dirs = torch.reshape(input_dirs, [-1, viewdirs.shape[-1]])
            input_dirs = dirs_embed(input_dirs)  # 131072, 27

            sigma = h[..., [0]].reshape(pts.shape[0], pts.shape[1], 1)  # 1024, 128, 1

            # color
            h = torch.cat([h[..., 1:], input_dirs], -1)
            for l in range(self.num_layers_color):
                h = self.color_net[l](h)
                if l != self.num_layers_color - 1:
                    h = F.relu(h, inplace=True)  # 131072, 3

            color = torch.sigmoid(h).reshape(pts.shape[0], pts.shape[1], h.shape[-1])  # 1024, 128, 3

            color, density_map, acc_map, weights, depth_map = self.raw2outputs(
                torch.cat([sigma, color], -1), z_vals, rays_d, raw_noise_std, is_train=is_train)

        return color, depth_map, acc_map, weights, feature_map


class VoxelNeRFRayFeatures(VoxelNeRFBase):

    def __init__(self, *args,
                 n_voxels=16777248,
                 rgb_activate="relu",
                 sigma_activate="relu",
                 extract_feature="after_sigma",
                 composite_feature=True,
                 app_actfn='none',
                 **kwargs):
        super(VoxelNeRFRayFeatures, self).__init__(
            *args,
            n_voxels=n_voxels,
            rgb_activate=rgb_activate,
            sigma_activate=sigma_activate,
            extract_feature=extract_feature,
            composite_feature=composite_feature,
            app_actfn=app_actfn,
            **kwargs
        )


class VoxelNeRFSampleFeatures(VoxelNeRFBase):

    def __init__(self, *args,
                 n_voxels=134217984,
                 rgb_activate='none',
                 sigma_activate="relu",
                 extract_feature="after_sigma",
                 composite_feature=False,
                 app_actfn='none',
                 **kwargs):
        super(VoxelNeRFSampleFeatures, self).__init__(
            *args,
            n_voxels=n_voxels,
            rgb_activate=rgb_activate,
            sigma_activate=sigma_activate,
            extract_feature=extract_feature,
            composite_feature=composite_feature,
            app_actfn=app_actfn,
            **kwargs
        )


class TVLoss(torch.nn.Module):

    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        count_w = max(count_w, 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
