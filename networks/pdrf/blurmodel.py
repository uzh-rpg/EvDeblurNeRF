import numpy as np
import torch
from torch import nn

from utils.misc import init_linear_weights
from networks.embedding import get_embedder


class BlurModel(nn.Module):
    def __init__(self, num_img, num_pt, kernel_hwindow, kernel_type, view_embed, img_wh=None, random_hwindow=0.25,
                 in_embed=3, random_mode='input', view_embed_cnl=32, spatial_embed=0, depth_embed=0,
                 num_hidden=3, num_wide=64, feat_cnl=15, short_cut=False, pattern_init_radius=0.1,
                 isglobal=False, optim_trans=False, optim_spatialvariant_trans=False, use_pattern_pos=True,
                 poses=None):
        """
        num_img: number of image, used for deciding the view embedding
        poses: the original poses, used for generating new rays, len(poses) == num_img
        num_pt: number of sparse point, we use 5 in the paper
        kernel_hwindow: the size of physically equivalent blur kernel, the sparse points are bounded inside the blur kernel.
                        Can be a very big number

        random_hwindow: in training, we randomly perturb the sparse point to model a smooth manifold
        random_mode: 'input' or 'output', it controls whether the random perturb is added to the input of DSK or output of DSK
        // the above two parameters do not have big impact on the results

        in_embed: embedding for the canonical kernel location
        img_embed: the length of the view embedding
        spatial_embed: embedding for the pixel location of the blur kernel inside an image
        depth_embed: (deprecated) the embedding for the depth of current rays

        num_hidden, num_wide, short_cut: control the structure of the MLP
        pattern_init_radius: the little gain add to the deform location described in Sec. 4.4
        isglobal: control whether the canonical kernel should be shared by all the input views or not, does not have big impact on the results
        optim_trans: whether to optimize the ray origin described in Sec. 4.3
        optim_spatialvariant_trans: whether to optimize the ray origin for each view or each kernel point.
        """
        super().__init__()
        self.num_pt = num_pt
        self.num_img = num_img
        self.short_cut = short_cut
        self.kernel_hwindow = kernel_hwindow
        self.random_hwindow = random_hwindow  # about 1 pix
        self.random_mode = random_mode
        self.kernel_type = kernel_type
        self.isglobal = isglobal
        self.feat_cnl = feat_cnl
        pattern_num = 1 if isglobal else num_img
        assert self.random_mode in ['input', 'output'], f"BlurModel::random_mode {self.random_mode} unrecognized, " \
                                                        f"should be input/output"
        if poses is not None:
            self.register_buffer("poses", poses)
        else:
            self.poses = None
        self.use_pattern_pos = use_pattern_pos
        if use_pattern_pos:
            self.register_parameter("pattern_pos",
                                    nn.Parameter(torch.randn(pattern_num, num_pt, 2)
                                                 .type(torch.float32) * pattern_init_radius, True))
        else:
            in_embed = 0

        self.optim_trans = optim_trans
        self.optim_sv_trans = optim_spatialvariant_trans

        if optim_trans:
            self.register_parameter("pattern_trans",
                                    nn.Parameter(torch.zeros(pattern_num, num_pt, 2)
                                                 .type(torch.float32), True))

        if in_embed > 0:
            self.in_embed_fn, self.in_embed_cnl = get_embedder(in_embed, input_dim=2)
        else:
            self.in_embed_fn, self.in_embed_cnl = None, 0

        self.img_embed = view_embed
        self.img_embed_cnl = view_embed_cnl

        if spatial_embed > 0:
            self.spatial_embed_fn, self.spatial_embed_cnl = get_embedder(spatial_embed, input_dim=2)
        else:
            self.spatial_embed_fn, self.spatial_embed_cnl = None, 0

        if depth_embed > 0:
            self.require_depth = True
            self.depth_embed_fn, self.depth_embed_cnl = get_embedder(depth_embed, input_dim=1)
        else:
            self.require_depth = False
            self.depth_embed_fn, self.depth_embed_cnl = None, 0

        in_cnl = self.in_embed_cnl + self.img_embed_cnl + self.depth_embed_cnl + \
                 self.spatial_embed_cnl
        if self.kernel_type == 'PBE':
            in_cnl += self.feat_cnl

        out_cnl = 1 + 2 + 2 if self.optim_sv_trans else 1 + 2  # u, v, w or u, v, w, dx, dy
        hiddens = [nn.Linear(num_wide, num_wide) if i % 2 == 0 else nn.ReLU()
                   for i in range((num_hidden - 1) * 2)]
        self.linears = nn.Sequential(
            nn.Linear(in_cnl, num_wide), nn.ReLU(),
            *hiddens,
        )
        self.linears1 = nn.Sequential(
            nn.Linear((num_wide + in_cnl) if short_cut else num_wide, num_wide), nn.ReLU(),
            nn.Linear(num_wide, out_cnl)
        )
        self.linears.apply(init_linear_weights)
        self.linears1.apply(init_linear_weights)

    def forward(self, H, W, K, rays, rays_info, feats=None,
                return_img_embed=False, **kwargs):
        """
        inputs: all input has shape (ray_num, cnl)
        outputs: output shape (ray_num, ptnum, 3, 2)  last two dim: [ray_o, ray_d]
        """

        device = rays_info["rays_x"].device
        img_idx = rays_info['images_idx'].squeeze(-1)
        img_embed = self.img_embed(img_idx)
        img_embed_expand = img_embed[:, None].expand(len(img_embed), self.num_pt, self.img_embed_cnl)

        if self.use_pattern_pos:
            pt_pos = self.pattern_pos.expand(len(img_idx), -1, -1) if self.isglobal \
                else self.pattern_pos[img_idx]
            pt_pos = torch.tanh(pt_pos) * self.kernel_hwindow

            if self.random_hwindow > 0 and self.random_mode == "input":
                random_pos = torch.randn_like(pt_pos) * self.random_hwindow
                pt_pos = pt_pos + random_pos

            input_pos = pt_pos  # the first point is the reference point
            if self.in_embed_fn is not None:
                pt_pos = pt_pos * (np.pi / self.kernel_hwindow)
                pt_pos = self.in_embed_fn(pt_pos)
        else:
            input_pos = 0  # dummy variable to be added later to the predicted deltas
            pt_pos = torch.empty([len(img_idx), self.num_pt, 0])

        if self.kernel_type == 'DSK':
            x = torch.cat([pt_pos, img_embed_expand], dim=-1)
        else:
            if feats == None:
                x = torch.cat([pt_pos, img_embed_expand,
                               torch.zeros(len(img_embed), self.num_pt, self.feat_cnl)], dim=-1)
            else:
                x = torch.cat([pt_pos, img_embed_expand,
                               feats.reshape(len(img_embed), self.num_pt, -1)], dim=-1)

        rays_x, rays_y = rays_info['rays_x'], rays_info['rays_y']
        if self.spatial_embed_fn is not None:
            spatialx = rays_x / (W / 2 / np.pi) - np.pi
            spatialy = rays_y / (H / 2 / np.pi) - np.pi  # scale 2pi to match the freq in the embedder
            spatial_save = torch.cat([spatialx, spatialy], dim=-1)
            spatial = self.spatial_embed_fn(spatial_save)
            spatial = spatial[:, None].expand(len(img_idx), self.num_pt, self.spatial_embed_cnl)
            x = torch.cat([x, spatial], dim=-1)

        if self.depth_embed_fn is not None:
            depth = rays_info['ray_depth']
            depth = depth * np.pi
            depth = self.depth_embed_fn(depth)
            depth = depth[:, None].expand(len(img_idx), self.num_pt, self.depth_embed_cnl)
            x = torch.cat([x, depth], dim=-1)

        # forward
        x1 = self.linears(x)
        x1 = torch.cat([x, x1], dim=-1) if self.short_cut else x1
        x1 = self.linears1(x1)

        delta_trans = None
        if self.optim_sv_trans:
            delta_trans, delta_pos, weight = torch.split(x1, [2, 2, 1], dim=-1)
        else:
            delta_pos, weight = torch.split(x1, [2, 1], dim=-1)

        if self.optim_trans:
            delta_trans = self.pattern_trans.expand(len(img_idx), -1, -1) if self.isglobal \
                else self.pattern_trans[img_idx]

        if delta_trans is None:
            delta_trans = torch.zeros_like(delta_pos)

        delta_trans = delta_trans * 0.01
        if not self.use_pattern_pos:
            delta_pos = torch.tanh(delta_pos) * self.kernel_hwindow

        new_rays_xy = delta_pos + input_pos
        if self.kernel_type == 'PBE':
            new_rays_xy[:, 0, :] = 0
            delta_trans[:, 0, :] = 0
            align = None
        else:
            align = new_rays_xy[:, 0, :].abs().mean()
            align += (delta_trans[:, 0, :].abs().mean() * 10)
        weight = torch.softmax(weight[..., 0], dim=-1)

        if self.random_hwindow > 0 and self.random_mode == 'output':
            raise NotImplementedError(f"{self.random_mode} for self.random_mode is not implemented")

        poses = rays_info["poses"] if self.poses is None else self.poses[img_idx]
        # get rays from offsetted pt position
        rays_x = (rays_x - K[0, 2] + new_rays_xy[..., 0]) / K[0, 0]
        rays_y = -(rays_y - K[1, 2] + new_rays_xy[..., 1]) / K[1, 1]
        dirs = torch.stack([rays_x - delta_trans[..., 0],
                            rays_y - delta_trans[..., 1],
                            -torch.ones_like(rays_x)], -1)

        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., None, :] * poses[..., None, :3, :3],
                           -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        translation = torch.stack([
            delta_trans[..., 0],
            delta_trans[..., 1],
            torch.zeros_like(rays_x),
            torch.ones_like(rays_x)
        ], dim=-1)
        rays_o = torch.sum(translation[..., None, :] * poses[:, None], dim=-1)

        extras = {}
        if return_img_embed:
            extras["img_embed"] = img_embed

        return torch.stack([rays_o, rays_d], dim=-1), weight, align, extras
