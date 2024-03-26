import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rigid_warping import SE3Field


class RigidBlurringModel(nn.Module):
    def __init__(self, W, D_r, W_r, D_v, W_v, D_w, W_w,
                 output_ch_r, output_ch_v, feat_ch, rv_window,
                 view_embed, num_motion=2, use_origin=True, use_view_embed=True):

        super(RigidBlurringModel, self).__init__()

        self.view_embed_module = view_embed
        self.use_view_embed = use_view_embed
        if self.use_view_embed is False:
            W = 0  # No view embedding

        self.warp_field = SE3Field()

        self.num_motion = num_motion
        # Add naive support for PDRF + RBK
        self.feat_ch = feat_ch * ((num_motion + 1) if use_origin else num_motion)

        self.use_origin = use_origin
        self.num_motion = num_motion

        self.output_ch_r = output_ch_r * num_motion
        self.output_ch_v = output_ch_v * num_motion
        self.output_ch_w = num_motion
        self.rv_window = rv_window

        self.r_branch = nn.ModuleList([nn.Linear(W+self.feat_ch, W_r)] +
                                      [nn.Linear(W_r, W_r) for i in range(D_r - 1)])
        self.r_linear = nn.Linear(W_r, self.output_ch_r)
        r_gain = 0.00001 / (math.sqrt((W_r + self.output_ch_r) / 6))  # for Uniform(-1.0e-5, 1.0e-5)
        torch.nn.init.xavier_uniform_(self.r_linear.weight, gain=r_gain)  # -1e-5, 1e-5

        self.v_branch = nn.ModuleList([nn.Linear(W+self.feat_ch, W_v)] +
                                      [nn.Linear(W_v, W_v) for i in range(D_v - 1)])
        self.v_linear = nn.Linear(W_v, self.output_ch_v)
        v_gain = 0.00001 / (math.sqrt((W_v + self.output_ch_v) / 6))  # for Uniform(-1.0e-5, 1.0e-5)
        torch.nn.init.xavier_uniform_(self.v_linear.weight, gain=v_gain)  # -1e-5, 1e-5

        self.w_branch = nn.ModuleList([nn.Linear(W+self.feat_ch, W_w)] +
                                      [nn.Linear(W_w, W_w) for i in range(D_w - 1)])
        self.w_linear = nn.Linear(W_w, self.output_ch_w + 1)

    def rbk_warp(self, rays, r, v, return_transform=False):
        r = r.reshape(r.shape[0], 3, self.num_motion)
        v = v.reshape(v.shape[0], 3, self.num_motion)
        rays_o = rays[..., 0]
        rays_d = rays[..., 1]
        pts_rays_end = rays_o + rays_d

        if self.use_origin:
            new_rays = torch.cat([rays_o[..., None], rays_d[..., None]], dim=-1).unsqueeze(1) \
                .repeat(1, self.num_motion + 1, 1, 1)
            new_transform = torch.eye(4)[None, None, :, :].repeat(r.shape[0], self.num_motion + 1, 1, 1) \
                            if return_transform else None
        else:
            new_rays = torch.zeros_like(rays.unsqueeze(1).repeat(1, self.num_motion, 1, 1))
            new_transform = torch.eye(4)[None, None, :, :].repeat(r.shape[0], self.num_motion, 1, 1)\
                            if return_transform else None

        for i in range(self.num_motion):
            warp_transform = self.warp_field.get_transform(rot=r[:, :, i], trans=v[:, :, i])
            warped_rays_o = self.warp_field.warp(rays_o, transform=warp_transform)
            warped_pts_end = self.warp_field.warp(pts_rays_end, transform=warp_transform)
            warped_rays_d = warped_pts_end - warped_rays_o
            warped_rays = torch.cat([warped_rays_o[..., None], warped_rays_d[..., None]], dim=-1)

            idx = i + 1 if self.use_origin else i
            new_rays[:, idx] = warped_rays
            if return_transform:
                new_transform[:, idx] = warp_transform

        if return_transform:
            return new_rays, new_transform
        return new_rays

    def rbk_warp_pose(self, poses, r, v, return_transform=False):
        r = r.reshape(r.shape[0], 3, self.num_motion)
        v = v.reshape(v.shape[0], 3, self.num_motion)
        # convert poses to homogeneous poses, adding a row [0, 0, 0, 1] to the bottom of each of them
        poses = torch.cat([poses, torch.eye(4)[None, -1:].repeat(poses.shape[0], 1, 1)], dim=1)

        if self.use_origin:
            new_poses = poses.unsqueeze(1).repeat(1, self.num_motion + 1, 1, 1)
            new_transform = torch.eye(4)[None, None, :, :].repeat(poses.shape[0], self.num_motion + 1, 1, 1) \
                            if return_transform else None
        else:
            new_poses = torch.zeros_like(poses.unsqueeze(1).repeat(1, self.num_motion, 1, 1))
            new_transform = torch.eye(4)[None, None, :, :].repeat(poses.shape[0], self.num_motion, 1, 1) \
                            if return_transform else None

        for i in range(self.num_motion):
            warp_transform = self.warp_field.get_transform(rot=r[:, :, i], trans=v[:, :, i])
            warped_poses = self.warp_field.warp_pose(poses, transform=warp_transform)

            idx = i + 1 if self.use_origin else i
            new_poses[:, idx] = warped_poses
            if return_transform:
                new_transform[:, idx] = warp_transform

        if return_transform:
            return new_poses, new_transform
        return new_poses

    def rbk_weighted_sum(self, rgb, depth, acc, extras, ccw):
        num_motion = self.num_motion + 1 if self.use_origin else self.num_motion
        rgb = torch.sum((rgb.reshape(-1, num_motion, rgb.shape[-1]) * ccw[..., None]), dim=1)
        depth = torch.sum(depth.reshape(-1, num_motion) * ccw, dim=1)
        acc = torch.sum(acc.reshape(-1, num_motion) * ccw, dim=1)

        for k, v in extras.items():
            if len(v.shape) == 1:
                v = torch.sum(v.reshape(-1, num_motion) * ccw, dim=1)
            if len(v.shape) == 2:
                v = torch.sum((v.reshape(-1, num_motion, v.shape[-1]) * ccw[..., None]), dim=1)
            if len(v.shape) == 3:
                v = torch.sum((v.reshape(-1, num_motion, v.shape[-2], v.shape[-1]) * ccw[..., None][..., None]), dim=1)
            extras[k] = v

        return rgb, depth, acc, extras

    def forward(self, H, W, K, rays, rays_info, feats=None, return_img_embed=False, **kwargs):
        num_rays = rays_info['images_idx'].shape[0]
        device = rays_info['images_idx'].device

        view_feature = self.view_embed_module(rays_info['images_idx'].squeeze(-1))
        if self.use_view_embed:
            h = view_feature
        else:
            h = torch.empty([num_rays, 0], device=device, dtype=torch.float32)

        if feats is None:
            feats = torch.zeros(num_rays, self.feat_ch)
        else:
            # Naive integration. We are given features for each ray and we concatenate all of them
            # to get a feature for the network. This is necessary since Deblur-NeRF-like networks
            # predict motino params separately, but here we predict them all at once.
            feats = feats.view(num_rays, self.feat_ch)
        h_branch = torch.cat([h, feats], axis=-1)

        for i, _ in enumerate(self.r_branch):
            h = self.r_branch[i](h_branch)
            h = F.relu(h)

        for i, _ in enumerate(self.v_branch):
            h_v = self.v_branch[i](h_branch)
            h_v = F.relu(h_v)

        for i, _ in enumerate(self.w_branch):
            h_w = self.w_branch[i](h_branch)
            h_w = F.relu(h_w)

        r = self.r_linear(h) * self.rv_window
        v = self.v_linear(h_v) * self.rv_window

        weight = torch.sigmoid(self.w_linear(h_w))
        weight = weight / (torch.sum(weight, dim=-1, keepdims=True) + 1e-10)

        new_rays = self.rbk_warp(rays.float(), r, v)

        align = None
        extras = {}
        if return_img_embed:
            extras["img_embed"] = view_feature

        return new_rays, weight, align, extras


