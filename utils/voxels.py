import numpy as np
import torch
import json

from utils.rays import get_ray_directions, get_rays_from_directions, get_ndc_rays

BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                               device='cuda')


def get_bbox3d_for_blenderobj(camera_transforms, H, W, near=2.0, far=6.0):
    camera_angle_x = float(camera_transforms['camera_angle_x'])
    focal = 0.5*W/np.tan(0.5 * camera_angle_x)

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []

    for frame in camera_transforms["frames"]:
        c2w = torch.FloatTensor(frame["transform_matrix"])
        rays_o, rays_d = get_rays_from_directions(directions, c2w)

        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([1.0, 1.0, 1.0]),
            torch.tensor(max_bound)+torch.tensor([1.0, 1.0, 1.0]))


def get_bbox3d_for_llff(poses, hwf, near=0.0, far=1.0, is_ndc=True):
    H, W, focal = hwf
    H, W = int(H), int(W)

    # ray directions in camera coordinates
    directions = get_ray_directions(H, W, focal)

    min_bound = [100, 100, 100]
    max_bound = [-100, -100, -100]

    points = []
    poses = torch.FloatTensor(poses)
    for pose in poses:
        rays_o, rays_d = get_rays_from_directions(directions, pose)
        if is_ndc:
            rays_o, rays_d = get_ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

        def find_min_max(pt):
            for i in range(3):
                if(min_bound[i] > pt[i]):
                    min_bound[i] = pt[i]
                if(max_bound[i] < pt[i]):
                    max_bound[i] = pt[i]
            return

        for i in [0, W-1, H*W-W, H*W-1]:
            min_point = rays_o[i] + near*rays_d[i]
            max_point = rays_o[i] + far*rays_d[i]
            points += [min_point, max_point]
            find_min_max(min_point)
            find_min_max(max_point)

    return (torch.tensor(min_bound)-torch.tensor([0.01, 0.01, 0.0001]),
            torch.tensor(max_bound)+torch.tensor([0.01, 0.01, 0.0001]))
