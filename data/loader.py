import os
import cv2
import torch
import imageio
import numpy as np

from torch.utils.data import Dataset

from utils.misc import unravel_index
from utils.rays import get_rays_pix, HALF_PIX
from utils.voxels import get_bbox3d_for_llff
from utils.data import _minify, normalize, recenter_poses, spherify_poses, poses_avg, \
    render_path_epi, render_path_spiral, _is_pure_rotation_matrix


def endless(iterable):
    while True:
        if iterable is None:
            yield None
        else:
            for x in iterable:
                yield x


class LLFFDataset(Dataset):

    def __init__(self, args, basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_epi=False,
                 device="cpu", pose_transform_allknown=False, **kwargs):
        super(LLFFDataset, self).__init__()
        self.args = args
        self.kwargs = kwargs

        # Standard arguments
        self.device = device
        self.basedir = basedir
        self.factor = factor

        self.recenter = recenter
        self.bd_factor = bd_factor
        self.spherify = spherify
        self.path_epi = path_epi
        self.pose_transform_allknown = pose_transform_allknown

        # Load data
        data = self.load_data()
        self.factor = data["factor"]

        if self.args.llffhold_end:
            i_test = np.arange(data["images"].shape[0])[-args.llffhold:]
            print(f"LLFF holdout, {args.llffhold} from the end")
        else:
            i_test = np.arange(data["images"].shape[0])[::args.llffhold]
            print(f"LLFF holdout, every {args.llffhold}")
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(data["images"].shape[0]))
                            if (i not in i_test and i not in i_val)])
        self.i_train, self.i_val, self.i_test = i_train, i_val, i_test

        self.K = data["K"]
        self.images = torch.tensor(data["images"][i_train], device=device)
        # self.bds = torch.tensor(data["bds"][data_idx], device=device)
        self.poses = torch.tensor(data["poses"][i_train][:, :3, :4], device=device)

        # Holds data for pts0 priors, set externally thorough set_pts0_prior
        self.pts0_images = None
        # Save validation and test data
        self.test_images = torch.tensor(data["images"][i_test], device=device)
        self.test_poses = torch.tensor(data["poses"][i_test][:, :3, :4], device=device)
        self.render_poses = torch.tensor(data["render_poses"][:, :3, :4], device=device)

        # Partial "state" for the recenter and shperify functions. We save this so that we can
        # reapply the same transformation to other poses from the same trajectory (eg., event data poses)
        self.scale = data["scale"]
        self.recenter_partial = data["recenter_partial"]
        self.spherify_partial = data["spherify_partial"]
        self.closest_bds = float(np.min(data["bds"]))
        self.furthest_bds = float(np.max(data["bds"]))

        self.n_imgs, self.h, self.w = self.images.shape[:3]
        self.n_rays = self.n_imgs * self.h * self.w

        if args.no_ndc:
            self.near = data.get("minbds", np.min(data["bds"])) * 0.9
            self.far = data.get("maxbds", np.max(data["bds"])) * 1.0
        else:
            self.near = 0.
            self.far = 1.

        self.bounding_box = get_bbox3d_for_llff(data["poses"][:, :3, :4], data["poses"][0, :3, -1],
                                                near=0, far=1, is_ndc=not args.no_ndc)

        print('Loaded llff', self.images.shape, self.render_poses.shape,
              (self.h, self.w, self.K[0, -1]), self.basedir, self.bounding_box)
        print('NEAR FAR', self.near, self.far)

        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

    @staticmethod
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    @staticmethod
    def get_imgfolder(sfx, foldername="images"):
        return foldername + sfx

    def set_pts0_prior(self, pts0_images):
        if not isinstance(pts0_images, torch.Tensor):
            pts0_images = torch.tensor(pts0_images, device=self.device)
        self.pts0_images = pts0_images.to(device=self.device)

        assert self.pts0_images.shape[0] == self.images.shape[0]

    def unravel_idx_from_rayid(self, ray_id):
        if isinstance(ray_id, torch.Tensor):
            i, y, x = unravel_index(ray_id, (self.n_imgs, self.h, self.w)).T
        else:
            i, y, x = np.unravel_index(ray_id, (self.n_imgs, self.h, self.w), order='C')
        return i, y, x

    def factor_images(self, factor):
        sfx = ''
        if factor is not None:
            sfx = '_{}'.format(self.factor)
            _minify(self.basedir, factors=[self.factor])
            factor = factor
        else:
            factor = 1
        return sfx, factor

    def load_intrinsics(self, H, W, H_scale, W_scale, focal):
        K = np.array([[focal * W_scale, 0, 0.5 * W * W_scale],
                      [0, focal * H_scale, 0.5 * H * H_scale],
                      [0, 0, 1]])
        return K

    def load_np_features(self, featuresfolder):
        featuresdir = os.path.join(self.basedir, featuresfolder)
        if not os.path.exists(featuresdir):
            print(featuresdir, 'does not exist, returning')
            raise FileNotFoundError(featuresdir)

        featuresfiles = [os.path.join(featuresdir, f) for f in sorted(os.listdir(featuresdir)) if
                         f.endswith('npy')]

        features = [np.load(f) for f in featuresfiles]
        features = np.stack(features, 0)
        return features

    def load_images(self, imgfolder, preload_imgs=True):
        imgdir = os.path.join(self.basedir, imgfolder)
        if not os.path.exists(imgdir):
            print(imgdir, 'does not exist, returning')
            raise FileNotFoundError(imgdir)

        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                    f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

        if preload_imgs:
            imgfiles = [self.imread(f)[..., :3].astype(np.float32) / 255. for f in imgfiles]
            imgfiles = np.stack(imgfiles, 0)

            if self.args.datadownsample > 0:
                imgfiles = np.stack([cv2.resize(
                    img_, None, None, 1 / self.args.datadownsample, 1 / self.args.datadownsample, cv2.INTER_AREA)
                    for img_ in imgfiles], axis=0)

            imgshape = imgfiles[0].shape
        else:
            imgshape = self.imread(imgfiles[0]).shape

        return imgfiles, imgshape

    def load_poses(self, factor, imgshape, bdsmin=None, bd_factor=.75, scale=None, filename="poses_bounds.npy"):
        poses_arr = np.load(os.path.join(self.basedir, filename))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # [N, 3, 5]
        assert _is_pure_rotation_matrix(poses[:, :3, :3])

        bds = poses_arr[:, -2:]  # [2, N]
        poses[:, :2, 4] = np.array(imgshape[:2]).reshape([1, 2])
        poses[:, 2, 4] = poses[:, 2, 4] * 1. / factor

        # Correct rotation matrix ordering and move variable dim to axis 0
        poses = np.concatenate([poses[..., 1:2], -poses[..., 0:1], poses[..., 2:]], -1)
        poses = poses.astype(np.float32)
        bds = bds.astype(np.float32)

        # Rescale if bd_factor is provided
        bdsmin = np.min(bds) if bdsmin is None else bdsmin
        if scale is None:
            sc = 1. if bd_factor is None else 1. / (bdsmin * bd_factor)
        else:
            sc = scale
        poses[:, :3, 3] *= sc
        bds *= sc

        return poses, bds, sc

    def recenter_spherify_poses(self, poses, bds, recenter, spherify,
                                render_focuspoint_scale, render_radius_scale,
                                path_epi=False, recenter_partial=None, spherify_partial=None):
        avg_pose, spherify_state = None, None

        if recenter:
            if recenter_partial is not None:
                poses = recenter_poses(poses, c2w=recenter_partial)
                avg_pose = recenter_partial
            else:
                bck_poses = poses.copy()  # For assertion
                poses, avg_pose = recenter_poses(poses, return_c2w=True)
                assert np.allclose(recenter_poses(bck_poses, c2w=avg_pose), poses)

        # generate render_poses for video generation
        if spherify:
            if spherify_partial is not None:
                poses, render_poses, bds = spherify_poses(poses, bds, state=spherify_partial)
                spherify_state = spherify_partial
            else:
                bck_poses, bck_bds = poses.copy(), bds.copy()  # For assertion
                poses, render_poses, bds, spherify_state = spherify_poses(poses, bds, return_state=True)
                poses_2, render_poses_2, bds_2 = spherify_poses(bck_poses, bck_bds, state=spherify_state)
                assert np.allclose(poses, poses_2) and np.allclose(render_poses, render_poses_2) and np.allclose(bds, bds_2)
        else:
            c2w = poses_avg(poses)
            print('recentered', c2w.shape)
            print(c2w[:3, :4])

            ## Get spiral
            # Get average pose
            up = normalize(poses[:, :3, 1].sum(0))

            # Find a reasonable "focus depth" for this dataset
            close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
            dt = .75
            mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
            focal = mean_dz
            focal = focal * render_focuspoint_scale

            # Get radii for spiral path
            shrink_factor = .8
            zdelta = close_depth * .2
            tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
            rads = np.percentile(np.abs(tt), 90, 0)
            rads[0] *= render_radius_scale
            rads[1] *= render_radius_scale
            c2w_path = c2w
            N_views = 120
            N_rots = 2

            # Generate poses for spiral path
            # rads = [0.7, 0.2, 0.7]
            render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

            if path_epi:
                # zloc = np.percentile(tt, 10, 0)[2]
                rads[0] = rads[0] / 2
                render_poses = render_path_epi(c2w_path, up, rads[0], N_views)

        render_poses = np.array(render_poses).astype(np.float32)
        return poses, render_poses, avg_pose, spherify_state

    def get_pose_transform_data(self, factor, imgshape):
        # This is not optimized, but we use it to be compatible with the "previous" loading method
        # i.e., pose_transform_allknown = False
        filename = "all_poses_bounds.npy" if self.pose_transform_allknown else "poses_bounds.npy"
        poses, bds, scale = self.load_poses(factor, imgshape, bd_factor=self.bd_factor, filename=filename)
        _, _, recenter_partial, spherify_partial = self.recenter_spherify_poses(
            poses, bds, self.recenter, self.spherify,
            self.args.render_focuspoint_scale, self.args.render_radius_scale,
            path_epi=self.path_epi
        )
        return scale, recenter_partial, spherify_partial, np.min(bds), np.max(bds)

    def load_poses_raw(self, factor, imgshape, nimages, scale, load_poses=True):
        retvals = dict()

        poses = None
        if load_poses:
            poses, bds, scale = self.load_poses(factor, imgshape, bd_factor=self.bd_factor, scale=scale)
            retvals["poses"] = poses
            retvals["bds"] = bds
            retvals["scale"] = scale

            assert poses.shape[0] == nimages, \
                'Mismatch between imgs {} and poses {} !!!!'.format(nimages, poses.shape[0])

        print('Loaded image data', (nimages, *imgshape), poses[0, :, -1] if load_poses is not None else "no poses")
        return retvals

    def load_data(self):
        data = dict()
        sfx, factor = self.factor_images(self.factor)
        data["images"], data["imgshape"] = self.load_images(self.get_imgfolder(sfx))
        scale, recenter_partial, spherify_partial, data["minbds"], data["maxbds"] = \
            self.get_pose_transform_data(factor, data["imgshape"])

        poses_data = self.load_poses_raw(factor, data["imgshape"], len(data["images"]), scale)
        assert poses_data["scale"] == scale
        data.update(poses_data)

        data["poses"], data["render_poses"], data["recenter_partial"], data["spherify_partial"] = \
            self.recenter_spherify_poses(
                data["poses"], data["bds"], self.recenter, self.spherify,
                self.args.render_focuspoint_scale, self.args.render_radius_scale,
                recenter_partial=recenter_partial, spherify_partial=spherify_partial, path_epi=self.path_epi,
            )
        assert (data["recenter_partial"], data["spherify_partial"]) == (recenter_partial, spherify_partial)

        data["render_poses"] = data["render_poses"][:, :3, :4]

        H, W, focal = data["poses"][0, :3, -1]
        H_scale, W_scale = data["imgshape"][0] / H, data["imgshape"][1] / W
        data["K"] = self.load_intrinsics(H, W, H_scale, W_scale, focal)
        data["factor"] = factor

        return data

    def __len__(self):
        return self.n_rays

    def __getitem__(self, ray_ids):
        """
        The dataset expects a list of ray ids, which are then used to create a batch of rays. This is different from
        a traditional dataset, where the __getitem__ is supposed to generate a single sample.

        :param ray_ids: list of ray ids
        :return: a batch of rays (dict)
        """
        if isinstance(ray_ids, int):
            ray_ids = [ray_ids]

        # Convert ray id (from 0 to Nimg * H * W) to image id and corresponding (x,y) pixel coordinates
        ray_ids = torch.tensor(ray_ids, device=self.device)
        img_id, ray_y, ray_x = self.unravel_idx_from_rayid(ray_ids)
        poses = self.poses[img_id]
        rgbs = self.images[img_id, ray_y, ray_x]
        rays_o, rays_d = get_rays_pix(torch.stack([ray_x, ray_y], dim=-1), self.K, poses)

        return_dict = {
            'rays': torch.stack([rays_o, rays_d], dim=-2).permute(0, 2, 1),  # [N_rays, 3, 2]
            'rays_x': (ray_x + HALF_PIX).reshape(-1, 1),  # [N_rays, 1]
            'rays_y': (ray_y + HALF_PIX).reshape(-1, 1),  # [N_rays, 1]
            'images_idx': img_id.reshape(-1, 1),  # [N_rays, 1]
            'rgbsf': rgbs.reshape(-1, 3),  # [N_rays, 3]
            'poses': poses.reshape(-1, 3, 4),  # [N_rays, 3, 4]
        }

        if self.pts0_images is not None:
            rgbs0 = self.pts0_images[img_id, ray_y, ray_x]
            return_dict["rgbsf_pts0"] = rgbs0.reshape(-1, 3)

        return return_dict
