import os
import torch
import numpy as np
import multiprocessing as mp

from torch.utils.data import Dataset

from utils.rays import get_rays_pix
from utils.misc import convert_unit, annealing_interpolator, torch_randint_vec
from utils.events import compute_successor, accumulate_events, can_be_int_dtype, \
    accumulate_events_at_time, load_events_h5, possibly_smallest_int, gather_successor
from utils.data import _is_pure_rotation_matrix, _get_slerp_interpolator, \
    recenter_poses, spherify_poses
from utils.edi import brightness_increment_image, deblur_double_integral


class LLFFEventsDataset(Dataset):

    def __init__(self, args, basedir, H, W, K, factor=8, recenter=True, bd_factor=.75, bd_scale=1.0, closest_bds=0.1,
                 furthest_bds=100.0, spherify=False, recenter_partial=None, spherify_partial=None,
                 events_tms_unit="ns", events_tms_files_unit="us", color_events=False, device="cpu", **kwargs):

        super(LLFFEventsDataset, self).__init__()
        self.args = args
        self.kwargs = kwargs

        # We assume these to be given (and equal to the ones used for the LLFF dataset)
        # In principle they can be different and loaded from the camera intrinsics calibration file
        self.h = H
        self.w = W
        self.K = K

        self.device = device
        self.basedir = basedir
        self.factor = factor
        self.bd_scale = bd_scale
        self.bd_factor = bd_factor

        self.closest_bds = closest_bds
        self.furthest_bds = furthest_bds

        self.recenter = recenter
        self.spherify = spherify
        self.recenter_partial = recenter_partial
        self.spherify_partial = spherify_partial

        self.color_events = color_events

        self.events_tms_unit = events_tms_unit
        self.events_tms_files_unit = events_tms_files_unit

        self.event_accumulate_step_range = self.args.event_accumulate_step_range
        self.event_accumulate_step_range_end = self.args.event_accumulate_step_range_end
        self.event_accumulate_step_end = self.args.event_accumulate_step_end
        self.event_accumulate_step_scheduler = self.args.event_accumulate_step_scheduler

        evdata = self.load_event_data()
        self.integer_coords = evdata["intcoords"]
        self.events_pose_bspl = evdata["events_pose_bspl"]
        self.allknown_poses = torch.tensor(evdata["allknown_poses"], device=self.device)
        self.allknown_poses_timestamps = torch.tensor(evdata["allknown_poses_timestamps"], device=self.device)
        self.images_poses_timestamps = torch.tensor(evdata["images_poses_timestamps"], device=self.device)
        self.images_tms_start = torch.tensor(evdata["images_timestamps_start"], device=self.device)
        self.images_tms_end = torch.tensor(evdata["images_timestamps_end"], device=self.device)

        self.id_to_coords = torch.tensor(evdata["id_to_coords"], device=self.device)
        self.id_to_color_map = torch.tensor(evdata["id_to_color_map"], device=self.device) \
            if evdata["id_to_color_map"] is not None else None
        self.coords_to_id = torch.tensor(evdata["coords_to_id"], device=self.device) \
            if isinstance(evdata["coords_to_id"], np.ndarray) else evdata["coords_to_id"]
        self.events = torch.tensor(evdata["events"], device=self.device)
        self.events_num_successors = torch.tensor(evdata["events_num_successors"], device=self.device)
        self.events_with_successor_idx = torch.tensor(evdata["events_with_successor_idx"], device=self.device)

        self.shared_global_step = mp.Value('i', 0)
        self.event_accum_min_step = annealing_interpolator(self.event_accumulate_step_range[0],
                                                           self.event_accumulate_step_range_end[0],
                                                           self.event_accumulate_step_end,
                                                           self.event_accumulate_step_scheduler)
        self.event_accum_max_step = annealing_interpolator(self.event_accumulate_step_range[1],
                                                           self.event_accumulate_step_range_end[1],
                                                           self.event_accumulate_step_end,
                                                           self.event_accumulate_step_scheduler)

    @property
    def global_step(self):
        return self.shared_global_step.value

    @global_step.setter
    def global_step(self, value):
        self.shared_global_step.value = value

    def global_step_plusplus(self):
        with self.shared_global_step.get_lock():
            global_step = self.shared_global_step.value
            self.shared_global_step.value += 1
        return global_step

    def compute_edi_prior(self, i_images, images, steps, cpos, cneg):
        img_n, img_h, img_w, _ = images.shape
        timestamps_start = self.images_tms_start[i_images]
        timestamps_end = self.images_tms_end[i_images]
        assert bool((timestamps_start < timestamps_end).all()) and bool((timestamps_start > 0).all())

        all_tms = []
        for tms_start, tms_end in zip(timestamps_start, timestamps_end):
            all_tms.append(np.linspace(tms_start, tms_end, steps))
        all_tms = torch.tensor(np.concatenate(all_tms), device=self.device)
        ev_tms = self.events[:, 1]

        idx_events_left = torch.searchsorted(ev_tms, all_tms).reshape(img_n, steps)
        idx_events_right = torch.searchsorted(ev_tms, all_tms, side="right").reshape(img_n, steps)
        images = images.cpu().numpy()

        priors = []
        for i in range(img_n):
            bii_images = []
            for j in range(steps - 1):
                idx_left = idx_events_left[i, j]
                idx_right = idx_events_right[i, j + 1]
                ev = self.events[idx_left:idx_right]
                x, y = self.id_to_coords[ev[:, 0].long()].T.cpu().numpy()
                p = ev[:, 2].cpu().numpy()

                bii = brightness_increment_image(x, y, p, img_w, img_h, cpos, cneg, interpolate=True)  # [H, W]
                bii = bii[..., np.newaxis].repeat(3, axis=-1)  # [H, W, 3]
                bii_images.append(bii)
            bii_images = np.stack(bii_images, axis=0)  # [N-1, H, W, 3]
            priors.append(deblur_double_integral(images[i], bii_images))

        return torch.tensor(priors, device=self.device)

    def interpolate_poses(self, t):
        int_poses, _ = self.events_pose_bspl(t)

        # Changes poses into the correct matrix format
        int_poses = np.concatenate([int_poses[..., 1:2], -int_poses[..., 0:1], int_poses[..., 2:]], -1)
        int_poses = int_poses.astype(np.float32)

        int_poses[..., :3, 3] *= self.bd_scale

        if self.recenter:
            int_poses = recenter_poses(int_poses, c2w=self.recenter_partial)
        if self.spherify:
            bds = np.array([[self.closest_bds, self.furthest_bds]]).repeat(int_poses.shape[0], axis=0)
            int_poses, _, _ = spherify_poses(int_poses, bds, state=self.spherify_partial)

        return int_poses

    def load_event_data(self):
        retvals = dict()

        tms_file_scale = convert_unit(self.events_tms_files_unit, "us")  # We work with us timestamps internally
        tms_arr = np.load(os.path.join(self.basedir, 'images_1/timestamps.npz'))
        img_timestamps = tms_arr["timestamps"] * tms_file_scale
        img_timestamps_start = tms_arr["timestamps_start"] * tms_file_scale
        img_timestamps_end = tms_arr["timestamps_end"] * tms_file_scale

        all_timestamps = np.load(os.path.join(self.basedir, "all_timestamps.npy")).astype(np.float64) * tms_file_scale
        if can_be_int_dtype(all_timestamps) and tms_file_scale == 1:
            all_timestamps = np.load(os.path.join(self.basedir, "all_timestamps.npy"))

        all_timestamps = possibly_smallest_int(all_timestamps)
        retvals["allknown_poses_timestamps"] = all_timestamps
        retvals["images_poses_timestamps"] = img_timestamps
        retvals["images_timestamps_start"] = img_timestamps_start
        retvals["images_timestamps_end"] = img_timestamps_end

        events_path = os.path.join(self.basedir, "events.h5")
        all_poses_bounds = np.load(os.path.join(self.basedir, "all_poses_bounds.npy"))
        all_poses = all_poses_bounds[:, :-2].reshape(-1, 3, 5)[:, :3, :4]
        retvals["allknown_poses"] = all_poses
        assert _is_pure_rotation_matrix(all_poses[:, :3, :3])

        interpolator = _get_slerp_interpolator(
            all_timestamps, all_poses[:, :3, :3], all_poses[:, :3, 3])
        def events_pose_bspl(t):
            # Cannot interpolate beyond the available timestamps
            t = np.clip(t, a_min=all_timestamps.min(), a_max=all_timestamps.max())
            irots, itrans = interpolator(t)
            bottom = np.array([0, 0, 0, 1]).reshape(1, 1, -1).repeat(t.shape[0], axis=0)
            iposes = np.block([[irots, itrans[..., np.newaxis]], [bottom]])
            return iposes, None  # None here replaces the interpolated lookat targets, which we don't need

        retvals["events_pose_bspl"] = events_pose_bspl

        # Read all the events in memory (may take some time and memory)
        events, zero_coord_ids, id_to_coords = load_events_h5(
            events_path, self.h, self.w, coords_decimals=None,
            optimize_ids=True, events_tms_unit=self.events_tms_unit)

        # Remove events outside range of known poses
        events = events[(events[:, -2] >= all_timestamps.min()) & (events[:, -2] <= all_timestamps.max())]
        print(f"Loaded {events.shape[0]} events")

        retvals["intcoords"] = np.all(id_to_coords.astype(np.int32) == id_to_coords)
        if retvals["intcoords"]:
            coords_to_id = np.full([self.h, self.w], fill_value=-1, dtype=np.int32)
            coords_to_id[np.int64(id_to_coords[:, 1]), np.int64(id_to_coords[:, 0])] = np.arange(id_to_coords.shape[0])
        else:
            coords_to_id = {(coord[0], coord[1]): idx for idx, coord in enumerate(id_to_coords)}

        if events[:, -1].min() == 0:
            # Make sure polarity is either -1 or 1
            events[events[:, -1] == 0, -1] = -1
        assert events[:, -1].max() == 1 and events[:, -1].min() == -1

        if self.color_events:
            color_map = np.zeros([self.h, self.w, 3], dtype=np.bool)
            color_map[0::2, 0::2, 0] = True  # r
            color_map[0::2, 1::2, 1] = True  # g
            color_map[1::2, 0::2, 1] = True  # g
            color_map[1::2, 1::2, 2] = True  # b

            if retvals["intcoords"]:  # If int coords, we did not do any rectification
                assert not os.path.exists((os.path.join(self.basedir, "ev_map.npz"))), \
                    "Int coordinates but ev_map.npz fund. Are coordinates rectified?"
                # Convert coordinates to ids
                id_to_color_map = color_map[np.int64(id_to_coords[:, 1]), np.int64(id_to_coords[:, 0])]
            else:
                assert os.path.exists((os.path.join(self.basedir, "ev_map.npz"))), \
                    "Float coordinates but no ev_map.npz fund. Are coordinates not rectified?"
                maps = np.load(os.path.join(self.basedir, "ev_map.npz"))
                invmap_x, invmap_y = maps["inv_mapx"], maps["inv_mapy"]
                assert invmap_x.shape == invmap_y.shape == (self.h, self.w)

                # Compute coordinates mapping
                id_to_color_map = np.zeros([id_to_coords.shape[0], 3], dtype=np.bool)
                for j in range(self.h):
                    for i in range(self.w):
                        if (invmap_x[j, i], invmap_y[j, i]) in coords_to_id:
                            id_to_color_map[coords_to_id[(invmap_x[j, i], invmap_y[j, i])]] = color_map[j, i]
                # Check that every coordinate, that is not a zero events coordinate, has been mapped
                ev_coords_ids_mask = np.ones([id_to_coords.shape[0]], dtype=np.bool)
                ev_coords_ids_mask[zero_coord_ids] = False
                assert (id_to_color_map[ev_coords_ids_mask].sum(axis=-1) == 1).all()
        else:
            id_to_color_map = None
        retvals["id_to_color_map"] = id_to_color_map

        # For each event compute the index of its successor in the same pixel (might take a while)
        events_next_idx, num_successors, first_pix_idx, last_pix_idx = compute_successor(events, flat_xy=True)
        print(f"Finished computing event successor graph")

        retvals["id_to_coords"] = id_to_coords
        retvals["coords_to_id"] = coords_to_id
        # Augment events with their successor and filter events with no successor
        retvals["events"] = np.concatenate([events, events_next_idx.reshape(-1, 1)], axis=-1)
        retvals["events_num_successors"] = num_successors

        if tuple(self.event_accumulate_step_range) != (0, 0):
            min_step = max(self.event_accumulate_step_range[0], self.event_accumulate_step_range_end[0])
            retvals['events_with_successor_idx'] = np.where(retvals["events_num_successors"] > min_step)[0]
        else:
            retvals['events_with_successor_idx'] = np.where(retvals["events_num_successors"] > 0)[0]

        return retvals

    def sample_events(self, events_ids, global_step):
        events_start_batch = self.events[events_ids]  # [nevents, 5]

        min_step = int(self.event_accum_min_step(global_step))
        max_step = int(self.event_accum_max_step(global_step))
        if (min_step, max_step) != (0, 0):
            num_successors = self.events_num_successors[events_ids]
            sampled_hops = torch_randint_vec(
                torch.tensor(min_step, device=self.device) - 1,
                torch.minimum(torch.tensor(max_step, device=self.device), num_successors) - 1 + 1e-5, torch.int64)
            succ_idx, events_neg_pol_cumsum, events_pos_pol_cumsum = gather_successor(
                events_ids, sampled_hops,
                self.events[:, -1].long(),  # successor_map
                self.events[:, -2].int())  # polarity
            events_end_batch = self.events[succ_idx]
        else:
            events_end_batch = self.events[events_start_batch[:, -1].long()]  # [nevents, 5]
            events_pos_mask = events_end_batch[:, -2] > 0
            events_pos_pol_cumsum = torch.where(events_pos_mask, events_end_batch[:, -2], 0)
            events_neg_pol_cumsum = torch.where(~events_pos_mask, events_end_batch[:, -2], 0)

        events_poses_start = torch.tensor(self.interpolate_poses(events_start_batch[:, -3].cpu().numpy()),
                                          device=self.device)
        events_poses_end = torch.tensor(self.interpolate_poses(events_end_batch[:, -3].cpu().numpy()),
                                        device=self.device)
        assert bool(torch.all(events_end_batch[:, 0] == events_start_batch[:, 0]))  # Check pixel ids are the same

        events_coords_ids = events_start_batch[:, 0].long()
        events_coords = self.id_to_coords[events_coords_ids]
        events_color_map = self.id_to_color_map[events_coords_ids] if self.color_events else None

        events_rays_start = torch.stack(get_rays_pix(events_coords, self.K, events_poses_start[:, :3, :4],
                                                     add_halfpix=self.integer_coords), 1)
        events_rays_end = torch.stack(get_rays_pix(events_coords, self.K, events_poses_end[:, :3, :4],
                                                   add_halfpix=self.integer_coords), 1)
        events_rays_start = events_rays_start.permute(0, 2, 1)
        events_rays_end = events_rays_end.permute(0, 2, 1)

        return {
            'events_pos_pol_cumsum': events_pos_pol_cumsum,
            'events_neg_pol_cumsum': events_neg_pol_cumsum,
            'events_rays_start': events_rays_start,
            'events_rays_end': events_rays_end,
            'events_coords_ids': events_coords_ids,
            'events_color_map': events_color_map
        }

    def __len__(self):
        return self.events_with_successor_idx.shape[0]

    def __getitem__(self, events_ids):
        """
        The dataset expects a list of ray ids, which are then used to create a batch of rays. This is different from
        a traditional dataset, where the __getitem__ is supposed to generate a single sample.

        :param events_ids: list of events ids
        :return: a batch of rays (dict)
        """
        global_step = self.global_step_plusplus()  # Atomic global_step++ operation

        if isinstance(events_ids, int):
            events_ids = [events_ids]

        # Convert ids to "absolute" event ids
        events_ids = self.events_with_successor_idx[events_ids]
        events_data = self.sample_events(events_ids, global_step)

        return events_data
