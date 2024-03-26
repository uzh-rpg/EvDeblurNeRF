import cv2
import h5py
import torch
import numpy as np
from numba import njit
from typing import Union, Tuple

from utils.misc import convert_unit, can_be_int_dtype, possibly_smallest_int, to_flattenvoid


def load_events_h5(events_path, h, w, coords_decimals=None, optimize_ids=False, events_tms_unit='ns'):
    """
    Load events from h5py file. Events can have either full integer coordinates (x,y) or floating point coordinates
        in case of rectifications or other processing. This function allows to convert coordinates into 1D flat
        coordinates where each (x,y) is replaced by a unique id to make float coords compatible with grid-based
        processing (e.g., successor computation) or make computation more memory efficient (empty coordinates are
        not assigned an id). In case optimize_ids=True, the function also optimizes the ids so that the minimum number
        of coordinate ids are used

    :param events_path: path of HDF5 events
    :param h: height of the image
    :param w: width of the image
    :param coords_decimals: if not None, rounds float coordinates to this number of decimal values
    :param optimize_ids: is True, only the ids of the occupied coordiantes are generated. This is done by default for
        float coordinates.  An additional [Ncoords, 2] array is returned, which allows to retrieve back the original
        coordinates given the ids
    :param events_tms_unit: the unit of the timestamps in the events file (ms, us, ns, etc.)
    :return: (events: np.ndarray, noevents: np.ndarray, id_to_coords: np.ndarray). The events and a mapping such that
        id_to_coords[events[:,0]] return the original coordinates.
    """

    tms_file_scale = convert_unit(events_tms_unit, "us")  # We work with usec timestamps internally
    events = h5py.File(events_path, "r")
    events = {k: events[k][:] for k in "xytp"}  # Reads from disk. Might be slow and memory expensive
    events["x"] = events["x"].astype(np.float32)
    events["y"] = events["y"].astype(np.float32)
    events["t"] = possibly_smallest_int(events["t"] * tms_file_scale)

    zero_pixels = np.ones((h, w), dtype=np.uint8)
    zero_pixels[np.clip(np.round(events["y"]).astype(np.int32), 0, h - 1),
                np.clip(np.round(events["x"]).astype(np.int32), 0, w - 1)] = False
    zeroev_coords = np.stack(np.where(zero_pixels), axis=-1)[:, ::-1]  # ::-1 to convert yx to xy coords

    float_coords = not can_be_int_dtype(events["x"]) or not can_be_int_dtype(events["y"])
    if float_coords and coords_decimals is not None:
        events["x"] = np.around(events["x"], decimals=coords_decimals)
        events["y"] = np.around(events["y"], decimals=coords_decimals)
    ev_coords = np.stack([events["x"], events["y"]], axis=-1)

    # Concatenates all regular and zero events to extract global unique coords
    num_ev, num_noev = ev_coords.shape[0], zeroev_coords.shape[0]
    all_ev_coords = np.concatenate([ev_coords, zeroev_coords], axis=0)

    # We compute unique coords also in case of float_coords since we need them to compute zero_coords
    if optimize_ids or float_coords:
        ev_coords_void = to_flattenvoid(all_ev_coords)
        _, idx, inv_idx = np.unique(ev_coords_void, return_index=True, return_inverse=True)
        id_to_coords = all_ev_coords[idx]
        all_ev_coord_ids = inv_idx
    else:
        # Coords are integers so we can directly convert them to ids
        assert can_be_int_dtype(all_ev_coords)
        id_to_coords = np.arange(h * w)
        all_ev_coord_ids = all_ev_coords[:, 1] * w + all_ev_coords[:, 0]

    ev_coords_ids, noev_coords_ids = all_ev_coord_ids[:num_ev], all_ev_coord_ids[num_ev:]

    events = np.stack([ev_coords_ids, events["t"], events["p"]], axis=-1)
    return events, noev_coords_ids, id_to_coords


@njit
def compute_successor(events: np.ndarray, flat_xy: bool = False) -> [np.ndarray, np.ndarray]:
    """
    Augments the event stream by adding to each event the index, in the original event stream,
    of the next event in the sequence in that same pixel location.

    :param events: array of shape [N, >=1] containing in the first position either (x,y) values or
        flattened xy values (e.g., y * w + x)
    :param flat_xy: if true, the first column of events is assumed to be a flattened xy value
    :param h: the height of the sensor size
    :param w: the width of the sensor size
    :param return_boundary_idx: if trflat_xyue, returns two additional arrays specifying the index of the first and last
            event in each active pixel
    :return: successor_idx: the indexes as a [N] array. In case there is no successor for the current
        event, the index of the event itself is provided instead
    """

    num_events = events.shape[0]
    h, w = (1, int(events[:, 0].max()+1)) if flat_xy \
        else (int(events[:, 1].max()+1), int(events[:, 0].max()+1))
    # Array keeping track of the most recent event's index in each pixel when
    # iterating from last to first event (at the end will contain the first event)
    latest_seen_idx = np.full((h, w), fill_value=-1, dtype=np.int64)
    # Array keeping track the index of the last event in each pixel
    first_seen_idx = np.full((h, w), fill_value=-1, dtype=np.int64)
    # Array containing the results
    successor_idx = np.full(num_events, fill_value=-1, dtype=np.int64)
    num_successors = np.zeros(num_events, dtype=np.int32)

    # Loops through the array in reverse
    for i in range(num_events-1, -1, -1):
        x = int(events[i, 0])
        y = 0 if flat_xy else int(events[i, 1])

        # If this is not the first seen (i.e., temporally last) event in this pixel
        if latest_seen_idx[y, x] != -1:
            successor_idx[i] = latest_seen_idx[y, x]
            num_successors[i] = num_successors[successor_idx[i]] + 1
        else:
            successor_idx[i] = i  # No successor, so the successor is the event itself
            num_successors[i] = 0

        # Update the latest seen event in this pixel
        latest_seen_idx[y, x] = i
        # Update the tensor only if it is the first seen event
        if first_seen_idx[y, x] == -1:
            first_seen_idx[y, x] = i

    return successor_idx, num_successors, latest_seen_idx, first_seen_idx


@njit
def accumulate_events(events: np.ndarray, n: int, flat_xy: bool = False) -> np.ndarray:
    """
    Group consecutive events into a single event by aggregating their polarity

    :param events: array of shape [N, >=1] containing in the first position either (x,y) values or
        flattened xy values (e.g., y * w + x)
    :param n: number of events to aggregate
    :param flat_xy: if true, the first column of events is assumed to be a flattened xy value
    :return: a new condensed event array of shape [N', 4] containing (x,y,t,p) values
    """
    num_events = events.shape[0]
    h, w = (1, int(events[:, 0].max() + 1)) if flat_xy \
        else (int(events[:, 1].max() + 1), int(events[:, 0].max() + 1))
    num_coords = 1 if flat_xy else 2

    num_events_out = 0
    running_seen = np.full((h, w), fill_value=-1, dtype=np.int32)
    running_pol = np.zeros((h, w), dtype=events.dtype)
    out_events = np.empty((num_events, 3 if flat_xy else 4), dtype=events.dtype)

    for i in range(num_events):
        x_, y_, t, p = (events[i, 0], 0, *events[i, 1:].tolist()) if flat_xy else events[i]
        x, y = int(x_), int(y_)
        xy_seen = running_seen[y, x]

        # Always generate the first event
        if xy_seen == -1:
            running_seen[y, x] = n-1

        # Time to generate an events
        if running_seen[y, x] == n-1:
            # Accumulate the current events
            running_pol[y, x] += p
            # Save the accumulated event
            out_events[num_events_out, 0] = x_
            if not flat_xy:
                out_events[num_events_out, 1] = y_
            out_events[num_events_out, num_coords+0] = t
            out_events[num_events_out, num_coords+1] = running_pol[y, x]
            # Reset running variables
            running_pol[y, x] = 0
            running_seen[y, x] = 0
            num_events_out += 1
        else:
            running_pol[y, x] += p
            running_seen[y, x] += 1

    return out_events[:num_events_out]


def accumulate_events_at_time(events: np.ndarray, timestamps: np.ndarray, n: int, flat_xy: bool = False,
                              return_zeroevents: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Group consecutive events at specified timestamps, meaning that events can only be triggered at the specified
    polarities. Events carry the aggregated polarity of all the events happened in between provided timestamps

    :param events: array of shape [N, >=1] containing in the first position either (x,y) values or
        flattened xy values (e.g., y * w + x)
    :param timestamps: array of shape [T] containing accumulation timestamps
    :param n: number of events to aggregate
    :param flat_xy: if true, the first column of events is assumed to be a flattened xy value
    :param return_zeroevents: if true, returns also the zero events
    :return: a new condensed event array of shape [N', 4] containing (x,y,t,p) values
    """
    sampled_timestamps = timestamps[::n+1]
    idx_timestamps = np.searchsorted(events[:, -2], sampled_timestamps-1e-6)
    h, w = (1, int(events[:, 0].max() + 1)) if flat_xy \
        else (int(events[:, 1].max() + 1), int(events[:, 0].max() + 1))
    num_coords = 1 if flat_xy else 2

    out_events, out_zeroevents = [], []
    for idx_start, idx_end, tms_start, tms_end in zip(idx_timestamps[:-1], idx_timestamps[1:],
                                                      sampled_timestamps[:-1], sampled_timestamps[1:]):
        assert (bool(np.all(events[idx_start:idx_end, -2] >= tms_start)) and
                bool(np.all(events[idx_start:idx_end, -2] < tms_end)))
        accum_pols = np.zeros([h, w], np.int32)
        np.add.at(accum_pols,
                  (0 if flat_xy else events[idx_start:idx_end, 1], events[idx_start:idx_end, 0]),
                  events[idx_start:idx_end, num_coords+1])
        nnz_y, nnz_x = np.nonzero(accum_pols)
        zero_y, zero_x = np.nonzero(accum_pols == 0)
        out_events.append(
            np.stack([nnz_x, nnz_y, [tms_end] * len(nnz_x), accum_pols[nnz_y, nnz_x]], axis=-1))
        out_zeroevents.append(
            np.stack([zero_x, zero_y, [tms_start] * len(zero_x), [tms_end] * len(zero_x)], axis=-1))

    out_events = np.concatenate(out_events, axis=0).astype(events.dtype)
    out_zeroevents = np.concatenate(out_zeroevents, axis=0).astype(events.dtype)
    if flat_xy:
        out_events = out_events[:, [0, 2, 3]]  # Remove redundant all-zero y column
        out_zeroevents = out_zeroevents[:, [0, 2, 3]]  # Remove redundant all-zero y column

    if return_zeroevents:
        return out_events, out_zeroevents
    return out_events


@torch.jit.script
def gather_successor(query_idx, query_hops, successor_map, polarities):
    """
    Follows the successor map connectivity map to gather the successor of a set of events given their indices and
    the number of hops to go forward (possibly different for each event). If the successor is not found, a -1 is
    returned in the output successor idx, and 0 pos/neg polarities are returned.

    :param query_idx: indices of the events to gather the successor of
    :param query_hops: number of hops to go forward
    :param successor_map: map of successor indices
    :param polarities: polarities of the events
    :return: a tuple containing the successor indices and polarities
    """

    device = query_idx.device
    max_hops = int(query_hops.max())
    invalid_successors = torch.zeros(query_idx.shape[0], dtype=torch.bool, device=device)
    out_pos_polarities = torch.zeros_like(query_idx, dtype=polarities.dtype, device=device)
    out_neg_polarities = torch.zeros_like(query_idx, dtype=polarities.dtype, device=device)
    out_successor_idx = query_idx.clone()

    for h in range(max_hops + 1):
        not_finished = h <= query_hops
        new_successors = successor_map[out_successor_idx[not_finished]]
        new_polarities = polarities[new_successors]
        invalid_successors[not_finished] += (new_successors < 0) + (new_successors >= successor_map.shape[0])

        out_successor_idx[not_finished] = new_successors
        out_pos_polarities[not_finished] += torch.where(new_polarities > 0, new_polarities, 0)
        out_neg_polarities[not_finished] += torch.where(new_polarities < 0, new_polarities, 0)

    # Overrides invalid successors
    out_successor_idx[invalid_successors] = -1
    out_neg_polarities[invalid_successors] = 0
    out_pos_polarities[invalid_successors] = 0

    return out_successor_idx, out_neg_polarities, out_pos_polarities


def egm_loss(luma_start, luma_end, bii, color_mask=None, color_weight=None, log_eps=1e-5):
    if color_mask is not None:
        assert luma_start.shape[-1] == luma_end.shape[-1] == color_mask.shape[-1] == 3 and color_mask.ndim == 2
        assert torch.all(color_mask.sum(-1) == 1), "Color mask must be one-hot"
    if color_weight is not None and not isinstance(color_weight, torch.Tensor):
        color_weight = torch.tensor(color_weight, dtype=torch.float32, device=luma_start.device)

    assert luma_start.ndim == 2 and luma_start.shape == luma_end.shape and luma_start.shape[0] == bii.shape[0]

    log_prev = torch.log(luma_start + log_eps)
    log_post = torch.log(luma_end + log_eps)
    pred_bii = (log_post - log_prev).squeeze(-1)

    if color_mask is not None:
        pred_bii = pred_bii[color_mask]  # Select only the correct channel
        if color_weight is not None:
            color_idx = torch.where(color_mask)[1]
            assert color_idx.shape[0] == color_mask.shape[0], "There is more than one True color mask per ray"
            color_weight = color_weight[color_idx]

    if color_weight is None:
        color_weight = torch.ones(pred_bii.shape[0], dtype=torch.float32, device=pred_bii.device)

    assert pred_bii.ndim == bii.ndim == 1
    return (((pred_bii - bii) ** 2) * color_weight).sum() / color_weight.sum()
