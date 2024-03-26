import numpy as np
import torch
import math
from torch import nn as nn

to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def exponential_scale_fine_loss_weight(N_iters, kernel_start_iter, start_ratio, end_ratio, iter):
    interval_len = N_iters - kernel_start_iter
    scale = (1 / interval_len) * np.log(end_ratio / start_ratio)
    return start_ratio * np.exp(scale * (iter - kernel_start_iter))


def annealing_interpolator(start_value, end_value, end_step, method='linear', start_step=0, **kwargs):
    """
    Returns a function that interpolates a value between start_value and end_value according to the specified method.

    Args:
        start_value (float): The initial value.
        end_value (float): The final value.
        end_step (int): The number of steps at which to end the interpolation.
        method (str): The interpolation method. Valid values are 'linear', and 'cosine'
        start_step (int): The number of steps at which to start the interpolation, assumes constant start_value before.
        **kwargs: Additional arguments required by the specified method.

    Returns:
        A function that takes an integer argument and returns a value interpolated between start_value and end_value.
    """
    if method == 'linear':
        def linear_interpolator(step):
            if step >= end_step:
                return end_value
            elif step < start_step:
                return start_value
            else:
                slope = (end_value - start_value) / (end_step - start_step)
                return start_value + slope * step

        return linear_interpolator
    elif method == 'cosine':
        def cosine_interpolator(step):
            if step >= end_step:
                return end_value
            elif step < start_step:
                return start_value
            else:
                cos_factor = (1 + math.cos(math.pi * (step - start_step) / (end_step - start_step))) / 2
                return start_value * cos_factor + end_value * (1 - cos_factor)

        return cosine_interpolator
    elif method == 'constant':
        return lambda step: start_value
    else:
        raise ValueError('Unsupported method: {}'.format(method))


def is_int_dtype(array):
    return np.issubdtype(array.dtype, np.integer)


def is_float_dtype(array):
    return np.issubdtype(array.dtype, np.floating)


def can_be_int_dtype(array, intdtype=np.int32):
    return is_int_dtype(array) or (is_float_dtype(array) and np.all(intdtype(array) == array))


def smallest_int_dtype(lower, upper, lib=torch):
    assert lib in [np, torch]
    dtypes = [lib.uint8, lib.int8, lib.int16, lib.int32, lib.int64]
    for dtype in dtypes:
        if upper <= lib.iinfo(dtype).max and lower >= lib.iinfo(dtype).min:
            return dtype
    return None


def possibly_smallest_int(array, round=True):
    if can_be_int_dtype(array):
        if round:
            array = np.round(array)
        return array.astype(smallest_int_dtype(array.min(), array.max(), lib=np))
    return array


def torch_randint_vec(mins, maxs, dtype):
    dist = torch.distributions.uniform.Uniform(mins.float(), maxs.float())
    values = dist.sample()
    if not torch.is_floating_point(torch.zeros(1, dtype=dtype, device=mins.device)):  # hacky way of testing dtype
        values = torch.round(values).to(dtype)
    return values


def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def convert_unit(from_unit, to_unit):
    powers = {'s': 0, 'ms': -3, 'us': -6, 'ns': -9}
    return 10 ** (powers[from_unit] - powers[to_unit])


def smallest_larger_then(values, target, equals=False, notfound_val=None):
    assert values.ndim == 1, values.shape
    values = np.sort(values)
    mask_idx = np.nonzero(values >= target if equals else values > target)[0]
    if mask_idx.size > 0:
        return values[mask_idx[0]]
    else:
        return notfound_val

smallest_larger_then_vec = np.vectorize(
    smallest_larger_then,
    excluded={0, 2, 3},
)


def largest_smaller_then(values, target, equals=False, notfound_val=None):
    assert values.ndim == 1, values.shape
    values = np.sort(values)
    mask_idx = np.nonzero(values <= target if equals else values < target)[0]
    if mask_idx.size > 0:
        return values[mask_idx[-1]]
    else:
        return notfound_val

largest_smaller_then_vec = np.vectorize(
    largest_smaller_then,
    excluded={0, 2, 3},
)


def to_flattenvoid(arr):
    """
    Converts a 2D array to a 1D array of voids of the proper size
    """
    assert arr.ndim == 2
    # From: https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array/16973510#16973510
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))


def from_flattenvoid(arr, dtype):
    """
    Converts a void 1D array back to a 2D array of the specified dtype
    """
    assert arr.ndim == 1
    return arr.view(dtype).reshape(arr.shape[0], -1)


def unravel_index(indices, shape):
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.

    This is a `torch` implementation of `numpy.unravel_index`.

    Taken from: https://github.com/francois-rozet/torchist

    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()
    return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]


def seed_everything(seed: int, deterministic=True, warn_only=False):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    np.random.default_rng(seed=seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


def smart_load_state_dict(model: nn.Module, state_dict: dict, network_key="network_state_dict"):
    if "network_fn_state_dict" in state_dict.keys():
        state_dict_fn = {k.lstrip("module."): v for k, v in state_dict["network_fn_state_dict"].items()}
        state_dict_fn = {"mlp_coarse." + k: v for k, v in state_dict_fn.items()}

        state_dict_fine = {k.lstrip("module."): v for k, v in state_dict["network_fine_state_dict"].items()}
        state_dict_fine = {"mlp_fine." + k: v for k, v in state_dict_fine.items()}
        state_dict_fn.update(state_dict_fine)
        state_dict = state_dict_fn
    # elif "network_state_dict" in state_dict.keys():
    # state_dict = {k[7:]: v for k, v in state_dict["network_state_dict"].items()}
    else:
        state_dict = state_dict

    # if isinstance(model, nn.DataParallel):
    # state_dict = {"module." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict[network_key])
