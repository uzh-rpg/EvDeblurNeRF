import cv2
import numpy as np

from itertools import product


def interpolate_subpixel(x, y, v, w, h, image=None):
    image = image if image is not None else np.zeros((h, w), dtype=np.float32)

    if x.size == 0:
        return image

    # Implement the equation:
    # V(x,y) = \sum_i{ value * kb(x - xi) * kb(y - yi)}
    # We just consider the 4 integer coordinates around
    # each event coordinate, which will give a nonzero k_b()
    round_fns = (np.floor, np.ceil)

    k_b = lambda a: np.maximum(0, 1 - np.abs(a))
    xy_round_fns = product(round_fns, round_fns)
    for x_round, y_round in xy_round_fns:
        x_ref = x_round(x)
        y_ref = y_round(y)

        # Avoid summing the same contribution multiple times if the
        # pixel or time coordinate is already an integer. In that
        # case both floor and ceil provide the same ref. If it is an
        # integer, we only add it if the case #_round is torch.floor
        # We also remove any out of frame or bin coordinate due to ceil
        valid_ref = np.logical_and.reduce([
            np.logical_or(x_ref != x, x_round is np.floor),
            np.logical_or(y_ref != y, y_round is np.floor),
            x_ref < w, y_ref < h])
        x_ref = x_ref[valid_ref]
        y_ref = y_ref[valid_ref]

        if x_ref.shape[0] > 0:
            val = v[valid_ref] * k_b(x_ref - x[valid_ref]) * k_b(y_ref - y[valid_ref])
            np.add.at(image, (y_ref.astype(np.int64), x_ref.astype(np.int64)), val)

    return image


def brightness_increment_image(x, y, p, w, h, c_pos, c_neg, interpolate=True, color_events=False):
    assert c_pos is not None and c_neg is not None

    image_pos = np.zeros((h, w), dtype=np.float32)
    image_neg = np.zeros((h, w), dtype=np.float32)
    events_vals = np.ones([x.shape[0]], dtype=np.float32)

    pos_events = p > 0
    neg_events = np.logical_not(pos_events)

    if interpolate:
        image_pos = interpolate_subpixel(x[pos_events], y[pos_events], events_vals[pos_events], w, h, image_pos)
        image_neg = interpolate_subpixel(x[neg_events], y[neg_events], events_vals[neg_events], w, h, image_neg)

        if color_events:
            image_neg = cv2.cvtColor(image_neg.astype(np.uint8), cv2.COLOR_BayerBG2BGR)
            image_pos = cv2.cvtColor(image_pos.astype(np.uint8), cv2.COLOR_BayerBG2BGR)
    else:
        np.add.at(image_pos, (y[pos_events].astype(np.int64), x[pos_events].astype(np.int64)), events_vals[pos_events])
        np.add.at(image_neg, (y[neg_events].astype(np.int64), x[neg_events].astype(np.int64)), events_vals[neg_events])

        if color_events:
            image_neg = cv2.cvtColor(image_neg.astype(np.uint8), cv2.COLOR_BayerBG2BGR)
            image_pos = cv2.cvtColor(image_pos.astype(np.uint8), cv2.COLOR_BayerBG2BGR)

    image = image_pos.astype(np.float32) * c_pos - image_neg.astype(np.float32) * c_neg
    return image


def inner_double_integral(bii):
    assert bii.shape[0] % 2 == 0
    N = bii.shape[0] // 2

    images = []
    # Left part of the interval from f-T/2 to f
    for i in range(N):
        images.append(- bii[i:N].sum(axis=0))
    # Frame at f
    images.append(np.zeros_like(images[0]))
    # Right part of the interval from f to f+T/2
    for i in range(N):
        images.append(+ bii[N:N + 1 + i].sum(axis=0))

    images = np.stack(images, axis=0)
    return images


def deblur_double_integral(blurry, bii):
    N = bii.shape[0] // 2
    images = inner_double_integral(bii)
    sharp = (2*N+1) * blurry / np.exp(images).sum(axis=0)
    return sharp


def slowmo_double_integral(sharp, bii):
    images = inner_double_integral(bii)
    slow_imgs = []
    for im in list(images):
        slow_imgs.append(sharp * np.exp(im))

    return slow_imgs
