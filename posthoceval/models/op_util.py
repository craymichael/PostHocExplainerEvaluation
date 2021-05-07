"""
op_util.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from collections.abc import Iterable
from itertools import chain

from math import ceil
from math import floor

import numpy as np

from posthoceval.expl_utils import standardize_effect


def pair(val):
    if not isinstance(val, str) and isinstance(val, Iterable):
        ret = tuple(val)
        if len(ret) != 2:
            raise ValueError(f'{ret} (derived from {val}) is not a '
                             f'length-2 pair!')
        return ret
    else:
        return val, val


def _unsupported_tf_arg(arg, default, provided):
    if default != provided:
        raise NotImplementedError(
            f'{arg}={provided} not yet supported! Please keep the default of '
            f'{default}'
        )


def tuplize(npy_data):
    return tuple(
        chain.from_iterable(x if isinstance(x, tuple) else (x,)
                            for x in npy_data.flat)
    )


def symbolic_conv2d(
        symbols: np.ndarray,
        filters: int,
        kernel_size,
        strides=(1, 1),
        padding='valid',
        data_format=None,
        dilation_rate=(1, 1),
        groups=1,
):
    if data_format is None:
        from tensorflow.keras.backend import image_data_format
        data_format = image_data_format()
    kernel_size = pair(kernel_size)
    strides = pair(strides)
    dilation_rate = pair(dilation_rate)
    _unsupported_tf_arg('data_format', 'channels_last', data_format)
    _unsupported_tf_arg('dilation_rate', (1, 1), dilation_rate)
    _unsupported_tf_arg('groups', 1, groups)

    assert symbols.ndim == 3, 'h x w x c'
    h, w, c = symbols.shape

    padding = padding.lower()
    if padding == 'valid':
        hpad = wpad = (0, 0)
        ho = ceil((h - kernel_size[0] + 1) / strides[0])
        wo = ceil((w - kernel_size[1] + 1) / strides[1])
    else:
        assert padding == 'same'
        ho = ceil(h / strides[0])
        wo = ceil(w / strides[1])

        hpad = max((ho - 1) * strides[0] + kernel_size[0] - h, 0) / 2
        wpad = max((wo - 1) * strides[1] + kernel_size[1] - w, 0) / 2
        hpad, wpad = (floor(hpad), ceil(hpad)), (floor(wpad), ceil(wpad))

    # output effects
    out = np.zeros((ho, wo, filters), dtype=object)
    for hoi, hi in enumerate(range(-hpad[0], h + hpad[1], strides[0])):
        hj = hi + kernel_size[0]
        hi, hj = min(max(hi, 0), h), min(max(hj, 0), h)
        if hi == hj or hoi >= ho:
            continue
        for woi, wi in enumerate(range(-wpad[0], w + wpad[1], strides[1])):
            wj = wi + kernel_size[1]
            wi, wj = min(max(wi, 0), w), min(max(wj, 0), w)
            if wi == wj or woi >= wo:
                continue
            out_i = standardize_effect(tuplize(symbols[hi:hj, wi:wj, :]))
            for fi in range(filters):
                out[hoi, woi, fi] = out_i
    return out
