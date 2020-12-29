"""
trig_func_test.py - A PostHocExplainerEvaluation file
Copyright (C) 2020  Zach Carmichael
"""
import numpy as np

from posthoceval.evaluate import UNSUPPORTED_FUNC_REPLACEMENTS


def unsupported_func_replacement_test():
    for s_f, r_f in UNSUPPORTED_FUNC_REPLACEMENTS.items():
        for v in [-np.pi, -2, -np.pi / 2, -1, -np.pi / 4, -.5, -.25,
                  0,
                  .25, .5, np.pi / 4, 1., np.pi / 2, 2, np.pi]:
            try:
                t = float(s_f(v))
            except TypeError:  # complex --> float
                continue
            if np.isinf(t):
                continue
            r = float(r_f(v))
            np.testing.assert_allclose(t, r,
                                       err_msg=f'{t} != {r} ({s_f}, v={v})')
