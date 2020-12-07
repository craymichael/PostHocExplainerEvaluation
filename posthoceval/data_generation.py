import numpy as np

import sympy as S
from sympy import stats

U = stats.Uniform('U', a, b)
a = np.fromiter(
    S.stats.sample_iter(-U, ii.contains(U), numsamples=10000), dtype=np.float32
)
