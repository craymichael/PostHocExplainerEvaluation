import numpy as np


def mse(true, pred):  # noqa
    return np.mean((true - pred) ** 2)


def rmse(true, pred):
    return np.sqrt(mse(true, pred))
