import numpy as np


def horizontal_flip(x):
    return np.flip(x, axis=1)


def vertical_flip(x):
    return np.flip(x, axis=0)
