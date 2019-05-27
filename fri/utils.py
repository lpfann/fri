import numpy as np


def distance(u, v):
    """
    Distance measure custom made for feature comparison.

    Parameters
    ----------
    u: first feature
    v: second feature

    Returns
    -------

    """
    u = np.asarray(u)
    v = np.asarray(v)
    # Euclidean differences
    diff = (u - v) ** 2
    # Nullify pairwise contribution
    diff[u == 0] = 0
    diff[v == 0] = 0
    return np.sqrt(np.sum(diff))

