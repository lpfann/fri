import numpy as np


def similarity(u, v):
    """
    Similarity measure custom made for feature comparison.
    We define similarity as the ability to replace another feature in the same context.
    Because of this we nullify the pairwise differences between u and v and only consider the relation to all other features.

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

