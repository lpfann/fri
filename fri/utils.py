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


def permutate_feature_in_data(data, feature_i, random_state):
    X, y = data
    X_copy = np.copy(X)
    # Permute selected feature
    permutated_feature = random_state.permutation(X_copy[:, feature_i])
    # Add permutation back to dataset
    X_copy[:, feature_i] = permutated_feature
    return X_copy, y
