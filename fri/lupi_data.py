import numpy as np
from sklearn.datasets import make_regression
from sklearn.utils import check_random_state
from sklearn.utils import shuffle


def _combFeat(n, size, strRelFeat, randomstate):
    # Split each strongly relevant feature into linear combination of it
    weakFeats = np.zeros((n, size))
    for x in range(size):
        cofact = 2 * randomstate.rand() - 1
        weakFeats[:, x] = cofact * strRelFeat
    return weakFeats


def _dummyFeat(n, randomstate):
    return randomstate.randn(n)


def _repeatFeat(feats, i, randomstate):
    i_pick = randomstate.choice(i)
    return feats[:, i_pick]


def _checkParam(n_samples: int = 100,
                n_features: int = 2,
                n_redundant: int = 0, n_strel: int = 1, n_repeated: int = 0,
                n_priv_features: int = 1,
                n_priv_redundant: int = 0, n_priv_strel: int = 1, n_priv_repeated: int = 0,
                flip_y: float = 0, noise: float = 1, partition=None, **kwargs):
    if not 1 < n_samples:
        raise ValueError("We need at least 2 samples.")
    if not 0 < n_features:
        raise ValueError("We need at least one feature.")
    if not 0 <= flip_y < 1:
        raise ValueError("Flip percentage has to be between 0 and 1.")
    if not n_redundant + n_repeated + n_strel <= n_features:
        raise ValueError("Inconsistent number of features")
    if n_strel + n_redundant < 1:
        raise ValueError("No informative features.")
    if n_strel == 0 and n_redundant < 2:
        raise ValueError("Redundant features have per definition more than one member.")
    if not 0 < n_priv_features:
        raise ValueError("We need at least one privileged feature.")
    if not n_priv_redundant + n_priv_repeated + n_priv_strel <= n_priv_features:
        raise ValueError("Inconsistent number of privileged features.")
    if n_priv_strel + n_priv_redundant < 1:
        raise ValueError("No informative privileged features.")
    if partition is not None:
        if sum(partition) != n_redundant:
            raise ValueError("Sum of partition values should yield number of redundant features.")
        if 0 in partition or 1 in partition:
            raise ValueError("Subset defined in Partition needs at least 2 features. 0 and 1 is not allowed.")


def _fillVariableSpaceLupi(X_informative, part_size, random_state: object, n_samples: int = 100,
                       n_features: int = 2, n_redundant: int = 0, n_strel: int = 1, n_repeated: int = 0,
                       n_priv_features: int = 1, n_priv_redundant: int = 0, n_priv_strel: int = 1, n_priv_repeated: int = 0,
                       noise: float = 1, partition=None, partition_priv=None, **kwargs):

    X = np.zeros((int(n_samples), int(n_features)))
    X_priv = np.zeros((int(n_samples), int(n_priv_features)))
    X[:, :n_strel] = X_informative[:, :n_strel]
    X_priv[:, :n_priv_strel] = X_informative[:, n_strel:n_strel + n_priv_strel]
    holdout = X_informative[:, n_strel + n_priv_strel:n_strel + n_priv_strel + part_size]
    holdout_priv = X_informative[:, n_strel + n_priv_strel + part_size:]
    i = n_strel
    j = n_priv_strel

    pi = 0
    for x in range(len(holdout.T)):
        size = partition[pi]
        X[:, i:i + size] = _combFeat(n_samples, size, holdout[:, x], random_state)
        i += size
        pi += 1

    pi_priv = 0
    for x in range(len(holdout_priv.T)):
        size_priv = partition_priv[pi_priv]
        X_priv[:, j:j + size_priv] = _combFeat(n_samples, size_priv, holdout_priv[:, x], random_state)
        j += size_priv
        pi_priv += 1

    for x in range(n_repeated):
        X[:, i] = _repeatFeat(X[:, :i], i, random_state)
        i += 1

    for x in range(n_priv_repeated):
        X_priv[:, j] = _repeatFeat(X_priv[:, :j], j, random_state)
        j += 1

    for x in range(n_features - i):
        X[:, i] = _dummyFeat(n_samples, random_state)
        i += 1

    for x in range(n_priv_features - j):
        X[:, j] = _dummyFeat(n_samples, random_state)
        j += 1

    return X, X_priv


def genLupiData(n_samples: int = 100,
                          n_features: int = 2,
                          n_redundant: int = 0, n_strel: int = 1, n_repeated: int = 0,
                          n_priv_features: int = 1,
                          n_priv_redundant: int = 0, n_priv_strel: int = 1, n_priv_repeated: int = 0,
                          flip_y: float = 0, random_state: object = None, partition=None, partition_priv=None):

    """Generate synthetic classification data

    Parameters
    ----------
    n_samples : int, optional
        Number of samples
    n_features : int, optional
        Number of features
    n_redundant : int, optional
        Number of features which are part of redundant subsets (weakly relevant)
    n_strel : int, optional
        Number of features which are mandatory for the underlying model (strongly relevant)
    n_repeated : int, optional
        Number of features which are clones of existing ones.
    n_priv_features : int, optional
        Number of privileged features
    n_priv_redundant : int, optional
        Number of weakly relevant privileged features
    n_priv_strel : int, optional
        Number of strongly relevant privileged features
    n_priv_repeated : int, optional
        Number of irrelevant privileged features
    flip_y : float, optional
        Ratio of samples randomly switched to wrong class.
    random_state : object, optional
        Randomstate object used for generation.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The output classes.

    Raises
    ------
    ValueError
        Description
    ValueError
    Wrong parameters for specified amonut of features/samples.

    """

    _checkParam(**locals())
    random_state = check_random_state(random_state)

    def genStrongRelFeaturesLupi(n_samples, strRel, random_state,
                             margin=1,
                             data_range=10):
        """ Generate data uniformly distributed in a square and perfectly separated by the hyperplane given by normal_vector and b.
        Keyword arguments:
        n_samples -- number of samples required (default 100)
        n_features -- number of features required
        normal_vector -- the normal vector of the separating hyperplane
        margin -- intrusion-free margin of the optimal separating hyperplane (default 1)
        data_range -- data is distributed between -data_range and data_range (default 10)

        """
        min_relevance = 0.2  # Minimum relevance for a feature
        normal_vector = random_state.uniform(min_relevance, 1, int(strRel))  # Generate absolute values
        normal_vector *= random_state.choice([1, -1], int(strRel))  # Add random sign for each relevance
        b = random_state.uniform(-1, 1)  # Hyperplane offset (bias) from origin

        # Sample data uniformly.
        data = random_state.uniform(-data_range, data_range,
                                    (n_samples, int(strRel))) + b

        # Re-roll margin intrusions.
        intruders = np.abs(np.inner(normal_vector, data) - b) < margin
        while np.sum(intruders) > 0:
            data[intruders] = random_state.uniform(
                -data_range, data_range, (np.sum(intruders), int(strRel))) + b
            intruders = np.abs(np.inner(normal_vector, data) - b) < margin

        # Label data according to placement relative to the hyperplane induced by normal_vector and b.
        labels = np.ones(n_samples)
        labels[np.inner(normal_vector, data) - b > 0] = 1
        labels[np.inner(normal_vector, data) - b < 0] = -1

        return data, labels

    X = np.zeros((n_samples, n_features))

    # Find partitions which defíne the weakly relevant subsets
    if partition is None and n_redundant > 0:
        partition = [n_redundant]
        part_size = 1
    elif partition is not None:
        part_size = len(partition)
    else:
        part_size = 0

    if partition_priv is None and n_priv_redundant > 0:
        partition_priv = [n_priv_redundant]
        part_size_priv = 1
    elif partition_priv is not None:
        part_size_priv = len(partition_priv)
    else:
        part_size_priv = 0


    X_informative, Y = genStrongRelFeaturesLupi(n_samples, n_strel + n_priv_strel + part_size + part_size_priv, random_state)

    X, X_priv = _fillVariableSpaceLupi(X_informative, part_size=part_size, random_state=random_state, n_samples=n_samples,
                           n_features=n_features, n_redundant=n_redundant, n_strel=n_strel, n_repeated=n_repeated,
                           n_priv_features=n_priv_features, n_priv_redundant=n_priv_redundant, n_priv_strel=n_priv_strel, n_priv_repeated=n_priv_repeated,
                           partition=partition, partition_priv=partition_priv)

    if flip_y > 0:
        n_flip = int(flip_y * n_samples)
        Y[random_state.choice(n_samples, n_flip)] *= -1

    return X, X_priv, Y