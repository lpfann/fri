import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.utils import check_random_state
from sklearn.utils import shuffle


def _combFeat(n, size, strRelFeat, randomstate):
    # Split each strongly relevant feature into linear combination of it
    weakFeats = np.tile(strRelFeat, (size, 1)).T
    weakFeats = randomstate.normal(loc=0, scale=1, size=size) + weakFeats
    return weakFeats


def _dummyFeat(n, randomstate):
    return randomstate.randn(n)


def _repeatFeat(feats, i, randomstate):
    i_pick = randomstate.choice(i)
    return feats[:, i_pick]


def _checkParam(n_samples: int = 100, n_features: int = 2,
                n_redundant: int = 0, n_strel: int = 1,
                n_repeated: int = 0,
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
        raise ValueError("We need more than 1 redundant feature.")
    if partition is not None:
        if sum(partition) != n_redundant:
            raise ValueError("Sum of partition values should yield number of redundant features.")
        if 0 in partition or 1 in partition:
            raise ValueError("Subset defined in Partition needs at least 2 features. 0 and 1 is not allowed.")


def _fillVariableSpace(X_informative, random_state: RandomState, n_samples: int = 100, n_features: int = 2,
                       n_redundant: int = 0, n_strel: int = 1,
                       n_repeated: int = 0,
                       noise: float = 1, partition=None, **kwargs):
    if partition is not None:
        assert n_redundant == np.sum(partition)
    X = np.zeros((int(n_samples), int(n_features)))
    X[:, :n_strel] = X_informative[:, :n_strel]
    holdout = X_informative[:, n_strel:]
    i = n_strel

    pi = 0
    for x in range(len(holdout.T)):
        size = partition[pi]
        X[:, i:i + size] = _combFeat(n_samples, size, holdout[:, x], random_state)
        i += size
        pi += 1

    for x in range(n_repeated):
        X[:, i] = _repeatFeat(X[:, :i], i, random_state)
        i += 1
    for x in range(n_features - i):
        X[:, i] = _dummyFeat(n_samples, random_state)
        i += 1

    return X


def generate_binary_classification_problem(n_samples: int, features: int, random_state: RandomState = None,
                                           data_range=1):
    """ Generate data uniformly distributed in a square and perfectly separated by the hyperplane given by normal_vector and b.
    Keyword arguments:
    n_samples -- number of samples required (default 100)
    n_features -- number of features required
    normal_vector -- the normal vector of the separating hyperplane
    data_range -- data is distributed between -data_range and data_range (default 10)

    """
    random_state = check_random_state(random_state)

    data = random_state.normal(size=(n_samples, features), scale=data_range)
    labels = np.sum(data, 1) > 0
    labels = labels.astype(int)
    labels[labels == 0] = -1
    return data, labels

def genClassificationData(n_samples: int = 100, n_features: int = 2,
                          n_redundant: int = 0, n_strel: int = 1,
                          n_repeated: int = 0, noise : float = 0.1,
                          flip_y: float = 0, random_state: object = None,
                          partition=None):
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
    noise : float
        Added gaussian noise to data. Parameter scales Std of normal distribution.
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
    

    Examples
    ---------
    >>> X,y = genClassificationData(n_samples=200)
    Generating dataset with d=2,n=200,strongly=1,weakly=0, partition of weakly=None
    >>> X.shape
    (200, 2)
    >>> y.shape
    (200,)
    """
    _checkParam(**locals())
    random_state = check_random_state(random_state)



    X = np.zeros((n_samples, n_features))

    # Find partitions which defíne the weakly relevant subsets
    if partition is None and n_redundant > 0:
        partition = [n_redundant]
        part_size = 1
    elif partition is not None:
        part_size = len(partition)
    else:
        part_size = 0

    X_informative, Y = generate_binary_classification_problem(n_samples, n_strel + part_size, random_state)

    X = _fillVariableSpace(X_informative, random_state, n_samples=n_samples, n_features=n_features,
                           n_redundant=n_redundant, n_strel=n_strel,
                           n_repeated=n_repeated, partition=partition)
    # Add target noise
    if flip_y > 0:
        n_flip = int(flip_y * n_samples)
        Y[random_state.choice(n_samples, n_flip)] *= -1

    # Add gaussian noise to data
    X = X + random_state.normal(size=(n_samples,n_features),scale=noise)

    return X, Y


def genRegressionData(n_samples: int = 100, n_features: int = 2, n_redundant: int = 0, n_strel: int = 1,
                      n_repeated: int = 0, noise: float = 0.0, random_state: object = None,
                      partition=None) -> object:
    """Generate synthetic regression data
    
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
    noise : float, optional
        Noise of the created samples around ground truth.
    random_state : object, optional
        Randomstate object used for generation.
    
    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The output values (target).
    
    Raises
    ------
    ValueError
    Wrong parameters for specified amonut of features/samples.
    """

    _checkParam(**locals())
    random_state = check_random_state(random_state)

    # Find partitions which defíne the weakly relevant subsets
    if partition is None and n_redundant > 0:
        partition = [n_redundant]
        part_size = 1
    elif partition is not None:
        part_size = len(partition)
    else:
        part_size = 0

    n_informative = n_strel + part_size

    X = random_state.randn(n_samples, n_informative)
    ground_truth = np.zeros((n_informative, 1))
    ground_truth[:n_informative, :] = 0.3
    bias = 0

    y = np.dot(X, ground_truth) + bias

    # Add noise
    if noise > 0.0:
        y += random_state.normal(scale=noise, size=y.shape)

    X = _fillVariableSpace(X, random_state, n_samples=n_samples, n_features=n_features,
                           n_redundant=n_redundant, n_strel=n_strel,
                           n_repeated=n_repeated,
                           noise=noise, partition=partition)
    y = np.squeeze(y)
    return X, y

def genOrdinalRegressionData(n_samples: int = 100, n_features: int = 2, n_redundant: int = 0, n_strel: int = 1,
                             n_repeated: int = 0, noise: float = 0.0, random_state: object = None,
                             partition=None, n_target_bins: int = 3):

    """
    Generate ordinal regression data

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
    noise : float, optional
        Noise of the created samples around ground truth.
    random_state : object, optional
        Randomstate object used for generation.
    n_target_bins : int, optional
        Number of bins in which the regressional target variable is split to form the ordinal classes

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.
    y : array of shape [n_samples]
        The output values (target).

    Raises
    ------
    ValueError
    Wrong parameters for specified amonut of features/samples.
    """

    _checkParam(**locals())
    random_state = check_random_state(random_state)

    if not n_target_bins > 1:
        raise ValueError("At least 2 target bins needed")

    # Use normal regression data as starting point
    X_regression, Y_regression = genRegressionData(n_samples=int(n_samples),
                                                   n_features=int(n_features),
                                                   n_redundant=int(n_redundant),
                                                   n_strel=int(n_strel),
                                                   n_repeated=int(n_repeated),
                                                   noise=0,
                                                   random_state=random_state,
                                                   partition=partition)

    bin_size = int(np.floor(n_samples / n_target_bins))
    rest = int(n_samples - (bin_size * n_target_bins))

    # Sort the target values and rearange the data accordingly
    sort_indices = np.argsort(Y_regression)
    X = X_regression[sort_indices]
    Y = Y_regression[sort_indices]

    # Assign ordinal classes as target values
    for i in range(n_target_bins):
        Y[bin_size*i:bin_size*(i+1)] = i

    # Put non divisable rest into last bin
    if rest > 0:
        Y[-rest:] = n_target_bins - 1

    X, Y = shuffle(X, Y, random_state=random_state)

    # Add gaussian noise to data
    X = X + random_state.normal(size=(n_samples,n_features),scale=noise)

    return X, Y


def _checkLupiParam(n_priv_features: int = 1,
                    n_priv_redundant: int = 0, n_priv_strel: int = 1, n_priv_repeated: int = 0,
                    partition_priv=None):
    if not 0 < n_priv_features:
        raise ValueError("We need at least one privileged feature.")
    if not n_priv_redundant + n_priv_repeated + n_priv_strel <= n_priv_features:
        raise ValueError("Inconsistent number of priv. features")
    if n_priv_strel + n_priv_redundant < 1:
        raise ValueError("No informative priv. features.")
    if n_priv_strel == 0 and n_priv_redundant < 2:
        raise ValueError("Redundant features have per definition more than one member (per group).")
    if partition_priv is not None:
        if sum(partition_priv) != n_priv_redundant:
            raise ValueError("Sum of partition_priv values should yield number of redundant features.")
        if 0 in partition_priv or 1 in partition_priv:
            raise ValueError("Subset defined in Partition needs at least 2 features. 0 and 1 is not allowed.")


def genLupiData(generator, n_priv_features: int = 1,
                n_priv_redundant: int = 0, n_priv_strel: int = 1, n_priv_repeated: int = 0,
                partition_priv=None, n_features: int = 2,
                n_redundant: int = 0, n_strel: int = 1,
                n_repeated: int = 0, partition=None, random_state=None, rettruth=False, **kwargs):
    if generator in [genClassificationData, genRegressionData, genOrdinalRegressionData]:
        _checkParam(n_features=n_features,
                    n_redundant=n_redundant, n_strel=n_strel,
                    n_repeated=n_repeated, partition=partition, **kwargs)
        _checkLupiParam(n_priv_features=n_priv_features,
                        n_priv_redundant=n_priv_redundant, n_priv_strel=n_priv_strel,
                        n_priv_repeated=n_priv_repeated,
                        partition_priv=partition_priv)
        c_features = n_features + n_priv_features
        c_strel = n_strel + n_priv_strel
        c_redundant = n_redundant + n_priv_redundant
        c_repeated = n_repeated + n_priv_repeated
        if partition_priv is not None:
            partition = partition_priv.extend(partition)  # Take priv partitions first for later indexing

        X, y = generator(n_features=c_features,
                         n_redundant=c_redundant, n_strel=c_strel,
                         n_repeated=c_repeated, partition=partition, random_state=random_state, **kwargs)

        # Build index list of features
        ix = range(c_features)
        ix_priv = []
        # Pick index for strel features from the beginning of the generated array
        ix_priv.extend(ix[:n_priv_strel])
        # Pick index for redundant
        ix_priv.extend(ix[c_strel:c_strel + n_priv_redundant])
        # Pick index for repeated features
        ix_priv.extend(ix[c_strel + c_redundant:c_strel + c_redundant + n_priv_repeated])
        # Pick index for irrelevant
        n_irrelevant_priv_features = n_priv_features - n_priv_strel - n_priv_redundant - n_priv_repeated
        if n_irrelevant_priv_features > 0:
            # notice the '-', we slice from the back, where irrelevant features are
            ix_priv.extend((ix[-n_irrelevant_priv_features:]))

        ix_not_priv = [index for index in ix if index not in ix_priv]

        X_priv = X[:, ix_priv]
        X = X[:, ix_not_priv]

        if rettruth:
            # Create truth vector
            truth = np.zeros(c_features)
            rel_X = n_strel + n_redundant + n_repeated
            truth[:rel_X] = 1
            rel_X_priv = n_priv_strel + n_priv_redundant + n_priv_repeated
            truth[n_features:n_features + rel_X_priv] = 1
            return X, X_priv, y, truth.astype(bool)
        
        return X, X_priv, y


def quick_generate(problem, **kwargs):
    if problem is "regression":
        gen = genRegressionData
    elif problem is "classification":
        gen = genClassificationData
    elif problem is "ordreg":
        gen = genOrdinalRegressionData
    else:
        raise ValueError("Unknown problem type. Try 'regression', 'classification' or 'ordreg'")
    return gen(**kwargs)
