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
        raise ValueError("Redundant features have per definition more than one member.")
    if partition is not None:
        if sum(partition) != n_redundant:
            raise ValueError("Sum of partition values should yield number of redundant features.")
        if 0 in partition or 1 in partition:
            raise ValueError("Subset defined in Partition needs at least 2 features. 0 and 1 is not allowed.")


def _fillVariableSpace(X_informative, random_state: object, n_samples: int = 100, n_features: int = 2,
                       n_redundant: int = 0, n_strel: int = 1,
                       n_repeated: int = 0,
                       noise: float = 1, partition=None, **kwargs):
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

    def genStrongRelFeatures(n_samples, strRel, random_state,
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
        min_relevance = 0.2 # Minimum relevance for a feature
        normal_vector = random_state.uniform(min_relevance, 1, int(strRel)) # Generate absolute values
        normal_vector *= random_state.choice([1, -1], int(strRel)) #  Add random sign for each relevance
        b = random_state.uniform(-1, 1) # Hyperplane offset (bias) from origin
        
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

    X_informative, Y = genStrongRelFeatures(n_samples, n_strel + part_size, random_state)

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
                      n_repeated: int = 0, noise: float = 0.1, random_state: object = None,
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

    X = np.zeros((int(n_samples), int(n_features)))

    # Find partitions which defíne the weakly relevant subsets
    if partition is None and n_redundant > 0:
        partition = [n_redundant]
        part_size = 1
    elif partition is not None:
        part_size = len(partition)
    else:
        part_size = 0

    X_informative, Y = make_regression(n_features=int(n_strel + part_size),
                                       n_samples=int(n_samples),
                                       noise=0,
                                       n_informative=int(n_strel + part_size),
                                       random_state=random_state,
                                       shuffle=False)

    X = _fillVariableSpace(X_informative, random_state, n_samples=n_samples, n_features=n_features,
                           n_redundant=n_redundant, n_strel=n_strel,
                           n_repeated=n_repeated,
                           noise=noise, partition=partition)
    
    # Add gaussian noise to data
    X = X + random_state.normal(size=(n_samples,n_features),scale=noise)

    return X, Y

def genOrdinalRegressionData(n_samples: int = 100, n_features: int = 2, n_redundant: int = 0, n_strel: int = 1,
                             n_repeated: int = 0, noise: float = 0.1, random_state: object = None,
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

