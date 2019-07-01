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


def _checkParam(n_samples: int = 100,
                n_redundant: int = 0, n_strel: int = 1,
                n_repeated: int = 0,
                flip_y: float = 0, noise: float = 1, partition=None, **kwargs):
    if not 1 < n_samples:
        raise ValueError("We need at least 2 samples.")
    if not 0 <= flip_y < 1:
        raise ValueError("Flip percentage has to be between 0 and 1.")
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


#######################################################################################################################
#                                                                                                                     #
#                                                    New Stuff                                                        #
#                                                                                                                     #
#######################################################################################################################


def _checkLupiParam(problemType, lupiType, n_strel, n_weakrel, n_priv_weakrel, partition, partition_priv):

    """
        Checks if the parameters supplied to the genLupiData() function are okay.

        Parameters
        ----------
        problemType : Str
            Must be one of ['classification', 'regression', 'ordinalRegression']
        lupiType : Str
            Must be one of ['cleanLabels', 'cleanFeatures']
        n_strel : int
            Stands for the number of strongly relevant features to generate in genLupiData()
            Must be greater than 0
        n_weakrel : int
            Must be equal to the length of the partition list
        n_priv_weakrel : int
            Must be equal to the length of the partition_priv list
        partition : list of int
            The length of the list must be equal to n_weakrel
        partition_priv : list of int
            The length of the list must be equal to n_priv_weakrel
    """

    if problemType not in ['classification', 'regression', 'ordinalRegression']:
        raise ValueError("The problemType parameter must be a string out of ['classification', 'regression', 'ordinalRegression'].")
    if lupiType not in ['cleanLabels', 'cleanFeatures']:
        raise ValueError("The lupiType parameter must be a string out of ['cleanLabels', 'cleanFeatures'].")
    if n_strel < 1:
        raise ValueError("At least one strongly relevant feature is necessary (Parmeter 'n_strel' must be greater than 0).")
    if partition is not None:
        if sum(partition) != n_weakrel:
            raise ValueError("The sum over the entries in the partition list must be equal to the parameter 'n_weakrel'.")
        if 0 in partition or 1 in partition:
            raise ValueError("The entries in the partition list must be greater or equal to 2.")
    if partition_priv is not None:
        if sum(partition_priv) != n_priv_weakrel:
            raise ValueError("The sum over the entries in the partition_priv list must be equal to the parameter 'n_priv_weakrel'.")
        if 0 in partition_priv or 1 in partition_priv:
            raise ValueError("The entries in the partition_priv list must be greater or equal to 2.")
    if lupiType == 'cleanLabels' and n_priv_weakrel > 0:
        raise ValueError("The 'cleanLabels' data has only one strongly relevant feature by nature, this can be repeated ('n_priv_repeated'),"
                         "or useless information can be added ('n_priv_irrel') but it can not be weakend => n_priv_weakrel hast to be 0.")


def _genWeakFeatures(n_weakrel, X, random_state, partition):

    """
        Generate n_weakrel features out of the strRelFeature

        Parameters
        ----------
        n_weakrel : int
            Number of weakly relevant feature to be generated
        X : array of shape [n_samples, n_features]
            Contains the data out of which the weakly relevant features are created
        random_state : Random State object
            Used to generate the samples
        partition : list of int
            Used to define how many weak features are calculated from the same strong feature
            The sum of the entries in the partition list must be equal to n_weakrel


        Returns
        ----------
        X_weakrel : array of shape [n_samples, n_weakrel]
            Contains the data of the generated weak relevant features
    """

    X_weakrel = np.zeros([X.shape[0], n_weakrel])

    if partition is None:
        for i in range(n_weakrel):
            X_weakrel[:, i] = X[:, random_state.choice(X.shape[1])] + random_state.normal(loc=0, scale=1, size=1)
    else:
        idx = 0
        for j in range(len(partition)):
            X_weakrel[:, idx: idx + partition[j]] = np.tile(X[:, random_state.choice(X.shape[1])], (partition[j], 1)).T + random_state.normal(loc=0, scale=1, size=partition[j])
            idx += partition[j]

    return X_weakrel


def _genRepeatedFeatures(n_repeated, X, random_state):

    """
        Generate repeated features by picking a random existing feature out of X

        Parameters
        ----------
        n_repeated : int
            Number of repeated features to create
        X : array of shape [n_samples, n_features]
            Contains the data of which the repeated features are picked
        random_state : Random State object
            Used to randomly pick a feature out of X
    """

    X_repeated = np.zeros([X.shape[0], n_repeated])
    for i in range(n_repeated):
        X_repeated[:,i] = X[:, random_state.choice(X.shape[1])]

    return X_repeated


def _genCleanLabelsLupiData(problemType, n_samples, n_informative, noise, random_state, n_ordinal_bins):

    """
        Generate strongly relevant problem data (X_informative) alongside one strongly relevant privileged feature (X_priv_strel),
        the privileged feature consists of the clean (real) y-labels for the problem data. The actually returned
        y-values (y) are noisy and differ in form based on the problemType.

        Parameters
        ----------
        problemType : Str
            Must be one of ['classification', 'regression', 'ordinalRegression'], defines the y-values of the problem
        n_samples : int
            Number of samples to be created
        n_informative : int
            Number of strongly relevant features to be created
        noise : float
            Noise of the created samples around ground truth
        random_state : Random State object
            Used to randomly pick a feature out of X
        n_ordinal_bins : int
            Number of bins in which the regressional target variable is split to form the ordinal classes,
            Only has an effect if problemType == 'ordinalRegression'
    """

    w = random_state.normal(size=n_informative)
    X_informative = random_state.normal(size=(n_samples, n_informative))
    e = random_state.normal(size=n_samples, scale=noise)
    X_priv_strel = np.dot(X_informative[:, :n_informative], w)
    scores = (X_priv_strel + e)[:, np.newaxis]

    if problemType == 'classification':
        y = (scores > 0).astype(int)
    elif problemType == 'regression':
        y = scores
    elif problemType == 'ordinalRegression':
        bs = np.append(np.sort(random_state.normal(size=n_ordinal_bins - 1)), np.inf)
        y = np.sum(scores - bs >= 0, -1)

    return (X_informative, X_priv_strel[:, np.newaxis], y)


def _genCleanFeaturesLupiData(problemType, n_samples, n_strel, noise, random_state, n_ordinal_bins):

    """
        Generate strongly relevant problem data (X_strel) alongside the same number of strongly relevant privileged
        features (X_priv_strel). The privileged features are the actual clean versions of the data, while the data (X_strel)
        that is corresponding to the target variable y has noise in it.

        Parameters
        ----------
        problemType : Str
            Must be one of ['classification', 'regression', 'ordinalRegression'], defines the y-values of the problem
        n_samples : int
            Number of samples to be created
        n_strel : int
            Number of strongly relevant features to be created
        noise : float
            Noise of the created samples around ground truth
        random_state : Random State object
            Used to randomly pick a feature out of X
        n_ordinal_bins : int
            Number of bins in which the regressional target variable is split to form the ordinal classes,
            Only has an effect if problemType == 'ordinalRegression'
    """

    w = random_state.normal(size=n_strel)
    X_priv_strel = random_state.normal(size=(n_samples, n_strel))
    e = np.random.normal(size=(n_samples, n_strel), scale=noise)
    X_strel = X_priv_strel + e
    scores = np.dot(X_priv_strel, w)[:, np.newaxis]

    if problemType == 'classification':
        y = (scores > 0).astype(int)
    elif problemType == 'regression':
        y = scores
    elif problemType == 'ordinalRegression':
        bs = np.append(np.sort(random_state.normal(size=n_ordinal_bins - 1)), np.inf)
        y = np.sum(scores - bs >= 0, -1)

    return (X_strel, X_priv_strel, y)


def genLupiData(problemType: str, lupiType: str, n_samples: int = 100, random_state: object = None, noise: float = 0.1, n_ordinal_bins: int = 3,
                 n_strel: int = 1, n_weakrel: int = 0, n_repeated: int = 0, n_irrel: int = 0,
                 n_priv_weakrel: int = 0, n_priv_repeated: int = 0, n_priv_irrel: int = 0,
                 partition = None, partition_priv = None):

    """
        Generate Lupi Data for Classification, Regression and Ordinal Regression Problems

        Parameters
        ----------
        problemType : Str
            Must be one of ['classification', 'regression', 'ordinalRegression'], defines the y-values of the problem
        lupiType : Str
            Must be one of ['cleanLabels', 'cleanFeatures'], defines the strongly relevant features of the privileged data
        n_samples : int, optional
            Number of samples
        random_state : object, optional
            Randomstate object used for generation.
        noise : float, optional
            Noise of the created samples around ground truth.
        n_ordinal_bins : int, optional
            Number of bins in which the regressional target variable is split to form the ordinal classes,
            Only has an effect if problemType == 'ordinalRegression'
        n_strel : int, optional
            Number of features which are mandatory for the underlying model (strongly relevant)
        n_weakrel : int, optional
            Number of features which are part of redundant subsets (weakly relevant)
        n_repeated : int, optional
            Number of features which are clones of existing ones.
        n_irrel : int, optional
            Number of features which are irrelevant to the underlying model
        n_priv_weakrel : int, optional
            Number of features in the privileged data that are weakly relevant
        n_priv_repeated : int, optional
            Number of privileged features which are clones of existing privileged features
        n_priv_irrel: int, optional
            Number of privileged features which are irrelevant
        partition : list of int
            Entries of the list define weak subsets. So an entry of the list says how many weak features are calculated
            from the same strong feature.
            The sum over the list entries must be equal to n_weakrel
        partition_priv : list of int
            Entries of the list define privileged weak subsets. So an entry of the list says how many privileged weak
            features are calculated from the same privileged strong feature.
            The sum over the list entries must be equal to n_priv_weakrel


        Returns
        -------
        X : array of shape [n_samples, (n_strel + n_weakrel + n_repeated + n_irrel)]
            The generated samples
        X_priv :
            The generated privileged samples
            In case of lupiType == 'cleanLabels' : array of shape [n_samples, (n_priv_weakrel + n_priv_repeated + n_priv_irrel + 1)]
            In case of lupiType == 'cleanFeatures' : array of shape [n_samples, (n_priv_weakrel + n_priv_repeated + n_priv_irrel + n_strel)]
        y : array of shape [n_samples]
            The generated target values
            In case of problemType == 'classification' : values are in [0,1]
            In case of problemType == 'regression' : values are continious
            In case of problemType == 'ordinalRegression' : values are in [0, n_ordinal_bins]


    """

    _checkLupiParam(problemType=problemType, lupiType=lupiType, n_strel=n_strel, n_weakrel=n_weakrel, n_priv_weakrel=n_priv_weakrel,
                    partition=partition, partition_priv=partition_priv)
    random_state = check_random_state(random_state)


    if lupiType == 'cleanLabels':

        n_informative = n_strel + n_weakrel

        # X_strel : array of shape [n_samples, n_strel], contains the strongly relevant data features
        # X_priv_strel : array of shape [n_samples], contains the strongly relevant privileged data feature
        # y : array of shape [n_samples], contains the target values to the problem
        X_informative, X_priv_strel, y = _genCleanLabelsLupiData(problemType=problemType, n_samples=n_samples, n_informative=n_informative,
                                                           noise=noise, random_state=random_state, n_ordinal_bins=n_ordinal_bins)

        X_priv_repeated = _genRepeatedFeatures(n_priv_repeated, X_priv_strel, random_state)
        X_priv_irrel = random_state.normal(size=(n_samples, n_priv_irrel))
        X_priv = np.hstack([X_priv_strel, X_priv_repeated, X_priv_irrel])

    elif lupiType == 'cleanFeatures':

        # X_strel : array of shape [n_samples, n_strel], contains the strongly relevant data features
        # X_priv_strel = array of shape [n_samples, n_strel], contains the strongly relevant privileged features
        # y : array of shape [n_samples], contains the target values to the problem
        X_strel, X_priv_strel, y = _genCleanFeaturesLupiData(problemType=problemType, n_samples=n_samples, n_strel=n_strel,
                                                             noise=noise, random_state=random_state, n_ordinal_bins=n_ordinal_bins)

        X_priv_weakrel = _genWeakFeatures(n_priv_weakrel, X_priv_strel, random_state, partition_priv)
        X_priv_repeated = _genRepeatedFeatures(n_priv_repeated, X_priv_strel, random_state)
        X_priv_irrel = random_state.normal(size=(n_samples, n_priv_irrel))
        X_priv = np.hstack([X_priv_strel, X_priv_weakrel, X_priv_repeated, X_priv_irrel])


    X_strel = X_informative[:, :n_strel]
    X_weakrel = _genWeakFeatures(n_weakrel, X_informative[:, n_strel:], random_state, partition)
    X_repeated = _genRepeatedFeatures(n_repeated, X_strel, random_state)
    X_irrel = random_state.normal(size=(n_samples, n_irrel))
    X = np.hstack([X_strel, X_weakrel, X_repeated, X_irrel])

    return X, X_priv, y



#######################################################################################################################
#                                                                                                                     #
#                                                    New Regression                                                   #
#                                                                                                                     #
#######################################################################################################################


def _checkParam2(n_samples, n_strel, n_weakrel, flip_y: float = 0, partition=None):

    if not 1 < n_samples:
        raise ValueError("We need at least 2 samples.")
    if not 0 <= flip_y < 1:
        raise ValueError("Flip percentage has to be between 0 and 1.")
    if n_strel + n_weakrel < 1:
        raise ValueError("No informative features.")
    if n_strel == 0 and n_weakrel < 2:
        raise ValueError("If we have no strong features, we need more than 1 weak feature.")
    if partition is not None:
        if sum(partition) != n_weakrel:
            raise ValueError("Sum of partition values should yield number of redundant features.")
        if 0 in partition or 1 in partition:
            raise ValueError("Subset defined in Partition needs at least 2 features. 0 and 1 is not allowed.")


def genRegressionData2(n_samples: int = 100, random_state: object = None, noise: float = 0.0,
                      n_strel: int = 1, n_weakrel: int = 0, n_repeated: int = 0, n_irrel: int = 0,
                      partition = None) -> object:

    """Generate synthetic regression data

    Parameters
    ----------
    n_samples : int, optional
        Number of samples
    random_state : object, optional
        Randomstate object used for generation
    noise : float, optional
        Noise of the created samples around ground truth
    n_strel : int, optional
        Number of features which are mandatory for the underlying model (strongly relevant)
    n_weakrel : int, optional
        Number of weakly relevant features
    n_repeated : int, optional
        Number of features which are clones of existing ones.
    n_irrel : int, optional
        Number of features that are irrelevant
    partition: list of int
        Entries of the list define how many weak relevant features are based on the same strongly relevant feature
        The sum over all list entries must be equal to n_weakrel



    Returns
    -------
    X : array of shape [n_samples, (n_strel + n_weakrel + n_repeated + n_irrel]
        The generated samples.
    y : array of shape [n_samples]
        The output values (target).

    Raises
    ------
    ValueError
    Wrong parameters for specified amonut of features/samples.
    """

    _checkParam2(n_samples=n_samples, n_strel=n_strel, n_weakrel=n_weakrel, partition=partition)
    random_state = check_random_state(random_state)

    if partition is None:
        n_informative = n_strel + n_weakrel
    else:
        n_informative = n_strel + len(partition)

    X_informative = random_state.randn(n_samples, n_informative)
    ground_truth = np.zeros((n_informative, 1))
    ground_truth[:n_informative, :] = 0.3
    bias = 0

    y = np.dot(X_informative, ground_truth) + bias

    if noise > 0.0:
        y += random_state.normal(scale=noise, size=y.shape)

    X_strel = X_informative[:, :n_strel]
    X_weakrel = _genWeakFeatures(n_weakrel, X_informative[:, n_strel:], random_state, partition)
    X_repeated = _genRepeatedFeatures(n_repeated, np.hstack([X_strel, X_weakrel]), random_state)
    X_irrel = random_state.normal(size=(n_samples, n_irrel))
    X = np.hstack([X_strel, X_weakrel, X_repeated, X_irrel])
    y = np.squeeze(y)

    return X, y