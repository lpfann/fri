import numpy as np
from sklearn.utils import check_random_state

from fri import ProblemName


def _checkLupiParam(problemName, lupiType, n_strel, n_weakrel, n_priv_weakrel, partition, partition_priv):
    """
        Checks if the parameters supplied to the genLupiData() function are okay.

        Parameters
        ----------
        problemName : Str
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

    if type(problemName) is not ProblemName:
        raise ValueError("Not of Type ProblemName")
    if lupiType not in ['cleanLabels', 'cleanFeatures']:
        raise ValueError("The lupiType parameter must be a string out of ['cleanLabels', 'cleanFeatures'].")
    if n_strel < 1:
        raise ValueError(
            "At least one strongly relevant feature is necessary (Parmeter 'n_strel' must be greater than 0).")
    if partition is not None:
        if sum(partition) != n_weakrel:
            raise ValueError(
                "The sum over the entries in the partition list must be equal to the parameter 'n_weakrel'.")
        if 0 in partition or 1 in partition:
            raise ValueError("The entries in the partition list must be greater or equal to 2.")
    if partition_priv is not None:
        if sum(partition_priv) != n_priv_weakrel:
            raise ValueError(
                "The sum over the entries in the partition_priv list must be equal to the parameter 'n_priv_weakrel'.")
        if 0 in partition_priv or 1 in partition_priv:
            raise ValueError("The entries in the partition_priv list must be greater or equal to 2.")
    if lupiType == 'cleanLabels' and n_priv_weakrel > 0:
        raise ValueError(
            "The 'cleanLabels' data has only one strongly relevant feature by nature, this can be repeated ('n_priv_repeated'),"
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
            X_weakrel[:, idx: idx + partition[j]] = np.tile(X[:, random_state.choice(X.shape[1])],
                                                            (partition[j], 1)).T + random_state.normal(loc=0, scale=1,
                                                                                                       size=partition[
                                                                                                           j])
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
        X_repeated[:, i] = X[:, random_state.choice(X.shape[1])]

    return X_repeated


# def _genCleanLabelsLupiData(problemType, n_samples, n_informative, noise, random_state, n_ordinal_bins):
#
#     """
#         Generate strongly relevant problem data (X_informative) alongside one strongly relevant privileged feature (X_priv_strel),
#         the privileged feature consists of the clean (real) y-labels for the problem data. The actually returned
#         y-values (y) are noisy and differ in form based on the problemType.
#
#         Parameters
#         ----------
#         problemType : Str
#             Must be one of ['classification', 'regression', 'ordinalRegression'], defines the y-values of the problem
#         n_samples : int
#             Number of samples to be created
#         n_informative : int
#             Number of strongly relevant features to be created
#         noise : float
#             Noise of the created samples around ground truth
#         random_state : Random State object
#             Used to randomly pick a feature out of X
#         n_ordinal_bins : int
#             Number of bins in which the regressional target variable is split to form the ordinal classes,
#             Only has an effect if problemType == 'ordinalRegression'
#     """
#
#     w = random_state.normal(size=n_informative)
#     X_informative = random_state.normal(size=(n_samples, n_informative))
#     e = random_state.normal(size=n_samples, scale=noise)
#     X_priv_strel = np.dot(X_informative[:, :n_informative], w)
#     scores = (X_priv_strel + e)[:, np.newaxis]
#
#     if problemType == 'classification':
#         y = (scores > 0).astype(int)
#     elif problemType == 'regression':
#         y = scores
#     elif problemType == 'ordinalRegression':
#         bs = np.append(np.sort(random_state.normal(size=n_ordinal_bins - 1)), np.inf)
#         y = np.sum(scores - bs >= 0, -1)
#
#     return (X_informative, X_priv_strel[:, np.newaxis], y)
#
#
# def _genCleanFeaturesLupiData(problemType, n_samples, n_strel, noise, random_state, n_ordinal_bins):
#
#     """
#         Generate strongly relevant problem data (X_strel) alongside the same number of strongly relevant privileged
#         features (X_priv_strel). The privileged features are the actual clean versions of the data, while the data (X_strel)
#         that is corresponding to the target variable y has noise in it.
#
#         Parameters
#         ----------
#         problemType : Str
#             Must be one of ['classification', 'regression', 'ordinalRegression'], defines the y-values of the problem
#         n_samples : int
#             Number of samples to be created
#         n_strel : int
#             Number of strongly relevant features to be created
#         noise : float
#             Noise of the created samples around ground truth
#         random_state : Random State object
#             Used to randomly pick a feature out of X
#         n_ordinal_bins : int
#             Number of bins in which the regressional target variable is split to form the ordinal classes,
#             Only has an effect if problemType == 'ordinalRegression'
#     """
#
#     w = random_state.normal(size=n_strel)
#     X_priv_strel = random_state.normal(size=(n_samples, n_strel))
#     e = np.random.normal(size=(n_samples, n_strel), scale=noise)
#     X_strel = X_priv_strel + e
#     scores = np.dot(X_priv_strel, w)[:, np.newaxis]
#
#     if problemType == 'classification':
#         y = (scores > 0).astype(int)
#     elif problemType == 'regression':
#         y = scores
#     elif problemType == 'ordinalRegression':
#         bs = np.append(np.sort(random_state.normal(size=n_ordinal_bins - 1)), np.inf)
#         y = np.sum(scores - bs >= 0, -1)
#
#     return (X_strel, X_priv_strel, y)


# def genLupiData(problemName: ProblemName, lupiType: str = "cleanFeatures", n_samples: int = 100, random_state: object = None, noise: float = 0.1,
#                 n_ordinal_bins: int = 3, n_strel: int = 1, n_weakrel: int = 0, n_repeated: int = 0, n_irrel: int = 0,
#                 n_priv_weakrel: int = 0, n_priv_repeated: int = 0, n_priv_irrel: int = 0, partition=None,
#                 partition_priv=None):
#
#     """
#         Generate Lupi Data for Classification, Regression and Ordinal Regression Problems
#
#         Parameters
#         ----------
#         problemName : ProblemName
#             Defines the type of y-values of the problem. Example `ProblemName.CLASSIFICATION`.
#         lupiType : Str
#             Must be one of ['cleanLabels', 'cleanFeatures'], defines the strongly relevant features of the privileged data
#         n_samples : int, optional
#             Number of samples
#         random_state : object, optional
#             Randomstate object used for generation.
#         noise : float, optional
#             Noise of the created samples around ground truth.
#         n_ordinal_bins : int, optional
#             Number of bins in which the regressional target variable is split to form the ordinal classes,
#             Only has an effect if problemType == 'ordinalRegression'
#         n_strel : int, optional
#             Number of features which are mandatory for the underlying model (strongly relevant)
#         n_weakrel : int, optional
#             Number of features which are part of redundant subsets (weakly relevant)
#         n_repeated : int, optional
#             Number of features which are clones of existing ones.
#         n_irrel : int, optional
#             Number of features which are irrelevant to the underlying model
#         n_priv_weakrel : int, optional
#             Number of features in the privileged data that are weakly relevant
#         n_priv_repeated : int, optional
#             Number of privileged features which are clones of existing privileged features
#         n_priv_irrel: int, optional
#             Number of privileged features which are irrelevant
#         partition : list of int
#             Entries of the list define weak subsets. So an entry of the list says how many weak features are calculated
#             from the same strong feature.
#             The sum over the list entries must be equal to n_weakrel
#         partition_priv : list of int
#             Entries of the list define privileged weak subsets. So an entry of the list says how many privileged weak
#             features are calculated from the same privileged strong feature.
#             The sum over the list entries must be equal to n_priv_weakrel
#
#
#         Returns
#         -------
#         X : array of shape [n_samples, (n_strel + n_weakrel + n_repeated + n_irrel)]
#             The generated samples
#         X_priv :
#             The generated privileged samples
#             In case of lupiType == 'cleanLabels' : array of shape [n_samples, (n_priv_weakrel + n_priv_repeated + n_priv_irrel + 1)]
#             In case of lupiType == 'cleanFeatures' : array of shape [n_samples, (n_priv_weakrel + n_priv_repeated + n_priv_irrel + n_strel)]
#         y : array of shape [n_samples]
#             The generated target values
#             In case of problemType == 'classification' : values are in [0,1]
#             In case of problemType == 'regression' : values are continious
#             In case of problemType == 'ordinalRegression' : values are in [0, n_ordinal_bins]
#
#
#     """
#
#     _checkLupiParam(problemName=problemName, lupiType=lupiType, n_strel=n_strel, n_weakrel=n_weakrel,
#                     n_priv_weakrel=n_priv_weakrel, partition=partition, partition_priv=partition_priv)
#     random_state = check_random_state(random_state)
#
#
#     if lupiType == 'cleanLabels':
#
#         n_informative = n_strel + n_weakrel
#
#         # X_strel : array of shape [n_samples, n_strel], contains the strongly relevant data features
#         # X_priv_strel : array of shape [n_samples], contains the strongly relevant privileged data feature
#         # y : array of shape [n_samples], contains the target values to the problem
#         X_informative, X_priv_strel, y = _genCleanLabelsLupiData(problemType=problemName, n_samples=n_samples, n_informative=n_informative,
#                                                                  noise=noise, random_state=random_state, n_ordinal_bins=n_ordinal_bins)
#
#         X_priv_repeated = _genRepeatedFeatures(n_priv_repeated, X_priv_strel, random_state)
#         X_priv_irrel = random_state.normal(size=(n_samples, n_priv_irrel))
#         X_priv = np.hstack([X_priv_strel, X_priv_repeated, X_priv_irrel])
#
#     elif lupiType == 'cleanFeatures':
#
#         # X_strel : array of shape [n_samples, n_strel], contains the strongly relevant data features
#         # X_priv_strel = array of shape [n_samples, n_strel], contains the strongly relevant privileged features
#         # y : array of shape [n_samples], contains the target values to the problem
#         X_strel, X_priv_strel, y = _genCleanFeaturesLupiData(problemType=problemName, n_samples=n_samples, n_strel=n_strel,
#                                                              noise=noise, random_state=random_state, n_ordinal_bins=n_ordinal_bins)
#
#         X_priv_weakrel = _genWeakFeatures(n_priv_weakrel, X_priv_strel, random_state, partition_priv)
#         X_priv_repeated = _genRepeatedFeatures(n_priv_repeated, X_priv_strel, random_state)
#         X_priv_irrel = random_state.normal(size=(n_samples, n_priv_irrel))
#         X_priv = np.hstack([X_priv_strel, X_priv_weakrel, X_priv_repeated, X_priv_irrel])
#
#
#     X_strel = X_informative[:, :n_strel]
#     X_weakrel = _genWeakFeatures(n_weakrel, X_informative[:, n_strel:], random_state, partition)
#     X_repeated = _genRepeatedFeatures(n_repeated, X_strel, random_state)
#     X_irrel = random_state.normal(size=(n_samples, n_irrel))
#     X = np.hstack([X_strel, X_weakrel, X_repeated, X_irrel])
#
#     return X, X_priv, y


def genLupiData(problemName: ProblemName, n_samples: int = 100, random_state: object = None, noise: float = 0.1,
                n_ordinal_bins: int = 3, n_strel: int = 1, n_weakrel_groups: int = 0,
                n_repeated: int = 0, n_irrel: int = 0, label_noise=0.0):
    """
            Generate Lupi Data for Classification, Regression and Ordinal Regression Problems

            Parameters
            ----------
            problemName : ProblemName
                Defines the type of y-values of the problem. Example `ProblemName.CLASSIFICATION`.
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
            n_weakrel_groups : int, optional
                Number of 2 feature groups which are part of redundant subsets (weakly relevant)
            n_repeated : int, optional
                Number of features which are clones of existing ones.
            n_irrel : int, optional
                Number of features which are irrelevant to the underlying model
            label_noise: float, optional
                Percentage of labels which get permutated.


            Returns
            -------
            X : array of shape [n_samples, (n_strel + n_weakrel + n_repeated + n_irrel)]
                The generated samples
            X_priv : array with same shape as X
                The generated privileged samples
            y : array of shape [n_samples]
                The generated target values
                In case of problemType == 'classification' : values are in [0,1]
                In case of problemType == 'regression' : values are continious
                In case of problemType == 'ordinalRegression' : values are in [0, n_ordinal_bins]


        """

    random_state = check_random_state(random_state)

    n_informative = n_strel + n_weakrel_groups
    w = random_state.normal(size=n_informative)
    X_informative = random_state.normal(size=(n_samples, n_informative))
    X_priv_strel = X_informative[:, :n_strel]

    X_priv_weakrel = np.zeros([n_samples, n_weakrel_groups * 2])
    idx = 0
    for i in range(n_weakrel_groups):
        X_priv_weakrel[:, idx:idx + 2] = np.tile(X_informative[:, n_strel + i], (2, 1)).T + random_state.normal(loc=0,
                                                                                                                scale=np.std(
                                                                                                                    X_informative[
                                                                                                                    :,
                                                                                                                    n_strel + i]),
                                                                                                                size=2)
        idx += 2

    X_priv_repeated = _genRepeatedFeatures(n_repeated, np.hstack([X_priv_strel, X_priv_weakrel]), random_state)

    X_priv = np.hstack([X_priv_strel, X_priv_weakrel, X_priv_repeated])

    e = random_state.normal(size=(n_samples, X_priv.shape[1]), scale=noise * np.std(X_priv))
    X = X_priv + e
    scores = np.dot(X_informative, w)

    if problemName == 'classification' or problemName == ProblemName.LUPI_CLASSIFICATION:
        y = scores > 0
    elif problemName == 'regression' or problemName == ProblemName.LUPI_REGRESSION:
        y = scores
    elif problemName == 'ordinalRegression' or problemName == ProblemName.LUPI_ORDREGRESSION:
        step = 1 / (n_ordinal_bins)
        quantiles = [i * step for i in range(1, n_ordinal_bins)]
        bs = np.quantile(scores, quantiles)
        bs = np.append(bs, np.inf)
        scores = scores[:, np.newaxis]
        y = np.sum(scores - bs >= 0, -1)

    if n_irrel > 0:
        X = np.hstack([X, random_state.normal(size=(n_samples, n_irrel))])
        X_priv = np.hstack([X_priv, random_state.normal(size=(n_samples, n_irrel))])

    if label_noise > 0:
        sample = random_state.choice(len(y), int(len(y) * label_noise))
        y[sample] = random_state.permutation(y[sample])

    return (X, X_priv, y.squeeze())


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
                       partition=None) -> object:
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
