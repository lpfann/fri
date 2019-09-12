import numpy as np
from sklearn.utils import check_random_state

from fri import ProblemName
from .gen_data import _fillVariableSpace


def _checkLupiParam(
    problemName, lupiType, n_strel, n_weakrel, n_priv_weakrel, partition, partition_priv
):
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
    if lupiType not in ["cleanLabels", "cleanFeatures"]:
        raise ValueError(
            "The lupiType parameter must be a string out of ['cleanLabels', 'cleanFeatures']."
        )
    if n_strel < 1:
        raise ValueError(
            "At least one strongly relevant feature is necessary (Parmeter 'n_strel' must be greater than 0)."
        )
    if partition is not None:
        if sum(partition) != n_weakrel:
            raise ValueError(
                "The sum over the entries in the partition list must be equal to the parameter 'n_weakrel'."
            )
        if 0 in partition or 1 in partition:
            raise ValueError(
                "The entries in the partition list must be greater or equal to 2."
            )
    if partition_priv is not None:
        if sum(partition_priv) != n_priv_weakrel:
            raise ValueError(
                "The sum over the entries in the partition_priv list must be equal to the parameter 'n_priv_weakrel'."
            )
        if 0 in partition_priv or 1 in partition_priv:
            raise ValueError(
                "The entries in the partition_priv list must be greater or equal to 2."
            )
    if lupiType == "cleanLabels" and n_priv_weakrel > 0:
        raise ValueError(
            "The 'cleanLabels' data has only one strongly relevant feature by nature, this can be repeated ('n_priv_repeated'),"
            "or useless information can be added ('n_priv_irrel') but it can not be weakend => n_priv_weakrel hast to be 0."
        )


# def _genWeakFeatures(n_weakrel, X, random_state, partition):
#     """
#         Generate n_weakrel features out of the strRelFeature
#
#         Parameters
#         ----------
#         n_weakrel : int
#             Number of weakly relevant feature to be generated
#         X : array of shape [n_samples, n_features]
#             Contains the data out of which the weakly relevant features are created
#         random_state : Random State object
#             Used to generate the samples
#         partition : list of int
#             Used to define how many weak features are calculated from the same strong feature
#             The sum of the entries in the partition list must be equal to n_weakrel
#
#
#         Returns
#         ----------
#         X_weakrel : array of shape [n_samples, n_weakrel]
#             Contains the data of the generated weak relevant features
#     """
#
#     X_weakrel = np.zeros([X.shape[0], n_weakrel])
#
#     if partition is None:
#         for i in range(n_weakrel):
#             X_weakrel[:, i] = X[
#                 :, random_state.choice(X.shape[1])
#             ] + random_state.normal(loc=0, scale=1, size=1)
#     else:
#         idx = 0
#         for j in range(len(partition)):
#             X_weakrel[:, idx : idx + partition[j]] = np.tile(
#                 X[:, random_state.choice(X.shape[1])], (partition[j], 1)
#             ).T + random_state.normal(loc=0, scale=1, size=partition[j])
#             idx += partition[j]
#
#     return X_weakrel
#
#
# def _genRepeatedFeatures(n_repeated, X, random_state):
#     """
#         Generate repeated features by picking a random existing feature out of X
#
#         Parameters
#         ----------
#         n_repeated : int
#             Number of repeated features to create
#         X : array of shape [n_samples, n_features]
#             Contains the data of which the repeated features are picked
#         random_state : Random State object
#             Used to randomly pick a feature out of X
#     """
#
#     X_repeated = np.zeros([X.shape[0], n_repeated])
#     for i in range(n_repeated):
#         X_repeated[:, i] = X[:, random_state.choice(X.shape[1])]
#
#     return X_repeated


def genLupiData(
    problemName: ProblemName,
    n_samples: int = 100,
    random_state: object = None,
    n_ordinal_bins: int = 3,
    n_strel: int = 1,
    n_weakrel: int = 0,
    n_repeated: int = 0,
    n_irrel: int = 0,
    label_noise=0.0,
):
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
            n_ordinal_bins : int, optional
                Number of bins in which the regressional target variable is split to form the ordinal classes,
                Only has an effect if problemType == 'ordinalRegression'
            n_strel : int, optional
                Number of features which are mandatory for the underlying model (strongly relevant)
            n_weakrel : int, optional
                Number of weakly relevant features
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
    n_informative = n_strel + (n_weakrel > 0)

    # Create truth (prototype) vector which contains true feature contributions
    # We enforce minimum of 0.1 to circumvent problems when testing for relevance
    w = random_state.uniform(low=0.5, high=1, size=n_informative)
    X_informative = random_state.normal(size=(n_samples, n_informative))
    scores = np.dot(X_informative, w)

    n_features = n_strel + n_weakrel + n_repeated + n_irrel
    X_priv = _fillVariableSpace(
        X_informative,
        random_state,
        n_features=n_features,
        n_redundant=n_weakrel,
        n_strel=n_strel,
        n_repeated=n_repeated,
        partition=[n_weakrel],
    )

    if (
        problemName == "classification"
        or problemName == ProblemName.LUPI_CLASSIFICATION
    ):
        e = random_state.normal(
            size=(n_samples, X_priv.shape[1]), scale=0.65 / X_priv.std()
        )
        X = X_priv + e
        y = np.zeros_like(scores)
        y[scores > 0] = 1
        y[scores <= 0] = -1

    elif problemName == "regression" or problemName == ProblemName.LUPI_REGRESSION:
        e = random_state.normal(
            size=(n_samples, X_priv.shape[1]), scale=0.5 / X_priv.std()
        )
        X = X_priv + e
        y = scores
    elif (
        problemName == "ordinalRegression"
        or problemName == ProblemName.LUPI_ORDREGRESSION
    ):
        e = random_state.normal(
            size=(n_samples, X_priv.shape[1]), scale=0.2 / X_priv.std()
        )
        X = X_priv + e

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
