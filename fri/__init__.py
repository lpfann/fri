import logging

# Get version from versioneer
from fri._version import get_versions

__version__ = get_versions()['version']
del get_versions

logging.basicConfig(level=logging.INFO)
from enum import Enum

from fri.genData import genRegressionData, genClassificationData, genOrdinalRegressionData, quick_generate
from fri.main import FRIBase
from fri.plot import plot_intervals
import fri.model


class ProblemName(Enum):
    """
    Enum which contains usable models for which feature relevance intervals can be computed in :func:`~FRI`.
    """
    CLASSIFICATION = fri.model.Classification
    REGRESSION = fri.model.Regression
    ORDINALREGRESSION = fri.model.OrdinalRegression
    ORDINALREGRESSION_IMP = fri.model.OrdinalRegression_Imp
    LUPI_CLASSIFICATION = fri.model.LUPI_Classification
    LUPI_REGRESSION = fri.model.LUPI_Regression
    LUPI_ORDREGRESSION = fri.model.LUPI_OrdinalRegression
    LUPI_ORDREGRESSION_IMP = fri.model.LUPI_OrdinalRegression_IMP


def FRI(problemName: ProblemName, random_state=None, n_jobs=1, verbose=0, n_param_search=50,
        n_probe_features=80, slack_regularization=0.1, slack_loss=0.1, normalize=True, **kwargs):
    """

    Parameters
    ----------
    problemName: :class:`~ProblemName` or str
        Type of Problem
    random_state: object or int
        Random state object or int
    n_jobs: int or None
        Number of threads or -1 for automatic.
    verbose: int
        Verbosity if > 0
    n_param_search: int
        Number of parameter samples in random search for hyperparameters.
    n_probe_features: int
        Number of probes to generate to improve feature selection.
    slack_regularization: float
        Allow deviation from optimal L1 norm.
    slack_loss: float
        Allow deviation of loss.
    normalize: boolean
        Normalize relevace bounds to range of [0,1] depending on L1 norm.


    Returns
    -------
    `FRIBase`
        Model initalized with type in `problemName`.

    """
    if isinstance(problemName, ProblemName):
        problemtype = problemName.value
    else:
        if problemName == "classification" or problemName == "class":
            problemtype = ProblemName.CLASSIFICATION
        elif problemName == "regression" or problemName == "reg":
            problemtype = ProblemName.REGRESSION
        elif problemName == "ordinalregression" or problemName == "ordreg":
            problemtype = ProblemName.ORDINALREGRESSION
        elif problemName == "lupi_classification" or problemName == "lupi_class":
            problemtype = ProblemName.LUPI_CLASSIFICATION
        else:
            names = [enum.name.lower() for enum in ProblemName]

            print(f"Parameter 'problemName' was not recognized or unset. Try one of {names}.")
            return None
    return FRIBase(problemtype, random_state=random_state, n_jobs=n_jobs, verbose=verbose,
                   n_param_search=n_param_search,
                   n_probe_features=n_probe_features,
                   w_l1_slack=slack_regularization,
                   loss_slack=slack_loss,
                   normalize=normalize,
                   **kwargs)


__all__ = ["genRegressionData", "genClassificationData", "genOrdinalRegressionData",
           "quick_generate", "plot_intervals", "ProblemName", "FRI"]

# def FRIClassification(**kwargs):
#     warnings.warn(
#         "This class call format is deprecated.",
#         DeprecationWarning
#     )
#
#     typeprob = Classification
#     kwargs = {k: v for k, v in kwargs.items() if v is not None}
#     return call_main_catch_old(typeprob,
#                                **kwargs)
#
#
# def FRIRegression(**kwargs):
#     warnings.warn(
#         "This class call format is deprecated.",
#         DeprecationWarning
#     )
#
#     typeprob = Regression
#     kwargs = {k: v for k, v in kwargs.items() if v is not None}
#     return call_main_catch_old(typeprob,
#                                **kwargs)
#
#
# def FRIOrdinalRegression(**kwargs):
#     warnings.warn(
#         "This class call format is deprecated.",
#         DeprecationWarning
#     )
#     typeprob = OrdinalRegression
#     kwargs = {k: v for k, v in kwargs.items() if v is not None}
#     return call_main_catch_old(typeprob,
#                                **kwargs)
#
#
# def call_main_catch_old(typeprob, optimum_deviation=0.1, n_resampling=80, iter_psearch=50, **kwargs):
#     return FRIBase(typeprob, w1_l1_slack=optimum_deviation,
#                    n_probe_features=n_resampling,
#                    n_param_search=iter_psearch,
#                    **kwargs)
