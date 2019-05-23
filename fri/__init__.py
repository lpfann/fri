"""
 FRI module for inferring relevance intervals for linear classification and regression data
"""
from enum import Enum

from .genData import genRegressionData, genClassificationData, genOrdinalRegressionData
from .main import FRIBase
from .model.classification import Classification
from .model.ordinal_regression import OrdinalRegression
from .model.regression import Regression
from .plot import plot_intervals

__all__ = ["genRegressionData", "genClassificationData", "genOrdinalRegressionData",
           "FRIClassification", "FRIRegression", "FRIOrdinalRegression", "plot_intervals"]

# Get version from versioneer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import warnings


class ProblemType(Enum):
    CLASSIFICATION = Classification
    REGRESSION = Regression
    ORDINALREGRESSION = OrdinalRegression


def FRI(problem: ProblemType, **kwargs):
    """

    Parameters
    ----------
    problem : ProblemType or str
    Type of problem at hand.
    E.g. "classification", "regression", "ordinalregression"
    """

    if isinstance(problem, ProblemType):
        problemtype = problem.value
    else:
        if problem == "classification" or problem == "class":
            problemtype = Classification
        elif problem == "regression" or problem == "reg":
            problemtype = Classification
        elif problem == "ordinalregression" or problem == "ordreg":
            problemtype = OrdinalRegression
        else:
            names = [enum.name.lower() for enum in ProblemType]

            print(f"Parameter 'problem' was not recognized or unset. Try one of {names}.")
            return None
    return FRIBase(problemtype, **kwargs)



def FRIClassification(**kwargs):
    warnings.warn(
        "This class call format is deprecated.",
        DeprecationWarning
    )

    typeprob = Classification
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return call_main_catch_old(typeprob,
                               **kwargs)


def FRIRegression(**kwargs):
    warnings.warn(
        "This class call format is deprecated.",
        DeprecationWarning
    )

    typeprob = Regression
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return call_main_catch_old(typeprob,
                               **kwargs)


def FRIOrdinalRegression(**kwargs):
    warnings.warn(
        "This class call format is deprecated.",
        DeprecationWarning
    )
    typeprob = OrdinalRegression
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return call_main_catch_old(typeprob,
                               **kwargs)


def call_main_catch_old(typeprob, optimum_deviation=0.001, n_resampling=40, iter_psearch=30, **kwargs):
    return FRIBase(typeprob, w1_l1_slack=optimum_deviation,
                   n_probe_features=n_resampling,
                   n_param_search=iter_psearch,
                   **kwargs)
