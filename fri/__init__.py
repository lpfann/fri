"""
 FRI module for inferring relevance intervals for linear classification and regression data
"""
import warnings
from enum import Enum

from fri.genData import genRegressionData, genClassificationData, genOrdinalRegressionData, quick_generate
from fri.main import FRIBase
from fri.model.classification import Classification
from fri.model.lupi_classification import LUPI_Classification
from fri.model.lupi_regression import LUPI_Regression
from fri.model.ordinal_regression import OrdinalRegression
from fri.model.regression import Regression
from fri.plot import plot_intervals


class ProblemType(Enum):
    CLASSIFICATION = Classification
    REGRESSION = Regression
    ORDINALREGRESSION = OrdinalRegression
    LUPI_CLASSIFICATION = LUPI_Classification
    LUPI_REGRESSION = LUPI_Regression


__all__ = ["genRegressionData", "genClassificationData", "genOrdinalRegressionData", "quick_generate",
           "FRIClassification", "FRIRegression", "FRIOrdinalRegression", "plot_intervals", ProblemType]

# Get version from versioneer
from fri._version import get_versions
__version__ = get_versions()['version']
del get_versions


def FRI(problem: ProblemType, random_state=None, n_jobs=1, verbose=0, n_param_search=50,
        n_probe_features=50, **kwargs):
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
            problemtype = Regression
        elif problem == "ordinalregression" or problem == "ordreg":
            problemtype = OrdinalRegression
        elif problem == "lupi_classification" or problem == "lupi_class":
            problemtype = LUPI_Classification
        else:
            names = [enum.name.lower() for enum in ProblemType]

            print(f"Parameter 'problem' was not recognized or unset. Try one of {names}.")
            return None
    return FRIBase(problemtype, random_state=random_state, n_jobs=n_jobs, verbose=verbose,
                   n_param_search=n_param_search,
                   n_probe_features=n_probe_features, **kwargs)



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
