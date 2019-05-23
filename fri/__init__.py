"""
 FRI module for inferring relevance intervals for linear classification and regression data
"""
from .genData import genRegressionData, genClassificationData, genOrdinalRegressionData
from .main import FRIBase
from .model.classification import Classification
from .model.regression import Regression
from .plot import plot_intervals

__all__ = ["genRegressionData", "genClassificationData", "genOrdinalRegressionData",
           "FRIClassification", "FRIRegression", "FRIOrdinalRegression", "plot_intervals"]

# Get version from versioneer
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions


def FRIClassification(**kwargs):
    typeprob = Classification
    return FRIBase(typeprob, **kwargs)


def FRIRegression(**kwargs):
    typeprob = Regression
    return FRIBase(typeprob, **kwargs)

#        typeprob = Regression
#        return FRIBase(typeprob, **kwargs)

def FRIOrdinalRegression(**kwargs):
    raise NotImplementedError
#        typeprob = OrdinalRegression
#        return FRIBase(typeprob, **kwargs)
