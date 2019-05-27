"""
 FRI module for inferring relevance intervals for linear classification and regression data
"""
from .classification import FRIClassification
from .genData import genRegressionData, genClassificationData, genOrdinalRegressionData
from .ordinalregression import FRIOrdinalRegression
from .plot import plot_intervals
from .regression import FRIRegression

__all__ = ["genRegressionData", "genClassificationData", "genOrdinalRegressionData",
           "FRIClassification", "FRIRegression", "FRIOrdinalRegression", "plot_intervals"]

# Get version from versioneer
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
