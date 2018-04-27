"""
 FRI module for inferring relevance intervals for linear classification and regression data
"""
from .classification import FRIClassification
from .genData import genRegressionData, genClassificationData
from .plot import plotIntervals
from .regression import FRIRegression

__all__ = ["genRegressionData", "genClassificationData",
           "plotIntervals", "FRIClassification", "FRIRegression"]

# Get version from versioneer
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
