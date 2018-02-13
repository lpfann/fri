"""
 FRI module for inferring relevance intervals for linear classification and regression data
"""
from .ensemble import EnsembleFRI
from .regression import FRIRegression
from .classification import FRIClassification
from .genData import genRegressionData, genClassificationData
from .plot import plotIntervals
from .plot import plot_dendrogram_and_intervals

__all__ = ["genRegressionData", "genClassificationData",
           "plotIntervals","EnsembleFRI","FRIClassification","FRIRegression"]

# Get version from versioneer
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
