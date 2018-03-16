"""
 FRI module for inferring relevance intervals for linear classification and regression data
"""
from .classification import FRIClassification
from .ensemble import EnsembleFRI
from .genData import genRegressionData, genClassificationData
from .ordinalregression import FRIOrdinalRegression
from .plot import plotIntervals
from .plot import plot_dendrogram_and_intervals
from .regression import FRIRegression

__all__ = ["genRegressionData", "genClassificationData",
           "plotIntervals", "EnsembleFRI", "FRIClassification", "FRIRegression", "FRIOrdinalRegression"]

# Get version from versioneer
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
