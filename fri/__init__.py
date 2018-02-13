"""
 FRI module for inferring relevance intervals for linear classification and regression data
"""
from .fri import (FRIClassification, FRIRegression, EnsembleFRI)
from .genData import genRegressionData, genClassificationData
from .plot import plotIntervals
from .plot import plot_dendrogram_and_intervals

__all__ = ['FRIClassification', 'FRIRegression', 'EnsembleFRI', "genRegressionData", "genClassificationData",
           "plotIntervals"]

# Get version from versioneer
from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
