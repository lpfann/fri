from .fri import (FRIClassification,FRIRegression,EnsembleFRI)
from . import fri
from . import optproblems
from . import genData
from . import bounds
from . import plot
__all__ = ['FRIClassification', 'FRIRegression','fri.py','EnsembleFRI',"plot"]

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
