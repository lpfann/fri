from .fri import (FRIClassification)
from . import fri
from . import optproblems
from . import genData
from . import bounds
__all__ = ['FRIClassification', 'FRIRegression','fri.py']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
