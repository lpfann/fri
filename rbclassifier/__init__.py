from .rbclassifier import (RelevanceBoundsClassifier)
from . import rbclassifier
from . import optproblems
from . import genData
from . import bounds
__all__ = ['RelevanceBoundsClassifier', 'rbclassifier']

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
