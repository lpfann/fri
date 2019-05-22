from unittest import TestCase

import numpy as np

from .classification import Classification
from .fri import FRIBase


class TestFRIBase(TestCase):
    def test_fit(self):
        typeprob = Classification
        frib = FRIBase(typeprob, n_jobs=1)

        X = np.random.normal(size=(50, 2))
        d = X.shape[1]
        y = X[:, 0] > 0

        frib.fit(X, np.asarray(y))

        ranges = frib.interval_
        assert ranges.shape == (d, 2)
