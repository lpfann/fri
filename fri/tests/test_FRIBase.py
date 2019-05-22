from unittest import TestCase

import numpy as np

from fri import FRIBase
from fri.model import classification


class TestFRIBase(TestCase):
    def test_fit(self):
        typeprob = classification.Classification
        frib = FRIBase(typeprob, n_jobs=1)

        X = np.random.normal(size=(50, 2))
        d = X.shape[1]
        y = X[:, 0] > 0

        frib.fit(X, np.asarray(y))

        ranges = frib.interval_
        assert ranges.shape == (d, 2)
