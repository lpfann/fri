import pytest
import numpy as np

from pytest import approx
from fri.l1models import L1OrdinalRegressor

@pytest.mark.parametrize('error_type', ["mze", "mae", "mmae"])
def test_score_max(error_type):

    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [4, 1], [4, 3], [4, 2], [5, 1]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = L1OrdinalRegressor(error_type=error_type)

    model.fit(X, y)

    score = model.score(X, y)

    assert score == approx(1)



@pytest.mark.parametrize('error_type', ["mze", "mae", "mmae"])
def test_score_low(error_type):

    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [4, 1], [4, 3], [4, 2], [5, 1]])
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1])

    model = L1OrdinalRegressor(error_type=error_type)

    model.fit(X, y)

    score = model.score(X, y)

    assert score == approx(0.75)