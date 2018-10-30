import pytest
import numpy as np

from pytest import approx
from fri.l1models import L1OrdinalRegressor

@pytest.fixture()
def data():
    X = np.array([[1, 1], [2, 2], [3, 3],[4, 4] ])
    y = np.array([0, 1, 2, 3])
    return X, y

@pytest.fixture()
def imb_data():
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4] ])
    y = np.array([0, 1, 2, 2])
    return X, y

def reverse_label(X,y):
    return X, np.flip(y,axis=0)

def swap_first_last(X,y):
    X[[0,-1]] = X[[-1,0]]
    return X, y


def test_mze_score(data):
    error = "mze"

    model = L1OrdinalRegressor()
    model.fit(*data)

    score = model.score(*data, error_type=error)
    assert score == approx(1)

    score = model.score(*reverse_label(*data), error_type=error)
    assert score == approx(0)

    score = model.score(*swap_first_last(*data), error_type=error)
    assert score == approx(0.5)


def test_mae_score(data):
    error = "mae"

    model = L1OrdinalRegressor()
    model.fit(*data)

    score = model.score(*data, error_type=error)
    assert score == approx(1)

    score = model.score(*reverse_label(*data), error_type=error)
    assert score == approx(1/3)

    score = model.score(*swap_first_last(*data), error_type=error)
    assert score == approx(0.5)



def test_mmae_score(data,imb_data):
    error = "mmae"

    model = L1OrdinalRegressor()
    model.fit(*data)

    score = model.score(*data, error_type=error)
    assert score == approx(1)

    score = model.score(*reverse_label(*data), error_type=error)
    assert score == approx(1/3)

    score = model.score(*swap_first_last(*data), error_type=error)
    assert score == approx(0.5)


    score = model.score(*imb_data, error_type=error)
    assert score == approx(0.95833334) # TODO: richtiger score??