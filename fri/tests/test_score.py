import pytest
import numpy as np

from pytest import approx
from fri.l1models import ordinal_scores as score

@pytest.fixture()
def data():
    y = np.array([0, 1, 2, 3])
    return y

@pytest.fixture()
def imb_data():
    y = np.array([0, 1, 2, 2])
    return y

def reverse_label(y):
    y = np.copy(y)
    return np.flip(y[:],axis=0)

def swap_first_last(y):
    y = np.copy(y)
    y[[0,-1]] = y[[-1,0]]
    return y

@pytest.mark.parametrize('error', ["mze", "mae", "mmae"])
def test_ordinal_score(error, data, imb_data):

    score_perfect = score(data, data, error_type=error)
    score_mixed = score(data, swap_first_last(data), error_type=error)
    score_worst = score(data, reverse_label(data), error_type=error)

    assert score_perfect > score_mixed
    assert score_mixed > score_worst

    assert score_perfect == 1
