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
    y = np.array([0, 1, 2, 2, 2, 2])
    return y

def reverse_label(y):
    y = np.copy(y)
    return np.flip(y[:],axis=0)

def swap_first_last(y):
    y = np.copy(y)
    y[[0,-1]] = y[[-1,0]]
    return y

@pytest.mark.parametrize('error', ["mze", "mae", "mmae"])
def test_ordinal_score(error, data):

    score_perfect = score(data, data, error_type=error)
    score_mixed = score(data, swap_first_last(data), error_type=error)
    score_worst = score(data, reverse_label(data), error_type=error)

    assert score_perfect > score_mixed
    assert score_mixed > score_worst

    assert score_perfect == 1

def test_score_imbalanced(data,imb_data):

    score_mae = score(data, swap_first_last(data), error_type="mae")
    score_mmae = score(data, swap_first_last(data), error_type="mmae")

    assert score_mae == score_mmae

    score_mae = score(imb_data, swap_first_last(imb_data), error_type="mae")
    score_mmae = score(imb_data, swap_first_last(imb_data), error_type="mmae")

    assert score_mae != score_mmae