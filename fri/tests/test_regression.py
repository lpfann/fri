import unittest

import pytest

import fri.fri
from fri.fri import FRIRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLasso
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils import check_random_state
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_greater, assert_equal, assert_true,assert_false
from numpy.testing import assert_array_almost_equal, assert_array_equal,assert_raises
import numpy as np
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import StandardScaler
from  sklearn.exceptions import FitFailedWarning

@pytest.fixture(scope="module")
def randomstate():
   return check_random_state(1337)

@pytest.fixture(scope="module",
                params=[(1,0),(2,0),(0,2),(1,2)],
                ids=["1strong","2strong","weak","allrele"])
def data(request,randomstate):
    strong = request.param[0]
    weak = request.param[1]
    generator = randomstate
    data = fri.genData.genRegressionData(n_samples=1000, n_features=4, n_redundant=weak, strRel=strong,
                                         n_repeated=0, random_state=generator)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)
    X = X_orig
    # X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
    # y = list(y)
    return X,y,strong,weak


def test_simpleRegression(data,randomstate):
    X, y, strong, weak = data

    # Test using the score function
    rbc = FRIRegression(random_state=randomstate, shadow_features=False, C=1, epsilon=0.1)
    try:
        rbc.fit(X, y)
    except FitFailedWarning or NotFeasibleForParameters:
        print(rbc._best_clf_score,rbc._hyper_C,rbc._hyper_epsilon)
        assert False

    assert_equal(len(rbc.allrel_prediction_), X.shape[1])
    assert_equal(len(rbc.interval_), X.shape[1])

    X_r = rbc.transform(X)
    print(rbc._best_clf_score)
    print(rbc.interval_,rbc.allrel_prediction_,rbc._hyper_C,rbc._hyper_epsilon)

    # All strongly relevant features have a lower bound > 0
    assert np.all(rbc.interval_[0:strong,0] > 0)
    assert np.all(rbc.interval_[strong:weak,0] == 0)
    # Upper bound checks
    assert np.all(rbc.interval_[0:strong,1] > 0)
    assert np.all(rbc.interval_[0:strong,1] > rbc.interval_[0:strong,0])

    # Test prediction
    truth = np.zeros(X.shape[1],dtype=bool)
    truth[:strong+weak] = 1
    assert np.all(rbc.allrel_prediction_ == truth)
#
# def test_simple(randomstate):
#     generator = randomstate
#     data = fri.genData.genRegressionData(n_samples=300, n_features=1, n_redundant=0, strRel=1,
#                                                   n_repeated=0, random_state=generator)
#
#     X_orig, y = data
#     X_orig = StandardScaler().fit(X_orig).transform(X_orig)
#     X = X_orig
#
#
#     # Test using the score function
#     rbc = FRIRegression(random_state=randomstate, shadow_features=False)
#     try:
#         rbc.fit(X, y)
#     except FitFailedWarning:
#         print(rbc._best_clf_score)
#         assert False
#
#     assert_equal(len(rbc.allrel_prediction_), X.shape[1])
#     assert_equal(len(rbc.interval_), X.shape[1])
#
#     X_r = rbc.transform(X)
#     print(rbc.interval_,rbc.allrel_prediction_,rbc._hyper_C,rbc._hyper_epsilon)
#
#     # All strongly relevant features have a lower bound > 0
#     assert rbc.interval_[0,0] > 0

# def test_simpleRegression2Strong():
#     strong = 2
#     weak = 0
#     generator = check_random_state(0)
#     data = fri.genData.genRegressionData(n_samples=100, n_features=4, n_redundant=weak, strRel=strong,
#                                                   n_repeated=0, random_state=generator)
#
#     X_orig, y = data
#     X_orig = StandardScaler().fit(X_orig).transform(X_orig)
#     X = X_orig
#     # X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
#     # y = list(y)
#
#     # Test using the score function
#     rbc = FRIRegression( random_state=generator, shadow_features=False)
#     rbc.fit(X, y)
#
#     assert_equal(len(rbc.allrel_prediction_), X.shape[1])
#     assert_equal(len(rbc.interval_), X.shape[1])
#
#     X_r = rbc.transform(X)
#     print(rbc.interval_, rbc.allrel_prediction_)
#
#     # All the noisy variable were filtered out
#     assert_array_equal(X_r, X_orig)
#
#     # All strongly relevant features have a lower bound > 0
#     assert_true(np.all(rbc.interval_[0:strong, 0] > 0))
#
#
# def test_simpleRegressionWeak():
#         strong = 0
#         weak = 2
#         generator = check_random_state(0)
#         data = fri.genData.genRegressionData(n_samples=100, n_features=4, n_redundant=weak,strRel=strong,
#                         n_repeated=0, random_state=generator)
#
#         X_orig, y = data
#         X_orig = StandardScaler().fit(X_orig).transform(X_orig)
#         X = X_orig
#         # X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
#         # y = list(y)
#
#         # Test using the score function
#         rbc = FRIRegression(random_state=generator, shadow_features=False)
#         rbc.fit(X, y)
#
#         assert_equal(len(rbc.allrel_prediction_), X.shape[1])
#         assert_equal(len(rbc.interval_), X.shape[1])
#
#         X_r = rbc.transform(X)
#         print(rbc.interval_,rbc.allrel_prediction_)
#
#         # All the noisy variable were filtered out
#         assert_array_equal(X_r, X_orig)
#
#         # All strongly relevant features have a lower bound > 0
#         assert_false(np.any(rbc.interval_[0:weak, 0] > 0))
#
#
# def test_simpleRegressionAllRelevant():
#     strong = 1
#     weak = 2
#     generator = check_random_state(0)
#     data = fri.genData.genRegressionData(n_samples=100, n_features=4, n_redundant=weak, strRel=strong,
#                                                   n_repeated=0, random_state=generator)
#
#     X_orig, y = data
#     X_orig = StandardScaler().fit(X_orig).transform(X_orig)
#     X = X_orig
#     # X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
#     # y = list(y)
#
#     # Test using the score function
#     rbc = FRIRegression(C=1, random_state=generator, shadow_features=False)
#     rbc.fit(X, y)
#
#     assert_equal(len(rbc.allrel_prediction_), X.shape[1])
#     assert_equal(len(rbc.interval_), X.shape[1])
#
#     X_r = rbc.transform(X)
#     print(rbc.interval_, rbc.allrel_prediction_)
#
#     # All the noisy variable were filtered out
#     assert_array_equal(X_r, X_orig)
#
#     # All strongly relevant features have a lower bound > 0
#     assert_true(np.all(rbc.interval_[0:strong,0]>0))
#     # All weakly relevant features should have a lower bound 0
#     assert_false(np.any(rbc.interval_[strong:weak,0]>0))
#
