import unittest

import rbclassifier.rbclassifier
from rbclassifier.rbclassifier import RelevanceBoundsRegressor
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

class TestRegression(unittest.TestCase):
    def test_simpleRegression1Strong(self):

        strong = 1
        weak = 0
        generator = check_random_state(0)
        data = rbclassifier.genData.genRegressionData(n_samples=100, n_features=4, n_redundant=weak,strRel=strong,
                        n_repeated=0, random_state=generator)

        X_orig, y = data
        X_orig = StandardScaler().fit(X_orig).transform(X_orig)
        X = X_orig
        # X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
        # y = list(y)

        # Test using the score function
        rbc = RelevanceBoundsRegressor(random_state=generator, shadow_features=False)
        rbc.fit(X, y)

        assert_equal(len(rbc.allrel_prediction_), X.shape[1])
        assert_equal(len(rbc.interval_), X.shape[1])

        X_r = rbc.transform(X)
        print(rbc.interval_,rbc.allrel_prediction_)

        # All the noisy variable were filtered out
        assert_array_equal(X_r, X_orig)

        # All strongly relevant features have a lower bound > 0
        assert_true(np.all(rbc.interval_[0:strong,0]>0))


    def test_simpleRegression2Strong(self):
        strong = 2
        weak = 0
        generator = check_random_state(0)
        data = rbclassifier.genData.genRegressionData(n_samples=100, n_features=4, n_redundant=weak, strRel=strong,
                                                      n_repeated=0, random_state=generator)

        X_orig, y = data
        X_orig = StandardScaler().fit(X_orig).transform(X_orig)
        X = X_orig
        # X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
        # y = list(y)

        # Test using the score function
        rbc = RelevanceBoundsRegressor(C=1, random_state=generator, shadow_features=False)
        rbc.fit(X, y)

        assert_equal(len(rbc.allrel_prediction_), X.shape[1])
        assert_equal(len(rbc.interval_), X.shape[1])

        X_r = rbc.transform(X)
        print(rbc.interval_, rbc.allrel_prediction_)

        # All the noisy variable were filtered out
        assert_array_equal(X_r, X_orig)

        # All strongly relevant features have a lower bound > 0
        assert_true(np.all(rbc.interval_[0:strong, 0] > 0))


    def test_simpleRegressionWeak(self):
            strong = 0
            weak = 2
            generator = check_random_state(0)
            data = rbclassifier.genData.genRegressionData(n_samples=100, n_features=4, n_redundant=weak,strRel=strong,
                            n_repeated=0, random_state=generator)

            X_orig, y = data
            X_orig = StandardScaler().fit(X_orig).transform(X_orig)
            X = X_orig
            # X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
            # y = list(y)

            # Test using the score function
            rbc = RelevanceBoundsRegressor(C=1,random_state=generator, shadow_features=False)
            rbc.fit(X, y)

            assert_equal(len(rbc.allrel_prediction_), X.shape[1])
            assert_equal(len(rbc.interval_), X.shape[1])

            X_r = rbc.transform(X)
            print(rbc.interval_,rbc.allrel_prediction_)

            # All the noisy variable were filtered out
            assert_array_equal(X_r, X_orig)

            # All strongly relevant features have a lower bound > 0
            assert_false(np.any(rbc.interval_[0:weak, 0] > 0))


    def test_simpleRegressionAllRelevant(self):
        strong = 1
        weak = 2
        generator = check_random_state(0)
        data = rbclassifier.genData.genRegressionData(n_samples=100, n_features=4, n_redundant=weak, strRel=strong,
                                                      n_repeated=0, random_state=generator)

        X_orig, y = data
        X_orig = StandardScaler().fit(X_orig).transform(X_orig)
        X = X_orig
        # X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
        # y = list(y)

        # Test using the score function
        rbc = RelevanceBoundsRegressor(C=1, random_state=generator, shadow_features=False)
        rbc.fit(X, y)

        assert_equal(len(rbc.allrel_prediction_), X.shape[1])
        assert_equal(len(rbc.interval_), X.shape[1])

        X_r = rbc.transform(X)
        print(rbc.interval_, rbc.allrel_prediction_)

        # All the noisy variable were filtered out
        assert_array_equal(X_r, X_orig)

        # All strongly relevant features have a lower bound > 0
        assert_true(np.all(rbc.interval_[0:strong,0]>0))
        # All weakly relevant features should have a lower bound 0
        assert_false(np.any(rbc.interval_[strong:weak,0]>0))


if __name__ == '__main__':
    unittest.main()
