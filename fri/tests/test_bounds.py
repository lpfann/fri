""" Test low level bound functions
"""
import numpy as np
import pytest
from cvxpy import OPTIMAL
from pytest import approx
from sklearn.utils import check_random_state

from fri import FRIClassification, FRIRegression, FRIOrdinalRegression
from fri.bounds import LowerBound, UpperBound
from fri.l1models import L1HingeHyperplane, L1EpsilonRegressor, L1OrdinalRegressor


@pytest.fixture(scope="function")
def randomstate():
    """ pytest fixture providing numpy randomstate

    Returns:
         numpy randomstate
    """
    return check_random_state(1337)


class TestClassifBounds(object):
    def test_bounds(randomstate):
        """Test bound functions using very easy data with perfect separation
        
        Args:
            randomstate  pytest fixture providing numpy randomstate:
        """
        n_samples = 4
        n_features = 2
        X = [
            [1, 0],
            [2, 0],
            [-1, 0],
            [-2, 0],
        ]
        X = np.asarray(X)
        y = np.asarray([1, 1, -1, -1])

        C = 1
        l1init = L1HingeHyperplane(C=C)
        l1init.fit(X, y)

        bias = l1init.intercept_
        coef = l1init.coef_[0]
        L1 = np.linalg.norm(coef, ord=1)
        loss = np.abs(l1init.slack).sum()

        # Test the init parameters
        assert bias == approx(0, rel=1e-3, abs=1e-3)
        assert L1 == approx(1, rel=1e-3, abs=1e-3)
        assert loss == approx(0, rel=1e-3, abs=1e-3)
        assert abs(coef[0]) > abs(coef[1])
        assert coef[0] == approx(1, rel=1e-3, abs=1e-3)
        assert coef[1] == approx(0, rel=1e-3, abs=1e-3)

        kwargs = {"verbose": False, "solver": "ECOS", "max_iters": 1000}
        
        model = FRIClassification()
        model._best_params = {"C":C}

        for current_dim in range(n_features):
            for b in (LowerBound, UpperBound):
                bound = b(model, current_dim, kwargs, 
                          loss, L1, X, y)
                bound.solve()

                prob = bound.prob_instance.problem
                assert prob.status == OPTIMAL
                assert prob.value - abs(coef[current_dim]) <= 0.1
                assert bound.prob_instance.loss.value <= max(0.01,
                                                             loss * 1.001)  # add a constant factor to handle numerical instabilities
                assert bound.prob_instance.weight_norm.value <= L1 * 1.00001

    def test_bounds_twoMiss(randomstate):
        """ Test bound functions using data with two missclassifications
        
        Args:
            randomstate pytest fixture providing numpy randomstate:
        """
        n_samples = 6
        n_features = 2
        X = [
            [1, 0],
            [2, 0],
            [-1, 0],
            [-2, 0],
            [1, 0],  # Missclassified
            [-1, 0],  # Missclassified
        ]
        X = np.asarray(X)
        y = np.asarray([1, 1, -1, -1, -1, 1])

        C = 1
        l1init = L1HingeHyperplane(C=C)
        l1init.fit(X, y)

        bias = l1init.intercept_
        coef = l1init.coef_[0]
        L1 = np.linalg.norm(coef, ord=1)
        loss = np.abs(np.maximum(0, l1init.slack)).sum()

        # Test the init parameters
        assert bias == approx(0, rel=1e-3, abs=1e-3)
        assert L1 == approx(0.5, rel=1e-3, abs=1e-3)
        assert loss == approx(4, rel=1e-3, abs=1e-3)
        assert abs(coef[0]) > abs(coef[1])
        assert coef[0] == approx(0.5, rel=1e-3, abs=1e-3)
        assert coef[1] == approx(0, rel=1e-3, abs=1e-3)

        kwargs = {"verbose": False, "solver": "ECOS", "max_iters": 1000}

        model = FRIClassification()
        model._best_params = {"C":C}

        for current_dim in range(n_features):
            for b in (LowerBound, UpperBound):
                bound = b(model, current_dim, kwargs, 
                          loss, L1, X, y)
                bound.solve()

                prob = bound.prob_instance.problem
                assert prob.status == OPTIMAL
                assert prob.value - abs(coef[current_dim]) <= 0.1
                assert bound.prob_instance.loss.value <= max(0.011,
                                                             loss * 1.001)  # add a constant factor to handle numerical instabilities
                assert bound.prob_instance.weight_norm.value <= L1 * 1.00001


class TestRegressionBounds(object):
    def test_bounds(randomstate):
        """Test bound functions using very easy data without noise
        
        Args:
            randomstate  pytest fixture providing numpy randomstate:
        """
        n_samples = 5
        n_features = 2
        X = [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
        ]
        X = np.asarray(X)
        y = np.asarray([0, 1, 2, 3, 4])

        C = 1
        epsilon = 0
        l1init = L1EpsilonRegressor(C=C, epsilon=epsilon)
        l1init.fit(X, y)

        bias = l1init.intercept_
        coef = l1init.coef_[0]
        L1 = np.linalg.norm(coef, ord=1)
        loss = np.abs(l1init.slack).sum()

        # Test the init parameters
        assert bias == approx(0, rel=1e-3, abs=1e-3)
        assert L1 == approx(1, rel=1e-3, abs=1e-3)
        assert loss == approx(0, rel=1e-3, abs=1e-3)
        assert abs(coef[0]) > abs(coef[1])
        assert coef[0] == approx(1, rel=1e-3, abs=1e-3)
        assert coef[1] == approx(0, rel=1e-3, abs=1e-3)

        model = FRIRegression()
        model._best_params = {"C":C,"epsilon":epsilon}

        kwargs = {"verbose": False, "solver": "ECOS", "max_iters": 1000}
        for current_dim in range(n_features):
            for b in (LowerBound, UpperBound):
                bound = b(model, current_dim, kwargs, 
                          loss, L1, X, y)
                bound.solve()

                prob = bound.prob_instance.problem
                assert prob.status == OPTIMAL
                assert prob.value - abs(coef[current_dim]) <= 0.1
                assert bound.prob_instance.loss.value <= max(0.01,
                                                             loss * 1.001)  # add a constant factor to handle numerical instabilities
                assert bound.prob_instance.weight_norm.value <= L1 * 1.00001

    def test_bounds_twoMiss(randomstate):
        """ Test bound functions using data with two noisy samples
        
        Args:
            randomstate pytest fixture providing numpy randomstate:
        """
        n_samples = 5
        n_features = 2
        X = [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
        ]
        X = np.asarray(X)
        y = np.asarray([0, -1, 2, -3, 4])

        C = 1
        epsilon = 0
        l1init = L1EpsilonRegressor(C=C, epsilon=epsilon)
        l1init.fit(X, y)

        bias = l1init.intercept_
        coef = l1init.coef_[0]
        L1 = np.linalg.norm(coef, ord=1)
        loss = np.abs(l1init.slack).sum()

        # Test the init parameters
        assert bias == approx(0, rel=1e-3, abs=1e-3)
        assert L1 == approx(1, rel=1e-3, abs=1e-3)
        assert loss == approx(8, rel=1e-3, abs=1e-3)
        assert abs(coef[0]) > abs(coef[1])
        assert coef[0] == approx(1, rel=1e-3, abs=1e-3)
        assert coef[1] == approx(0, rel=1e-3, abs=1e-3)

        model = FRIRegression()
        model._best_params = {"C":C,"epsilon":epsilon}

        kwargs = {"verbose": False, "solver": "ECOS", "max_iters": 1000}
        for current_dim in range(n_features):
            for b in (LowerBound, UpperBound):
                bound = b(model, current_dim, kwargs, 
                          loss, L1, X, y)
                bound.solve()

                prob = bound.prob_instance.problem
                assert prob.status == OPTIMAL
                assert prob.value - abs(coef[current_dim]) <= 0.1
                assert bound.prob_instance.loss.value <= max(0.01,
                                                             loss * 1.001)  # add a constant factor to handle numerical instabilities
                assert bound.prob_instance.weight_norm.value <= L1 * 1.00001


class TestOrdinalRegressionBounds(object):
    def test_bounds(randomstate):
        """Test bound functions using very easy data without noise

        Args:
            randomstate  pytest fixture providing numpy randomstate:
        """
        n_samples = 6
        n_features = 2
        X = [
            [-1, 0],
            [-1, 1],
            [-1, 2],
            [1, 0],
            [1, 1],
            [1, 2],
        ]
        X = np.asarray(X)
        y = np.asarray([0, 0, 0, 1, 1, 1])

        C = 1
        l1init = L1OrdinalRegressor(C=C)
        l1init.fit(X, y)

        bias = l1init.intercept_
        coef = l1init.coef_[0]
        L1 = np.linalg.norm(coef, ord=1)
        loss = np.abs(l1init.slack).sum()

        # Test the init parameters
        assert bias == approx(0, abs=1e-9)
        assert L1 == approx(1)
        assert loss == approx(0, abs=1e-7)
        assert abs(coef[0]) > abs(coef[1])
        assert coef[0] == approx(1)
        assert coef[1] == approx(0)

        model = FRIOrdinalRegression()
        model._best_params = {"C": C}

        kwargs = {"verbose": False, "solver": "ECOS", "max_iters": 1000}
        for current_dim in range(n_features):
            for b in (LowerBound, UpperBound):
                bound = b(model, current_dim, kwargs,
                          loss, L1, X, y)
                bound.solve()

                prob = bound.prob_instance.problem
                assert prob.status == OPTIMAL
                assert prob.value - abs(coef[current_dim]) <= 0.1
                assert bound.prob_instance.loss.value <= max(0.011,
                                                             loss * 1.001)  # add a constant factor to handle numerical instabilities
                assert bound.prob_instance.weight_norm.value <= L1 * 1.00001




    def test_bounds_twoMiss(randomstate):
        """Test bound functions using very easy data with noise

        Args:
            randomstate  pytest fixture providing numpy randomstate:
        """
        n_samples = 8
        n_features = 2
        X = [
            [-1, 0],
            [-1, 1],
            [-1, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [-1, 0],
            [1, 0],
        ]
        X = np.asarray(X)
        y = np.asarray([0, 0, 0, 1, 1, 1, 1, 0])

        C = 1
        l1init = L1OrdinalRegressor(C=C)
        l1init.fit(X, y)

        bias = l1init.intercept_
        coef = l1init.coef_[0]
        L1 = np.linalg.norm(coef, ord=1)
        loss = np.abs(l1init.slack).sum()

        # Test the init parameters
        assert bias == approx(0, abs=1e-9)
        assert L1 == approx(1)
        assert loss == approx(4, abs=1e-7)
        assert abs(coef[0]) > abs(coef[1])
        assert coef[0] == approx(1)
        assert coef[1] == approx(0)

        model = FRIOrdinalRegression()
        model._best_params = {"C": C}

        kwargs = {"verbose": False, "solver": "ECOS", "max_iters": 1000}
        for current_dim in range(n_features):
            for b in (LowerBound, UpperBound):
                bound = b(model, current_dim, kwargs,
                          loss, L1, X, y)
                bound.solve()

                prob = bound.prob_instance.problem
                assert prob.status == OPTIMAL
                assert prob.value - abs(coef[current_dim]) <= 0.1
                assert bound.prob_instance.loss.value <= max(0.011,
                                                             loss * 1.001)  # add a constant factor to handle numerical instabilities
                assert bound.prob_instance.weight_norm.value <= L1 * 1.00001