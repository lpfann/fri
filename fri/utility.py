import numpy as np
import cvxpy as cvx
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, RegressorMixin, LinearModel
from cvxpy.error import SolverError

class L1HingeHyperplane(BaseEstimator, LinearClassifierMixin):
    """
    Determine a separating hyperplane using L1-regularization and hinge loss
    """
    def __init__(self, C=1):
        self.C = C

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        (n, d) = X.shape
        w = cvx.Variable(d)
        xi = cvx.Variable(n)
        b = cvx.Variable()

        # Prepare problem.
        objective = cvx.Minimize(cvx.norm(w, 1) + self.C * cvx.sum(xi))
        constraints = [
            cvx.multiply(y.T, X * w - b) >= 1 - xi,
            xi >= 0
        ]
        # Solve problem.
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="ECOS")

        # Prepare output and convert from matrices to flattened arrays.
        self.coef_ = np.array(w.value)[np.newaxis]
        self.intercept_ = b.value
        self.slack = np.asarray(xi.value).flatten()

        return self

class L1EpsilonRegressor(LinearModel, RegressorMixin):
    """
    Determine a L1 regularized regression hyperplane with linear loss function
    """
    def __init__(self, C = 1, epsilon = 0):
        self.C = C
        self.epsilon = epsilon
        
    def fit(self, X, y):
        
        (n, d) = X.shape
        w = cvx.Variable(d)
        xi = cvx.Variable(n)
        b = cvx.Variable()
        
        # Prepare problem.
        objective = cvx.Minimize(cvx.norm(w, 1) + self.C * cvx.sum(xi))
        constraints = [
            cvx.abs(y - (X * w + b )) <= self.epsilon + xi,
            xi >= 0
        ]
        # Solve problem.
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="ECOS")

        # Prepare output and convert from matrices to flattened arrays.
        self.coef_ = np.array(w.value)[np.newaxis]
        self.intercept_ = b.value
        self.slack = np.asarray(xi.value).flatten()

        return self