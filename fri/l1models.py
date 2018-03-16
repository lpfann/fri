""" 
    Class housing all initialisation models used in fri
    We use the models paramaters to calculate the loss and L1 bounds.
"""
import cvxpy as cvx
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, RegressorMixin, LinearModel
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import check_X_y


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
        xi = cvx.Variable(n, nonneg=True)
        b = cvx.Variable()

        # Prepare problem.
        objective = cvx.Minimize(cvx.norm(w, 1) + self.C * cvx.sum(xi))
        constraints = [
            cvx.multiply(y.T, X * w - b) >= 1 - xi
        ]
        # Solve problem.
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="ECOS", max_iters=5000)

        # Prepare output and convert from matrices to flattened arrays.
        self.coef_ = np.array(w.value)[np.newaxis]
        self.intercept_ = b.value
        self.slack = np.asarray(xi.value).flatten()

        return self

    def score(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Negative class is set to -1 for decision surface
        y = LabelEncoder().fit_transform(y)
        y[y == 0] = -1

        X = StandardScaler().fit_transform(X)
        prediction = self.predict(X)
        # Using weighted f1 score to have a stable score for imbalanced datasets
        score = f1_score(y, prediction, average="weighted")
        return score

class L1EpsilonRegressor(LinearModel, RegressorMixin):
    """
    Determine a L1 regularized regression hyperplane with linear loss function
    """

    def __init__(self, C=1, epsilon=1):
        self.C = C
        self.epsilon = epsilon

    def fit(self, X, y):
        (n, d) = X.shape
        w = cvx.Variable(d)
        xi = cvx.Variable(n, nonneg=True)
        b = cvx.Variable()

        # Prepare problem.
        objective = cvx.Minimize(cvx.norm(w, 1) + self.C * cvx.sum(xi))
        constraints = [
            cvx.abs(y - (X * w + b)) <= self.epsilon + xi
        ]
        # Solve problem.
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="ECOS", max_iters=5000)

        self.coef_ = np.array(w.value)[np.newaxis]
        self.intercept_ = b.value
        self.slack = np.asarray(xi.value).flatten()

        return self

class L1OrdinalRegressor(LinearModel):

    #TODO: Connect error_type in higher levels
    # TODO: score mze hat intuitiv richtigen score, die anderen beiden nicht?
    def __init__(self, C=1, error_type="mze"):
        self.C = C
        self.error_type = error_type
        self.coef_ = None
        self.intercept_ = None
        self.slack = None


    def fit(self, X, y):

        (n, d) = X.shape
        n_bins = len(np.unique(y))

        w = cvx.Variable(d)
        b = cvx.Variable(n_bins - 1)
        chi = cvx.Variable(n, nonneg=True)
        xi = cvx.Variable(n, nonneg=True)

        # Prepare problem.
        objective = cvx.Minimize(0.5 * cvx.norm(w, 1) + self.C * cvx.sum(chi + xi))
        constraints = []

        for i in range(n_bins - 1):
            indices = np.where(y == i)
            constraints.append(X[indices] * w - chi[indices] <= b[i] - 1)

        for i in range(1, n_bins):
            indices = np.where(y == i)
            constraints.append(X[indices] * w + xi[indices] >= b[i - 1] + 1)

        for i in range(n_bins - 2):
            constraints.append(b[i] <= b[i + 1])


        # Solve problem.
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="ECOS", max_iters=5000)

        self.coef_ = np.array(w.value)[np.newaxis]
        self.intercept_ = np.array(b.value).flatten()
        self.slack = np.append(chi.value, xi.value)


        return self

    def score(self, X, y):

        X, y = check_X_y(X, y)

        (n, d) = X.shape
        n_bins = len(np.unique(y))
        w = self.coef_[0]
        b = np.append(self.intercept_, np.inf)
        sum = 0

        # Score based on mean zero-one error
        if self.error_type == "mze":
            for i in range(n):
                val = np.matmul(w, X[i])
                pos = np.argmax(np.less_equal(val, b))
                if y[i] != pos:
                    sum += 1

            score = 1 - (sum / n)

        # Score based on mean absolute error
        elif self.error_type == "mae":
            for i in range(n):
                val = np.matmul(w, X[i])
                sum += np.abs(y[i] - np.argmax(np.less_equal(val, b)))

            error = sum / n

            # TODO: Check how the error has to be scaled to transform it to a adequate score in case of n_bins == 1
            if n_bins == 1:
                score = 0
            else:
                score = 1 - (error / (n_bins - 1))

        # Score based on macro-averaged mean absolute error
        elif self.error_type == "mmae":
            for i in range(n_bins):
                indices = np.where(y == i)
                X_re = X[indices]
                y_re = y[indices]
                n_c = X_re.shape[0]

                if n_c == 0:
                    error_c = 0

                else:
                    sum_c = 0

                    for j in range(n_c):
                        val = np.matmul(w, X_re[j])
                        sum_c += np.abs(y_re[j] - np.argmax(np.less_equal(val, b)))

                    error_c = sum_c / n_c

                sum += error_c

            error = sum / n_bins

            #TODO: Check how the error has to be scaled to transform it to a adequate score in case of n_bins == 1
            if n_bins == 1:
                score = 0
            else:
                score = 1 - (error / (n_bins - 1))

        # error message if no correct error type has been specified
        else:
            #TODO: Send Error message (wrong input on 'error_type')
            pass

        return score



