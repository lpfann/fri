""" 
    Class housing all initialisation models used in fri
    We use the models paramaters to calculate the loss and L1 bounds.
"""
import cvxpy as cvx
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, RegressorMixin, LinearModel
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import check_X_y,check_array
from sklearn.exceptions import NotFittedError

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

    def score(self, X, y, debug=False):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Negative class is set to -1 for decision surface
        y = LabelEncoder().fit_transform(y)
        y[y == 0] = -1

        X = StandardScaler().fit_transform(X)
        prediction = self.predict(X)
        # Using weighted f1 score to have a stable score for imbalanced datasets
        score = fbeta_score(y, prediction, beta=1, average="weighted")
        if debug:
            precision = precision_score(y, prediction)
            recall = recall_score(y, prediction)
            print("precision: {}, recall: {}".format(precision, recall))
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

    def __init__(self, error_type="mmae", C=1, l1_ratio=1):
        self.C = C
        self.l1_ratio = l1_ratio
        self.error_type = error_type
        self.coef_ = None
        self.intercept_ = None
        self.slack = None

    def fit(self, X, y):

        (n, d) = X.shape
        n_bins = len(np.unique(y))
        self.classes_ = np.unique(y)
        original_bins = sorted(np.unique(y))
        n_bins = len(original_bins)
        bins = np.arange(n_bins)
        get_old_bin = dict(zip(bins, original_bins))

        w = cvx.Variable(d)
        b = cvx.Variable(n_bins - 1)
        chi = cvx.Variable(n, nonneg=True)
        xi = cvx.Variable(n, nonneg=True)

        # Prepare problem.
        norm = self.l1_ratio * cvx.pnorm(w, 1) + (1 - self.l1_ratio) * cvx.pnorm(w, 2)**2
        objective = cvx.Minimize(norm + self.C * cvx.sum(chi + xi))
        constraints = []

        for i in range(n_bins - 1):
            indices = np.where(y == get_old_bin[i])
            constraints.append(X[indices] * w - chi[indices] <= b[i] - 1)

        for i in range(1, n_bins):
            indices = np.where(y == get_old_bin[i])
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

    def decision_function(self, X):
        """Compute predicted scores for samples in X
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        array, shape=(n_samples,)
        """
        if not hasattr(self, 'coef_') or self.coef_ is None:
            raise NotFittedError("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})

        X = check_array(X,)

        n_features = self.coef_.shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))

        scores = np.dot(X, self.coef_[0].T)[np.newaxis]
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        """Predict class labels for samples in X.
        Parameters
        ----------
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.
        Returns
        -------
        C : array, shape [n_samples]
            Predicted class label per sample.
        """
        bin_thresholds = np.append(self.intercept_, np.inf)

        # If thresholds are smaller than score they value belongs to bigger bin
        # after subtracting we check for positive elements
        indices = np.sum(self.decision_function(X).T - bin_thresholds >= 0, -1)

        return self.classes_[indices]

    def score(self, X, y, error_type="mmae"):

        X, y = check_X_y(X, y)

        prediction = self.predict(X)
        score = ordinal_scores(y, prediction, error_type)

        return score

def ordinal_scores(y, prediction, error_type, return_error=False):
        """Score function for ordinal problems.
        
        Parameters
        ----------
        y : target class vector
            Truth vector
        prediction : prediction class vector
            Predicted classes
        error_type : str
            Error type "mze","mae","mmae"
        return_error : bool, optional
            Return error (lower is better) or score (inverted, higher is better)
        
        Returns
        -------
        float
            Error or score depending on 'return_error'
        
        Raises
        ------
        ValueError
            When using wrong error_type
        """
        n = len(y)
        classes = np.unique(y)
        n_bins = len(classes)
        max_dist = n_bins-1

        def mze(prediction, y):
            return np.sum(prediction != y)

        def mae(prediction, y):
            return np.sum(np.abs(prediction - y))

        # Score based on mean zero-one error
        if error_type == "mze":
            error = mze(prediction, y) / n
            score = 1 - error

        # Score based on mean absolute error
        elif error_type == "mae":
            error = mae(prediction, y) / n
            score = (max_dist-error) / max_dist

        # Score based on macro-averaged mean absolute error
        elif error_type == "mmae":
            sum = 0
            for i in range(n_bins):
                samples = y == i
                n_samples = np.sum(samples)
                if n_samples > 0:
                    bin_error = mae(prediction[samples],y[samples]) / n_samples
                    sum += bin_error

            error = sum / n_bins
            score = (max_dist-error)/max_dist
        else:
            raise ValueError("error_type {} not available!'".format(error_type))

        if return_error:
            return error
        else:
            return score

