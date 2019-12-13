import cvxpy as cvx
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y

from .base_cvxproblem import Relevance_CVXProblem
from .base_initmodel import InitModel
from .base_type import ProblemType


class OrdinalRegression(ProblemType):
    @classmethod
    def parameters(cls):
        return ["C"]

    @property
    def get_initmodel_template(cls):
        return OrdinalRegression_SVM

    @property
    def get_cvxproblem_template(cls):
        return OrdinalRegression_Relevance_Bound

    def relax_factors(cls):
        return ["loss_slack", "w_l1_slack"]

    def preprocessing(self, data, **kwargs):
        X, y = data

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        if np.min(y) > 0:
            print("First ordinal class has index > 0. Shifting index...")
            y = y - np.min(y)

        return X, y


class OrdinalRegression_SVM(InitModel):
    HYPERPARAMETER = ["C"]

    def __init__(self, C=1):
        super().__init__()
        self.C = C

    def fit(self, X, y, **kwargs):
        (n, d) = X.shape

        C = self.get_params()["C"]

        self.classes_ = np.unique(y)
        original_bins = sorted(self.classes_)
        n_bins = len(original_bins)
        bins = np.arange(n_bins)
        get_old_bin = dict(zip(bins, original_bins))

        w = cvx.Variable(shape=(d), name="w")
        # For ordinal regression we use two slack variables, we observe the slack in both directions
        slack_left = cvx.Variable(shape=(n), name="slack_left")
        slack_right = cvx.Variable(shape=(n), name="slack_right")
        # We have an offset for every bin boundary
        b_s = cvx.Variable(shape=(n_bins - 1), name="bias")

        objective = cvx.Minimize(cvx.norm(w, 1) + C * cvx.sum(slack_left + slack_right))
        constraints = [slack_left >= 0, slack_right >= 0]

        # Add constraints for slack into left neighboring bins
        for i in range(n_bins - 1):
            indices = np.where(y == get_old_bin[i])
            constraints.append(X[indices] * w - slack_left[indices] <= b_s[i] - 1)

        # Add constraints for slack into right neighboring bins
        for i in range(1, n_bins):
            indices = np.where(y == get_old_bin[i])
            constraints.append(X[indices] * w + slack_right[indices] >= b_s[i - 1] + 1)

        # Add explicit constraint, that all bins are ascending
        for i in range(n_bins - 2):
            constraints.append(b_s[i] <= b_s[i + 1])

        # Solve problem.

        problem = cvx.Problem(objective, constraints)
        problem.solve(**self.SOLVER_PARAMS)

        w = w.value
        b_s = b_s.value
        slack_left = np.asarray(slack_left.value).flatten()
        slack_right = np.asarray(slack_right.value).flatten()
        self.model_state = {"w": w, "b_s": b_s, "slack": (slack_left, slack_right)}

        loss = np.sum(slack_left + slack_right)
        w_l1 = np.linalg.norm(w, ord=1)
        self.constraints = {"loss": loss, "w_l1": w_l1}
        return self

    def predict(self, X):
        w = self.model_state["w"]
        b_s = self.model_state["b_s"]

        scores = np.dot(X, w.T)[np.newaxis]
        bin_thresholds = np.append(b_s, np.inf)

        # If thresholds are smaller than score the value belongs to the bigger bin
        # after subtracting we check for positive elements
        indices = np.sum(scores.T - bin_thresholds >= 0, -1)
        return self.classes_[indices]

    def score(self, X, y, error_type="mmae", return_error=False, **kwargs):

        X, y = check_X_y(X, y)

        prediction = self.predict(X)
        score = ordinal_scores(y, prediction, error_type, return_error=return_error)

        return score

    def make_scorer(self):
        # Use multiple scores for ordinal regression
        mze = make_scorer(ordinal_scores, error_type="mze")
        mae = make_scorer(ordinal_scores, error_type="mae")
        mmae = make_scorer(ordinal_scores, error_type="mmae")
        scorer = {"mze": mze, "mae": mae, "mmae": mmae}
        return scorer, "mmae"


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
    max_dist = n_bins - 1
    # If only one class available, we dont need to average
    if max_dist == 0:
        error_type = "mze"

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
        score = (max_dist - error) / max_dist

    # Score based on macro-averaged mean absolute error
    elif error_type == "mmae":
        sum = 0
        for i in range(n_bins):
            samples = y == i
            n_samples = np.sum(samples)
            if n_samples > 0:
                bin_error = mae(prediction[samples], y[samples]) / n_samples
                sum += bin_error

        error = sum / n_bins
        score = (max_dist - error) / max_dist
    else:
        raise ValueError("error_type {} not available!'".format(error_type))

    if return_error:
        return error
    else:
        return score


class OrdinalRegression_Relevance_Bound(Relevance_CVXProblem):
    def init_objective_UB(self, sign=None, **kwargs):

        self.add_constraint(
            self.feature_relevance <= sign * self.w[self.current_feature]
        )
        self._objective = cvx.Maximize(self.feature_relevance)

    def init_objective_LB(self, **kwargs):
        self.add_constraint(
            cvx.abs(self.w[self.current_feature]) <= self.feature_relevance
        )
        self._objective = cvx.Minimize(self.feature_relevance)

    def _init_constraints(self, parameters, init_model_constraints):

        n_bins = len(np.unique(self.y))

        # Upper constraints from initial model
        l1_w = init_model_constraints["w_l1"]
        init_loss = init_model_constraints["loss"]

        C = parameters["C"]

        # New Variables
        self.w = cvx.Variable(shape=(self.d), name="w")

        # For ordinal regression we use two slack variables, we observe the slack in both directions
        self.slack_left = cvx.Variable(shape=(self.n), name="slack_left", nonneg=True)
        self.slack_right = cvx.Variable(shape=(self.n), name="slack_right", nonneg=True)

        # We have an offset for every bin boundary
        self.b_s = cvx.Variable(shape=(n_bins - 1), name="bias")

        # New Constraints
        self.loss = cvx.sum(self.slack_left + self.slack_right)
        self.weight_norm = cvx.norm(self.w, 1)

        for i in range(n_bins - 1):
            indices = np.where(self.y == i)
            self.add_constraint(
                self.X[indices] * self.w - self.slack_left[indices] <= self.b_s[i] - 1
            )

        for i in range(1, n_bins):
            indices = np.where(self.y == i)
            self.add_constraint(
                self.X[indices] * self.w + self.slack_right[indices]
                >= self.b_s[i - 1] + 1
            )

        for i in range(n_bins - 2):
            self.add_constraint(self.b_s[i] <= self.b_s[i + 1])

        self.add_constraint(self.weight_norm <= l1_w)
        self.add_constraint(C * self.loss <= C * init_loss)

        self.feature_relevance = cvx.Variable(nonneg=True, name="Feature Relevance")
