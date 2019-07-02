import cvxpy as cvx
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y

from .base_cvxproblem import Relevance_CVXProblem
from .base_initmodel import InitModel
from .base_type import ProblemType
from .lupi_ordinal_regression import get_bin_mapping


class OrdinalRegression_Imp(ProblemType):

    @classmethod
    def parameters(cls):
        return ["C"]

    @property
    def get_initmodel_template(cls):
        return OrdinalRegression_Imp_SVM

    @property
    def get_cvxproblem_template(cls):
        return OrdinalRegression_Imp_Relevance_Bound

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


class OrdinalRegression_Imp_SVM(InitModel):

    @classmethod
    def hyperparameter(cls):
        return ["C"]

    def fit(self, X, y, **kwargs):
        (n, d) = X.shape
        self.classes_ = np.unique(y)

        # Get parameters from CV model without any feature contstraints
        C = self.hyperparam["C"]

        get_original_bin_name, n_bins = get_bin_mapping(y)
        n_boundaries = n_bins - 1

        # Initalize Variables in cvxpy
        w = cvx.Variable(shape=(d), name="w")
        b_s = cvx.Variable(shape=(n_boundaries), name="bias")
        slack_left = [cvx.Variable(shape=(n_bins, n), name="slack_left", nonneg=True) for j in range(n_bins)]
        slack_right = [cvx.Variable(shape=(n_bins, n), name="slack_right", nonneg=True) for j in range(n_bins)]

        # L1 norm regularization of both functions with 1 scaling constant
        w_l1 = cvx.norm(w, 1)
        weight_regularization = 0.5 * w_l1

        constraints = []
        loss = 0
        for j in range(0, n_boundaries):
            print("boundary ", j)
            # Add constraints for slack into right neighboring bins
            for k in range(0, j + 1):
                print(f"Left samples in bin:{k}")
                indices = np.where(y == get_original_bin_name[k])
                constraints.append(X[indices] * w - b_s[j] <= -1 + slack_left[j][k, indices][0])
                loss += cvx.sum(slack_left[j][k, indices][0])

            # Add constraints for slack into left neighboring bins
            for k in range(j + 1, n_bins):
                print(f"Right samples in bin:{k}")
                indices = np.where(y == get_original_bin_name[k])
                constraints.append(X[indices] * w - b_s[j] >= +1 - slack_right[j][k, indices][0])
                loss += cvx.sum(slack_right[j][k, indices][0])

        print("END loop")
        # loss = cvx.sum(slack_left+slack_right)
        objective = cvx.Minimize(C * loss + weight_regularization)

        # Solve problem.
        solver_params = self.solver_params
        problem = cvx.Problem(objective, constraints)
        problem.solve(**solver_params)

        w = w.value
        b_s = b_s.value
        self.model_state = {
            "w": w,
            "b_s": b_s,
            "bin_boundaries": n_boundaries
        }

        self.constraints = {
            "loss": loss.value,
            "w_l1": w_l1.value,
        }
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

    def score(self, X, y, **kwargs):

        X, y = check_X_y(X, y)

        prediction = self.predict(X)
        error_type = kwargs.get("error_type", "mmae")
        score = ordinal_scores(y, prediction, error_type)

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


class OrdinalRegression_Imp_Relevance_Bound(Relevance_CVXProblem):

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

        # Upper constraints from initial model
        init_w_l1 = init_model_constraints["w_l1"]
        init_loss = init_model_constraints["loss"]

        get_original_bin_name, n_bins = get_bin_mapping(self.y)
        n_boundaries = n_bins - 1

        # Initalize Variables in cvxpy
        w = cvx.Variable(shape=(self.d), name="w")
        b_s = cvx.Variable(shape=(n_boundaries), name="bias")
        slack_left = [cvx.Variable(shape=(n_bins, self.n), name="slack_left", nonneg=True) for j in range(n_bins)]
        slack_right = [cvx.Variable(shape=(n_bins, self.n), name="slack_right", nonneg=True) for j in range(n_bins)]

        w_l1 = cvx.norm(w, 1)

        constraints = []
        loss = 0
        for j in range(n_boundaries):
            # Add constraints for slack into right neighboring bins
            for k in range(0, j + 1):
                indices = np.where(self.y == get_original_bin_name[k])
                self.add_constraint(self.X[indices] * w - b_s[j] <= -1 + slack_left[j][k, indices][0])
                loss += cvx.sum(slack_left[j][k, indices][0])

            # Add constraints for slack into left neighboring bins
            for k in range(j + 1, n_bins):
                indices = np.where(self.y == get_original_bin_name[k])
                self.add_constraint(self.X[indices] * w - b_s[j] >= +1 - slack_right[j][k, indices][0])
                loss += cvx.sum(slack_right[j][k, indices][0])

        self.add_constraint(w_l1 <= init_w_l1)
        self.add_constraint(loss <= init_loss)

        self.w = w
        self.feature_relevance = cvx.Variable(nonneg=True, name="Feature Relevance")
