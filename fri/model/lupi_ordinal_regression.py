from itertools import product

import cvxpy as cvx
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y

from fri.model.base_lupi import (
    LUPI_Relevance_CVXProblem,
    split_dataset,
    is_lupi_feature,
)
from fri.model.ordinal_regression import (
    OrdinalRegression_Relevance_Bound,
    ordinal_scores,
)
from .base_initmodel import LUPI_InitModel
from .base_type import ProblemType


class LUPI_OrdinalRegression(ProblemType):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lupi_features = None

    @property
    def lupi_features(self):
        return self._lupi_features

    @classmethod
    def parameters(cls):
        return ["C", "scaling_lupi_w"]

    @property
    def get_initmodel_template(cls):
        return LUPI_OrdinalRegression_SVM

    @property
    def get_cvxproblem_template(cls):
        return LUPI_OrdinalRegression_Relevance_Bound

    def relax_factors(cls):
        return ["loss_slack", "w_l1_slack"]

    def preprocessing(self, data, lupi_features=None):
        X, y = data
        d = X.shape[1]

        if lupi_features is None:
            raise ValueError("Argument 'lupi_features' missing in fit() call.")
        if not isinstance(lupi_features, int):
            raise ValueError("Argument 'lupi_features' is not type int.")
        if not 0 < lupi_features < d:
            raise ValueError(
                "Argument 'lupi_features' looks wrong. We need at least 1 priviliged feature (>0) or at least one normal feature."
            )

        self._lupi_features = lupi_features

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        if np.min(y) > 0:
            print("First ordinal class has index > 0. Shifting index...")
            y = y - np.min(y)

        return X, y


class LUPI_OrdinalRegression_SVM(LUPI_InitModel):
    HYPERPARAMETER = ["C", "scaling_lupi_w"]

    def __init__(self, C=1, scaling_lupi_w=1, lupi_features=None):
        super().__init__()
        self.scaling_lupi_w = scaling_lupi_w
        self.C = C
        self.lupi_features = lupi_features

    def fit(self, X_combined, y, lupi_features=None):
        """

        Parameters
        ----------
        lupi_features : int
            Number of features in dataset which are considered privileged information (PI).
            PI features are expected to be the last features in the dataset.

        """
        if lupi_features is None:
            try:
                lupi_features = self.lupi_features
                self.lupi_features = lupi_features
            except:
                raise ValueError("No amount of lupi features given.")
        X, X_priv = split_dataset(X_combined, self.lupi_features)
        (n, d) = X.shape
        self.classes_ = np.unique(y)

        # Get parameters from CV model without any feature contstraints
        C = self.get_params()["C"]
        scaling_lupi_w = self.get_params()["scaling_lupi_w"]

        get_original_bin_name, n_bins = get_bin_mapping(y)
        n_boundaries = n_bins - 1

        # Initalize Variables in cvxpy
        w = cvx.Variable(shape=(d), name="w")
        b_s = cvx.Variable(shape=(n_boundaries), name="bias")
        w_priv = cvx.Variable(shape=(self.lupi_features, 2), name="w_priv")
        d_priv = cvx.Variable(shape=(2), name="bias_priv")

        def priv_function(bin, sign):
            indices = np.where(y == get_original_bin_name[bin])
            return X_priv[indices] * w_priv[:, sign] + d_priv[sign]

        # L1 norm regularization of both functions with 1 scaling constant
        priv_l1_1 = cvx.norm(w_priv[:, 0], 1)
        priv_l1_2 = cvx.norm(w_priv[:, 1], 1)
        w_priv_l1 = priv_l1_1 + priv_l1_2
        w_l1 = cvx.norm(w, 1)
        weight_regularization = 0.5 * (w_l1 + scaling_lupi_w * w_priv_l1)

        constraints = []
        loss = 0

        for left_bin in range(0, n_bins - 1):
            indices = np.where(y == get_original_bin_name[left_bin])
            constraints.append(
                X[indices] * w - b_s[left_bin] <= -1 + priv_function(left_bin, 0)
            )
            constraints.append(priv_function(left_bin, 0) >= 0)
            loss += cvx.sum(priv_function(left_bin, 0))

        # Add constraints for slack into right neighboring bins
        for right_bin in range(1, n_bins):
            indices = np.where(y == get_original_bin_name[right_bin])
            constraints.append(
                X[indices] * w - b_s[right_bin - 1] >= +1 - priv_function(right_bin, 1)
            )
            constraints.append(priv_function(right_bin, 1) >= 0)
            loss += cvx.sum(priv_function(right_bin, 1))

        for i_boundary in range(0, n_boundaries - 1):
            constraints.append(b_s[i_boundary] <= b_s[i_boundary + 1])

        objective = cvx.Minimize(C * loss + weight_regularization)

        # Solve problem.

        problem = cvx.Problem(objective, constraints)
        problem.solve(**self.SOLVER_PARAMS)

        w = w.value
        b_s = b_s.value
        self.model_state = {
            "w": w,
            "b_s": b_s,
            "w_priv": w_priv.value,
            "d_priv": d_priv.value,
            "lupi_features": lupi_features,  # Number of lupi features in the dataset TODO: Move this somewhere else
            "bin_boundaries": n_boundaries,
        }

        self.constraints = {
            "loss": loss.value,
            "w_l1": w_l1.value,
            "w_priv_l1": w_priv_l1.value,
        }
        return self

    def predict(self, X):

        X, X_priv = split_dataset(X, self.lupi_features)
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


def get_bin_mapping(y):
    """
    Get ordered unique classes and corresponding mapping from old names
    Parameters
    ----------
    y: array of discrete values (int, str)

    Returns

    -------

    """
    classes_ = np.unique(y)
    original_bins = sorted(classes_)
    n_bins = len(original_bins)
    bins = np.arange(n_bins)
    get_old_bin = dict(zip(bins, original_bins))
    return get_old_bin, n_bins


class LUPI_OrdinalRegression_Relevance_Bound(
    LUPI_Relevance_CVXProblem, OrdinalRegression_Relevance_Bound
):
    @classmethod
    def generate_lower_bound_problem(
        cls,
        best_hyperparameters,
        init_constraints,
        best_model_state,
        data,
        di,
        preset_model,
        probeID=-1,
    ):
        is_priv = is_lupi_feature(
            di, data, best_model_state
        )  # Is it a lupi feature where we need additional candidate problems?

        if not is_priv:
            yield from super().generate_lower_bound_problem(
                best_hyperparameters,
                init_constraints,
                best_model_state,
                data,
                di,
                preset_model,
                probeID=probeID,
            )
        else:
            for sign in [1, -1]:
                problem = cls(
                    di,
                    data,
                    best_hyperparameters,
                    init_constraints,
                    preset_model=preset_model,
                    best_model_state=best_model_state,
                    probeID=probeID,
                )
                problem.init_objective_LB(sign=sign)
                problem.isLowerBound = True
                yield problem

    @classmethod
    def generate_upper_bound_problem(
        cls,
        best_hyperparameters,
        init_constraints,
        best_model_state,
        data,
        di,
        preset_model,
        probeID=-1,
    ):
        is_priv = is_lupi_feature(
            di, data, best_model_state
        )  # Is it a lupi feature where we need additional candidate problems?

        if not is_priv:
            yield from super().generate_upper_bound_problem(
                best_hyperparameters,
                init_constraints,
                best_model_state,
                data,
                di,
                preset_model,
                probeID=probeID,
            )
        else:
            for sign, pos in product([1, -1], [0, 1]):
                problem = cls(
                    di,
                    data,
                    best_hyperparameters,
                    init_constraints,
                    preset_model=preset_model,
                    best_model_state=best_model_state,
                    probeID=probeID,
                )
                problem.init_objective_UB(sign=sign, pos=pos)
                yield problem

    @classmethod
    def aggregate_min_candidates(cls, min_problems_candidates):
        vals = [candidate.solved_relevance for candidate in min_problems_candidates]
        # We take the max of mins because we need the necessary contribution over all functions
        min_value = max(vals)
        return min_value

    def _init_objective_LB_LUPI(self, sign=None, bin_index=None, **kwargs):

        self.add_constraint(
            sign * self.w_priv[self.lupi_index, :] <= self.feature_relevance
        )

        self._objective = cvx.Minimize(self.feature_relevance)

    def _init_objective_UB_LUPI(self, sign=None, pos=None, **kwargs):

        self.add_constraint(
            self.feature_relevance <= sign * self.w_priv[self.lupi_index, pos]
        )

        self._objective = cvx.Maximize(self.feature_relevance)

    def _init_constraints(self, parameters, init_model_constraints):

        # Upper constraints from initial model
        init_w_l1 = init_model_constraints["w_l1"]
        init_w_priv_l1 = init_model_constraints["w_priv_l1"]
        init_loss = init_model_constraints["loss"]
        scaling_lupi_w = parameters["scaling_lupi_w"]

        get_original_bin_name, n_bins = get_bin_mapping(self.y)
        n_boundaries = n_bins - 1

        # Initalize Variables in cvxpy
        w = cvx.Variable(shape=(self.d), name="w")
        b_s = cvx.Variable(shape=(n_boundaries), name="bias")

        w_priv = cvx.Variable(shape=(self.d_priv, 2), name="w_priv")
        d_priv = cvx.Variable(shape=(2), name="bias_priv")

        def priv_function(bin, sign):
            indices = np.where(self.y == get_original_bin_name[bin])
            return self.X_priv[indices] * w_priv[:, sign] + d_priv[sign]

        # L1 norm regularization of both functions with 1 scaling constant
        priv_l1_1 = cvx.norm(w_priv[:, 0], 1)
        priv_l1_2 = cvx.norm(w_priv[:, 1], 1)
        w_priv_l1 = priv_l1_1 + priv_l1_2
        w_l1 = cvx.norm(w, 1)

        loss = 0
        for left_bin in range(0, n_bins - 1):
            indices = np.where(self.y == get_original_bin_name[left_bin])
            self.add_constraint(
                self.X[indices] * w - b_s[left_bin] <= -1 + priv_function(left_bin, 0)
            )
            self.add_constraint(priv_function(left_bin, 0) >= 0)
            loss += cvx.sum(priv_function(left_bin, 0))

        # Add constraints for slack into right neighboring bins
        for right_bin in range(1, n_bins):
            indices = np.where(self.y == get_original_bin_name[right_bin])
            self.add_constraint(
                self.X[indices] * w - b_s[right_bin - 1]
                >= +1 - priv_function(right_bin, 1)
            )
            self.add_constraint(priv_function(right_bin, 1) >= 0)
            loss += cvx.sum(priv_function(right_bin, 1))

        for i_boundary in range(0, n_boundaries - 1):
            self.add_constraint(b_s[i_boundary] <= b_s[i_boundary + 1])

        self.add_constraint(
            w_l1 + scaling_lupi_w * w_priv_l1
            <= init_w_l1 + scaling_lupi_w * init_w_priv_l1
        )
        self.add_constraint(loss <= init_loss)

        self.w = w
        self.w_priv = w_priv
        self.feature_relevance = cvx.Variable(nonneg=True, name="Feature Relevance")
