import cvxpy as cvx
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y

from fri.baseline import InitModel
from fri.model.base import Relevance_CVXProblem
from .base import MLProblem


class LUPI_OrdinalRegression(MLProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lupi_features = None

    @property
    def lupi_features(self):
        return self._lupi_features

    @classmethod
    def parameters(cls):
        return ["C", "scaling_lupi_w", "scaling_lupi_loss"]

    @classmethod
    def get_init_model(cls):
        return LUPI_OrdinalRegression_SVM

    @classmethod
    def get_bound_model(cls):
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
                "Argument 'lupi_features' looks wrong. We need at least 1 priviliged feature (>0) or at least one normal feature.")

        self._lupi_features = lupi_features

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        if np.min(y) > 0:
            print("First ordinal class has index > 0. Shifting index...")
            y = y - np.min(y)

        return X, y


class LUPI_OrdinalRegression_SVM(InitModel):

    @classmethod
    def hyperparameter(cls):
        return ["C", "scaling_lupi_w", "scaling_lupi_loss"]

    def fit(self, X_combined, y, lupi_features=None):
        """

        Parameters
        ----------
        lupi_features : int
            Number of features in dataset which are considered privileged information (PI).
            PI features are expected to be the last features in the dataset.

        """
        if lupi_features is None:
            raise ValueError("No lupi_features argument given.")
        self.lupi_features = lupi_features
        X, X_priv = split_dataset(X_combined, lupi_features)
        (n, d) = X.shape
        self.classes_ = np.unique(y)

        # Get parameters from CV model without any feature contstraints
        C = self.hyperparam["C"]
        scaling_lupi_w = self.hyperparam["scaling_lupi_w"]

        get_original_bin_name, n_bins = get_bin_mapping(y)
        J = n_bins - 1

        # Initalize Variables in cvxpy
        w = cvx.Variable(shape=(d), name="w")
        b_s = cvx.Variable(shape=(J), name="bias")

        w_priv = cvx.Variable(shape=(J, self.lupi_features), name="w_priv")
        d_priv = cvx.Variable(shape=(J), name="bias_priv")

        def priv_function(j, k):
            indices = np.where(y == get_original_bin_name[k])
            return X_priv[indices] * w_priv[j, :] + d_priv[j]

        # L1 norm regularization of both functions with 1 scaling constant
        w_priv_l1 = cvx.norm(w_priv, 1)
        w_l1 = cvx.norm(w, 1)
        weight_regularization = 0.5 * (w_l1 + scaling_lupi_w * w_priv_l1)

        constraints = []
        loss = 0
        for j in range(J):
            # Add constraints for slack into right neighboring bins
            for k in range(0, j):
                indices = np.where(y == get_original_bin_name[k])
                constraints.append(X[indices] * w - b_s[j] <= -1 + priv_function(j, k))
                constraints.append(priv_function(j, k) >= 0)
                loss += cvx.sum(priv_function(j, k))

            # Add constraints for slack into left neighboring bins
            for k in range(j + 1, n_bins):
                indices = np.where(y == get_original_bin_name[k])
                constraints.append(X[indices] * w - b_s[j] >= +1 - priv_function(j, k))
                constraints.append(priv_function(j, k) >= 0)
                loss += cvx.sum(priv_function(j, k))

        # loss = C * loss
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
            "w_priv": w_priv.value,
            "d_priv": d_priv.value,
            "lupi_features": lupi_features  # Number of lupi features in the dataset TODO: Move this somewhere else
        }

        self.constraints = {
            "loss": loss.value,
            "w_l1": w_l1.value,
            "w_priv_l1": w_priv_l1.value
        }
        return self

    def predict(self, X):
        # TODO: remove this when not needed
        # Check if passed dataset X is combined with PI features or if only non-PI features are present.
        if X.shape[1] > self.lupi_features:
            # Take only the non PI features
            X = X[:, :-self.lupi_features]

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


def split_dataset(X_combined, lupi_features):
    assert X_combined.shape[1] > lupi_features
    X = X_combined[:, :-lupi_features]
    X_priv = X_combined[:, -lupi_features:]
    return X, X_priv


class LUPI_OrdinalRegression_Relevance_Bound(Relevance_CVXProblem):

    def preprocessing_data(self, data, best_model_state):
        lupi_features = best_model_state["lupi_features"]

        X_combined, y = data
        X, X_priv = split_dataset(X_combined, lupi_features)
        self.X_priv = X_priv

        assert lupi_features == X_priv.shape[1]
        self.d_priv = lupi_features

        return super().preprocessing_data((X, y), best_model_state)

    def _init_objective_UB(self):

        if self.sign:
            factor = -1
        else:
            factor = 1
        # We have two models basically with different indexes
        if self.current_feature < self.d:
            # Normal model, we use w and normal index
            self.add_constraint(
                self.feature_relevance <= factor * self.w[self.current_feature]
            )
        else:
            # LUPI model, we need to ofset the index
            relative_index = self.current_feature - self.d
            self.add_constraint(
                self.feature_relevance <= factor * self.w_priv[:, relative_index]
            )

        self._objective = cvx.Maximize(self.feature_relevance)

    def _init_objective_LB(self):
        # We have two models basically with different indexes
        if self.current_feature < self.d:
            # Normal model, we use w and normal index
            self.add_constraint(
                cvx.abs(self.w[self.current_feature]) <= self.feature_relevance
            )
        else:
            # LUPI model, we need to ofset the index
            relative_index = self.current_feature - self.d
            self.add_constraint(
                cvx.abs(self.w_priv[:, relative_index]) <= self.feature_relevance
            )

        self._objective = cvx.Minimize(self.feature_relevance)

    def _init_constraints(self, parameters, init_model_constraints):

        # Upper constraints from initial model
        init_w_l1 = init_model_constraints["w_l1"]
        init_w_priv_l1 = init_model_constraints["w_priv_l1"]
        init_loss = init_model_constraints["loss"]

        C = parameters["C"]
        scaling_lupi_w = parameters["scaling_lupi_w"]

        get_original_bin_name, n_bins = get_bin_mapping(self.y)
        J = n_bins - 1

        # Initalize Variables in cvxpy
        w = cvx.Variable(shape=(self.d), name="w")
        b_s = cvx.Variable(shape=(J), name="bias")

        w_priv = cvx.Variable(shape=(J, self.d_priv), name="w_priv")
        d_priv = cvx.Variable(shape=(J), name="bias_priv")

        def priv_function(j, k):
            indices = np.where(self.y == get_original_bin_name[k])
            return self.X_priv[indices] * w_priv[j, :] + d_priv[j]

        # L1 norm regularization of both functions with 1 scaling constant
        w_priv_l1 = cvx.norm(w_priv, 1)
        w_l1 = cvx.norm(w, 1)

        constraints = []
        loss = 0
        for j in range(J):
            # Add constraints for slack into right neighboring bins
            for k in range(0, j):
                indices = np.where(self.y == get_original_bin_name[k])
                constraints.append(self.X[indices] * w - b_s[j] <= -1 + priv_function(j, k))
                constraints.append(priv_function(j, k) >= 0)
                loss += cvx.sum(priv_function(j, k))

            # Add constraints for slack into left neighboring bins
            for k in range(j + 1, n_bins):
                indices = np.where(self.y == get_original_bin_name[k])
                constraints.append(self.X[indices] * w - b_s[j] >= +1 - priv_function(j, k))
                constraints.append(priv_function(j, k) >= 0)
                loss += cvx.sum(priv_function(j, k))

        # loss = C * loss
        for c in constraints:
            self.add_constraint(c)
        self.add_constraint(w_priv_l1 <= init_w_priv_l1)
        self.add_constraint(w_l1 <= init_w_l1)
        self.add_constraint(loss <= init_loss)

        self.w = w
        self.w_priv = w_priv
        self.feature_relevance = cvx.Variable(nonneg=True, name="Feature Relevance")
