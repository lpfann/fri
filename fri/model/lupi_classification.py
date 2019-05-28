import cvxpy as cvx
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import fbeta_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from fri.baseline import InitModel
from .base import MLProblem
from .base import Relevance_CVXProblem


class LUPI_Classification(MLProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lupi_features = None

    @property
    def lupi_features(self):
        return self._lupi_features

    @classmethod
    def parameters(cls):
        return ["C", "gamma", "beta"]

    @classmethod
    def get_init_model(cls):
        return LUPI_Classification_SVM

    @classmethod
    def get_bound_model(cls):
        return LUPI_Classification_Relevance_Bound

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

        # Store the classes seen during fit
        classes_ = unique_labels(y)

        if len(classes_) > 2:
            raise ValueError("Only binary class data supported")

        # Negative class is set to -1 for decision surface
        y = preprocessing.LabelEncoder().fit_transform(y)
        y[y == 0] = -1

        return X, y


class LUPI_Classification_SVM(InitModel):

    @classmethod
    def hyperparameter(cls):
        return ["C", "gamma", "beta"]

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

        C = self.hyperparam["C"]
        gamma = self.hyperparam["gamma"]
        beta = self.hyperparam["beta"]

        w = cvx.Variable(shape=(d), name="w")
        w_priv = cvx.Variable(lupi_features, name="w_priv")
        b = cvx.Variable(name="bias")
        b_priv = cvx.Variable(name="bias_priv")
        slack = cvx.Variable(shape=(n), name="slack")

        priv_function = X_priv * w_priv + b_priv
        L1 = 0.5 * (cvx.norm(w, 1) + gamma * cvx.norm(w_priv, 1))

        loss = C * (cvx.sum(slack) + beta * cvx.sum(priv_function))
        constraints = [
            cvx.multiply(y.T, X * w + b) >= 1 - priv_function - slack,
            priv_function >= 0,
            slack >= 0,
        ]
        objective = cvx.Minimize(loss + L1)

        # Solve problem.
        solver_params = self.solver_params
        problem = cvx.Problem(objective, constraints)
        problem.solve(**solver_params)

        w = w.value
        w_priv = w_priv.value
        b = b.value
        b_priv = b_priv.value
        slack = np.asarray(slack.value).flatten()
        self.model_state = {
            "w": w,
            "w_priv": w_priv,
            "b": b,
            "b_priv": b_priv,
            "slack": slack,
            "lupi_features": lupi_features
        }

        loss = loss.value
        w_l1 = np.linalg.norm(w, ord=1)
        w_priv_l1 = np.linalg.norm(w_priv, ord=1)
        self.constraints = {
            "loss": loss,
            "w_l1": w_l1,
            "w_priv_l1": w_priv_l1,
        }
        return self

    def predict(self, X):
        # Check if passed dataset X is combined with PI features or if only non-PI features are present.
        if X.shape[1] > self.lupi_features:
            # Take only the non PI features
            X = X[:, :-self.lupi_features]

        w = self.model_state["w"]
        assert len(w) == X.shape[1]
        b = self.model_state["b"]

        # Simple hyperplane classification rule
        y = np.dot(X, w) + b >= 0
        y = y.astype(int)

        # Format binary as signed unit vector
        y[y == 0] = -1

        return y

    def score(self, X, y, **kwargs):
        prediction = self.predict(X)

        # Negative class is set to -1 for decision surface
        y = LabelEncoder().fit_transform(y)
        y[y == 0] = -1

        # Using weighted f1 score to have a stable score for imbalanced datasets
        score = fbeta_score(y, prediction, beta=1, average="weighted")
        if "verbose" in kwargs:
            return classification_report(y, prediction)
        return score


def split_dataset(X_combined, lupi_features):
    assert X_combined.shape[1] > lupi_features
    X = X_combined[:, :-lupi_features]
    X_priv = X_combined[:, -lupi_features:]
    return X, X_priv


class LUPI_Classification_Relevance_Bound(Relevance_CVXProblem):

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
                self.feature_relevance <= factor * self.w_priv[relative_index]
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
                cvx.abs(self.w_priv[relative_index]) <= self.feature_relevance
            )

        self._objective = cvx.Minimize(self.feature_relevance)

    def _init_constraints(self, parameters, init_model_constraints):
        # Upper constraints from initial model
        l1_w = init_model_constraints["w_l1"]
        l1_priv_w = init_model_constraints["w_priv_l1"]
        init_loss = init_model_constraints["loss"]

        C = parameters["C"]
        # gamma = parameters["gamma"]
        beta = parameters["beta"]

        # New Variables
        self.w = cvx.Variable(shape=(self.d), name="w")
        self.w_priv = cvx.Variable(shape=(self.d_priv), name="w_priv")
        self.b = cvx.Variable(name="b")
        self.b_priv = cvx.Variable(name="b_priv")
        self.slack = cvx.Variable(shape=(self.n), nonneg=True, name="slack")

        # New Constraints
        distance_from_plane = cvx.multiply(self.y, self.X * self.w + self.b)
        priv_function = self.X_priv * self.w_priv + self.b_priv

        self.loss = C * (cvx.sum(self.slack) + beta * cvx.sum(priv_function))
        self.weight_norm = cvx.norm(self.w, 1)
        self.weight_norm_priv = cvx.norm(self.w_priv, 1)

        self.add_constraint(distance_from_plane >= 1 - priv_function - self.slack)
        self.add_constraint(self.weight_norm <= l1_w)
        self.add_constraint(self.weight_norm_priv <= l1_priv_w)
        self.add_constraint(self.loss <= init_loss)
        self.add_constraint(self.slack >= 0)
        self.add_constraint(priv_function >= 0)

        self.feature_relevance = cvx.Variable(nonneg=True, name="Feature Relevance")
