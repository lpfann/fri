import cvxpy as cvx
import numpy as np
from sklearn.utils import check_X_y

from fri.baseline import InitModel
from .base import MLProblem
from .base import Relevance_CVXProblem


class LUPI_Regression(MLProblem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lupi_features = None

    @property
    def lupi_features(self):
        return self._lupi_features

    @classmethod
    def parameters(cls):
        return ["C", "epsilon", "scaling_lupi_w"]

    @classmethod
    def get_init_model(cls):
        return LUPI_Regression_SVM

    @classmethod
    def get_bound_model(cls):
        return LUPI_Regression_Relevance_Bound

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

        return X, y

    def generate_upper_bound_problem(self, best_hyperparameters, init_constraints, best_model_state, data, di,
                                     preset_model, isProbe=False):
        lupi_features = best_model_state["lupi_features"]
        if di
            for sign, pos in zip([True, False], [True, False]):
                yield self.get_bound_model()(False, di, data, best_hyperparameters, init_constraints,
                                             preset_model=preset_model,
                                             best_model_state=best_model_state, isProbe=isProbe, sign=sign, pos=pos)

    def aggregate_max_candidates(self, max_problems_candidates):
        return super().aggregate_max_candidates(max_problems_candidates)


class LUPI_Regression_SVM(InitModel):

    @classmethod
    def hyperparameter(cls):
        return ["C", "epsilon", "scaling_lupi_w"]

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

        # Get parameters from CV model without any feature contstraints
        C = self.hyperparam["C"]
        epsilon = self.hyperparam["epsilon"]
        scaling_lupi_w = self.hyperparam["scaling_lupi_w"]

        # Initalize Variables in cvxpy
        w = cvx.Variable(shape=(d), name="w")
        b = cvx.Variable(name="bias")
        w_priv_pos = cvx.Variable(lupi_features, name="w_priv_pos")
        b_priv_pos = cvx.Variable(name="bias_priv_pos")
        w_priv_neg = cvx.Variable(lupi_features, name="w_priv_neg")
        b_priv_neg = cvx.Variable(name="bias_priv_neg")

        # Define functions for better readability
        priv_function_pos = X_priv * w_priv_pos + b_priv_pos
        priv_function_neg = X_priv * w_priv_neg + b_priv_neg

        # Combined loss of lupi function and normal slacks, scaled by two constants
        priv_loss = cvx.sum(priv_function_pos + priv_function_neg)
        loss = C * priv_loss

        # L1 norm regularization of both functions with 1 scaling constant
        weight_regularization = 0.5 * (
                    cvx.norm(w, 1) + scaling_lupi_w * (cvx.norm(w_priv_pos, 1) + cvx.norm(w_priv_neg, 1)))

        constraints = [
            y - X * w - b <= epsilon + priv_function_pos,
            X * w + b - y <= epsilon + priv_function_neg,
            priv_function_pos >= 0,
            priv_function_neg >= 0,
        ]
        objective = cvx.Minimize(loss + weight_regularization)

        # Solve problem.
        solver_params = self.solver_params
        problem = cvx.Problem(objective, constraints)
        problem.solve(**solver_params)

        self.model_state = {
            "w": w.value,
            "w_priv_pos": w_priv_pos.value,
            "w_priv_neg": w_priv_neg.value,
            "w_priv": w_priv_pos.value + w_priv_neg.value,
            "b": b.value,
            "b_priv_pos": b_priv_pos.value,
            "b_priv_neg": b_priv_neg.value,
            "b_priv": b_priv_pos.value + b_priv_neg.value,
            "lupi_features": lupi_features  # Number of lupi features in the dataset TODO: Move this somewhere else
        }

        w_l1 = np.linalg.norm(w.value, ord=1)
        w_priv_pos_l1 = np.linalg.norm(w_priv_pos.value, ord=1)
        w_priv_neg_l1 = np.linalg.norm(w_priv_neg.value, ord=1)
        self.constraints = {
            "loss_priv": priv_loss.value,
            "loss": loss.value,
            "w_l1": w_l1,
            "w_priv_pos_l1": w_priv_pos_l1,
            "w_priv_neg_l1": w_priv_neg_l1,
            "w_priv_l1": w_priv_pos_l1 + w_priv_neg_l1
        }
        return self

    @property
    def solver_params(cls):
        return {"solver": "ECOS", "verbose": False}

    def predict(self, X):
        """
        Method to predict points using svm classification rule.
        We use both normal and priv. features.
        This function is mainly used for CV purposes to find the best parameters according to score.

        Parameters
        ----------
        X : numpy.ndarray
        """
        # TODO: remove this when not needed
        ## Check if passed dataset X is combined with PI features or if only non-PI features are present.
        # if X.shape[1] > self.lupi_features:
        #    # Take only the non PI features
        #    X = X[:, :-self.lupi_features]
        X, X_priv = split_dataset(X, self.lupi_features)
        w = self.model_state["w"]
        b = self.model_state["b"]
        w_priv = self.model_state["w_priv"]
        b_priv = self.model_state["b_priv"]

        # Combine both models
        # w = np.concatenate([w, w_priv])
        #b += b_priv

        # Simple hyperplane classification rule
        y = np.dot(X, w) + b - (np.dot(X_priv, w_priv) + b_priv)

        return y

    def score(self, X, y, **kwargs):
        prediction = self.predict(X)

        from sklearn.metrics import r2_score
        from sklearn.metrics.regression import _check_reg_targets

        _check_reg_targets(y, prediction, None)

        # Using weighted f1 score to have a stable score for imbalanced datasets
        score = r2_score(y, prediction)

        return score


def split_dataset(X_combined, lupi_features):
    assert X_combined.shape[1] > lupi_features
    X = X_combined[:, :-lupi_features]
    X_priv = X_combined[:, -lupi_features:]
    return X, X_priv


class LUPI_Regression_Relevance_Bound(Relevance_CVXProblem):

    def preprocessing_data(self, data, best_model_state):
        lupi_features = best_model_state["lupi_features"]

        X_combined, y = data
        X, X_priv = split_dataset(X_combined, lupi_features)
        self.X_priv = X_priv

        assert lupi_features == X_priv.shape[1]
        self.d_priv = lupi_features

        return super().preprocessing_data((X, y), best_model_state)

    def _init_objective_UB(self, sign=None, pos=None, **kwargs):

        # We have two models basically with different indexes
        if self.current_feature < self.d:
            # Normal model, we use w and normal index
            self.add_constraint(
                self.feature_relevance <= sign * self.w[self.current_feature]
            )
        else:
            # LUPI model, we need to offset the index
            relative_index = self.current_feature - self.d
            if pos:
                self.add_constraint(
                    self.feature_relevance <= sign * self.w_priv_pos[relative_index],
                )
            else:
                self.add_constraint(
                    self.feature_relevance <= -1 * sign * self.w_priv_neg[relative_index],
                )

        self._objective = cvx.Maximize(self.feature_relevance)

    def _init_objective_LB(self, **kwargs):
        # We have two models basically with different indexes
        if self.current_feature < self.d:
            # Normal model, we use w and normal index
            self.add_constraint(
                cvx.abs(self.w[self.current_feature]) <= self.feature_relevance
            )
        else:
            # LUPI model, we need to ofset the index
            relative_index = self.current_feature - self.d
            self.add_constraint(cvx.abs(self.w_priv_pos[relative_index]) <= self.feature_relevance)
            self.add_constraint(cvx.abs(self.w_priv_neg[relative_index]) <= self.feature_relevance)

        self._objective = cvx.Minimize(self.feature_relevance)

    def _init_constraints(self, parameters, init_model_constraints):
        # Upper constraints from best initial model
        l1_w = init_model_constraints["w_l1"]
        l1_priv_w_pos = init_model_constraints["w_priv_pos_l1"]
        l1_priv_w_neg = init_model_constraints["w_priv_neg_l1"]
        init_loss = init_model_constraints["loss"]
        # Parameters from best model
        C = parameters["C"]
        epsilon = parameters["epsilon"]

        # New Variables
        w = cvx.Variable(shape=(self.d), name="w")
        b = cvx.Variable(name="b")
        w_priv_pos = cvx.Variable(self.d_priv, name="w_priv_pos")
        b_priv_pos = cvx.Variable(name="bias_priv_pos")
        w_priv_neg = cvx.Variable(self.d_priv, name="w_priv_pos")
        b_priv_neg = cvx.Variable(name="bias_priv_pos")

        # Define functions for better readability
        priv_function_pos = self.X_priv * w_priv_pos + b_priv_pos
        priv_function_neg = self.X_priv * w_priv_neg + b_priv_neg
        priv_loss = cvx.sum(priv_function_pos + priv_function_neg)
        # New Constraints

        loss = C * priv_loss
        weight_norm = cvx.norm(w, 1)
        weight_norm_priv_pos = cvx.norm(w_priv_pos, 1)
        weight_norm_priv_neg = cvx.norm(w_priv_neg, 1)

        self.add_constraint(self.y - self.X * w - b <= epsilon + priv_function_pos)
        self.add_constraint(self.X * w + b - self.y <= epsilon + priv_function_neg )
        self.add_constraint(priv_function_pos >= 0)
        self.add_constraint(priv_function_neg >= 0)
        self.add_constraint(loss <= init_loss)
        self.add_constraint(weight_norm <= l1_w)
        self.add_constraint(weight_norm_priv_pos <= l1_priv_w_pos)
        self.add_constraint(weight_norm_priv_neg <= l1_priv_w_neg)

        # Save values for object use later
        self.w = w
        self.w_priv_pos = w_priv_pos
        self.w_priv_neg = w_priv_neg
        self.feature_relevance = cvx.Variable(nonneg=True, name="Feature Relevance")
