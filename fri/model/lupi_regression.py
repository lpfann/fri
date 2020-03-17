from itertools import product

import cvxpy as cvx
import numpy as np
from sklearn.metrics import r2_score
from sklearn.utils import check_X_y

from fri.model.base_lupi import (
    LUPI_Relevance_CVXProblem,
    split_dataset,
    is_lupi_feature,
)
from fri.model.regression import Regression_Relevance_Bound
from .base_initmodel import LUPI_InitModel
from .base_type import ProblemType


class LUPI_Regression(ProblemType):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lupi_features = None

    @property
    def lupi_features(self):
        return self._lupi_features

    @classmethod
    def parameters(cls):
        return ["C", "epsilon", "scaling_lupi_w", "scaling_lupi_loss"]

    @property
    def get_initmodel_template(cls):
        return LUPI_Regression_SVM

    @property
    def get_cvxproblem_template(cls):
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
                "Argument 'lupi_features' looks wrong. We need at least 1 priviliged feature (>0) or at least one normal feature."
            )

        self._lupi_features = lupi_features

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        return X, y


class LUPI_Regression_SVM(LUPI_InitModel):
    HYPERPARAMETER = ["C", "epsilon", "scaling_lupi_w", "scaling_lupi_loss"]

    def __init__(
        self,
        C=1,
        epsilon=0.1,
        scaling_lupi_w=1,
        scaling_lupi_loss=1,
        lupi_features=None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.scaling_lupi_loss = scaling_lupi_loss
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

        # Get parameters from CV model without any feature contstraints
        C = self.get_params()["C"]
        epsilon = self.get_params()["epsilon"]
        scaling_lupi_w = self.get_params()["scaling_lupi_w"]
        scaling_lupi_loss = self.get_params()["scaling_lupi_loss"]

        # Initalize Variables in cvxpy
        w = cvx.Variable(shape=(d), name="w")
        b = cvx.Variable(name="bias")
        w_priv_pos = cvx.Variable(lupi_features, name="w_priv_pos")
        b_priv_pos = cvx.Variable(name="bias_priv_pos")
        w_priv_neg = cvx.Variable(lupi_features, name="w_priv_neg")
        b_priv_neg = cvx.Variable(name="bias_priv_neg")
        slack = cvx.Variable(shape=(n), name="slack")

        # Define functions for better readability
        priv_function_pos = X_priv * w_priv_pos + b_priv_pos
        priv_function_neg = X_priv * w_priv_neg + b_priv_neg

        # Combined loss of lupi function and normal slacks, scaled by two constants
        priv_loss_pos = cvx.sum(priv_function_pos)
        priv_loss_neg = cvx.sum(priv_function_neg)
        priv_loss = priv_loss_pos + priv_loss_neg
        slack_loss = cvx.sum(slack)
        loss = scaling_lupi_loss * priv_loss + slack_loss

        # L1 norm regularization of both functions with 1 scaling constant
        weight_regularization = 0.5 * (
            cvx.norm(w, 1)
            + scaling_lupi_w
            * (0.5 * cvx.norm(w_priv_pos, 1) + 0.5 * cvx.norm(w_priv_neg, 1))
        )

        constraints = [
            y - X * w - b <= epsilon + priv_function_pos + slack,
            X * w + b - y <= epsilon + priv_function_neg + slack,
            priv_function_pos >= 0,
            priv_function_neg >= 0,
            # priv_loss_pos >= 0,
            # priv_loss_neg >= 0,
            # slack_loss >= 0,
            slack >= 0,
            # loss >= 0,
        ]
        objective = cvx.Minimize(C * loss + weight_regularization)

        # Solve problem.

        problem = cvx.Problem(objective, constraints)
        problem.solve(**self.SOLVER_PARAMS)

        self.model_state = {
            "signs_pos": priv_function_pos.value > 0,
            "signs_neg": priv_function_neg.value > 0,
            "w": w.value,
            "w_priv_pos": w_priv_pos.value,
            "w_priv_neg": w_priv_neg.value,
            "b": b.value,
            "b_priv_pos": b_priv_pos.value,
            "b_priv_neg": b_priv_neg.value,
            "lupi_features": lupi_features,  # Number of lupi features in the dataset TODO: Move this somewhere else,
        }
        w_l1 = np.linalg.norm(w.value, ord=1)
        w_priv_pos_l1 = np.linalg.norm(w_priv_pos.value, ord=1)
        w_priv_neg_l1 = np.linalg.norm(w_priv_neg.value, ord=1)
        # We take the mean to combine all submodels (for priv) into a single normalization factor
        w_priv_l1 = w_priv_pos_l1 + w_priv_neg_l1
        self.constraints = {
            "priv_loss": priv_loss.value,
            "scaling_lupi_loss": scaling_lupi_loss,
            # "loss_slack": slack_loss.value,
            "loss": loss.value,
            "w_l1": w_l1,
            "w_priv_l1": w_priv_l1,
            "w_priv_pos_l1": w_priv_pos_l1,
            "w_priv_neg_l1": w_priv_neg_l1,
        }
        return self

    @property
    def SOLVER_PARAMS(cls):
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
        X, X_priv = split_dataset(X, self.lupi_features)
        w = self.model_state["w"]
        b = self.model_state["b"]

        y = np.dot(X, w) + b

        return y

    def score(self, X, y, **kwargs):
        prediction = self.predict(X)

        score = r2_score(y, prediction)
        return score


class LUPI_Regression_Relevance_Bound(
    LUPI_Relevance_CVXProblem, Regression_Relevance_Bound
):
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
            for sign, pos in product([1, -1], [True, False]):
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

    def _init_objective_LB_LUPI(self, **kwargs):
        self.add_constraint(
            cvx.abs(self.w_priv_pos[self.lupi_index]) <= self.feature_relevance
        )
        self.add_constraint(
            cvx.abs(self.w_priv_neg[self.lupi_index]) <= self.feature_relevance
        )

        self._objective = cvx.Minimize(self.feature_relevance)

    def _init_objective_UB_LUPI(self, pos=None, sign=None, **kwargs):
        if pos:
            self.add_constraint(
                self.feature_relevance <= sign * self.w_priv_pos[self.lupi_index]
            )
        else:
            self.add_constraint(
                self.feature_relevance <= sign * self.w_priv_neg[self.lupi_index]
            )

        self._objective = cvx.Maximize(self.feature_relevance)

    def _init_constraints(self, parameters, init_model_constraints):
        # Upper constraints from best initial model
        l1_w = init_model_constraints["w_l1"]
        self.l1_priv_w_pos = init_model_constraints["w_priv_pos_l1"]
        self.l1_priv_w_neg = init_model_constraints["w_priv_neg_l1"]
        init_loss = init_model_constraints["loss"]
        epsilon = parameters["epsilon"]
        scaling_lupi_loss = init_model_constraints["scaling_lupi_loss"]

        # New Variables
        w = cvx.Variable(shape=(self.d), name="w")
        b = cvx.Variable(name="b")
        w_priv_pos = cvx.Variable(self.d_priv, name="w_priv_pos")
        b_priv_pos = cvx.Variable(name="bias_priv_pos")
        w_priv_neg = cvx.Variable(self.d_priv, name="w_priv_neg")
        b_priv_neg = cvx.Variable(name="bias_priv_neg")
        slack = cvx.Variable(shape=(self.n))

        priv_function_pos = self.X_priv * w_priv_pos + b_priv_pos
        priv_function_neg = self.X_priv * w_priv_neg + b_priv_neg
        priv_loss = cvx.sum(priv_function_pos + priv_function_neg)
        loss = priv_loss + cvx.sum(slack)
        weight_norm = cvx.norm(w, 1)
        self.weight_norm_priv_pos = cvx.norm(w_priv_pos, 1)
        self.weight_norm_priv_neg = cvx.norm(w_priv_neg, 1)

        self.add_constraint(
            self.y - self.X * w - b <= epsilon + priv_function_pos + slack
        )
        self.add_constraint(
            self.X * w + b - self.y <= epsilon + priv_function_neg + slack
        )
        self.add_constraint(priv_function_pos >= 0)
        self.add_constraint(priv_function_neg >= 0)
        self.add_constraint(loss <= init_loss)
        self.add_constraint(slack >= 0)
        sum_norms = weight_norm + self.weight_norm_priv_pos + self.weight_norm_priv_neg
        self.add_constraint(sum_norms <= l1_w)
        # self.add_constraint(self.weight_norm_priv_pos <= self.l1_priv_w_pos)
        # self.add_constraint(self.weight_norm_priv_neg <= self.l1_priv_w_neg)

        # Save values for object use later
        self.w = w
        self.w_priv_pos = w_priv_pos
        self.w_priv_neg = w_priv_neg
        self.feature_relevance = cvx.Variable(nonneg=True, name="Feature Relevance")
