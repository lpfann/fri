import cvxpy as cvx
import numpy as np
from sklearn.utils import check_X_y
from sklearn.metrics import r2_score

from .base_cvxproblem import Relevance_CVXProblem
from .base_initmodel import InitModel
from .base_type import ProblemType


class Regression(ProblemType):
    @classmethod
    def parameters(cls):
        return ["C", "epsilon"]

    @property
    def get_initmodel_template(cls):
        return Regression_SVR

    @property
    def get_cvxproblem_template(cls):
        return Regression_Relevance_Bound

    def relax_factors(cls):
        return ["loss_slack", "w_l1_slack"]

    def preprocessing(self, data, **kwargs):
        X, y = data

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        return X, y


class Regression_SVR(InitModel):
    HYPERPARAMETER = ["C", "epsilon"]

    def __init__(self, C=1, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.C = C

    def fit(self, X, y, **kwargs):
        (n, d) = X.shape

        C = self.get_params()["C"]
        epsilon = self.get_params()["epsilon"]

        w = cvx.Variable(shape=(d), name="w")
        slack = cvx.Variable(shape=(n), name="slack")
        b = cvx.Variable(name="bias")

        objective = cvx.Minimize(cvx.norm(w, 1) + C * cvx.sum(slack))
        constraints = [cvx.abs(y - (X * w + b)) <= epsilon + slack, slack >= 0]

        # Solve problem.

        problem = cvx.Problem(objective, constraints)
        problem.solve(**self.SOLVER_PARAMS)

        w = w.value
        b = b.value
        slack = np.asarray(slack.value).flatten()
        self.model_state = {"w": w, "b": b, "slack": slack}

        loss = np.sum(slack)
        w_l1 = np.linalg.norm(w, ord=1)
        self.constraints = {"loss": loss, "w_l1": w_l1}
        return self

    def predict(self, X):
        w = self.model_state["w"]
        b = self.model_state["b"]
        y = np.dot(X, w) + b
        return y

    def score(self, X, y, **kwargs):
        prediction = self.predict(X)

        # Using weighted f1 score to have a stable score for imbalanced datasets
        score = r2_score(y, prediction)

        return score


class Regression_Relevance_Bound(Relevance_CVXProblem):
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
        l1_w = init_model_constraints["w_l1"]
        init_loss = init_model_constraints["loss"]

        C = parameters["C"]
        epsilon = parameters["epsilon"]

        # New Variables
        self.w = cvx.Variable(shape=(self.d), name="w")
        self.b = cvx.Variable(name="b")
        self.slack = cvx.Variable(shape=(self.n), nonneg=True, name="slack")

        # New Constraints
        distance_from_plane = cvx.abs(self.y - (self.X * self.w + self.b))
        self.loss = cvx.sum(self.slack)
        self.weight_norm = cvx.norm(self.w, 1)

        self.add_constraint(distance_from_plane <= epsilon + self.slack)
        self.add_constraint(self.weight_norm <= l1_w)
        self.add_constraint(C * self.loss <= C * init_loss)

        self.feature_relevance = cvx.Variable(nonneg=True, name="Feature Relevance")
