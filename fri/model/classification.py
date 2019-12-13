import cvxpy as cvx
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import fbeta_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from fri.model.base_cvxproblem import Relevance_CVXProblem
from fri.model.base_initmodel import InitModel
from .base_type import ProblemType


class Classification(ProblemType):
    @classmethod
    def parameters(cls):
        return ["C"]

    @property
    def get_initmodel_template(cls):
        return Classification_SVM

    @property
    def get_cvxproblem_template(cls):
        return Classification_Relevance_Bound

    def relax_factors(cls):
        return ["loss_slack", "w_l1_slack"]

    def preprocessing(self, data, **kwargs):
        X, y = data
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


class Classification_SVM(InitModel):
    def __init__(self, C=1):
        super().__init__()
        self.C = C

    def fit(self, X, y, **kwargs):
        (n, d) = X.shape

        C = self.get_params()["C"]

        w = cvx.Variable(shape=(d), name="w")
        slack = cvx.Variable(shape=(n), name="slack")
        b = cvx.Variable(name="bias")

        objective = cvx.Minimize(cvx.norm(w, 1) + C * cvx.sum(slack))
        constraints = [cvx.multiply(y.T, X * w + b) >= 1 - slack, slack >= 0]

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
        y = np.dot(X, w) + b >= 0
        y = y.astype(int)
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


class Classification_Relevance_Bound(Relevance_CVXProblem):
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

        # New Variables
        self.w = cvx.Variable(shape=(self.d), name="w")
        self.b = cvx.Variable(name="b")
        self.slack = cvx.Variable(shape=(self.n), nonneg=True, name="slack")

        # New Constraints
        distance_from_plane = cvx.multiply(self.y, self.X * self.w + self.b)
        self.loss = cvx.sum(self.slack)
        self.weight_norm = cvx.norm(self.w, 1)

        self.add_constraint(distance_from_plane >= 1 - self.slack)
        self.add_constraint(self.weight_norm <= l1_w)
        self.add_constraint(C * self.loss <= C * init_loss)

        self.feature_relevance = cvx.Variable(nonneg=True, name="Feature Relevance")
