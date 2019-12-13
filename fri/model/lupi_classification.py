import cvxpy as cvx
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import fbeta_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

from .base_initmodel import InitModel
from .base_lupi import LUPI_Relevance_CVXProblem, split_dataset
from .base_type import ProblemType
from .classification import Classification_Relevance_Bound


class LUPI_Classification(ProblemType):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._lupi_features = None

    @property
    def lupi_features(self):
        return self._lupi_features

    @classmethod
    def parameters(cls):
        return ["C", "scaling_lupi_w", "scaling_lupi_loss"]

    @property
    def get_initmodel_template(cls):
        return LUPI_Classification_SVM

    @property
    def get_cvxproblem_template(cls):
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
                "Argument 'lupi_features' looks wrong. We need at least 1 priviliged feature (>0) or at least one normal feature."
            )

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
    HYPERPARAMETER = ["C", "scaling_lupi_w", "scaling_lupi_loss"]

    def __init__(self, C=1, scaling_lupi_w=1, scaling_lupi_loss=1, lupi_features=None):
        super().__init__()
        self.lupi_features = lupi_features
        self.scaling_lupi_loss = scaling_lupi_loss
        self.scaling_lupi_w = scaling_lupi_w
        self.C = C

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
        scaling_lupi_w = self.get_params()["scaling_lupi_w"]
        scaling_lupi_loss = self.get_params()["scaling_lupi_loss"]

        # Initalize Variables in cvxpy
        w = cvx.Variable(shape=(d), name="w")
        w_priv = cvx.Variable(lupi_features, name="w_priv")
        b = cvx.Variable(name="bias")
        b_priv = cvx.Variable(name="bias_priv")

        # Define functions for better readability
        function = X * w + b
        priv_function = X_priv * w_priv + b_priv
        slack = cvx.Variable(shape=(n))

        # Combined loss of lupi function and normal slacks, scaled by two constants
        loss = scaling_lupi_loss * cvx.sum(priv_function) + cvx.sum(slack)

        # L1 norm regularization of both functions with 1 scaling constant
        w_l1 = cvx.norm(w, 1)
        w_priv_l1 = cvx.norm(w_priv, 1)
        weight_regularization = 0.5 * (w_l1 + scaling_lupi_w * w_priv_l1)

        constraints = [
            cvx.multiply(y.T, function) >= 1 - cvx.multiply(y.T, priv_function) - slack,
            priv_function >= 0,
            slack >= 0,
        ]
        objective = cvx.Minimize(C * loss + weight_regularization)

        # Solve problem.

        problem = cvx.Problem(objective, constraints)
        problem.solve(**self.SOLVER_PARAMS)

        w = w.value
        w_priv = w_priv.value
        b = b.value
        b_priv = b_priv.value
        self.model_state = {
            "w": w,
            "w_priv": w_priv,
            "b": b,
            "b_priv": b_priv,
            "lupi_features": lupi_features,  # Number of lupi features in the dataset TODO: Move this somewhere else
        }

        loss = loss.value
        w_l1 = w_l1.value
        w_priv_l1 = w_priv_l1.value
        self.constraints = {"loss": loss, "w_l1": w_l1, "w_priv_l1": w_priv_l1}
        return self

    def predict(self, X):
        X, X_priv = split_dataset(X, self.lupi_features)
        w = self.model_state["w"]
        b = self.model_state["b"]

        # Simple hyperplane classification rule
        f = np.dot(X, w) + b
        y = f >= 0
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


class LUPI_Classification_Relevance_Bound(
    LUPI_Relevance_CVXProblem, Classification_Relevance_Bound
):
    def _init_objective_UB_LUPI(self, sign=None, **kwargs):
        self.add_constraint(
            self.feature_relevance <= sign * self.w_priv[self.lupi_index]
        )
        self._objective = cvx.Maximize(self.feature_relevance)

    def _init_objective_LB_LUPI(self, **kwargs):
        self.add_constraint(
            cvx.abs(self.w_priv[self.lupi_index]) <= self.feature_relevance
        )
        self._objective = cvx.Minimize(self.feature_relevance)

    def _init_constraints(self, parameters, init_model_constraints):
        # Upper constraints from best initial model
        l1_w = init_model_constraints["w_l1"]
        l1_priv_w = init_model_constraints["w_priv_l1"]
        init_loss = init_model_constraints["loss"]

        # New Variables
        w = cvx.Variable(shape=(self.d), name="w")
        w_priv = cvx.Variable(shape=(self.d_priv), name="w_priv")
        b = cvx.Variable(name="b")
        b_priv = cvx.Variable(name="b_priv")
        slack = cvx.Variable(shape=(self.n))

        # New Constraints
        function = cvx.multiply(self.y.T, self.X * w + b)
        priv_function = self.X_priv * w_priv + b_priv
        loss = cvx.sum(priv_function) + cvx.sum(slack)

        weight_norm = cvx.norm(w, 1)
        weight_norm_priv = cvx.norm(w_priv, 1)

        self.add_constraint(
            function >= 1 - cvx.multiply(self.y.T, priv_function) - slack
        )
        self.add_constraint(priv_function >= 0)
        self.add_constraint(loss <= init_loss)
        self.add_constraint(weight_norm + weight_norm_priv <= l1_w + l1_priv_w)
        self.add_constraint(slack >= 0)

        # Save values for object use later
        self.w = w
        self.w_priv = w_priv
        self.feature_relevance = cvx.Variable(nonneg=True, name="Feature Relevance")
