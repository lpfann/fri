import cvxpy as cvx
import numpy as np

from .lupi_ordinal_regression import get_bin_mapping
from .ordinal_regression import OrdinalRegression, OrdinalRegression_SVM, OrdinalRegression_Relevance_Bound


class OrdinalRegression_Imp(OrdinalRegression):


    @property
    def get_initmodel_template(cls):
        return OrdinalRegression_Imp_SVM

    @property
    def get_cvxproblem_template(cls):
        return OrdinalRegression_Imp_Relevance_Bound


class OrdinalRegression_Imp_SVM(OrdinalRegression_SVM):

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
            # Add constraints for slack into right neighboring bins
            for k in range(0, j + 1):
                indices = np.where(y == get_original_bin_name[k])
                constraints.append(X[indices] * w - b_s[j] <= -1 + slack_left[j][k, indices][0])
                loss += cvx.sum(slack_left[j][k, indices][0])

            # Add constraints for slack into left neighboring bins
            for k in range(j + 1, n_bins):
                indices = np.where(y == get_original_bin_name[k])
                constraints.append(X[indices] * w - b_s[j] >= +1 - slack_right[j][k, indices][0])
                loss += cvx.sum(slack_right[j][k, indices][0])

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


class OrdinalRegression_Imp_Relevance_Bound(OrdinalRegression_Relevance_Bound):


    def _init_constraints(self, parameters, init_model_constraints):

        # Upper constraints from initial model
        init_w_l1 = init_model_constraints["w_l1"]
        init_loss = init_model_constraints["loss"]

        C = parameters["C"]

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
