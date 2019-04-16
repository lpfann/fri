import cvxpy as cvx
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, RegressorMixin, LinearModel
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import check_X_y



class L1LupiHyperplane(BaseEstimator, LinearClassifierMixin):


    def __init__(self, C=1, gamma=1):
        self.C = C
        self.gamma = gamma



    def fit(self, data, y):
        self.classes_ = np.unique(y)
        self.X = data.X
        self.X_priv = data.X_priv
        (n, d) = self.X.shape
        d_priv = self.X_priv.shape[1]
        y = y.reshape((n,1))
        w = cvx.Variable((d,1))
        w_priv = cvx.Variable((d_priv,1))
        b = cvx.Variable()
        b_priv = cvx.Variable()

        # Prepare problem.
        objective = cvx.Minimize( 0.5 * (cvx.norm(w, 1) + self.gamma * cvx.norm(w_priv, 1)) + self.C * cvx.sum(self.X_priv * w_priv + b_priv))
        constraints = [

            cvx.multiply(y, self.X * w + b) >= 1 - (self.X_priv * w_priv + b_priv),
            self.X_priv * w_priv + b_priv >= 0

        ]

        # Solve problem.
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="ECOS", max_iters=5000)

        # Prepare output and convert from matrices to flattened arrays.
        self.coef_ = np.array(w.value)[np.newaxis]
        self.coef_priv_ = np.array(w_priv.value)[np.newaxis]
        self.intercept_ = b.value
        self.intercept_priv_ = b_priv.value
        self.slack = np.dot(self.X_priv, self.coef_priv_.T) + self.intercept_priv_

        return self





    def score(self, X, y):

        X, y = check_X_y(X, y)

        y = LabelEncoder().fit_transform(y)
        y[y == 0] = -1

        X = StandardScaler().fit_transform(X)
        prediction = self.predict(X)

        score = fbeta_score(y, prediction, beta=1, average="weighted")

        return score