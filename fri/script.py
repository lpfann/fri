import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, RegressorMixin, LinearModel
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cvxpy as cvx
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels

X = np.array([[1,1],[1,2],[2,1],[2,2], [4,1],[4,3],[4,2],[5,1]])
y = np.array([0,0,0,0,1,1,1,1])

C=1
(n, d) = X.shape
n_bins = len(np.unique(y))

w = cvx.Variable(d)
b = cvx.Variable(n_bins - 1)
chi = cvx.Variable(n, nonneg=True)
xi = cvx.Variable(n, nonneg=True)

# Prepare problem.
objective = cvx.Minimize(0.5 * cvx.norm(w, 1) + C * cvx.sum(chi + xi))
constraints = []

for i in range(n_bins - 1):
    indices = np.where(y == i)
    constraints.append(X[indices] * w - chi[indices] <= b[i] - 1)

for i in range(1, n_bins):
    indices = np.where(y == i)
    constraints.append(X[indices] * w + xi[indices] >= b[i - 1] + 1)

for i in range(n_bins - 2):
    constraints.append(b[i] <= b[i + 1])


# Solve problem.
problem = cvx.Problem(objective, constraints)
problem.solve(solver="ECOS", max_iters=5000)



coef_ = np.array(w.value)[np.newaxis]
intercept_ = np.array(b.value).flatten()
slack = np.append(chi.value, xi.value)

intercept_2 = b.value
slack2 = np.asarray(xi.value).flatten()