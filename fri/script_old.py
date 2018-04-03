import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, RegressorMixin, LinearModel
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cvxpy as cvx
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from genData import genOrdinalRegressionData

X,y = genOrdinalRegressionData(n_samples=1000, n_features=6, n_target_bins = 4)

X = np.array([[1,1],[1,2],[2,1],[2,2], [4,1],[4,3],[4,2],[5,1]])
y = np.array([0,0,0,0,1,1,1,1])

C = 1
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




w = coef_[0]
b = np.append(intercept_, np.inf)
sum = 0

# Score based on mean zero-one error
for i in range(n):
    val = np.matmul(w, X[i])
    pos = np.argmax(np.less_equal(val, b))
    if y[i] != pos:
        sum += 1

score = 1 - (sum / n)

sum = 0
# Score based on mean absolute error
for i in range(n):
    val = np.matmul(w, X[i])
    sum += np.abs(y[i] - np.argmax(np.less_equal(val, b)))

error = sum / n

# TODO: Check how the error has to be scaled to transform it to a adequate score in case of n_bins == 1
if n_bins == 1:
    score2 = 0
else:
    score2 = 1 - (error / (n_bins - 1))


sum = 0
# Score based on macro-averaged mean absolute error
for i in range(n_bins):
    indices = np.where(y == i)
    X_re = X[indices]
    y_re = y[indices]
    n_c = X_re.shape[0]

    if n_c == 0:
        error_c = 0

    else:
        sum_c = 0

        for j in range(n_c):
            val = np.matmul(w, X_re[j])
            sum_c += np.abs(y_re[j] - np.argmax(np.less_equal(val, b)))

        error_c = sum_c / n_c

    sum += error_c

error3 = sum / n_bins

#TODO: Check how the error has to be scaled to transform it to a adequate score in case of n_bins == 1
if n_bins == 1:
    score3 = 0
else:
    score3 = 1 - (error3 / (n_bins - 1))