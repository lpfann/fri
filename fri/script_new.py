import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, RegressorMixin, LinearModel
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cvxpy as cvx
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from genData import genOrdinalRegressionData

X,y = genOrdinalRegressionData(n_samples=28, n_features=2, n_target_bins = 4)

#X = np.array([[1,1],[1,2],[2,1],[2,2], [4,1],[4,3],[4,2],[5,1]])
#y = np.array([0,0,0,0,1,1,1,1])

C = 1
(n, d) = X.shape
n_bins = len(np.unique(y))
bin_size = int(np.floor(n / n_bins))

X_re = []
y = np.array(y)
for i in range(n_bins):
     indices = np.where(y == i)
     X_re.append(X[indices])

w = cvx.Variable(shape=(d,1))
b = cvx.Variable(shape=(n_bins - 1,1))

chi = []
xi = []
for i in range(n_bins):
    n_x = len(np.where(y == i)[0])
    chi.append(cvx.Variable(shape=(n_x,1)))
    xi.append(cvx.Variable(shape=(n_x,1)))

objective = cvx.Minimize(0.5 * cvx.norm(w, 1) + C * cvx.sum(cvx.hstack(chi) + cvx.hstack(xi)))

constraints = []
for i in range(n_bins - 1):
    constraints.append(X_re[i] * w - chi[i] <= b[i] - 1)
    constraints.append(chi[i] >= 0)
    constraints.append(xi[i] >= 0)

for i in range(1, n_bins):
    constraints.append(X_re[i] * w + xi[i] >= b[i - 1] + 1)

for i in range(n_bins - 2):
    constraints.append(b[i] <= b[i + 1])


problem = cvx.Problem(objective, constraints)
problem.solve(solver="ECOS", max_iters=5000)

print(problem.status)
print("b = ", b.value)
print("w = ", w.value)
print("chi[0] = ", chi[0].value)
print(X_re[0][0])
