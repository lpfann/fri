import cvxpy as cvx
import numpy as np

from fri.lupi_data import genLupiData
from fri.lupi_model import L1LupiHyperplane



X, X_priv, Y = genLupiData(n_samples=100, n_features=3, n_strel=2, n_redundant=1, n_repeated=0,
                           n_priv_features=1, n_priv_strel=1, n_priv_redundant=0, n_priv_repeated=0)


gamma = 1
C = 1
classes_ = np.unique(Y)
(n, d) = X.shape
d_priv = X_priv.shape[1]


##################################################################################
#                                   WORKS !!!!!                                  #
##################################################################################


y = Y.reshape((n ,1))
w = cvx.Variable((d ,1))
w_priv = cvx.Variable((d_priv ,1))
b = cvx.Variable()
b_priv = cvx.Variable()

# Prepare problem.
objective = cvx.Minimize( 0.5 * (cvx.norm(w, 1) + gamma * cvx.norm(w_priv, 1)) + C * cvx.sum(X_priv * w_priv + b_priv))
constraints = [

    cvx.multiply(y, X * w + b) >= 1 - (X_priv * w_priv + b_priv),
    X_priv * w_priv + b_priv >= 0

]

# Solve problem.
problem = cvx.Problem(objective, constraints)
problem.solve(solver="ECOS", max_iters=5000)

# Prepare output and convert from matrices to flattened arrays.
coef_ = np.array(w.value)[np.newaxis]
coef_priv_ = np.array(w_priv.value)[np.newaxis]
intercept_ = b.value
intercept_priv_ = b_priv.value
slack = np.dot(X_priv, coef_priv_.T) + intercept_priv_


##################################################################################
#                                   FAILS !!!!!                                  #
##################################################################################



y2 = Y
w2 = cvx.Variable(d)
w_priv2 = cvx.Variable(d_priv)
b2 = cvx.Variable()
b_priv2 = cvx.Variable()

# Prepare problem.
objective2 = cvx.Minimize( 0.5 * (cvx.norm(w2, 1) + gamma * cvx.norm(w_priv2, 1)) + C * cvx.sum(X_priv * w_priv2 + b_priv2))
constraints2 = [

    cvx.multiply(y2, X * w2 + b2) >= 1 - (X_priv * w_priv2 + b_priv2),
    X_priv * w_priv2 + b_priv2 >= 0

]

# Solve problem.
problem2 = cvx.Problem(objective2, constraints2)
problem2.solve(solver="ECOS", max_iters=5000)

# Prepare output and convert from matrices to flattened arrays.
coef_2 = np.array(w2.value)[np.newaxis]
coef_priv_2 = np.array(w_priv2.value)[np.newaxis]
intercept_2 = b2.value
intercept_priv_2 = b_priv2.value
slack2 = np.dot(X_priv, coef_priv_2.T) + intercept_priv_2

