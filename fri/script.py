import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, RegressorMixin, LinearModel
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import cvxpy as cvx
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels


def fit(X, y):
        C = 1
        (n, d) = X.shape
        n_bins = len(np.unique(y))
        bin_size = int(np.floor(n / n_bins))


    
        X_re = np.zeros([bin_size,d,n_bins])


        
        y = np.array(y)  
        for i in range(n_bins):
            index = np.where(y == i)
            X_re[0:bin_size, 0:d, i] = X[index]

        w = cvx.Variable(d)
        chi = cvx.Variable(n_bins, bin_size, nonneg=True)
        xi = cvx.Variable(n_bins, bin_size, nonneg=True)
        b = cvx.Variable(n_bins - 1)

        # Prepare problem.
        objective = cvx.Minimize(0.5 * cvx.norm(w, 1) + C * cvx.sum(chi + xi))
        '''
        constraints = []
        for i in range(max_bin_size):
            for j in range(n_bins):
                #constraints.append(w.T * X_re[i,:,j] <= b[j] - 1 + chi[i,j])
                #constraints.append(w.T * X_re[i,:,j+1] >= b[j] + 1 - xi[i,j+1])
                constraints.append(b[j] <= b[j+1])
                constraints.append(chi.value[i,j] >= 0)
                constraints.append(xi[i,j] >= 0)
        
        constraints = [
            [X_re[:,:,i] * w - chi[i] <= b[i] - 1 for i in range(n_bins)],
            [X_re[:,:,i] * w + xi[i] >= b[i - 1] + 1 for i in range(1, n_bins)],
            [b[i] <= b[i + 1] for i in range(n_bins - 1)],
            [chi[i] >= 0 for i in range(n_bins)],
            [xi[i] >= 0 for i in range(n_bins)]
            ]
        '''

        constraints = []
        for i in range(n_bins - 1):
            constraints.append(X_re[:,:,i] * w - chi[i] <= b[i] - 1)
            constraints.append(chi[i] >= 0)
            constraints.append(xi[i] >= 0)

        for i in range(1, n_bins):
            constraints.append(X_re[:,:,i] * w + xi[i] >= b[i - 1] + 1)

        for i in range(n_bins - 2):
            constraints.append(b[i] <= b[i + 1])

            #[[w.T * X_re[i, :, j] <= b[j] - 1 + chi[i, j] for i in range(max_bin_size)] for j in range(n_bins)],
            #[[w.T * X_re[i, :, j + 1] >= b[j] + 1 - xi.value[i, j + 1] for i in range(max_bin_size)] for j in
            # range(n_bins)],
            #[b[j] <= b[j + 1] for j in range(n_bins - 1)],
            #[[chi.value[i, j] >= 0 for i in range(max_bin_size)] for j in range(n_bins)],
            #[[xi.value[i, j] >= 0 for i in range(max_bin_size)] for j in range(n_bins)]

        # Solve problem.
        problem = cvx.Problem(objective, constraints)
        problem.solve(solver="ECOS", max_iters=5000)

        print(problem.status)
        print(b.value)
        print(w.value)
        print(X_re[0,:,0])
        print(np.matmul(w.value, X_re[0,:,0]))
        print(np.matmul(w.value, X_re[0, :, 1]))
'''
        self.coef_ = np.array(w.value)[np.newaxis]
        self.intercept_ = b.value
        self.slack = np.asarray(xi.value).flatten()
'''
        
