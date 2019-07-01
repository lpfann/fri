import numpy as np

import fri
from fri import FRI
from fri.genData import genLupiData
from fri.plot import plot_lupi_intervals


# Clean labels as privileged information

def synthetic_1(n, d, b):
    # n = number of samples
    # d = number of dimensions
    # b = number of ordinal bins

    # Returns: X = data, Xs = privileged feature, y = noisy labels

    w = np.random.normal(size=d)
    bs = np.append(np.sort(np.random.normal(size=b - 1)), np.inf)

    X = np.random.normal(size=(n, d))
    e = np.random.normal(size=n, scale=0.1)
    Xs = np.dot(X, w)

    scores = (Xs + e)[:, np.newaxis]

    y = np.sum(scores - bs >= 0, -1)

    return (X, Xs, y)


# Clean features as privileged information

def synthetic_2(n, d, b):
    # n = number of samples
    # d = number of dimensions
    # b = number of ordinal bins

    # Returns: X = noisy data, Xs = real data, y = labels

    w = np.random.normal(size=d)
    bs = np.append(np.sort(np.random.normal(size=b - 1)), np.inf)

    Xs = np.random.normal(size=(n, d))
    e = np.random.normal(size=(n, d), scale=0.1)
    X = Xs + e

    scores = np.dot(Xs, w)[:, np.newaxis]

    y = np.sum(scores - bs >= 0, -1)

    return (X, Xs, y)

'''
X, Xs, y = synthetic_1(100, 4, 4)
data = np.hstack([X, Xs[:,np.newaxis]])

f = FRI(fri.ProblemName.LUPI_ORDREGRESSION, n_probe_features=3, n_jobs=1, n_param_search=5)

f.fit(data, y, lupi_features=1)

plot_lupi_intervals(f)

'''
randomstate = np.random.seed(123)
X, Xs, y = genLupiData(problemType='ordinalRegression', lupiType='cleanLabels', n_samples=500,
                       n_strel=4, n_weakrel=4, n_repeated=1, n_irrel=1,
                       n_priv_weakrel=0, n_priv_repeated=1, n_priv_irrel=1,
                       random_state=randomstate)



data = np.hstack([X, Xs])

f = FRI(fri.ProblemName.LUPI_ORDREGRESSION, n_probe_features=50, n_jobs=-1, n_param_search=30, verbose=1,
        random_state=randomstate)

f.fit(data, y, lupi_features=3)
f.print_interval_with_class()
plot_lupi_intervals(f)
