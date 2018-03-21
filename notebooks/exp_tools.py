import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from fri import genClassificationData


def gen_split_feature():
    X, y = genClassificationData(n_samples=200, n_features=5, n_strel=2, n_redundant=0,
                                 n_repeated=0, flip_y=0, random_state=3)

    n = len(X)
    s1 = np.zeros(n)
    s2 = np.zeros(n)

    division = int(n / 2)

    s1[:division] = X[:division, 0]

    s2[division:] = X[division:, 0]

    X[:, 2] = s1
    X[:, 3] = s2

    X_scaled = StandardScaler().fit_transform(X)

    return X_scaled, y


def plotbars(bars, names, X, di):
    d = X.shape[1]
    n_bars = len(bars)

    xticks = np.arange(d)
    width = 0.2
    plt.figure(figsize=(8, 5))
    plt.title("Dimension {}".format(di))

    def plotbar(bar1, name, i):
        if bar1.ndim > 1:
            upper_vals = bar1[:, 1]
            lower_vals = bar1[:, 0]
        else:
            upper_vals = bar1
            lower_vals = np.zeros(d)
        height = upper_vals - lower_vals
        height[height < 0.004] = 0.004
        plt.bar(xticks + i * width, height, width, lower_vals, label=name)

    for bar, name, i in zip(bars, names, range(n_bars)):
        plotbar(bar, name, i)

    plt.axvline(x=di - width / 2, linestyle="--")
    plt.axvline(x=di + n_bars * width - 0.5 * width, linestyle="--")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
