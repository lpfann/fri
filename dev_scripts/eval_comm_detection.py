from numpy.random import RandomState
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from fri import *

RANDOM_STATE = RandomState(seed=4)


def get_truth(d, informative, redundant):
    rest = d - informative - redundant
    truth = [0] * informative + [1] * redundant + [2] * rest
    return truth


def get_toy_set_params(name):
    datasets = {
        "toy_red_sparse": [50, 4, 4],
        "toy_red_dense": [20, 12, 6],
        "toy_nored_sparse": [50, 4, 0],
        "toy_nored_dense": [20, 16, 0]
    }
    return datasets[name]


def test_sets(n_repeats=5):
    repeats = []
    d = 20
    informative = 5
    redundant = 5
    for x in range(n_repeats):
        X, y = genClassificationData(n_samples=200, n_features=d, n_strel=informative, n_redundant=redundant,
                                     random_state=RANDOM_STATE)
        truth = get_truth(d, informative, redundant)
        repeats.append((X, y, truth))

    return repeats


def evaluate(true, pred):
    print("True ", true)
    print("predicted ", pred)
    return metrics.fowlkes_mallows_score(true, pred)


def test(X, y, true):
    X_scaled = StandardScaler().fit_transform(X)
    fri = FRIClassification(optimum_deviation=0.1, parallel=True)
    fri.fit(X_scaled, y)
    clust, link, feat_points, dist_mat = fri.community_detection2(X_scaled, y)

    return evaluate(true, clust)


if __name__ == '__main__':
    sets = test_sets()

    for s in sets:
        result = test(*s)
        print(result)

# result output
# community
