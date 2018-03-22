import numpy as np
from numpy.random import RandomState
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

from fri import *

RANDOM_STATE = RandomState(seed=4)
datasets = {
        "onlyStrong": {"n_features":20,"n_strel":5, "n_redundant":0},
        "onlyRedundant": {"n_features":20,"n_strel":0, "n_redundant":5},
        "both": {"n_features":20,"n_strel":5, "n_redundant":5},
        "partitions": {"n_features":20,"n_strel":3,"n_redundant":12,"partition":[3,3,3,3]}
    }
def gen_set(params):
    return genClassificationData(**params,random_state=RANDOM_STATE)

def get_truth(dataset):
    d = dataset["n_features"]
    informative = dataset["n_strel"]
    redundant = dataset["n_redundant"]
    rest = d - informative - redundant
    truth = [0] * informative + [1] * redundant + [0] * rest
    return truth


def test_sets(dataset, random_state, n_repeats=5):
    repeats = []
    for x in range(n_repeats):
        X, y = genClassificationData(**dataset, random_state=random_state)
        repeats.append((X, y))

    return repeats


def evaluate(true, pred):
    print("True ", true)
    print("predicted ", pred)
    return metrics.fowlkes_mallows_score(true, pred)


def test(X, y, dataset, mode):
    X_scaled = StandardScaler().fit_transform(X)
    fri = FRIClassification(parallel=True)
    fri.fit(X_scaled, y)
    clust, link, feat_points, dist_mat = fri.community_detection2(X_scaled, y, mode=mode)
    
    truth = get_truth(dataset)

    measure =  evaluate(truth, clust)
    # print(measure)
    return measure

if __name__ == '__main__':
    n_repeats = 5
    set = datasets["partitions"]
    data_sets = test_sets(set, RANDOM_STATE, n_repeats=n_repeats)
    results = np.zeros((3, n_repeats))
    for i, s in enumerate(data_sets):
        results[0, i] = test(*s, set, "both")
        results[1, i] = test(*s, set, "min")
        results[2, i] = test(*s, set, "max")
    print(results)
# result output
# community
