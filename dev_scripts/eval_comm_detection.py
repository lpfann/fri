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
    truth = [0] * informative + [1] * redundant + [2] * rest
    return truth

def test_sets(n_repeats=5):
    repeats = []
    d = 20
    informative = 5
    redundant = 5
    for x in range(n_repeats):
        X, y = genClassificationData(n_samples=200, n_features=d, n_strel=informative, n_redundant=redundant,
                                     random_state=RANDOM_STATE,)
        truth = get_truth(d, informative, redundant)
        repeats.append((X, y, truth))

    return repeats


def evaluate(true, pred):
    print("True ", true)
    print("predicted ", pred)
    return metrics.fowlkes_mallows_score(true, pred)


def test(dataset,mode):
    X,y = gen_set(dataset)

    X_scaled = StandardScaler().fit_transform(X)
    fri = FRIClassification(parallel=True)
    fri.fit(X_scaled, y)
    clust, link, feat_points, dist_mat = fri.community_detection2(X_scaled, y)
    
    truth = get_truth(dataset)

    measure =  evaluate(truth, clust)
    print(measure)
    return measure

if __name__ == '__main__':
    test(datasets["partitions"],"both")
    test(datasets["partitions"],"min")
    test(datasets["partitions"],"max")

# result output
# community
