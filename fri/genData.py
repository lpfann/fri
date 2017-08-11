import numpy as np
from sklearn.utils import check_random_state
from sklearn.datasets import make_regression

def genData(n_samples=100, n_features=2, n_redundant=0,strRel=1,
     n_repeated=0,class_sep=0.2,flip_y=0,random_state=None):
    if not 0 < n_samples:
        raise ValueError("We need at least one sample.")
    if not 0 < n_features:
        raise ValueError("We need at least one feature.")
    if not 0 <= flip_y < 1:
        raise ValueError("Flip percentage has to be between 0 and 1.")
    if not n_redundant%2 == 0:
        raise ValueError("Number of redundant features has to be even.")
    if not n_redundant+n_repeated+strRel<= n_features:
        raise ValueError("Inconsistent number of features")

    randomstate  = check_random_state(random_state)
    weakRel =  n_redundant

    X = np.zeros((n_samples,n_features))
    n = n_samples
    width = 10

    def dummyFeat(n,scale=2):
        return  randomstate.rand(n)*scale - scale/2

    def repeatFeat(feats, i):
        i_pick = randomstate.choice(i)
        return feats[:, i_pick]

    def genStrongRelFeatures(n, strRel, width=10, epsilon=0.05):
        Y = np.ones(n)
        # Generate hyperplane consiting of strongly relevant features
        base = 0 # origin for now # TODO
        n_vec = randomstate.uniform(0.2, 1, int(strRel)) * randomstate.choice([1, -1], int(strRel))
        candidates = randomstate.uniform(-width, width, (n, int(strRel)))
        distPlane = (np.inner(n_vec, candidates) - base)
        # TODO fix this scaling issue
        # epsilon = width*epsilon
        #epsilon = 0.01
        close_candiate_mask = np.abs(distPlane) < epsilon


        # reroll points which are too cloos to hyperplane
        # makes classif. easier
        while np.sum(close_candiate_mask) > 0:
            candidates[close_candiate_mask] = \
                randomstate.uniform(-width, width, (np.sum(close_candiate_mask), int(strRel)))
            distPlane = (np.inner(n_vec, candidates) - base)
            close_candiate_mask = np.abs(distPlane) < epsilon

        Y[distPlane > epsilon] = 1
        Y[distPlane < -epsilon] = -1

        return candidates,Y

    def combFeat(n,strRelFeat):
        # Split each strongly relevant feature into linear combination of it
        weakFeats = np.zeros((n,2))
        for x in range(2):
            cofact = 2 * randomstate.rand() - 1
            weakFeats[:,x] = cofact  * strRelFeat
        return weakFeats

    if strRel+weakRel/2 > 0:
        f_strong, Y = genStrongRelFeatures(n,strRel+weakRel/2,width=width, epsilon=class_sep)
        X[:,:strRel] = f_strong[:,:strRel]
        holdout = f_strong[:,strRel:]
        i = strRel

        for x in range(len(holdout.T)):
            X[:,i:i+2] = combFeat(n_samples,holdout[:,x])
            i += 2

        for x in range(n_repeated):
            X[:,i ] = repeatFeat(X[:,:i],i)
            i += 1
    else:
        Y = randomstate.choice(2,size=n)
        i = 0



    for x in range(n_features-i):
        X[:,i ] = dummyFeat(n_samples,width)
        i += 1

    if flip_y > 0:
        n_flip = np.rint(flip_y * n_samples)
        Y[randomstate.choice(n_samples,n_flip)] *= -1

    return X, Y


def genRegressionData(n_samples: int = 100, n_features: int = 2, n_redundant: int = 0, strRel: int = 1,
                      n_repeated: int = 0, random_state: object = None) -> object:
    if not 0 < n_samples:
        raise ValueError("We need at least one sample.")
    if not 0 < n_features:
        raise ValueError("We need at least one feature.")
    if not n_redundant % 2 == 0:
        raise ValueError("Number of redundant features has to be even.")
    if not n_redundant + n_repeated + strRel <= n_features:
        raise ValueError("Inconsistent number of features")

    randomstate = check_random_state(random_state)
    weakRel = n_redundant

    X = np.zeros((int(n_samples), int(n_features)))
    n = n_samples
    width = 10

    def dummyFeat(n, scale=width):
        return randomstate.rand(int(n)) * scale - scale / 2

    def repeatFeat(feats, i):
        i_pick = randomstate.choice(int(i))
        return feats[:, i_pick]

    def genStrongRelFeatures(n, strRel, width=1):
        candidates, Y = make_regression(n_features=int(strRel),
                                        n_samples=int(n),
                                        noise=width,
                                        n_informative=int(strRel),
                                        random_state=random_state,
                                        shuffle=False)
        return candidates, Y

    def combFeat(n, strRelFeat):
        # Split each strongly relevant feature into linear combination of it
        weakFeats = np.zeros((int(n), 2))
        for x in range(2):
            cofact = 2 * randomstate.rand() - 1
            weakFeats[:, x] = cofact * strRelFeat
        return weakFeats

    if strRel + weakRel / 2 > 0:
        f_strong, Y = genStrongRelFeatures(n, strRel + weakRel / 2, width=width)
        X[:, :strRel] = f_strong[:, :strRel]
        holdout = f_strong[:, strRel:]
        i = strRel

        for x in range(len(holdout.T)):
            X[:, i:i + 2] = combFeat(n_samples, holdout[:, x])
            i += 2

        for x in range(n_repeated):
            X[:, i] = repeatFeat(X[:, :i], i)
            i += 1
    else:
        Y = randomstate.rand(n)
        i = 0

    for x in range(n_features - i):
        X[:, i] = dummyFeat(n_samples, width)
        i += 1

    return X, Y
