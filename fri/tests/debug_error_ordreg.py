import numpy
import numpy as np

import fri
from fri import FRI


def synthetic_1(n: int = 500, d: int = 1, irr: int = 1, b: int = 3, seed: int = 123) -> [numpy.ndarray, numpy.ndarray,
                                                                                         numpy.ndarray]:
    """

    Parameters
    ----------
    n number of samples
    d  number of dimensions
    irr number of irrelevant features
    b  number of ordinal bins
    seed random seed

    Returns
    -------
    X = data, Xs = privileged feature, y = noisy labels
    """

    random_state = np.random.RandomState(seed)

    w = random_state.normal(size=d)
    bs = np.append(np.sort(random_state.normal(size=b - 1)), np.inf)

    X = random_state.normal(size=(n, d + irr))
    e = random_state.normal(size=n, scale=0.1)
    Xs = np.dot(X[:, :d], w)

    scores = (Xs + e)[:, np.newaxis]

    y = np.sum(scores - bs >= 0, -1)

    return (X, Xs[:, np.newaxis], y)


# Clean features as privileged information

def synthetic_2(n: int = 500, d: int = 1, irr: int = 1, b: int = 3, noise: float = 0.1, seed: int = 123) -> [
    numpy.ndarray, numpy.ndarray,
    numpy.ndarray]:
    """

    Parameters
    ----------
    n number of samples
    d  number of dimensions
    irr number of irrelevant features
    b  number of ordinal bins
    seed random seed

    Returns
    -------
     X = noisy data, Xs = real data, y = labels
    """

    random_state = np.random.RandomState(seed)

    w = random_state.normal(size=d)
    bs = np.append(np.sort(random_state.normal(size=b - 1)), np.inf)

    Xs = random_state.normal(size=(n, d))
    e = random_state.normal(size=(n, d), scale=noise)

    X = Xs + e
    X_irr = random_state.normal(size=(n, irr))
    X = np.hstack([X, X_irr])

    scores = np.dot(Xs, w)[:, np.newaxis]

    y = np.sum(scores - bs >= 0, -1)

    return (X, Xs, y)


def test_debug():
    rs = 1337
    X, X_priv, y = synthetic_2(n=500, d=2, irr=5, b=5, noise=0.3, seed=rs)
    print(X.shape, X_priv.shape)
    f = FRI(fri.ProblemName.LUPI_ORDREGRESSION, n_probe_features=20, n_jobs=-1, n_param_search=20,
            random_state=rs, verbose=1)
    # X = StandardScaler().fit(X).transform(X)
    # X_priv = StandardScaler().fit(X_priv).transform(X_priv)
    combined = np.hstack([X, X_priv])

    f.fit(combined, y, lupi_features=2)
    assert f.interval_ is not None
    print(f.interval_)
    print(f.allrel_prediction_)
    print(f.relevance_classes_)
