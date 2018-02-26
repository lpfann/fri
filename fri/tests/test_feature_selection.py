import numpy as np
import pytest
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from fri import FRIClassification, FRIRegression, EnsembleFRI
from fri.genData import genRegressionData, genClassificationData


@pytest.fixture(scope="function")
def randomstate():
    return check_random_state(1337)


def check_interval(interval, n_strong):
    # All strongly relevant features have a lower bound > 0
    assert np.all(interval[0:n_strong, 0] > 0)
    # All other features are zero or very close to it
    np.testing.assert_allclose(interval[n_strong:, 0], 0, atol=1e-02)

    # Upper bounds of relevant features also bigger than zero
    assert np.all(interval[0:n_strong, 1] > 0)
    assert np.all(interval[0:n_strong, 1] >= interval[0:n_strong, 0])  # TODO: check what consequences this has


@pytest.mark.parametrize('problem', ["regression", "classification"])
@pytest.mark.parametrize('model', [
    "Ensemble",
    "Single"
])
@pytest.mark.parametrize('n_strong', [0, 1, 2])
@pytest.mark.parametrize('n_weak', [0, 2, 3])
def test_model(problem, model, n_strong, n_weak, randomstate):
    n_samples = 300
    n_features = 8

    if problem is "regression":
        gen = genRegressionData
        fri = FRIRegression(random_state=randomstate, C=1, debug=True)
    else:
        gen = genClassificationData
        fri = FRIClassification(random_state=randomstate, C=1, debug=True)

    if n_strong + n_weak == 0:
        with pytest.raises(ValueError):
            gen(n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
                n_repeated=0, random_state=randomstate)

    else:
        data = gen(n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
                   n_repeated=0, random_state=randomstate)

        X_orig, y = data
        X_orig = StandardScaler().fit(X_orig).transform(X_orig)
        X = X_orig

        if model is "Ensemble":
            fri = EnsembleFRI(fri, random_state=randomstate)

        fri.fit(X, y)

        # Check the interval output
        interval = fri.interval_
        assert len(fri.allrel_prediction_) == X.shape[1]
        assert len(interval) == X.shape[1]
        check_interval(interval, n_strong)
        # Check the score which should be good
        assert fri.score(X[:30], y[:30]) >= 0.8
        # Check the feature selection methods providing boolean masks and #n of selected features
        assert fri.n_features_() == n_strong + n_weak
        truth = np.zeros(n_features)
        truth[:n_strong + n_weak] = 1
        assert all(fri._get_support_mask() == truth)


def test_multiprocessing(randomstate):
    data = genClassificationData(n_samples=500, n_features=10, n_redundant=2, n_strel=2, random_state=randomstate)

    X_orig, y = data
    X = StandardScaler().fit(X_orig).transform(X_orig)

    fri = FRIClassification(random_state=randomstate, parallel=True)
    fri.fit(X, y)
    check_interval(fri.interval_, 2)

    efri = EnsembleFRI(FRIClassification(), random_state=randomstate, n_jobs=2)
    efri.fit(X, y)
    check_interval(fri.interval_, 2)


@pytest.mark.parametrize('problem', ["regression", "classification"])
@pytest.mark.parametrize('hyperParam', [1, None])
def test_random_data(randomstate, problem, hyperParam):
    n = 100
    d = 5
    X = randomstate.rand(n, d)

    if problem is "regression":
        fri = FRIRegression(random_state=randomstate, C=hyperParam, epsilon=hyperParam)
        y = randomstate.rand(n)
    else:
        fri = FRIClassification(random_state=randomstate, C=hyperParam)
        y = np.ones(n)
        # invert first half of labels
        half = int(n / 2)
        y[:half] *= -1

    # Test with expectation to get warning
    with pytest.raises(FitFailedWarning):
        fri.fit(X, y)


def test_nonbinaryclasses(randomstate):
    n = 90
    d = 2
    X = randomstate.rand(n, d)
    firstclass = [1] * 30
    secondclass = [2] * 30
    thirdclass = [3] * 30
    y = np.array([firstclass, secondclass, thirdclass]).ravel()

    fri = FRIClassification()
    with pytest.raises(ValueError):
        fri.fit(X, y)
