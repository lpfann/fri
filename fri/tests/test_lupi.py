import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import fri
from fri import FRI
from fri.genData import genClassificationData, genLupiData, genRegressionData


@pytest.fixture(scope="function")
def randomstate():
    return check_random_state(1337)


def test_strongly_relevant_class(randomstate):
    lupi_features = 1
    X, X_priv, y = genLupiData(genClassificationData, random_state=randomstate, n_samples=200, n_features=1, n_strel=1,
                               n_redundant=0,
                               n_repeated=0,
                               n_priv_features=lupi_features, n_priv_strel=1, n_priv_redundant=0, n_priv_repeated=0)

    f = FRI(fri.ProblemName.LUPI_CLASSIFICATION, n_probe_features=3, n_jobs=1, n_param_search=5,
            random_state=randomstate)
    X = StandardScaler().fit(X).transform(X)
    X_priv = StandardScaler().fit(X_priv).transform(X_priv)
    combined = np.hstack([X, X_priv])


    f.fit(combined, y, lupi_features=lupi_features)
    assert f.interval_ is not None
    print(f.interval_)
    assert f.interval_[0, 0] > 0, "Normal SRel feature lower bound error"
    assert f.interval_[1, 0] > 0, "Priv SRel feature lower bound error"


def test_error_class():
    lupi_features = 2
    X, X_priv, y = genLupiData(genClassificationData, n_samples=500, n_features=4, n_strel=3, n_redundant=1,
                               n_repeated=0,
                               n_priv_features=lupi_features, n_priv_strel=1, n_priv_redundant=0, n_priv_repeated=0)

    f = FRI("lupi_classification")

    combined = np.hstack([X, X_priv])

    with pytest.raises(ValueError):
        f.fit(combined, y, lupi_features=0)
    d = combined.shape[1]
    with pytest.raises(ValueError):
        f.fit(combined, y, lupi_features=d)
    with pytest.raises(ValueError):
        f.fit(combined, y)


@pytest.mark.parametrize('n_strong', [1, 2])
@pytest.mark.parametrize('n_weak', [0, 2, 3])
@pytest.mark.parametrize('n_priv_strong', [1, 2])
@pytest.mark.parametrize('n_priv_weak', [0, 2, 3])
def test_lupi_model_class(n_strong, n_weak, n_priv_strong, n_priv_weak, randomstate):
    n_samples = 500
    n_features = 8

    gen = genClassificationData
    model = FRI(fri.ProblemName.LUPI_CLASSIFICATION, random_state=randomstate, verbose=1, n_param_search=20, n_jobs=1)

    n_priv_features = n_priv_strong + n_priv_weak
    data = genLupiData(gen, n_priv_strel=n_priv_strong, n_priv_redundant=n_priv_weak,
                       n_priv_features=n_priv_features,
                       n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
                       n_repeated=0, random_state=randomstate, rettruth=True
                       )

    X, X_priv, y, truth = data
    X = StandardScaler().fit(X).transform(X)
    X_priv = StandardScaler().fit(X_priv).transform(X_priv)
    combined = np.hstack([X, X_priv])

    model.fit(combined, y, lupi_features=n_priv_features)

    # Check the interval output
    interval = model.interval_
    print(interval)
    assert len(model.allrel_prediction_) == X.shape[1] + X_priv.shape[1]
    assert len(interval) == X.shape[1] + X_priv.shape[1]

    n_f = n_strong + n_weak + n_priv_strong + n_priv_weak  # Number of relevant features

    # Check how many are selected
    selected = model._n_features()
    print(model._get_support_mask())
    assert n_f == selected or selected == n_f + 1, "Feature Selection not accurate"

    # Check if all relevant features are selected
    assert np.all(model._get_support_mask() == truth)



@pytest.mark.parametrize('n_strong', [2, 4])
@pytest.mark.parametrize('n_weak', [0, 2])
@pytest.mark.parametrize('n_priv_strong', [1, 2])
@pytest.mark.parametrize('n_priv_weak', [0, 2])
def test_lupi_model_regression(n_strong, n_weak, n_priv_strong, n_priv_weak, randomstate):
    n_samples = 100
    n_features = 10

    gen = genRegressionData
    model = FRI(fri.ProblemName.LUPI_REGRESSION, random_state=randomstate, verbose=1, n_param_search=20,
                n_jobs=-1,
                n_probe_features=20)

    n_priv_features = n_priv_strong + n_priv_weak
    data = genLupiData(gen, n_priv_strel=n_priv_strong, n_priv_redundant=n_priv_weak,
                       n_priv_features=n_priv_features,
                       n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
                       n_repeated=0, random_state=randomstate, rettruth=True
                       )

    X, X_priv, y, truth = data
    X = StandardScaler().fit(X).transform(X)
    X_priv = StandardScaler().fit(X_priv).transform(X_priv)
    combined = np.hstack([X, X_priv])

    model.fit(combined, y, lupi_features=n_priv_features)

    # Check the interval output
    interval = model.interval_
    print(interval)
    assert len(model.allrel_prediction_) == X.shape[1] + X_priv.shape[1]
    assert len(interval) == X.shape[1] + X_priv.shape[1]

    n_f = n_strong + n_weak + n_priv_strong + n_priv_weak  # Number of relevant features

    # Check how many are selected
    selected = model._n_features()
    print(model._get_support_mask())
    assert n_f == selected or selected == n_f + 1, "Feature Selection not accurate"

    # Check if all relevant features are selected
    assert np.all(model._get_support_mask() == truth)

def test_strongly_relevant_regression(randomstate):
    lupi_features = 1
    X, X_priv, y = genLupiData(genRegressionData, random_state=randomstate, n_samples=100, n_features=1, n_strel=1,
                               n_redundant=0,
                               n_repeated=0,
                               n_priv_features=lupi_features, n_priv_strel=1, n_priv_redundant=0, n_priv_repeated=0)

    f = FRI(fri.ProblemName.LUPI_REGRESSION, n_probe_features=10, n_jobs=-1,
            n_param_search=100,
            random_state=randomstate, verbose=1)
    X = StandardScaler().fit(X).transform(X)
    X_priv = StandardScaler().fit(X_priv).transform(X_priv)
    combined = np.hstack([X, X_priv])

    f.fit(combined, y, lupi_features=lupi_features)
    assert f.interval_ is not None
    print(f.interval_)
    print(f.allrel_prediction_)
    assert f.interval_[0, 0] > 0, "Normal SRel feature lower bound error"
    assert f.interval_[1, 0] > 0, "Priv SRel feature lower bound error"
