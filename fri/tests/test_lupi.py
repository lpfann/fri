import fri
import numpy as np
import pytest
from fri import FRI
from fri.genData import genCleanFeaturesAsPrivData
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


@pytest.fixture(scope="function")
def randomstate():
    return check_random_state(1337)


@pytest.mark.parametrize('problem', [fri.ProblemName.LUPI_CLASSIFICATION, fri.ProblemName.LUPI_REGRESSION,
                                     fri.ProblemName.LUPI_ORDREGRESSION])
def test_error_class(problem):
    X, X_priv, y = genCleanFeaturesAsPrivData(n_samples=500, n_strel=3)

    f = FRI(problem)

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
def test_lupi_model_classification(n_strong, n_weak, randomstate):
    n_samples = 10

    model = FRI(fri.ProblemName.LUPI_CLASSIFICATION, random_state=randomstate, verbose=1, n_param_search=100,
                n_probe_features=70, n_jobs=1)

    data = genCleanFeaturesAsPrivData("classification", n_strel=n_strong, n_weakrel_groups=n_weak,
                                      n_samples=n_samples, n_irrel=1, noise=0.0,
                                      n_repeated=0, random_state=randomstate
                                      )

    X, X_priv, y = data
    n_priv_features = X_priv.shape[1]
    X = StandardScaler().fit(X).transform(X)
    X_priv = StandardScaler().fit(X_priv).transform(X_priv)
    combined = np.hstack([X, X_priv])

    model.fit(combined, y, lupi_features=n_priv_features)

    # Check the interval output
    interval = model.interval_
    print(interval)
    assert len(model.allrel_prediction_) == X.shape[1] + X_priv.shape[1]
    assert len(interval) == X.shape[1] + X_priv.shape[1]

    n_f = n_strong + n_weak * 2  # Number of relevant features
    n_f = n_f * 2  # We have two sets with same relevance

    # Check how many are selected
    selected = model._n_selected_features()
    print(model._get_support_mask())
    assert n_f == selected or selected == n_f + 1, "Feature Selection not accurate"


@pytest.mark.parametrize('n_strong', [2, 4])
@pytest.mark.parametrize('n_weak', [0, 2])
def test_lupi_model_regression(n_strong, n_weak, randomstate):
    n_samples = 500

    model = FRI(fri.ProblemName.LUPI_REGRESSION, random_state=randomstate, verbose=1, n_param_search=100,
                n_probe_features=50, n_jobs=-1
                )

    data = genCleanFeaturesAsPrivData("regression", n_strel=n_strong, n_weakrel_groups=n_weak,
                                      n_samples=n_samples, n_irrel=2,
                                      n_repeated=0, random_state=randomstate
                       )

    X, X_priv, y = data
    n_priv_features = X.shape[1]
    X = StandardScaler().fit(X).transform(X)
    X_priv = StandardScaler().fit(X_priv).transform(X_priv)
    combined = np.hstack([X, X_priv])

    model.fit(combined, y, lupi_features=n_priv_features)

    # Check the interval output
    interval = model.interval_
    print(interval)
    assert len(model.allrel_prediction_) == X.shape[1] + X_priv.shape[1]
    assert len(interval) == X.shape[1] + X_priv.shape[1]

    n_f = n_strong + n_weak * 2  # Number of relevant features
    n_f = n_f * 2  # We have two sets with same relevance

    # Check how many are selected
    selected = model._n_selected_features()
    print(model._get_support_mask())
    assert n_f == selected or selected == n_f + 1, "Feature Selection not accurate"


def test_strongly_relevant_ordregression(randomstate):
    lupi_features = 1
    X, X_priv, y = genCleanFeaturesAsPrivData("ordinalRegression", n_strel=1, n_weakrel_groups=0,
                                              n_samples=200, n_irrel=1,
                                              n_repeated=0, random_state=randomstate
                                              )
    f = FRI(fri.ProblemName.LUPI_ORDREGRESSION, n_probe_features=3, n_jobs=1, n_param_search=100,
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


@pytest.mark.parametrize('n_strong', [1, 2])
@pytest.mark.parametrize('n_weak', [0, 1, 2])
def test_lupi_model_ord_regression(n_strong, n_weak, randomstate):
    n_samples = 500

    model = FRI(fri.ProblemName.LUPI_ORDREGRESSION, random_state=randomstate, verbose=1, n_param_search=50,
                n_probe_features=50, n_jobs=-1)

    data = genCleanFeaturesAsPrivData("ordinalRegression", n_strel=n_strong, n_weakrel_groups=n_weak,
                                      n_samples=n_samples, n_irrel=2, noise=0.05,
                       n_repeated=0, random_state=randomstate
                       )

    X, X_priv, y = data
    n_priv_features = X.shape[1]
    X = StandardScaler().fit(X).transform(X)
    X_priv = StandardScaler().fit(X_priv).transform(X_priv)
    combined = np.hstack([X, X_priv])

    model.fit(combined, y, lupi_features=n_priv_features)

    # Check the interval output
    interval = model.interval_
    print(interval)
    assert len(model.allrel_prediction_) == X.shape[1] + X_priv.shape[1]
    assert len(interval) == X.shape[1] + X_priv.shape[1]

    n_f = n_strong + n_weak * 2  # Number of relevant features
    n_f = n_f * 2  # We have two sets with same relevance

    # Check how many are selected
    selected = model._n_selected_features()
    print(model._get_support_mask())
    assert n_f == selected or selected == n_f + 1, "Feature Selection not accurate"
