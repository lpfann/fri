import numpy as np
import pytest
from fri import FRI, ProblemName
from fri.genData import genOrdinalRegressionData
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


@pytest.fixture(scope="function")
def randomstate():
    return check_random_state(1337)


@pytest.mark.parametrize('n_weak', [0, 2, 3])
@pytest.mark.parametrize('n_strong', [1, 2])
def test_model(n_strong, n_weak, randomstate):
    n_samples = 300
    n_features = 8
    noise = 0

    gen = genOrdinalRegressionData

    data = gen(n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
               n_repeated=0, noise=noise, random_state=randomstate)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)
    X = X_orig

    model = FRI(ProblemName.ORDINALREGRESSION_IMP, random_state=randomstate, verbose=1, n_jobs=1, n_param_search=10)
    model.fit(X, y, )
    assert model.optim_model_.score(X, y) > 0.9
    # Check the interval output
    interval = model.interval_
    assert len(model.allrel_prediction_) == X.shape[1]
    assert len(interval) == X.shape[1]

    n_f = n_strong + n_weak  # Number of relevant features

    # Check how many are selected
    selected = model._n_selected_features()
    # we allow one more false positive
    print(model._get_support_mask())
    print(model.interval_)
    assert n_f == selected

    # Check if all relevant features are selected
    truth = np.ones(n_f)
    assert all(model._get_support_mask()[:n_f] == truth)


def test_model_no_irrelevant(randomstate):
    n_samples = 200
    n_features = 21
    noise = 0

    gen = genOrdinalRegressionData
    n_strong = 1
    n_weak = 20
    data = gen(n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
               n_repeated=0, noise=noise, random_state=randomstate)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)
    X = X_orig

    model = FRI(ProblemName.ORDINALREGRESSION_IMP, random_state=randomstate, verbose=1, n_jobs=1, n_param_search=10)
    model.fit(X, y, )
    assert model.optim_model_.score(X, y) > 0.9
    # Check the interval output
    interval = model.interval_
    assert len(model.allrel_prediction_) == X.shape[1]
    assert len(interval) == X.shape[1]

    n_f = n_strong + n_weak  # Number of relevant features

    # Check how many are selected
    selected = model._n_selected_features()
    # we allow one more false positive
    print(model._get_support_mask())
    print(model.interval_)
    assert n_f == selected

    # Check if all relevant features are selected
    truth = np.ones(n_f)
    assert all(model._get_support_mask()[:n_f] == truth)
