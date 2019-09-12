import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import fri
from fri import FRI, genLupiData


@pytest.fixture(scope="function")
def randomstate():
    return check_random_state(1337)


@pytest.mark.parametrize("problem", fri.LUPI_MODELS)
def test_error_class(problem):
    X, X_priv, y = genLupiData(problem, n_samples=500, n_strel=3)

    f = FRI(problem)

    combined = np.hstack([X, X_priv])

    with pytest.raises(ValueError):
        f.fit(combined, y, lupi_features=0)
    d = combined.shape[1]
    with pytest.raises(ValueError):
        f.fit(combined, y, lupi_features=d)
    with pytest.raises(ValueError):
        f.fit(combined, y)


@pytest.mark.parametrize("n_weak", [0, 2])
@pytest.mark.parametrize("problem", fri.LUPI_MODELS)
def test_lupi_model_correctness(problem, n_weak, randomstate):
    n_samples = 100

    model = FRI(
        problem,
        random_state=randomstate,
        verbose=0,
        # slack_regularization=0.1,
        # slack_loss=0.1,
        n_param_search=50,
        n_probe_features=70,
        n_jobs=1,
    )
    print(model)

    data = genLupiData(
        problem,
        n_strel=1,
        n_weakrel=n_weak,
        n_samples=n_samples,
        n_irrel=2,
        # label_noise=0.1,
        n_repeated=0,
        random_state=randomstate,
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
    print(model.print_interval_with_class())
    assert len(model.allrel_prediction_) == X.shape[1] + X_priv.shape[1]
    assert len(interval) == X.shape[1] + X_priv.shape[1]
