import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import fri
from fri import genLupiData
from fri.parameter_searcher import find_best_model


@pytest.fixture(scope="session")
def randomstate():
    return check_random_state(1337)


@pytest.mark.parametrize("n_weak", [0, 2])
@pytest.mark.parametrize("problem", fri.LUPI_MODELS)
def test_baseline_lupi(problem, n_weak, randomstate):
    n_samples = 300

    template = problem.value[0]().get_initmodel_template
    params = problem.value[0]().get_all_parameters()
    data = genLupiData(
        problem,
        n_strel=1,
        n_weakrel=n_weak,
        n_samples=n_samples,
        n_irrel=1,
        n_repeated=0,
        random_state=randomstate,
    )

    X, X_priv, y = data
    X = StandardScaler().fit(X).transform(X)
    X_priv = StandardScaler().fit(X_priv).transform(X_priv)
    combined = np.hstack([X, X_priv])

    iter = 50

    best_model, best_score = find_best_model(
        template,
        params,
        (combined, y),
        randomstate,
        iter,
        verbose=1,
        n_jobs=-2,
        lupi_features=X_priv.shape[1],
    )

    assert best_score > 0.5
