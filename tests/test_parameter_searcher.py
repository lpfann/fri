import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import fri
from fri.parameter_searcher import find_best_model
from fri.toydata import genLupiData


@pytest.fixture(scope="session")
def randomstate():
    return check_random_state(1337)


@pytest.mark.parametrize("n_strong", [5, 10])
@pytest.mark.parametrize("n_weak", [0, 2])
def test_baseline_lupi_reg(n_strong, n_weak, randomstate):
    n_samples = 100

    problem = fri.ProblemName.LUPI_REGRESSION.value(
        scaling_lupi_loss=1, scaling_lupi_w=1
    )
    template = problem.get_initmodel_template
    params = problem.get_all_parameters()
    data = genLupiData(
        fri.ProblemName.LUPI_REGRESSION,
        n_strel=n_strong,
        n_weakrel_groups=n_weak,
        n_samples=n_samples,
        n_irrel=1,
        noise=0.0,
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

    assert best_score > 0.9
