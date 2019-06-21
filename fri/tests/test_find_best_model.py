import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

import fri
from fri.genData import genLupiData, genRegressionData
from fri.parameter_searcher import find_best_model


@pytest.fixture(scope="session")
def randomstate():
    return check_random_state(1337)


@pytest.mark.parametrize('n_strong', [5, 10])
@pytest.mark.parametrize('n_weak', [0, 2])
@pytest.mark.parametrize('n_priv_strong', [1, 2])
@pytest.mark.parametrize('n_priv_weak', [0, 2])
def test_baseline_lupi_reg(n_strong, n_weak, n_priv_strong, n_priv_weak, randomstate):
    n_samples = 100
    n_features = max(8, n_strong + n_weak)

    gen = genRegressionData
    problem = fri.ProblemName.LUPI_REGRESSION.value(scaling_lupi_loss=1, scaling_lupi_w=1)
    template = problem.get_initmodel_template
    params = problem.get_all_parameters()
    n_priv_features = n_priv_strong + n_priv_weak
    data = genLupiData(gen, n_priv_strel=n_priv_strong, n_priv_redundant=n_priv_weak,
                       n_priv_features=n_priv_features,
                       n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
                       n_repeated=0, random_state=randomstate
                       )

    X, X_priv, y = data
    X = StandardScaler().fit(X).transform(X)
    X_priv = StandardScaler().fit(X_priv).transform(X_priv)
    combined = np.hstack([X, X_priv])

    iter = 50

    best_model, best_score = find_best_model(template, params, (combined, y), randomstate, iter, verbose=1, n_jobs=-2,
                                             lupi_features=n_priv_features)

    assert best_score > 0.9
