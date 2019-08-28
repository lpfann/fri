import numpy as np
import pytest
from sklearn.utils import check_random_state

from fri import FRI, NORMAL_MODELS, LUPI_MODELS
from fri import quick_generate


@pytest.fixture(scope="function")
def random_state():
    return check_random_state(1337)


@pytest.fixture()
def data(prob):
    return quick_generate(prob)


@pytest.mark.parametrize("problem", NORMAL_MODELS)
def test_normal_model(problem, random_state):
    model = FRI(problem, random_state=random_state)

    X, y = quick_generate(problem, random_state=random_state)

    model.fit(X, y)

    assert len(model.allrel_prediction_) == X.shape[1]


@pytest.mark.parametrize("problem", LUPI_MODELS)
def test_lupi_model(problem, random_state):
    model = FRI(problem, random_state=random_state)

    X, X_p, y = quick_generate(problem, random_state=random_state)
    combined = np.hstack([X, X_p])
    model.fit(combined, y, lupi_features=X.shape[1])

    assert len(model.allrel_prediction_) == combined.shape[1]
