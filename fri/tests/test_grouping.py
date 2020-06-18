import numpy as np
import pytest
from sklearn.utils import check_random_state

from fri import FRI, NORMAL_MODELS, LUPI_MODELS
from fri import quick_generate
import fri

@pytest.fixture(scope="function")
def random_state():
    return check_random_state(1337)


@pytest.fixture()
def data(prob):
    return quick_generate(prob)


def test_normal_model(random_state):
    problem = fri.ProblemName.CLASSIFICATION
    model = FRI(problem, random_state=random_state)

    X, y = quick_generate(problem, random_state=random_state,n_features=5)

    model.fit(X, y)

    assert len(model.allrel_prediction_) == X.shape[1]

    groups,links = model.get_grouping()
    print(groups)
    assert len(groups)==X.shape[1]