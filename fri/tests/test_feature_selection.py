import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from fri import FRI, ProblemName
from fri import genRegressionData, genClassificationData, genOrdinalRegressionData


@pytest.fixture(scope="function")
def randomstate():
    return check_random_state(1337)


@pytest.mark.parametrize("n_weak", [0, 2, 3])
@pytest.mark.parametrize("n_strong", [0, 1, 2])
@pytest.mark.parametrize("problem", ["regression", "classification", "ordreg"])
def test_model(problem, n_strong, n_weak, randomstate):
    n_samples = 300
    n_features = 8

    if problem is "regression":
        gen = genRegressionData
        model = FRI(ProblemName.REGRESSION, random_state=randomstate, verbose=1)

    elif problem is "classification":
        gen = genClassificationData
        model = FRI(ProblemName.CLASSIFICATION, random_state=randomstate, verbose=1)
    elif problem is "ordreg":
        gen = genOrdinalRegressionData
        model = FRI(ProblemName.ORDINALREGRESSION, random_state=randomstate, verbose=1)

    if n_strong + n_weak == 0:
        with pytest.raises(ValueError):
            gen(
                n_samples=n_samples,
                n_features=n_features,
                n_redundant=n_weak,
                n_strel=n_strong,
                n_repeated=0,
                random_state=randomstate,
            )

    else:
        data = gen(
            n_samples=n_samples,
            n_features=n_features,
            n_redundant=n_weak,
            n_strel=n_strong,
            n_repeated=0,
            noise=0,
            random_state=randomstate,
        )

        X_orig, y = data
        X_orig = StandardScaler().fit(X_orig).transform(X_orig)
        X = X_orig

        model.fit(X, y)

        # Check the interval output
        interval = model.interval_
        assert len(model.allrel_prediction_) == X.shape[1]
        assert len(interval) == X.shape[1]

        # Check the score which should be good
        if problem is not "ordreg":
            assert model.score(X[:30], y[:30]) >= 0.8

        n_f = n_strong + n_weak  # Number of relevant features

        # Check how many are selected
        selected = model._n_selected_features()
        # we allow one more false positive
        print(model._get_support_mask())
        print(model.interval_)
        print(model.print_interval_with_class())
        assert n_f == selected

        # Check if all relevant features are selected
        truth = np.ones(n_f)
        assert all(model._get_support_mask()[:n_f] == truth)


def test_multiprocessing(randomstate):
    data = genClassificationData(
        n_samples=100, n_features=3, n_redundant=2, n_strel=1, random_state=randomstate
    )

    X_orig, y = data
    X = StandardScaler().fit(X_orig).transform(X_orig)

    model = FRI(ProblemName.CLASSIFICATION, random_state=randomstate, n_jobs=1)
    model.fit(X, y)

    model = FRI(ProblemName.CLASSIFICATION, random_state=randomstate, n_jobs=2)
    model.fit(X, y)

    model = FRI(ProblemName.CLASSIFICATION, random_state=randomstate, n_jobs=-1)
    model.fit(X, y)


def test_nonbinaryclasses(randomstate):
    n = 90
    d = 2
    X = randomstate.rand(n, d)
    firstclass = [1] * 30
    secondclass = [2] * 30
    thirdclass = [3] * 30
    y = np.array([firstclass, secondclass, thirdclass]).ravel()

    fri = FRI(ProblemName.CLASSIFICATION)
    with pytest.raises(ValueError):
        fri.fit(X, y)
