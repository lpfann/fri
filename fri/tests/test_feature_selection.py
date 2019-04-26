import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from fri import FRIClassification, FRIRegression, FRIOrdinalRegression
from fri.genData import genRegressionData, genClassificationData, genOrdinalRegressionData, genLupiData


@pytest.fixture(scope="function")
def randomstate():
    return check_random_state(1337)

@pytest.mark.parametrize('problem', ["regression", "classification", "ordreg"])
@pytest.mark.parametrize('n_strong', [0, 1, 2])
@pytest.mark.parametrize('n_weak', [0, 2, 3])
def test_model(problem, n_strong, n_weak, randomstate):
    n_samples = 300
    n_features = 8

    if problem is "regression":
        gen = genRegressionData
        fri = FRIRegression(random_state=randomstate, verbose=1, C=0.1, optimum_deviation=0.0)
    elif problem is "classification":
        gen = genClassificationData
        fri = FRIClassification(random_state=randomstate, verbose=1, C=0.1, optimum_deviation=0.0)
    elif problem is "ordreg":
        gen = genOrdinalRegressionData
        fri = FRIOrdinalRegression(random_state=randomstate, verbose=1, C=0.1, optimum_deviation=0.0)
    if n_strong + n_weak == 0:
        with pytest.raises(ValueError):
            gen(n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
                n_repeated=0, random_state=randomstate)

    else:
        data = gen(n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
                   n_repeated=0, random_state=randomstate)

        X_orig, y = data
        X_orig = StandardScaler().fit(X_orig).transform(X_orig)
        X = X_orig


        fri.fit(X, y)

        # Check the interval output
        interval = fri.interval_
        assert len(fri.allrel_prediction_) == X.shape[1]
        assert len(interval) == X.shape[1]

        # Check the score which should be good
        if problem is not "ordreg":
            assert fri.score(X[:30], y[:30]) >= 0.8

        n_f =  n_strong + n_weak # Number of relevant features

        # Check how many are selected
        selected = fri._n_features()
        # we allow one more false positive
        print(fri._get_support_mask())
        assert n_f == selected or selected == n_f+1

        # Check if all relevant features are selected
        truth = np.ones(n_f)
        assert all(fri._get_support_mask()[:n_f] == truth)

def test_multiprocessing(randomstate):
    data = genClassificationData(n_samples=500, n_features=10, n_redundant=2, n_strel=2, random_state=randomstate)

    X_orig, y = data
    X = StandardScaler().fit(X_orig).transform(X_orig)


    fri = FRIClassification(random_state=randomstate, n_jobs=1)
    fri.fit(X, y)
    
    fri = FRIClassification(random_state=randomstate, n_jobs=2)
    fri.fit(X, y)

    fri = FRIClassification(random_state=randomstate, n_jobs=-1)
    fri.fit(X, y)

def test_nonbinaryclasses(randomstate):
    n = 90
    d = 2
    X = randomstate.rand(n, d)
    firstclass = [1] * 30
    secondclass = [2] * 30
    thirdclass = [3] * 30
    y = np.array([firstclass, secondclass, thirdclass]).ravel()

    fri = FRIClassification()
    with pytest.raises(ValueError):
        fri.fit(X, y)


@pytest.mark.parametrize('problem', ["regression", "classification", "ordreg"])
@pytest.mark.parametrize('n_strong', [0, 1, 2])
@pytest.mark.parametrize('n_weak', [0, 2, 3])
@pytest.mark.parametrize('n_priv_strong', [1, 2])
@pytest.mark.parametrize('n_priv_weak', [0, 2, 3])
def test_lupi_model(problem, n_strong, n_weak, n_priv_strong, n_priv_weak, randomstate):
    n_samples = 300
    n_features = 8

    if problem is "regression":
        gen = genRegressionData
        fri = FRIRegression(random_state=randomstate, verbose=1, C=0.1, optimum_deviation=0.0)
    elif problem is "classification":
        gen = genClassificationData
        fri = FRIClassification(random_state=randomstate, verbose=1, C=0.1, optimum_deviation=0.0)
    elif problem is "ordreg":
        gen = genOrdinalRegressionData
        fri = FRIOrdinalRegression(random_state=randomstate, verbose=1, C=0.1, optimum_deviation=0.0)
    if n_strong + n_weak == 0:
        with pytest.raises(ValueError):
            gen(n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
                n_repeated=0, random_state=randomstate)

    else:
        n_priv_features = n_priv_strong + n_priv_weak
        data = genLupiData(gen, n_priv_strel=n_priv_strong, n_priv_redundant=n_priv_weak,
                           n_priv_features=n_priv_features,
                           n_samples=n_samples, n_features=n_features, n_redundant=n_weak, n_strel=n_strong,
                           n_repeated=0, random_state=randomstate
                           )

        X, X_priv, y = data
        X = StandardScaler().fit(X).transform(X)
        X_priv = StandardScaler().fit(X_priv).transform(X_priv)

        fri.fit(X, y, X_priv=X_priv)

        # Check the interval output
        interval = fri.interval_
        assert len(fri.allrel_prediction_) == X.shape[1] + X_priv.shape[1]
        assert len(interval) == X.shape[1] + X_priv.shape[1]

        # Check the score which should be good
        if problem is not "ordreg":
            assert fri.score(X[:30], y[:30]) >= 0.8

        n_f = n_strong + n_weak + n_priv_strong + n_priv_weak  # Number of relevant features

        # Check how many are selected
        selected = fri._n_features()
        # we allow one more false positive
        print(fri._get_support_mask())
        assert n_f == selected or selected == n_f + 1

        # Check if all relevant features are selected
        # truth = np.ones(n_f)
        # TODO: rework truth vector check for lupi test
        # assert all(fri._get_support_mask()[:n_f] == truth)
