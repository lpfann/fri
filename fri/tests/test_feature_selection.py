import pytest
from fri.genData import genData, genRegressionData
from fri.fri import FRIRegression,  FRIClassification, EnsembleFRI
from sklearn.utils import check_random_state
import numpy as np
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="function")
def randomstate():
   return check_random_state(1337)

def check_interval(interval,n_strong):
        # All strongly relevant features have a lower bound > 0
        assert np.all(interval[0:n_strong,0] > 0)
        np.testing.assert_allclose(interval[n_strong:, 0], 0,atol=1e-02)
        
        # Upper bound checks 
        assert np.all(interval[0:n_strong,1] > 0)
        assert np.all(interval[0:n_strong,1] > interval[0:n_strong,0])


@pytest.mark.parametrize('problem', ["regression","classification"])
@pytest.mark.parametrize('model', [
                                    "Ensemble",
                                    "Single"
                                    ])
@pytest.mark.parametrize('n_strong', [0,1,2])
@pytest.mark.parametrize('n_weak', [0,2,3])
def test_model(problem, model, n_strong, n_weak, randomstate):
    
    n_samples = 500
    n_features = 10

    if problem is "regression":
        gen = genRegressionData
        fri = FRIRegression(random_state = randomstate, C = 1, epsilon = 0)
    else:
        gen = genData
        fri = FRIClassification(random_state = randomstate, C = 1)

    if n_strong + n_weak == 0:
        with pytest.raises(ValueError):
            gen(n_samples=n_samples, n_features=n_features, n_redundant=n_weak, strRel=n_strong,
                                             n_repeated=0, random_state = randomstate)

    else:
        data = gen(n_samples=n_samples, n_features=n_features, n_redundant=n_weak, strRel=n_strong,
                                         n_repeated=0, random_state = randomstate)

        X_orig, y = data
        X_orig = StandardScaler().fit(X_orig).transform(X_orig)
        X = X_orig

        if model is "Ensemble":
            fri = EnsembleFRI(fri, random_state = randomstate)

        try:
            fri.fit(X, y)
        except FitFailedWarning:
            assert False

        interval = fri.interval_

        assert len(fri.allrel_prediction_) == X.shape[1]
        assert len(interval) == X.shape[1]

        check_interval(interval, n_strong)

def test_multiprocessing(randomstate):

    data = genData(n_samples=500, n_features=4, n_redundant=2,strRel=2,
                    n_repeated=0, flip_y=0, random_state=randomstate)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)

    X = np.c_[X_orig, randomstate.normal(size=(len(X_orig), 6))]
    y = list(y)   # regression test: list should be supported

    # Test using the score function
    fri = EnsembleFRI(FRIClassification(random_state=randomstate),n_bootstraps=5,n_jobs=2, random_state=randomstate)
    fri.fit(X, y)
    # non-regression test for missing worst feature:
    assert len(fri.allrel_prediction_) == X.shape[1]
    assert len(fri.interval_) == X.shape[1]

    check_interval(fri.interval_, 2)
