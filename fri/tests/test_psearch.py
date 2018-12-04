import pytest
from fri import FRIClassification, FRIRegression, FRIOrdinalRegression
from fri.genData import genRegressionData, genClassificationData, genOrdinalRegressionData
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


@pytest.fixture(scope="function")
def randomstate():
    return check_random_state(1337)

@pytest.mark.parametrize('problem', ["regression", "classification", "ordreg"])
@pytest.mark.parametrize('C', [None, 1])
@pytest.mark.parametrize('iter_psearch', [None, 3, 30])
def test_psearch(problem,C,iter_psearch, randomstate):
    n_samples = 300
    n_features = 8

    if problem is "regression":
        gen = genRegressionData
        fri = FRIRegression(random_state=randomstate, debug=True, C=C,iter_psearch=iter_psearch, optimum_deviation=0.0)
    elif problem is "classification":
        gen = genClassificationData
        fri = FRIClassification(random_state=randomstate, debug=True, C=C,iter_psearch=iter_psearch, optimum_deviation=0.0)
    elif problem is "ordreg":
        gen = genOrdinalRegressionData
        fri = FRIOrdinalRegression(random_state=randomstate, debug=True, C=C,iter_psearch=iter_psearch, optimum_deviation=0.0)

    data = gen(n_samples=n_samples, n_features=n_features, n_redundant=2, n_strel=2,
               n_repeated=0, random_state=randomstate)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)
    X = X_orig

    fri.fit(X, y)

    # Check the interval output
    interval = fri.interval_
    assert len(fri.allrel_prediction_) == X.shape[1]
    assert len(interval) == X.shape[1]

