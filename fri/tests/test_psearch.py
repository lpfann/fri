import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from fri import FRIClassification, FRIRegression, FRIOrdinalRegression
from fri.genData import genRegressionData, genClassificationData, genOrdinalRegressionData


@pytest.mark.parametrize('problem', ["regression", "classification", "ordreg"])
@pytest.mark.parametrize('C', [None, 1])
@pytest.mark.parametrize('iter_psearch', [None, 30])
def test_psearch(problem, C, iter_psearch):

    randomstate = check_random_state(1337)
    n_samples = 300
    n_features = 8
    optimum_deviation = 0.3

    if problem is "regression":
        gen = genRegressionData
        model = FRIRegression(random_state=randomstate, verbose=1, C=C, iter_psearch=iter_psearch,
                              optimum_deviation=optimum_deviation)
    elif problem is "classification":
        gen = genClassificationData
        model = FRIClassification(random_state=randomstate, verbose=1, C=C, iter_psearch=iter_psearch,
                                  optimum_deviation=optimum_deviation)
    elif problem is "ordreg":
        gen = genOrdinalRegressionData
        model = FRIOrdinalRegression(random_state=randomstate, verbose=1, C=C, iter_psearch=iter_psearch,
                                     optimum_deviation=optimum_deviation)

    data = gen(n_samples=n_samples, n_features=n_features, n_redundant=2, n_strel=2,
               n_repeated=0, random_state=randomstate)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)
    X = X_orig
    print(model)
    model.fit(X, y)

    # Check the interval output
    interval = model.interval_
    assert len(model.allrel_prediction_) == X.shape[1]
    assert len(interval) == X.shape[1]

