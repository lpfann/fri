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

@pytest.fixture(scope="function",
                params=[(1,0),(2,0),(0,2),(1,2)],
                ids=["1strong","2strong","weak","allrele"])
def regressiondata(request,randomstate):
    strong = request.param[0]
    weak = request.param[1]
    generator = randomstate
    data = genRegressionData(n_samples=200, n_features=4, n_redundant=weak, strRel=strong,
                                         n_repeated=0, random_state=generator)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)
    X = X_orig
    # X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
    # y = list(y)
    return X,y,strong,weak

@pytest.fixture(scope="function",
                params=[(1,0),(2,0),(0,2),(1,2)],
                ids=["1strong","2strong","weak","allrele"])
def classdata(request,randomstate):
    strong = request.param[0]
    weak = request.param[1]
    generator = randomstate
    data = genData(n_samples=200, n_features=4, n_redundant=weak, strRel=strong,
                                         n_repeated=0, random_state=generator)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)
    X = X_orig
    # X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
    # y = list(y)
    return X,y,strong,weak

@pytest.fixture(scope='function', params=[(FRIRegression,None),
                                          (EnsembleFRI,FRIRegression)],
                ids=["Regression","EnsembleRegression"])
def regressionmodel(request,randomstate):
    if request.param[1] is None:
        model = request.param[0](random_state=randomstate)
    else:
        model = request.param[0](request.param[1](random_state=randomstate),random_state=randomstate )
    return model

@pytest.fixture(scope='function', params=[(FRIClassification,None),
                                          (EnsembleFRI,FRIClassification)],
                ids=["Classification","EnsembleClassification"])
def classmodel(request,randomstate):
    if request.param[1] is None:
        model = request.param[0](random_state=randomstate)
    else:
        model = request.param[0](request.param[1](random_state=randomstate),random_state=randomstate )
    return model


def test_regression(regressiondata,randomstate,regressionmodel):
    X, y, strong, weak = regressiondata

    # Test using the score function
    fri = regressionmodel
    try:
        fri.fit(X, y)
    except FitFailedWarning:
        #print(fri._best_clf_score,fri._hyper_C,fri._hyper_epsilon)
        assert False

    assert len(fri.allrel_prediction_) == X.shape[1]
    assert len(fri.interval_) == X.shape[1]

    print(fri._best_clf_score)

    # All strongly relevant features have a lower bound > 0
    assert np.all(fri.interval_[0:strong,0] > 0)
    assert np.all(fri.interval_[strong:weak,0] == 0)
    # Upper bound checks
    assert np.all(fri.interval_[0:strong,1] > 0)
    assert np.all(fri.interval_[0:strong,1] > fri.interval_[0:strong,0])

    # Test prediction
    truth = np.zeros(X.shape[1],dtype=bool)
    truth[:strong+weak] = 1
    # TODO: feat.sel. predicition neu machen für feature elimination oder shadow features
    #assert np.all(fri.allrel_prediction_ == truth)

def test_classification(classdata,randomstate,classmodel):
    X, y, strong, weak = classdata

    # Test using the score function
    fri = classmodel
    try:
        fri.fit(X, y)
    except FitFailedWarning:
        #print(fri._best_clf_score,fri._hyper_C,fri._hyper_epsilon)
        assert False

    assert len(fri.allrel_prediction_) == X.shape[1]
    assert len(fri.interval_) == X.shape[1]

    print(fri._best_clf_score)
    print("Intervals",fri.interval_)
    # All strongly relevant features have a lower bound > 0
    assert np.all(fri.interval_[0:strong,0] > 0)
    assert np.all(fri.interval_[strong:weak,0] == 0)
    # Upper bound checks
    assert np.all(fri.interval_[0:strong,1] > 0)
    assert np.all(fri.interval_[0:strong,1] > fri.interval_[0:strong,0])

    # Test prediction
    truth = np.zeros(X.shape[1],dtype=bool)
    truth[:strong+weak] = 1
    # TODO: feat.sel. predicition neu machen für feature elimination oder shadow features
    #assert np.all(fri.allrel_prediction_ == truth) 


def test_multiprocessing():
    generator = check_random_state(0)
    data = genData(n_samples=200, n_features=4, n_redundant=2,strRel=2,
                    n_repeated=0, class_sep=1, flip_y=0, random_state=generator)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)

    X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
    y = list(y)   # regression test: list should be supported

    # Test using the score function
    fri = EnsembleFRI(FRIClassification(random_state=generator),n_bootstraps=5,n_jobs=2, random_state=generator)
    fri.fit(X, y)
    # non-regression test for missing worst feature:
    assert len(fri.allrel_prediction_) == X.shape[1]
    assert len(fri.interval_) == X.shape[1]

    # All strongly relevant features have a lower bound > 0
    assert np.all(fri.interval_[0:2,0]>0)
    # All weakly relevant features should have a lower bound 0
    assert np.any(fri.interval_[2:4,0]>0) == False
    
