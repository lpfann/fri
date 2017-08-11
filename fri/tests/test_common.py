from sklearn.utils.estimator_checks import check_estimator
from fri import (FRIClassification, genData)
from sklearn.utils import check_random_state
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_greater, assert_equal, assert_true,assert_false
from numpy.testing import assert_array_almost_equal, assert_array_equal,assert_raises
import numpy as np
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import StandardScaler
#def test_estimator():
#    return check_estimator(FRIClassification)


def test_strongRelevant():
    generator = check_random_state(0)
    data = genData.genData(n_samples=100, n_features=2, n_redundant=0,strRel=2,
                    n_repeated=0, class_sep=1, flip_y=0, random_state=generator)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)

    X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
    y = list(y)   # regression test: list should be supported

    # Test using the score function
    rbc = FRIClassification(random_state=generator)
    rbc.fit(X, y)
    # non-regression test for missing worst feature:
    assert_equal(len(rbc.allrel_prediction_), X.shape[1])
    assert_equal(len(rbc.interval_), X.shape[1])
    X_r = rbc.transform(X)
    print(rbc.interval_,rbc.allrel_prediction_)
    
    # All the noisy variable were filtered out
    assert_array_equal(X_r, X_orig)

    # All strongly relevant features have a lower bound > 0
    assert_true(np.all(rbc.interval_[0:2,0]>0))

def test_weakRelevant():
    generator = check_random_state(0)
    data = genData.genData(n_samples=100, n_features=2, n_redundant=2,strRel=0,
                    n_repeated=0, class_sep=1, flip_y=0, random_state=generator)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)
    X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
    y = list(y)   # regression test: list should be supported

    # Test using the score function
    rbc = FRIClassification(random_state=generator)
    rbc.fit(X, y)
    # non-regression test for missing worst feature:
    assert_equal(len(rbc.allrel_prediction_), X.shape[1])
    assert_equal(len(rbc.interval_), X.shape[1])
    X_r = rbc.transform(X)
    print(rbc.interval_,rbc.allrel_prediction_)
    # All the noisy variable were filtered out
    assert_array_equal(X_r, X_orig)

    # All weakly relevant features should have a lower bound 0
    assert_false(np.any(rbc.interval_[0:2,0]>0))

def test_allRelevant():
    generator = check_random_state(0)
    data = genData.genData(n_samples=100, n_features=4, n_redundant=2,strRel=2,
                    n_repeated=0, class_sep=1, flip_y=0, random_state=generator)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)

    X = np.c_[X_orig, generator.normal(size=(len(X_orig), 6))]
    y = list(y)   # regression test: list should be supported

    # Test using the score function
    rbc = FRIClassification(random_state=generator)
    rbc.fit(X, y)
    # non-regression test for missing worst feature:
    assert_equal(len(rbc.allrel_prediction_), X.shape[1])
    assert_equal(len(rbc.interval_), X.shape[1])
    X_r = rbc.transform(X)
    print(rbc.interval_,rbc.allrel_prediction_)

    # All the noisy variable were filtered out
    assert_array_equal(X_r, X_orig)

    # All strongly relevant features have a lower bound > 0
    assert_true(np.all(rbc.interval_[0:2,0]>0))
    # All weakly relevant features should have a lower bound 0
    assert_false(np.any(rbc.interval_[2:4,0]>0))
    

def test_norelevant(capsys):
    generator = check_random_state(0)
    data = genData.genData(n_samples=100, n_features=10, n_redundant=0, strRel=0,
                    n_repeated=0, class_sep=1, flip_y=0, random_state=generator)

    X_orig, y = data
    X_orig = StandardScaler().fit(X_orig).transform(X_orig)
    # Test using the score function
    rbc = FRIClassification(random_state=generator)
    #assert_raises(FitFailedWarning,rbc.fit,X_orig,y)
    rbc.fit(X_orig, y)
    assert rbc._best_clf_score < 0.6
    out, err = capsys.readouterr()
    assert out == "WARNING: Bad Model performance!\n"

