import pytest
from fri.genData import genData, genClassificationData
from sklearn.utils import check_random_state
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_greater, assert_equal, assert_true
from numpy.testing import assert_array_almost_equal, assert_array_equal,assert_raises
import numpy as np

@pytest.fixture(scope="function")
def randomstate():
   return check_random_state(1337)

def test_legacy_method(randomstate):

    generator = randomstate
    n = 10
    d = 4
    strRel = 2
    weakRel  = 2
    class_sep = 0.2
    flip_y = 0
    args = {"n_samples":n, "n_features":d, "n_redundant":weakRel,"strRel":strRel,
    "n_repeated":0, "class_sep":class_sep, "flip_y":flip_y, "random_state":generator}
    X,y = genData(**args)
    # Equal length
    assert_equal(len(X),len(y))
    assert_equal(len(X),n)
    assert_equal(X.shape[1],d)

@pytest.mark.parametrize('n_samples', [2,100,10000])
@pytest.mark.parametrize('n_dim', [1,5,30,100,1000])
def test_shape(n_samples,n_dim):

    X,y = genClassificationData(n_samples = n_samples,n_features = n_dim)
    
    # Equal length
    assert_equal(len(X),len(y))
    # Correct parameters
    assert_equal(len(X),n_samples)
    assert_equal(X.shape[1],n_dim)

@pytest.mark.parametrize('wrong_param', [
    {"n_samples" : 0},
    {"n_features" : 0},
    {"n_samples" : -1},
    {"n_features" : -1},
    ])
def test_wrong_values(wrong_param):
    with pytest.raises(ValueError) as exc:
        genClassificationData(**wrong_param)

@pytest.mark.parametrize('strong', [0,1,2,20,50])
@pytest.mark.parametrize("weak", [0,1,2,20])
@pytest.mark.parametrize('repeated', [0,1,2,5])
@pytest.mark.parametrize('flip_y', [0,0.1,1])
@pytest.mark.parametrize('class_sep', [0,0.5,10])
def test_all_feature_types(strong, weak, repeated, flip_y, class_sep):
    n_samples = 10
    n_features = 100

    if strong == 0 and weak <2:
            with pytest.raises(ValueError):
                X, y = genClassificationData(n_samples = n_samples, n_features = n_features, strRel = strong, n_redundant = weak,
                            flip_y = flip_y, class_sep = class_sep, n_repeated = repeated)
            return  
    if flip_y == 1:
            with pytest.raises(ValueError):
                X, y = genClassificationData(n_samples = n_samples, n_features = n_features, strRel = strong, n_redundant = weak,
                            flip_y = flip_y, class_sep = class_sep, n_repeated = repeated)
            return
    if class_sep == 10:
            with pytest.raises(ValueError):
                X, y = genClassificationData(n_samples = n_samples, n_features = n_features, strRel = strong, n_redundant = weak,
                            flip_y = flip_y, class_sep = class_sep, n_repeated = repeated)
            return  
    X, y = genClassificationData(n_samples = n_samples, n_features = n_features, strRel = strong, n_redundant = weak,
                            flip_y = flip_y, class_sep = class_sep, n_repeated = repeated)
    # Equal length
    assert_equal(len(X),len(y))
    # Correct parameters
    assert_equal(len(X),n_samples)
    assert_equal(X.shape[1],n_features)

def test_class_balance(randomstate):

    X,y = genData(n_samples= 100, random_state=randomstate)

    import collections
    c = collections.Counter(y)
    first_class = c[-1] 
    second_class = c[1] 
    assert np.abs(first_class - second_class) <= 1
