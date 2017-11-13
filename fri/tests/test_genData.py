from fri.genData import genData
from sklearn.utils import check_random_state
from sklearn.utils.testing import ignore_warnings
from sklearn.utils.testing import assert_greater, assert_equal, assert_true
from numpy.testing import assert_array_almost_equal, assert_array_equal,assert_raises

generator = check_random_state(0)
n = 10
d = 4
strRel = 2
weakRel  = 2
class_sep = 0.2
flip_y = 0

args = {"n_samples":n, "n_features":d, "n_redundant":weakRel,"strRel":strRel,
"n_repeated":0, "class_sep":class_sep, "flip_y":flip_y, "random_state":generator}


def test_shape():

    X,y = genData(**args)
    # Equal length
    assert_equal(len(X),len(y))
    assert_equal(len(X),n)
    assert_equal(X.shape[1],d)

    #assert_raises(ValueError,genData.genData,)

def test_wrong_values():

    m_args = dict(args)
    m_args["n_samples"] = 0
    assert_raises(ValueError,genData,**m_args)

    m_args = dict(args)
    m_args["n_features"] = 0
    assert_raises(ValueError,genData,**m_args)

    m_args = dict(args)
    m_args["flip_y"] = 2
    assert_raises(ValueError,genData,**m_args)

    m_args = dict(args)
    m_args["strRel"] = 1
    m_args["n_redundant"] = 2
    m_args["n_repeated"] = 1
    m_args["n_features"] = 2 # less total features then specified
    assert_raises(ValueError,genData,**m_args)


def test_noise_features():

    m_args = dict(args)
    m_args["strRel"] = 0
    m_args["n_redundant"] = 0
    m_args["n_repeated"] = 0
    m_args["n_features"] = 5 # less total features then specified
    
    X,y = genData(**args)
    # Equal length
    assert_equal(len(X),len(y))
    assert_equal(len(X),n)
    assert_equal(X.shape[1],d)
