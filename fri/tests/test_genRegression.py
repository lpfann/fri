from sklearn.model_selection import train_test_split

from fri import fri, genData
from sklearn.utils import check_random_state
from sklearn import linear_model

def test_shape():
    n = 100
    d = 10
    strRel = 2

    generator = check_random_state(1337)
    X, Y = genData.genRegressionData(n_samples=n, n_features=d, n_redundant=0, strRel=strRel,
                                                  n_repeated=0, random_state=generator)

    assert X.shape == (n, d)

    X, Y = genData.genRegressionData(n_samples=n, n_features=d, n_redundant=2, strRel=strRel,
                                                  n_repeated=1, random_state=generator)

    assert X.shape == (n, d)

    X, Y = genData.genRegressionData(n_samples=n, n_features=d, n_redundant=2, strRel=0,
                                                  n_repeated=1, random_state=generator)

    assert X.shape == (n, d)

def test_data_truth():
    n = 200
    d = 10
    strRel = 5

    generator = check_random_state(1337)
    X, Y = genData.genRegressionData(n_samples=n, n_features=d, n_redundant=0, strRel=strRel,
                                                  n_repeated=0, random_state=generator)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=generator)
    reg = linear_model.LinearRegression(normalize=True)
    reg.fit(X_train, y_train)

    testscore = reg.score(X_test,y_test)
    assert testscore > 0.98