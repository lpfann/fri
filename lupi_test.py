from fri.lupi_data import genLupiData
from fri.lupi_model import L1LupiHyperplane
from fri.lupi import FRILupi

from fri.l1models import L1OrdinalRegressor, L1HingeHyperplane
from fri.genData import genOrdinalRegressionData, genClassificationData



class DataHandler(object):
    # Package class to give X and X_priv to gridsearch.fit()
    def __init__(self, X, X_priv):
        self.X = X
        self.X_priv = X_priv
        self.shape = self.X.shape

    def __getitem__(self, x):
        return self.X[x], self.X_priv[x]



X, X_priv, y = genLupiData(n_samples=1000, n_features=6, n_strel=3, n_redundant=2, n_repeated=0,
                           n_priv_features=4, n_priv_strel=3, n_priv_redundant=1, n_priv_repeated=0)




data = DataHandler(X=X, X_priv=X_priv)

model = L1LupiHyperplane(C=1, gamma=10)

model.fit(data=data, y=y)

score = model.score(X, y)

'''

f = FRILupi()
f.fit(X, X_priv, y)
'''
