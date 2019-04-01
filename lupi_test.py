from fri.lupi_data import genLupiData
from fri.lupi_model import L1LupiHyperplane
from fri.lupi import FRILupi

from fri.l1models import L1OrdinalRegressor, L1HingeHyperplane
from fri.genData import genOrdinalRegressionData, genClassificationData




X, X_priv, y = genLupiData(n_samples=100, n_features=3, n_strel=2, n_redundant=1, n_repeated=0,
                           n_priv_features=2, n_priv_strel=2, n_priv_redundant=0, n_priv_repeated=0)


'''
model = L1LupiHyperplane(C=0.1, gamma=0.01)

model.fit(X=X, X_priv=X_priv, y=y)

score = model.score(X, y)

'''

f = FRILupi()
f.fit(X, X_priv, y)