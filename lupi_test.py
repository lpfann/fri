from fri.classification import FRIClassification
from fri.genData import genClassificationData, genLupiData
from fri.l1models import DataHandler, L1HingeHyperplane

X, X_priv, y = genLupiData(genClassificationData, n_samples=1000, n_features=4, n_strel=3, n_redundant=1, n_repeated=0,
                           n_priv_features=2, n_priv_strel=1, n_priv_redundant=1, n_priv_repeated=0)

'''

data = DataHandler(X=X, X_priv=X_priv)

model = L1HingeHyperplane(C=1, gamma=10)

model.fit(data=data, y=y)

score = model.score(X, y)

print(score)

'''

f = FRIClassification()
f.fit(X, y, X_priv)

