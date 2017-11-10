"""
========================================================
Ensemble on Classification data
========================================================

An example plot created using :class:`fri.fri.EnsembleFRI` on Classification data
"""


from fri.genData import genData
X,y = genData(n_samples=100, n_features=6,strRel=2, n_redundant=2,
                    n_repeated=0, flip_y=0,random_state=123)

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)


from fri.fri import EnsembleFRI,FRIClassification
model = FRIClassification()
fri = EnsembleFRI(model)

fri.fit(X_scaled,y)

from fri import plot
plot.plotIntervals(fri.interval_)

