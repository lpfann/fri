"""
========================================================
Ensemble on Classification data
========================================================

An example plot created using :class:`fri.fri.EnsembleFRI` on Classification data
"""


from fri import genClassificationData
X,y = genClassificationData(n_samples=100, n_features=6,n_strel=2, n_redundant=2,
                    n_repeated=0, flip_y=0,random_state=123)

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

from fri import plot, FRIClassification, EnsembleFRI
model = FRIClassification()
fri = EnsembleFRI(model)

fri.fit(X_scaled,y)


plot.plotIntervals(fri.interval_)

