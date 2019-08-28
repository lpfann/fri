"""
========================================================
FRI on Regression data
========================================================

An example plot created using :class:`fri.fri.FRIRegression` on Regression data
"""


from fri.genData import genRegressionData
X,y = genRegressionData(n_samples=100, n_features=6, n_strel=2, n_redundant=2,
                        n_repeated=0, random_state=123)

from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(X)

from fri import plot, FRIRegression
fri = FRIRegression()

fri.fit(X_scaled,y)


plot.plotIntervals(fri.interval_)

