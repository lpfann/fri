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

from fri import FRIRegression
fri_model = FRIRegression()
fri_model.fit(X_scaled,y)

from fri.plot import plot_relevance_bars
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1,figsize=(6,4))

# plot the bars on the axis, colored according to fri
plot_relevance_bars(ax,fri_model.interval_,classes=fri_model.relevance_classes_)
