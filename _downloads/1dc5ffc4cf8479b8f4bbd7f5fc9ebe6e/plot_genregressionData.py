"""
========================================================
Generating toy regression data
========================================================

An example plot created using :class:`fri.dataGen.genData` to generate Toy data
"""


from fri import genRegressionData
import matplotlib.pyplot as plt

X,y = genRegressionData(n_samples=50, n_features=10, n_strel=1, n_redundant=0,
                        n_repeated=0, random_state=123)

plt.scatter(X[:,0],y)

