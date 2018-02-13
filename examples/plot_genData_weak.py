"""
========================================================
Generating toy data with weakly relevant features
========================================================

An example plot created using :class:`fri.dataGen.genData` to generate Toy data
"""


from fri import genClassificationData
import matplotlib.pyplot as plt

X,y = genClassificationData(n_samples=50, n_features=2,n_strel=0, n_redundant=2,
                    n_repeated=0, flip_y=0,random_state=123)

plt.scatter(X[:,0],X[:,1],c=y)

