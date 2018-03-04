import numpy as np
from genData import genOrdinalRegressionData
from script import fit

#x,y = genOrdinalRegressionData(n_samples=10, n_features=2, n_target_bins = 2)

x = np.array([[1,1],[1,2],[2,1],[2,2], [4,1],[4,3],[4,2],[5,1]])

y = np.array([0,0,0,0,1,1,1,1])
