import numpy as np
from fri.genData import genOrdinalRegressionData
from fri.script import fit
from fri.ordinalregression import FRIOrdinalRegression
import fri.plot as plot
import cvxpy as cvx
from fri.l1models import L1OrdinalRegressor

#x,y = genOrdinalRegressionData(n_samples=1000, n_features=2, n_target_bins = 2)

x = np.array([[1,1],[1,2],[2,1],[2,2], [4,1],[4,3],[4,2],[5,1]])

y = np.array([0,0,0,0,1,1,1,1])

'''
fri_model = FRIOrdinalRegression()

# Fit to data
fri_model.fit(x,y)

# Print out feature relevance intervals
print(fri_model.interval_)

# ### Plot results
plot.plotIntervals(fri_model.interval_)

'''
model = L1OrdinalRegressor()

model.fit(x,y)

x = model.score(x,y)
