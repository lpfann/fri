import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle

from fri.genData import genOrdinalRegressionData
from fri.ordinalregression import FRIOrdinalRegression
import fri.plot as plot
import cvxpy as cvx
from fri.l1models import L1OrdinalRegressor


def freq_binning(X_reg, y_reg):

    n, d = X_reg.shape
    n_bins = 10
    bin_size = int(np.floor(n / n_bins))
    rest = int(n - (bin_size * n_bins))

    # Sort the target values and rearange the data accordingly
    sort_indices = np.argsort(y_reg)
    X = X_reg[sort_indices]
    y = y_reg[sort_indices]

    # Assign ordinal classes as target values
    for i in range(n_bins):
        if i < rest:
            y[(bin_size + 1) * i:] = i
        else:
            y[(bin_size * i) + rest:] = i

    X, y = shuffle(X, y, random_state = 0)

    return X, y



#data = np.loadtxt('../../Data/MachineCPU/machine.data', delimiter=',')
#X, y = freq_binning(data[:,0:6], data[:,-1])

#data = np.loadtxt('../../Data/Pyrimidines/pyrim.data', delimiter=',')
#X, y = freq_binning(data[:,0:27], data[:,-1])

#data = np.loadtxt('../../Data/Boston/housing.data', delimiter=',')
#X, y = freq_binning(data[:,0:13], data[:,-1])

#data1 = np.loadtxt('../../Data/Bank/bank32nh.data')
#data2 = np.loadtxt('../../Data/Bank/bank32nh.test')
#data = np.append(data1, data2, axis=0)
#X, y = freq_binning(data[:,0:32], data[:,-1])

'''
col_names = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"]
dataset = pd.read_csv("../../Data/Abalone/abalone.data", names=col_names)
X = dataset.drop("rings", axis=1)

#X = pd.get_dummies(X_pre, columns=["sex"])
X = X.replace(to_replace='M', value=1)
X = X.replace(to_replace='F', value=2)
X = X.replace(to_replace='I', value=3)

y = dataset["rings"]
y = np.subtract(y, 1)
y[np.where(y==28)[0]] = 27
X, y = shuffle(X, y, random_state = 0)
'''

#data = np.loadtxt('../../Data/Computer/cpu_act.data', delimiter=',')
#X, y = freq_binning(data[:,0:21], data[:,-1])

#data = np.loadtxt('../../Data/California/cal_housing.data', delimiter=',')
#X, y = freq_binning(data[:,0:8], data[:,-1])

data = np.loadtxt('../../Data/Census/census-house/house-price-8L/Prototask.data')
X, y = freq_binning(data[:,0:7], data[:,-1])


X = preprocessing.scale(X)


fri_model = FRIOrdinalRegression()

fri_model.fit(X,y)

print(fri_model.interval_)

plot.plotIntervals(fri_model.interval_)

