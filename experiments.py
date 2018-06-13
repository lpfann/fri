import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
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



#data = np.loadtxt('../Data/Pyrimidines/pyrim.data', delimiter=',')
#X, y = freq_binning(data[:,0:27], data[:,-1])
#train_size = 50
#test_size = 24

#data = np.loadtxt('../Data/MachineCPU/machine.data', delimiter=',')
#X, y = freq_binning(data[:,0:6], data[:,-1])
#train_size = 150
#test_size = 59

#data = np.loadtxt('../Data/Boston/housing.data', delimiter=',')
#X, y = freq_binning(data[:,0:13], data[:,-1])
#train_size = 300
#test_size = 206

#data1 = np.loadtxt('../Data/Bank/bank32nh.data')
#data2 = np.loadtxt('../Data/Bank/bank32nh.test')
#data = np.append(data1, data2, axis=0)
#X, y = freq_binning(data[:,0:32], data[:,-1])
#train_size = 3000
#test_size = 5182

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

train_size = 1000
test_size = 3177
'''

#data = np.loadtxt('../Data/Computer/cpu_act.data', delimiter=',')
#X, y = freq_binning(data[:,0:21], data[:,-1])
#train_size = 4000
#test_size = 4182

#data = np.loadtxt('../Data/California/cal_housing.data', delimiter=',')
#X, y = freq_binning(data[:,0:8], data[:,-1])
#train_size = 5000
#test_size = 15640

#data = np.loadtxt('../Data/Census/census-house/house-price-8L/Prototask.data')
#X, y = freq_binning(data[:,0:8], data[:,-1])
#train_size = 6000
#test_size = 16784


# ---------------------------------------------------------------------------------------------


data = np.loadtxt('../Data/Wine/winequality-red.csv', delimiter=';', skiprows=1)
X = data[:,0:11]
y = data[:,-1]
y = y - 3  # classes should start at 0
X, y = shuffle(X, y, random_state = 0)
train_size = 0.75
test_size = 0.25


# ----------------------------------------------------------------------------------------------








scaler = StandardScaler().fit(X)
X = scaler.transform(X)

#fri_model = FRIOrdinalRegression(C=None, debug=False)
#fri_model.fit(X,y)
#plot.plotIntervals(fri_model.interval_)


#rs = ShuffleSplit(n_splits=20, test_size=test_size, train_size=train_size)

'''
C = fri_model.tuned_C_
n_bins = 6
mze = []
mae = []
mmae = []
for train_idx, test_idx in rs.split(X):
    X_train = X[train_idx]
    y_train = y[train_idx]
    model_mze = L1OrdinalRegressor(error_type="mze", C=C)
    model_mze.fit(X_train, y_train)

    model_mae = L1OrdinalRegressor(error_type="mae", C=C)
    model_mae.fit(X_train, y_train)

    model_mmae = L1OrdinalRegressor(error_type="mmae", C=C)
    model_mmae.fit(X_train, y_train)

    X_test = X[test_idx]
    y_test = y[test_idx]
    score_mze = model_mze.score(X_test, y_test)
    score_mae = model_mae.score(X_test, y_test)
    score_mmae = model_mmae.score(X_test, y_test)

    mze.append(1 - score_mze)
    mae.append((1 - score_mae) * (n_bins - 1))
    mmae.append((1 - score_mmae) * (n_bins - 1))

avg_mze = np.average(mze)
avg_mae = np.average(mae)
avg_mmae = np.average(mmae)


std_mze = np.std(mze)
std_mae = np.std(mae)
std_mmae = np.std(mmae)


print("mze:", avg_mze, "+/-", std_mze)
print("mae:", avg_mae, "+/-", std_mae)
print("mmae:", avg_mmae, "+/-", std_mmae)

'''

#############################################################


C_params = []
bounds = []
for train_idx, test_idx in rs.split(X):
    X_train = X[train_idx]
    y_train = y[train_idx]

    fri_model = FRIOrdinalRegression(C=0.1, debug=False)
    fri_model.fit(X_train, y_train)

    C_params.append(fri_model.tuned_C_)
    bounds.append(fri_model.interval_)
    
avg_C = np.average(C_params)
avg_bounds = np.average(bounds, axis=0)

