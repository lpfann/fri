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

#data1 = np.loadtxt('../Data/ordinal-classification-datasets/automobile/gpor/test_automobile.0', delimiter=' ')
#data2 = np.loadtxt('../Data/ordinal-classification-datasets/automobile/gpor/train_automobile.0', delimiter=' ')
#data = np.append(data1, data2, axis=0)
#X = data[:,0:71]
#y = data[:,-1]
#y = y - 1  # classes should start at 0

#data1 = np.loadtxt('../Data/ordinal-classification-datasets/bondrate/gpor/test_bondrate.0', delimiter=' ')
#data2 = np.loadtxt('../Data/ordinal-classification-datasets/bondrate/gpor/train_bondrate.0', delimiter=' ')
#data = np.append(data1, data2, axis=0)
#X = data[:,0:37]
#y = data[:,-1]
#y = y - 1  # classes should start at 0

#data1 = np.loadtxt('../Data/ordinal-classification-datasets/contact-lenses/gpor/test_contact-lenses.0', delimiter=' ')
#data2 = np.loadtxt('../Data/ordinal-classification-datasets/contact-lenses/gpor/train_contact-lenses.0', delimiter=' ')
#data = np.append(data1, data2, axis=0)
#X = data[:,0:6]
#y = data[:,-1]
#y = y - 1  # classes should start at 0

data1 = np.loadtxt('../Data/ordinal-classification-datasets/eucalyptus/gpor/test_eucalyptus.0', delimiter=' ')
data2 = np.loadtxt('../Data/ordinal-classification-datasets/eucalyptus/gpor/train_eucalyptus.0', delimiter=' ')
data = np.append(data1, data2, axis=0)
X = data[:,0:91]
y = data[:,-1]
y = y - 1  # classes should start at 0

#data1 = np.loadtxt('../Data/ordinal-classification-datasets/newthyroid/gpor/test_newthyroid.0', delimiter=' ')
#data2 = np.loadtxt('../Data/ordinal-classification-datasets/newthyroid/gpor/train_newthyroid.0', delimiter=' ')
#data = np.append(data1, data2, axis=0)
#X = data[:,0:5]
#y = data[:,-1]
#y = y - 1  # classes should start at 0

#data1 = np.loadtxt('../Data/ordinal-classification-datasets/pasture/gpor/test_pasture.0', delimiter=' ')
#data2 = np.loadtxt('../Data/ordinal-classification-datasets/pasture/gpor/train_pasture.0', delimiter=' ')
#data = np.append(data1, data2, axis=0)
#X = data[:,0:25]
#y = data[:,-1]
#y = y - 1  # classes should start at 0

#data1 = np.loadtxt('../Data/ordinal-classification-datasets/squash-stored/gpor/test_squash-stored.0', delimiter=' ')
#data2 = np.loadtxt('../Data/ordinal-classification-datasets/squash-stored/gpor/train_squash-stored.0', delimiter=' ')
#data = np.append(data1, data2, axis=0)
#X = data[:,0:51]
#y = data[:,-1]
#y = y - 1  # classes should start at 0

#data1 = np.loadtxt('../Data/ordinal-classification-datasets/squash-unstored/gpor/test_squash-unstored.0', delimiter=' ')
#data2 = np.loadtxt('../Data/ordinal-classification-datasets/squash-unstored/gpor/train_squash-unstored.0', delimiter=' ')
#data = np.append(data1, data2, axis=0)
#X = data[:,0:52]
#y = data[:,-1]
#y = y - 1  # classes should start at 0

#data1 = np.loadtxt('../Data/ordinal-classification-datasets/tae/gpor/test_tae.0', delimiter=' ')
#data2 = np.loadtxt('../Data/ordinal-classification-datasets/tae/gpor/train_tae.0', delimiter=' ')
#data = np.append(data1, data2, axis=0)
#X = data[:,0:54]
#y = data[:,-1]
#y = y - 1  # classes should start at 0

#data1 = np.loadtxt('../Data/ordinal-classification-datasets/winequality-red/gpor/test_winequality-red.0', delimiter=' ')
#data2 = np.loadtxt('../Data/ordinal-classification-datasets/winequality-red/gpor/train_winequality-red.0', delimiter=' ')
#data = np.append(data1, data2, axis=0)
#X = data[:,0:11]
#y = data[:,-1]
#y = y - 1  # classes should start at 0



###############################################################################################################

X, y = shuffle(X, y, random_state = 0)
scaler = StandardScaler().fit(X)
X = scaler.transform(X)

fri_model = FRIOrdinalRegression(C=None, debug=False)
fri_model.fit(X,y)
#plot.plotIntervals(fri_model.interval_)

train_size = 0.75
test_size = 0.25
rs = ShuffleSplit(n_splits=30, test_size=test_size, train_size=train_size)

C = fri_model.tuned_C_
n_bins = 5
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

