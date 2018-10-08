import sys
import sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from fri import *
from fri.genData import genOrdinalRegressionData
from sklearn.feature_selection import RFECV
from fri.l1models import L1OrdinalRegressor
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV



def get_truth(d, informative, redundant):
    truth = [True]*(informative + redundant) + [False]*(d - (informative + redundant))
    return truth


#############################################################################################################

n = 512
d = 14
strong = 1
weak = 2

print("Used norm: L1 ", strong, weak, (d - (strong + weak)) )

truth = get_truth(d=d, informative=strong, redundant=weak)

A_f1 = []               # Our Method
A_precision = []
A_recall = []
B_f1 = []               # Backward Selection
B_precision = []
B_recall = []
C_f1 = []               # Lasso
C_precision = []
C_recall = []
D_f1 = []               # Ridge
D_precision = []
D_recall = []
for i in range(10):

    X,y = genOrdinalRegressionData(n_samples=n, n_features=d, n_strel=strong, n_redundant=weak, random_state=i)

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    f = FRIOrdinalRegression()
    f.fit(X, y)
    C = f.tuned_C_

    A_prediction = f.get_support()
    A_f1.append(f1_score(truth, A_prediction))
    A_precision.append(precision_score(truth, A_prediction))
    A_recall.append(recall_score(truth, A_prediction))

    ##########################################################################

    estimator = L1OrdinalRegressor(error_type="mae", C=C)
    selector = RFECV(estimator, step=1, cv=5)
    selector.fit(X, y)

    B_prediction = selector.get_support()
    B_f1.append(f1_score(truth, B_prediction))
    B_precision.append(precision_score(truth, B_prediction))
    B_recall.append(recall_score(truth, B_prediction))

    ##########################################################################

    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold=None) # should use 0.00001 as threshold
    sfm.fit(X, y)

    C_prediction = sfm.get_support()
    C_f1.append(f1_score(truth, C_prediction))
    C_precision.append(precision_score(truth, C_prediction))
    C_recall.append(recall_score(truth, C_prediction))

    ##########################################################################

    reg = RidgeCV()
    sfm2 = SelectFromModel(reg, threshold=None) # should use mean as threshold
    sfm2.fit(X, y)

    D_prediction = sfm2.get_support()
    D_f1.append(f1_score(truth, D_prediction))
    D_precision.append(precision_score(truth, D_prediction))
    D_recall.append(recall_score(truth, D_prediction))

    ###########################################################################

avg_A_f1 = np.average(A_f1)
avg_A_precision = np.average(A_precision)
avg_A_recall = np.average(A_recall)

avg_B_f1 = np.average(B_f1)
avg_B_precision = np.average(B_precision)
avg_B_recall = np.average(B_recall)

avg_C_f1 = np.average(C_f1)
avg_C_precision = np.average(C_precision)
avg_C_recall = np.average(C_recall)

avg_D_f1 = np.average(D_f1)
avg_D_precision = np.average(D_precision)
avg_D_recall = np.average(D_recall)

#############################################################################################################

print("-------------------------------------------")

print("A_f1 = ", avg_A_f1)
print("A_precision = ", avg_A_precision)
print("A_recall = ", avg_A_recall)

print("-------------------------------------------")

print("B_f1 = ", avg_B_f1)
print("B_precision = ", avg_B_precision)
print("B_recall = ", avg_B_recall)

print("-------------------------------------------")

print("C_f1 = ", avg_C_f1)
print("C_precision = ", avg_C_precision)
print("C_recall = ", avg_C_recall)

print("-------------------------------------------")

print("D_f1 = ", avg_D_f1)
print("D_precision = ", avg_D_precision)
print("D_recall = ", avg_D_recall)

print("-------------------------------------------")

