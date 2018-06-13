import sys
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from fri import *
import fri.plot as plot
from sklearn.feature_selection import RFECV
from fri.l1models import L1OrdinalRegressor
from sklearn.feature_selection import RFE



def get_truth(d, informative, redundant):
    truth = [True]*(informative + redundant) + [False]*(d - (informative + redundant))
    return truth



d = 14
strong = 1
weak = 2

X,y = genOrdinalRegressionData(n_samples=256, n_features=d, n_strel=strong, n_redundant=weak, random_state=0)

f = FRIOrdinalRegression()

f.fit(X,y)

prediction = f.get_support()

truth = get_truth(d=d, informative=strong, redundant=weak)

f1 = f1_score(truth, prediction)
preci = precision_score(truth, prediction)
reca = recall_score(truth, prediction)


#############################################################################################################


estimator = L1OrdinalRegressor(error_type="mae", C=0.1)
#estimator.fit(X,y)
selector = RFECV(estimator, step=1, cv=5, verbose=1)
#selector = RFE(estimator, step=1, verbose=1, n_features_to_select=1)
selector.fit(X, y)

pred = selector.get_support()

f1b = f1_score(truth, pred)
precib = precision_score(truth, pred)
recab = recall_score(truth, pred)

print("f1 = ", f1)
print("precision = ", preci)
print("recall = ", reca)

print("---------------------------------------------------------------")
print("---------------------------------------------------------------")


print("f1b = ", f1b)
print("precisionb = ", precib)
print("recallb = ", recab)