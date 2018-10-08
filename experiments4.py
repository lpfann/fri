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

expsets = {
    "Set 1" : {"n":150, "strong":6, "weak":0,"irr":6},
    "Set 2" : {"n":150, "strong":0, "weak":6,"irr":6},
    "Set 3" : {"n":150, "strong":3, "weak":4,"irr":3},
    "Set 4" : {"n":256, "strong":6, "weak":6,"irr":6},
    "Set 5" : {"n":512, "strong":1, "weak":2,"irr":11},
}

result = []

for name, expset in expsets.items():
    n = expset["n"]
    strong = expset["strong"]
    weak = expset["weak"]
    irr = expset["irr"]
    d = strong + weak + irr

    truth = get_truth(d=d, informative=strong, redundant=weak)

    for i in range(10):
        X, y = genOrdinalRegressionData(n_samples=n, n_features=d, n_strel=strong,
                                        n_redundant=weak, random_state=i)
        X = StandardScaler().fit_transform(X)

        f = FRIOrdinalRegression(optimum_deviation=0.1, parallel=True, shadow_features=True)
        f.fit(X, y)
        C = f.tuned_C_

        prediction = f.get_support()
        f1 = f1_score(truth, prediction)
        prec = precision_score(truth, prediction)
        rec = recall_score(truth, prediction)

        result.append([prec, rec, f1, name, "fri"])
        ##########################################################################

        estimator = L1OrdinalRegressor(error_type="mae", C=C)
        selector = RFECV(estimator, step=1, cv=5)
        selector.fit(X, y)

        prediction = selector.get_support()
        f1 = f1_score(truth, prediction)
        prec = precision_score(truth, prediction)
        rec = recall_score(truth, prediction)

        result.append([prec, rec, f1, name, "rfe"])
        ##########################################################################
'''
        clf = LassoCV()
        sfm = SelectFromModel(clf, threshold=None)  # should use 0.00001 as threshold
        sfm.fit(X, y)

        prediction = sfm.get_support()
        f1 = f1_score(truth, prediction)
        prec = precision_score(truth, prediction)
        rec = recall_score(truth, prediction)

        result.append([prec, rec, f1, name, "lasso"])
        ##########################################################################

        reg = RidgeCV()
        sfm2 = SelectFromModel(reg, threshold=None)  # should use mean as threshold
        sfm2.fit(X, y)

        prediction = sfm2.get_support()
        f1 = f1_score(truth, prediction)
        prec = precision_score(truth, prediction)
        rec = recall_score(truth, prediction)

        result.append([prec, rec, f1, name, "ridge"])
        ###########################################################################
'''

frame = pd.DataFrame(result,columns=["prec","rec","f1","set","method"])

pretty_frame = frame.groupby(["method","set"]).mean().T.stack().reorder_levels([1,0]).sort_index(level=0).round(decimals=2)