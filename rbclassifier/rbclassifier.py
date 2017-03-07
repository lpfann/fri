"""
This is a module to be used as a reference for building other modules
"""
from abc import abstractmethod
from multiprocessing.pool import Pool

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle, check_random_state, check_X_y
from sklearn import preprocessing, svm
from sklearn import svm
from collections import namedtuple
from sklearn.exceptions import FitFailedWarning
import rbclassifier.bounds
from multiprocessing import Pool
import cvxpy as cvx

from multiprocessing import Process, Queue, current_process, freeze_support

class NotFeasibleForParameters(Exception):
    """SVM cannot separate points with this parameters"""


class RelevanceBoundsBase(BaseEstimator, SelectorMixin):
    @abstractmethod
    def __init__(self, C=None, random_state=None, shadow_features=True,parallel=False):

        self.random_state = random_state
        self.C = C
        self.shadow_features = shadow_features
        self.parallel = parallel
        self.isRegression = None
        self._hyper_epsilon = None
        self._hyper_C = None

    @abstractmethod
    def fit(self, X, y):

        self.X_ = X
        self.y_ = y

        # Use SVM to get optimal solution
        self._initEstimator(X, y)

        # Main Optimization step
        self._main_opt(X, y)

        # Classify features
        self._get_relevance_mask()

        # Return the classifier
        return self

    def _get_relevance_mask(self,
                #upper_epsilon = 0.0606,
                upper_epsilon = 0.1,

                lower_epsilon = 0.0323):
        rangevector = self.interval_
        prediction = np.zeros(rangevector.shape[0], dtype=np.bool)
        # Treshold for relevancy
        #upper_epsilon = np.median(rangevector[:,1])
        #lower_epsilon = np.median(rangevector[:,0])

        # Weakly relevant ones have high upper bounds
        prediction[rangevector[:, 1] > upper_epsilon] = True
        # Strongly relevant bigger than 0 + some epsilon
        prediction[rangevector[:, 0] > lower_epsilon] = True

        #allrel_prediction = prediction.copy()
        #allrel_prediction[allrel_prediction == 2] = 1

        self.allrel_prediction_ = prediction

        return prediction

    def _get_support_mask(self):
        return self.allrel_prediction_

    def _opt_per_thread(self,bound):
        return bound.solve()

    def _main_opt(self, X, Y):
        n, d = X.shape
        rangevector = np.zeros((d, 2))
        shadowrangevector = np.zeros((d, 2))
        omegas = np.zeros((d, 2, d))
        biase = np.zeros((d, 2))

        svmloss = self._svm_loss
        L1 = self._svm_L1
        C = self._hyper_C

        """
        Solver Parameters
        """
        #kwargs = {"warm_start": False, "solver": "SCS", "gpu": True, "verbose": False, "parallel": False}
        #kwargs = { "solver": "GUROBI","verbose":False}
        kwargs = {"verbose":False}

        work = [self.LowerBound(di, d, n, kwargs, L1, svmloss, C, X, Y,regression=self.isRegression,epsilon=self._hyper_epsilon) for di in range(d)]
        work.extend([self.UpperBound(di, d, n, kwargs, L1, svmloss, C, X, Y,regression=self.isRegression,epsilon=self._hyper_epsilon) for di in range(d)])
        if self.shadow_features:
            work.extend([self.LowerBoundS(di, d, n, kwargs, L1, svmloss, C, X, Y,regression=self.isRegression,epsilon=self._hyper_epsilon,random_state=self.random_state) for di in range(d)])
            work.extend([self.UpperBoundS(di, d, n, kwargs, L1, svmloss, C, X, Y,regression=self.isRegression,epsilon=self._hyper_epsilon,random_state=self.random_state) for di in range(d)])

        def pmap(*args):
                with Pool() as p:
                    return p.map(*args)

        if self.parallel:
            newmap = pmap
        else:
            newmap = map

        done = newmap(self._opt_per_thread, work)

        for finished_bound in done:
            di = finished_bound.di
            i = finished_bound.type

            if not hasattr(finished_bound,"isShadow"):
                rangevector[di, i] = finished_bound.prob_instance.problem.value
                omegas[di, i] = finished_bound.prob_instance.omega.value.reshape(d)
                biase[di, i] =  finished_bound.prob_instance.b.value
            else:
                shadowrangevector[di, i] = finished_bound.prob_instance.problem.value

        #rangevector = np.abs(rangevector)
        self.unmod_interval_ = rangevector.copy()
        
        # Correction through shadow features

        if self.shadow_features:
            rangevector -= shadowrangevector
            rangevector[rangevector < 0] = 0

        # Scale to L1
        if L1 > 0:
            rangevector = rangevector / L1
            # shadowrangevector = shadowrangevector / L1

        # round mins to zero
        rangevector[np.abs(rangevector) < 1 * 10 ** -4] = 0

        self.interval_ = rangevector
        self._omegas = omegas
        self._biase = biase
        self._shadowintervals = shadowrangevector

    @abstractmethod
    def _initEstimator(self, X, Y):
        pass


class RelevanceBoundsClassifier( RelevanceBoundsBase):
    """ L1-relevance Bounds Classifier

    """
    def __init__(self,C=None, random_state=None, shadow_features=True,parallel=False):
        super().__init__(C=C, random_state=random_state, shadow_features=shadow_features,parallel=parallel)
        self.isRegression = False
        self.LowerBound = rbclassifier.bounds.LowerBound
        self.UpperBound = rbclassifier.bounds.UpperBound
        self.LowerBoundS = rbclassifier.bounds.ShadowLowerBound
        self.UpperBoundS = rbclassifier.bounds.ShadowUpperBound

    def _initEstimator(self, X, Y):
        estimator = svm.LinearSVC(penalty='l1', loss="squared_hinge", dual=False,
                                  random_state=self.random_state)
        if self.C is None:
            # Hyperparameter Optimization over C, starting from minimal C
            min_c = svm.l1_min_c(X, Y)
            tuned_parameters = [{'C': min_c * np.logspace(1, 4)}]
        else:
            # Fixed Hyperparameter
            tuned_parameters = [{'C': [self.C]}]

        gridsearch = GridSearchCV(estimator,
                                  tuned_parameters,
                                  scoring="f1",
                                  n_jobs=-1,
                                  cv=5,
                                  verbose=False)
        gridsearch.fit(X, Y)
        self._hyper_C = gridsearch.best_params_['C']
        self._best_clf_score = gridsearch.best_score_
        if self._best_clf_score < 0.7:
            raise FitFailedWarning()

        self._svm_clf = best_clf = gridsearch.best_estimator_
        self._svm_coef = best_clf.coef_
        self._svm_bias = best_clf.intercept_[0]
        self._svm_L1 = np.linalg.norm(self._svm_coef[0], ord=1)

        Y_vector = np.array([Y[:], ] * 1)
       
        prediction = best_clf.decision_function(X)
        self._svm_loss = np.sum(np.maximum(0, 1- Y_vector*prediction))

        self._svm_coef = self._svm_coef[0]

    def fit(self,X,y):
        """A reference implementation of a fitting function for a classifier.
                """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        if len(self.classes_) > 2:
            raise ValueError("Only binary class data supported")
        # Negative class is set to -1 for decision surface
        y = preprocessing.LabelEncoder().fit_transform(y)
        y[y == 0] = -1

        super().fit(X,y)

class RelevanceBoundsRegressor( RelevanceBoundsBase):
    """ L1-relevance Bounds Regressor

    """
    def __init__(self,C=None,epsilon=None, random_state=None, shadow_features=True,parallel=False):
        super().__init__(C=C,random_state=random_state, shadow_features=shadow_features,parallel=parallel)
        self.isRegression = True
        self.epsilon = epsilon
        self.LowerBound = rbclassifier.bounds.LowerBound
        self.UpperBound = rbclassifier.bounds.UpperBound
        self.LowerBoundS = rbclassifier.bounds.ShadowLowerBound
        self.UpperBoundS = rbclassifier.bounds.ShadowUpperBound

    def _initEstimator(self, X, Y):
        estimator = svm.SVR(kernel="linear")

        tuned_parameters = {'C': [self.C],'epsilon':[self.epsilon]}
        if self.C is None:
            tuned_parameters["C"] =  np.linspace(0.001, 100,num=10)
        if self.epsilon is None:
            tuned_parameters["epsilon"] = np.linspace(0.001, 2, num=10)

        gridsearch = GridSearchCV(estimator,
                                  tuned_parameters,
                                  scoring=None,
                                  n_jobs=-1,
                                  cv=3,
                                  verbose=False)
        gridsearch.fit(X, Y)
        self._hyper_C = gridsearch.best_params_['C']
        self._hyper_epsilon = gridsearch.best_params_['epsilon']
        self._best_clf_score = gridsearch.best_score_
        if self._best_clf_score < 0.7:
            raise FitFailedWarning()

        self._svm_clf = best_clf = gridsearch.best_estimator_
        self._svm_coef = best_clf.coef_
        self._svm_bias = best_clf.intercept_[0]
        self._svm_L1 = np.linalg.norm(self._svm_coef, ord=1)
        prediction = best_clf.predict(X)
        self._svm_loss = np.sum(np.abs(Y - prediction))

        self._svm_coef = self._svm_coef[0]

    def fit(self, X, y):
        """A reference implementation of a fitting function for a regressor.
                """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        super().fit(X, y)
