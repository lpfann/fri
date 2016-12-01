"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle, check_random_state
from sklearn import svm
from collections import namedtuple
import itertools


import cvxpy as cvx

class NotFeasibleForParameters(Exception):
    """SVM cannot separate points with this parameters"""


class RelevanceBoundsClassifier(BaseEstimator, ClassifierMixin):
    """ L1-relevance Bounds Classifier

    """

    def __init__(self, C=None, random_state=None,shadow_features=True):
        self.random_state = check_random_state(random_state)
        self.C = C
        self.shadow_f = shadow_features

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        # Use SVM to get optimal solution
        self._performSVM(X, Y)

        # Main Optimization step
        self._main_opt(X, y)

        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.y_[closest]

    def _main_opt(self, X, Y):
        n, d = X.shape()
        rangevector = np.zeros((d, 2))
        shadowrangevector = np.zeros((d, 2))
        omegas = np.zeros((d, 2, d))
        biase = np.zeros((d, 2))

        svmloss = self._svm_loss
        L1 = self._svm_L1
        C  = self.C

        """
        Solver Parameters
        """
        kwargs = {"warm_start": True, "solver": "SCS", "gpu": False, "verbose": False, "parallel": True}
        acceptableStati = [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]

        # Optimize for every dimension
        for di in range(d):
            rangevector[di, 0], \
            omegas[di, 0], \
            biase[di, 0] = self.opt_min(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)
            rangevector[di, 1], \
            omegas[di, 1], \
            biase[di, 1] = self.opt_max(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)
            if self.shadow_f:
                # Shuffle values for single feature
                Xshuffled = np.append(np.random.permutation(X[:, di]).reshape((n, 1)),X,axis=1)
                shadowrangevector[di, 0] = self._opt_min(acceptableStati, 0, d+1, n, kwargs, L1, svmloss, C, Xshuffled,
                                                        Y).bounds
                shadowrangevector[di, 1] = self._opt_max(acceptableStati, 0, d+1, n, kwargs, L1, svmloss, C, Xshuffled,
                                                        Y).bounds

        # Correction through shadow features
        if self.shadow_f:
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

    def _performSVM(self, X, Y):
        if self.C is None:
            # Hyperparameter Optimization over C, starting from minimal C
            min_c = sklearn.svm.l1_min_c(X, Y)
            tuned_parameters = [{'C': min_c * np.logspace(1, 4)}]
        else:
            # Fixed Hyperparameter
            tuned_parameters = [{'C': [self.C]}]

        gridsearch = GridSearchCV(
                             svm.LinearSVC(penalty='l1',
                             loss="squared_hinge",
                             dual=False,
                             random_state=self.randomstate),
                           tuned_parameters,
                           n_jobs=-1, cv=6, verbose=False)
        gridsearch.fit(X, Y)
        C = gridsearch.best_params_['C']

        self._svm_clf=  = best_clf = clf.best_estimator_
        self._svm_coef = best_clf.coef_
        self._svm_bias = best_clf.intercept_[0]
        self._svm_L1 = np.linalg.norm(beta[0], ord=1)

        Y_vector = np.array([Y[:], ] * 1)
        # Hinge loss
        # loss = np.sum(np.maximum(0, 1 - Y_vector * np.inner(beta, X1) - bias))
        # Squared hinge loss
        self._svm_loss = np.sum(np.maximum(0, 1 - Y_vector * (np.inner( self._svm_coef, X) - self._svm_bias))[0]**2)

        self._svm_coef = self._svm_coef[0]

    def _opt_max(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        Y = np.array([Y, ] * 1)
        M = 2 * L1
        xp = cvx.Variable(d)
        omega = cvx.Variable(d)
        omegai = Parameter(d)
        b = cvx.Variable()
        eps = cvx.Variable(n)

        constraints2 = [
            #cvx.abs(omega) <= xp,
            xp >= 0,
            cvx.mul_elemwise(Y.T, X * omega - b) >= 1 - eps,
            eps >= 0,
            cvx.norm(omega, 1) + C * cvx.sum_squares(eps) <= L1 + C * svmloss,
        ]

        # Max 1 Problem
        maxConst1 = [
            xp.T * omegai <= omega.T * omegai,
            xp.T * omegai <= -(omega.T * omegai) + M
        ]
        maxConst1.extend(constraints2[:])
        obj_max1 = cvx.Maximize(xp.T * omegai)
        prob_max1 = cvx.Problem(obj_max1, maxConst1)

        # Max 2 Problem
        maxConst2 = [
            xp.T * omegai <= -(omega.T * omegai),
            xp.T * omegai <= (omega.T * omegai) + M
        ]
        maxConst2.extend(constraints2[:])
        obj_max2 = cvx.Maximize(xp.T * omegai)
        prob_max2 = cvx.Problem(obj_max2, maxConst2)

        dim = np.zeros(d)
        dim[di] = 1
        omegai.value = dim
        valid = False
        prob_max1.solve(**kwargs)
        status = prob_max1.status
        opt_value = 0
        weights = None
        bias = None
        if status in acceptableStati:
            opt_value = np.abs(prob_max1.value)
            weights = omega.value.reshape(d)
            bias = b.value
            valid = True
        prob_max2.solve(**kwargs)
        status = prob_max2.status
        if status in acceptableStati and np.abs(prob_max2.value) > np.abs(opt_value):
            opt_value = np.abs(prob_max2.value)
            weights = omega.value.reshape(d)
            bias = b.value
            valid = True

        if not valid:
            # return softMarginLPOptimizer.Opt_output(0, 0, 0)
            raise NotFeasibleForParameters
        else:
            return softMarginLPOptimizer.Opt_output(opt_value, weights, bias)

    def _opt_min(self, acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y):
        Y = np.array([Y, ] * 1)
        xp = cvx.Variable(d)
        omega = cvx.Variable(d)
        omegai = Parameter(d)
        b = cvx.Variable()
        eps = cvx.Variable(n)

        constraints = [
            # absolute value
            cvx.abs(omega) <= xp,
            # points still correctly classified with soft margin
            cvx.mul_elemwise(Y.T, X * omega - b) >= 1 - eps,
            eps >= 0,
            # L1 reg. and allow slack
            cvx.norm(omega, 1) + C * cvx.sum_squares(eps) <= L1 + C * svmloss,
        ]

        # Min Problem
        obj_min = cvx.Minimize(xp.T * omegai)
        prob_min = cvx.Problem(obj_min, constraints)

        dim = np.zeros(d)
        dim[di] = 1
        omegai.value = dim
        prob_min.solve(**kwargs)
        status = prob_min.status

        if status in acceptableStati:
            return softMarginLPOptimizer.Opt_output(prob_min.value, omega.value.reshape(d), b.value)
        else:
            # return softMarginLPOptimizer.Opt_output(0, 0, 0)
            raise NotFeasibleForParameters