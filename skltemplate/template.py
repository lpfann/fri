"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


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

        self.interval_,\
            self._omegas,\
            self._biase,\
            self._shadowintervals = self._main_opt(x,y)

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
        n,d = X 
        rangevector = np.zeros((d, 2))
        shadowrangevector = np.zeros((d, 2))
        omegas = np.zeros((d, 2, d))
        biase = np.zeros((d, 2))

        #######
        kwargs = {"warm_start": True, "solver": "SCS", "gpu": False, "verbose": False, "parallel": True}
        acceptableStati = [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]

        for di in range(self.d):
            rangevector[di, 0], \
            omegas[di, 0], \
            biase[di, 0] = self.opt_min(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)
            rangevector[di, 1], \
            omegas[di, 1], \
            biase[di, 1] = self.opt_max(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y)
            if shadowF:
                # Shuffle values for single feature
                Xshuffled = np.append(np.random.permutation(X[:, di]).reshape((n,1)),X,axis=1)
                shadowrangevector[di, 0] = self.opt_min(acceptableStati, 0, d+1, n, kwargs, L1, svmloss, C, Xshuffled,
                                                        Y).bounds
                shadowrangevector[di, 1] = self.opt_max(acceptableStati, 0, d+1, n, kwargs, L1, svmloss, C, Xshuffled,
                                                        Y).bounds

        # Correction through shadow features
        if shadowF:
            rangevector -= shadowrangevector
            rangevector[rangevector < 0] = 0

        # Scale to L1
        if L1 > 0:
            rangevector = rangevector / L1
            # shadowrangevector = shadowrangevector / L1
        # round mins to zero
        rangevector[np.abs(rangevector) < 1 * 10 ** -4] = 0
        return rangevector, omegas, biase, shadowrangevector

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
        best_clf = clf.best_estimator_
        beta = best_clf.coef_
        bias = best_clf.intercept_[0]
        L1 = np.linalg.norm(beta[0], ord=1)

        Y_vector = np.array([Y[:], ] * 1)
        # Hinge loss
        # loss = np.sum(np.maximum(0, 1 - Y_vector * np.inner(beta, X1) - bias))
        # Squared hinge loss
        loss = np.sum(np.maximum(0, 1 - Y_vector * (np.inner(beta, X) - bias))[0]**2)

        self.beta = beta[0]
        return L1, loss, C, clf