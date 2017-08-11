"""
This is a module to be used as a reference for building other modules
"""
from abc import abstractmethod
from multiprocessing import Pool

import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_X_y ,check_random_state
from sklearn.utils.multiclass import unique_labels


import fri.bounds


class NotFeasibleForParameters(Exception):
    """SVM cannot separate points with this parameters
    """


class FRIBase(BaseEstimator, SelectorMixin):
    """Base class for interaction with program
    
    Attributes
    ----------
    allrel_prediction_ : array of booleans
        Truth value for each feature if it is relevant (weakly OR strongly)
    C : float , optional
        Regularization parameter, default obtains the hyperparameter through gridsearch optimizing accuracy
    interval_ : array of [float,float]
        Feature relevance intervals
    parallel : boolean, optional
        Enables parallel computation of feature intervals
    random_state : object
        Set seed for random number generation.
    shadow_features : boolean, optional
        Enables noise reduction using feature permutation results.
    """
    @abstractmethod
    def __init__(self,isRegression, C=None, random_state=None, shadow_features=True,parallel=False,n_resampling=3):
        """Summary
        
        Parameters
        ----------
        C : float , optional
            Regularization parameter, default obtains the hyperparameter through gridsearch optimizing accuracy
        random_state : object
            Set seed for random number generation.
        shadow_features : boolean, optional
            Enables noise reduction using feature permutation results.
        parallel : boolean, optional
            Enables parallel computation of feature intervals
        """

        self.random_state = check_random_state(random_state)
        self.C = C
        self.shadow_features = shadow_features
        self.parallel = parallel
        self.isRegression = isRegression
        self.n_resampling = n_resampling
        self._hyper_epsilon = None
        self._hyper_C = None

    @abstractmethod
    def fit(self, X, y):
        """Summary
        
        Parameters
        ----------
        X : array_like
            Data matrix
        y : array_like
            Response variable
        
        Returns
        -------
        FRIBase
            Instance
        """
        self.X_ = X
        self.y_ = y

        # Use SVM to get optimal solution
        self._initEstimator(X, y)

        if self._best_clf_score < 0.6:
             print("WARNING: Bad Model performance!")

        # Main Optimization step
        self._main_opt(X, y)

        # Classify features
        self._get_relevance_mask()

        # Return the classifier
        return self

    def _get_relevance_mask(self,
                upper_epsilon = 0.1,
                lower_epsilon = 0.0323):
        """Determines relevancy using feature relevance interval values
        
        Parameters
        ----------
        upper_epsilon : float, optional
            Threshold for upper bound of feature relevance interval
        lower_epsilon : float, optional
            Threshold for lower bound of feature relevance interval
        
        Returns
        -------
        boolean array
            Relevancy prediction for each feature
        """
        rangevector = self.interval_
        prediction = np.zeros(rangevector.shape[0], dtype=np.bool)

        # Weakly relevant ones have high upper bounds
        prediction[rangevector[:, 1] > upper_epsilon] = True
        # Strongly relevant bigger than 0 + some epsilon
        prediction[rangevector[:, 0] > lower_epsilon] = True

        self.allrel_prediction_ = prediction

        return prediction

    def n_features_(self):
        """

        Returns the number of selected features.
        -------

        """
        return sum(self.allrel_prediction_)

    def _get_support_mask(self):
        """Method for SelectorMixin
        
        Returns
        -------
        boolean array
            
        """
        return self.allrel_prediction_

    def _opt_per_thread(self,bound):
        """
        Worker thread method for parallel computation
        """
        return bound.solve()

    def _main_opt(self, X, Y):
        """ Main calculation function. 
            Creates LP for each bound and distributes them depending on parallel flag.
        
        Parameters
        ----------
        X : array_like
            standardized data matrix
        Y : array_like
            response vector
        
        """
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

        """
        Create tasks for
        """
        work = [self.LowerBound(di, d, n, kwargs, L1, svmloss, C, X, Y,regression=self.isRegression,epsilon=self._hyper_epsilon) for di in range(d)]
        work.extend([self.UpperBound(di, d, n, kwargs, L1, svmloss, C, X, Y,regression=self.isRegression,epsilon=self._hyper_epsilon) for di in range(d)])
        if self.shadow_features:
            for nr in range(self.n_resampling):
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
                shadowrangevector[di, i] += (finished_bound.prob_instance.problem.value / self.n_resampling) # Get the mean of all shadow samples

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


class FRIClassification(FRIBase):
    """Class for Classification data
    
    Attributes
    ----------
    LowerBound : LowerBound
        Class for lower bound
    LowerBoundS : ShadowLowerBound
        Class for lower bound noise reduction (shadow)
    UpperBound : UpperBound
        Class for upper Bound 
    UpperBoundS : ShadowUpperBound
        Class for upper bound noise reduction (shadow)
    
    """
    LowerBound = fri.bounds.LowerBound
    UpperBound = fri.bounds.UpperBound
    LowerBoundS = fri.bounds.ShadowLowerBound
    UpperBoundS = fri.bounds.ShadowUpperBound
    def __init__(self,C=None, random_state=None, shadow_features=True,parallel=False,n_resampling=3):
        """Initialize a solver for classification data
        
        
        Parameters
        ----------
        C : float , optional
            Regularization parameter, default obtains the hyperparameter through gridsearch optimizing accuracy
        random_state : object
            Set seed for random number generation.
        shadow_features : boolean, optional
            Enables noise reduction using feature permutation results.
        parallel : boolean, optional
            Enables parallel computation of feature intervals
        """
        super().__init__(C=C, random_state=random_state, shadow_features=shadow_features,parallel=parallel,n_resampling=n_resampling, isRegression=False)

    def _initEstimator(self, X, Y):
        estimator = svm.LinearSVC(penalty='l2', loss="squared_hinge", dual=False,
                                  random_state=self.random_state)
        if self.C is None:
            # Hyperparameter Optimization over C, starting from minimal C
            min_c = svm.l1_min_c(X, Y)
            tuned_parameters = [{'C': min_c * np.logspace(1, 4)}]
        else:
            # Fixed Hyperparameter
            tuned_parameters = [{'C': [self.C]}]

        n = len(X)
        if n <= 20:
            cv = 3
        else:
            cv = 7

        gridsearch = GridSearchCV(estimator,
                                  tuned_parameters,
                                  scoring="f1",
                                  n_jobs=-1 if self.parallel else 1,
                                  cv=cv,
                                  verbose=False)
        gridsearch.fit(X, Y)
        self._hyper_C = gridsearch.best_params_['C']
        self._best_clf_score = gridsearch.best_score_

        self._svm_clf = best_clf = gridsearch.best_estimator_
        self._svm_coef = best_clf.coef_
        self._svm_bias = -best_clf.intercept_[0]
        self._svm_L1 = np.linalg.norm(self._svm_coef[0], ord=1)

        Y_vector = np.array([Y[:], ] * 1)

        prediction = best_clf.decision_function(X)
        self._svm_loss = np.sum(np.maximum(0, 1- Y_vector*prediction))

        self._svm_coef = self._svm_coef[0]

    def fit(self,X,y):
        """A reference implementation of a fitting function for a classifier.
        
        Parameters
        ----------
        X : array_like
            standardized data matrix
        y : array_like
            label vector
        
        Raises
        ------
        ValueError
            Only binary classification.
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

class FRIRegression(FRIBase):
    """Class for regression data

        Attributes
        ----------
        epsilon : float, optional
            epsilon margin, default is using value provided by gridsearch
        LowerBound : LowerBound
            Class for lower bound
        LowerBoundS : ShadowLowerBound
            Class for lower bound noise reduction (shadow)
        UpperBound : UpperBound
            Class for upper Bound
        UpperBoundS : ShadowUpperBound
            Class for upper bound noise reduction (shadow)

        """
    LowerBound = fri.bounds.LowerBound
    UpperBound = fri.bounds.UpperBound
    LowerBoundS = fri.bounds.ShadowLowerBound
    UpperBoundS = fri.bounds.ShadowUpperBound

    def __init__(self,C=None,epsilon=None, random_state=None, shadow_features=True,parallel=False,n_resampling=3):
        super().__init__(C=C, random_state=random_state, shadow_features=shadow_features, parallel=parallel, n_resampling=n_resampling, isRegression=True)
        self.epsilon = epsilon

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
                                  cv=7,
                                  verbose=False)
        gridsearch.fit(X, Y)
        self._hyper_C = gridsearch.best_params_['C']
        self._hyper_epsilon = gridsearch.best_params_['epsilon']
        self._best_clf_score = gridsearch.best_score_

        self._svm_clf = best_clf = gridsearch.best_estimator_
        self._svm_coef = best_clf.coef_
        self._svm_bias = -best_clf.intercept_[0]
        self._svm_L1 = np.linalg.norm(self._svm_coef, ord=1)
        prediction = best_clf.predict(X)
        self._svm_loss = np.sum(np.abs(Y - prediction))

        self._svm_coef = self._svm_coef[0]

    def fit(self, X, y):
        """ Fit model to data and provide feature relevance intervals

        Parameters
        ----------
        X : array_like
            standardized data matrix
        y : array_like
            response vector
        
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        super().fit(X, y)
