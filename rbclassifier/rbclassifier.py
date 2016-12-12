"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle, check_random_state
from sklearn import preprocessing
from sklearn import svm
from collections import namedtuple
from sklearn.exceptions import FitFailedWarning
from rbclassifier.bounds import LowerBound,UpperBound, ShadowUpperBound, ShadowLowerBound
from multiprocessing import Pool
import cvxpy as cvx

from multiprocessing import Process, Queue, current_process, freeze_support

class NotFeasibleForParameters(Exception):
    """SVM cannot separate points with this parameters"""


class RelevanceBoundsClassifier(BaseEstimator, SelectorMixin):
    """ L1-relevance Bounds Classifier

    """
    Opt_output = namedtuple("OptOutput", ["bounds", "omegas", "b"])
    def __init__(self, C=None, random_state=None, shadow_features=True):
        self.random_state = random_state
        self.C = C
        self.shadow_features = shadow_features

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        if len(self.classes_)>2:
            raise ValueError("Only binary class data supported")
        # Negative class is set to -1 for decision surface
        y = preprocessing.LabelEncoder().fit_transform(y)
        y[y==0] = -1

        self.X_ = X
        self.y_ = y

        # Use SVM to get optimal solution
        self._performSVM(X, y)

        # Main Optimization step
        self._main_opt(X, y)

        # Classify features
        self._get_relevance_mask()

        # Return the classifier
        return self

    # def predict(self, X):
    #     """ A reference implementation of a prediction for a classifier.

    #     Parameters
    #     ----------
    #     X : array-like of shape = [n_samples, n_features]
    #         The input samples.

    #     Returns
    #     -------
    #     y : array of int of shape = [n_samples]
    #         The label for each sample is the label of the closest sample
    #         seen udring fit.
    #     """
    #     # Check is fit had been called
    #     check_is_fitted(self, ['X_', 'y_'])

    #     # Input validation
    #     X = check_array(X)

    #     return None

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
        #kwargs = {"warm_start": True, "solver": "SCS", "gpu": False, "verbose": False, "parallel": True}
        kwargs = { "solver": "ECOS"}
        acceptableStati = [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]

        work = [LowerBound(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y) for di in range(d)]
        work.extend([UpperBound(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y) for di in range(d)])
        if self.shadow_features:
            work.extend([ShadowLowerBound(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y) for di in range(d)])
            work.extend([ShadowUpperBound(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y) for di in range(d)])
     
        
        with Pool() as p:
            done = p.map(self._opt_per_thread, work)

        for finished_bound in done:
            di = finished_bound.di
            i = finished_bound.type

            if not hasattr(finished_bound,"isShadow"):
                rangevector[di, i] = finished_bound.prob_instance.problem.value
                omegas[di, i] = finished_bound.prob_instance.omega.value.reshape(d)
                biase[di, i] =  finished_bound.prob_instance.b.value                
            else:
                shadowrangevector[di, i] = finished_bound.prob_instance.problem.value        

        # done_queue = Queue()
        # task_queue = Queue()
        # # Submit tasks
        # for task in TASKS1:
        #     task_queue.put(task)

        # # Start worker processes
        # for i in range(NUMBER_OF_PROCESSES):
        #     Process(target=RelevanceBoundsClassifier._opt_per_thread, args=(task_queue, done_queue)).start()

        # # Get and print results
        # for di in range(d):
        #     finished_bound = done_queue.get()
        #     i = finished_bound.is_upper_Bound
        #     if not finished_bound.shadowF:
        #         rangevector[di, i] = finished_bound.prob_instance.problem.value
        #         omegas[di, i] = finished_bound.prob_instance.omega.value.reshape(d)
        #         biase[di, i] =  finished_bound.prob_instance.b.value                
        #     if finished_bound.shadowF:
        #         shadowrangevector[di, i] = finished_bound.prob_instance.problem.value

        # # Tell child processes to stop
        # for i in range(NUMBER_OF_PROCESSES):
        #     task_queue.put('STOP')

        # Optimize for every dimension
        # for di in range(d):
        #     lowerB = LowerBound(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y).solve()
        #     upperB = UpperBound(acceptableStati, di, d, n, kwargs, L1, svmloss, C, X, Y).solve()
        #     bounds = [lowerB, upperB]
        #     for i in range(2):
        #         rangevector[di, i] = bounds[i].prob_instance.problem.value
        #         omegas[di, i] = bounds[i].prob_instance.omega.value.reshape(d)
        #         biase[di, i] =  bounds[i].prob_instance.b.value

        #     if self.shadow_features:
        #         # Shuffle values for single feature
        #         Xshuffled = np.append(np.random.permutation(X[:, di]).reshape((n, 1)), X ,axis=1)
        #         lowerB = LowerBound(acceptableStati, 0, d+1, n, kwargs, L1, svmloss, C, Xshuffled, Y,shadowF=True).solve()
        #         upperB = UpperBound(acceptableStati, 0, d+1, n, kwargs, L1, svmloss, C, Xshuffled, Y,shadowF=True).solve()
        #         bounds = [lowerB, upperB]
        #         for i in range(2):
        #             shadowrangevector[di, i] = bounds[i].prob_instance.problem.value

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

    def _performSVM(self, X, Y):
        if self.C is None:
            # Hyperparameter Optimization over C, starting from minimal C
            min_c = svm.l1_min_c(X, Y)
            tuned_parameters = [{'C': min_c * np.logspace(1, 4)}]
        else:
            # Fixed Hyperparameter
            tuned_parameters = [{'C': [self.C]}]

        gridsearch = GridSearchCV(
                             svm.LinearSVC(penalty='l1',
                             loss="squared_hinge",
                             dual=False,
                             random_state=self.random_state),
                           tuned_parameters,scoring="f1",
                           n_jobs=-1, cv=3, verbose=False)
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
        # Hinge loss
        # loss = np.sum(np.maximum(0, 1 - Y_vector * np.inner(beta, X1) - bias))
        # Squared hinge loss
        self._svm_loss = np.sum(np.maximum(0, 1 - Y_vector * (np.inner( self._svm_coef, X) - self._svm_bias))[0]**2)

        self._svm_coef = self._svm_coef[0]

