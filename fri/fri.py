"""
This is a module to be used as a reference for building other modules
"""
from abc import abstractmethod
from multiprocessing import Pool

import numpy as np
from sklearn import preprocessing
from sklearn import svm
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection.base import SelectorMixin
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.utils import check_X_y, check_random_state, resample
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster,linkage
        
import fri.bounds
import copy
import math


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
        Regularization parameter, default obtains the hyperparameter
         through gridsearch optimizing accuracy
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
    def __init__(self, isRegression, C=None, random_state=None,
                 shadow_features=False, parallel=False, n_resampling=3, feat_elim=False):
        """Summary
        Parameters
        ----------
        C : float , optional
            Regularization parameter, default obtains the hyperparameter
             through gridsearch optimizing accuracy
        random_state : object
            Set seed for random number generation.
        shadow_features : boolean, optional
            Enables noise reduction using feature permutation results.
        parallel : boolean, optional
            Enables parallel computation of feature intervals
        """

        self._svm_clf = None
        self._best_clf_score = None
        self.random_state = check_random_state(random_state)
        self.C = C
        self.shadow_features = shadow_features
        self.parallel = parallel
        self.isRegression = isRegression
        self.n_resampling = n_resampling
        self.feat_elim = feat_elim
        self._hyper_epsilon = None
        self._hyper_C = None
        self._svm_L1 = None
        self._svm_loss = None
        self._ensemble = None
        self.allrel_prediction_ = None
        self.feature_clusters_ = None
        self.linkage_ = None

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

        if self._best_clf_score < 0.7:
            print("WARNING: Weak Model performance!")

        # Main Optimization step
        results = self._main_opt(X, y, self._svm_loss, self._svm_L1,
                                 self._hyper_C, self._hyper_epsilon,
                                 self.random_state, self.isRegression)

        self.interval_ = results[0]
        self._omegas = results[1]
        self._biase = results[2]
        self._shadowintervals = results[3]

        if not self._ensemble:
            self._get_relevance_mask()
            if X.shape[1] > 1:
                self.feature_clusters_, self.linkage_ = self.community_detection()
        

        # Return the classifier
        return self

    def _feature_elimination(self, X, y, estimator, intervals, minsize=1):
        # copy array to allow deletion
        intervals = copy.copy(intervals)
        assert intervals.shape[1] == 2

        # sort features by bounds
        low_bounds = intervals[:, 0]
        up_bounds = intervals[:, 1]
        # lower bounds are more important, used as primary sort key
        sorted_bounds = list(np.lexsort((up_bounds, low_bounds)))

        fs = np.zeros(intervals.shape[0], dtype=np.bool)
        elem_big_zero = np.where(np.any(intervals > 0, 1))[0]
        if sum(elem_big_zero) < 1:
            # All bounds zero, no relevant feature or error...
            return fs
        fs[elem_big_zero] = 1
        fs = np.where(fs)[0]
        fs = fs.tolist()

        #skip features with bounds==0
        bounds = intervals[intervals[:, 1].argsort(kind="mergesort")]
        bounds = bounds[bounds[:, 0].argsort(kind="mergesort")]
        skip = np.argmax(np.any(bounds > 0, 1))
        sorted_bounds = sorted_bounds[skip:]

        memory = []
        # iterate over all features who have low_bound >0 starting by lowest
        while len(fs) >= minsize:
            # for i in range(X.shape[1] - minsize - (skip + 1)):
            # 10cv for each subset
            #print(fs, sorted_bounds, skip)
            cv_score = cross_validate(estimator, X[:, fs],
                                      y=y, cv=10,
                                      scoring=None)["test_score"]
            mean_score = cv_score.mean()
            #print(fs, sorted_bounds, skip, mean_score)
            memory.append((mean_score, fs[:]))
            fs.remove(sorted_bounds.pop(0))
        if len(memory) < 1:
            return fs
        # Return only best scoring feature subset
        #memory = sorted(memory, key=lambda m: len(m[1]))
        best_fs = max(memory, key=lambda m: m[0])
        #print("bests fs socer {}, best fs {}".format(*best_fs))
        return best_fs[1]

    def community_detection(self,cutoff_threshold=0.55):
        '''
        Finding communities of features using pairwise differences of solutions aquired in the main LP step.
        '''
        svm_solution = self._svm_coef
        abs_svm_sol = np.abs(svm_solution)

        om = self._omegas
        mins = om[:,0,:]
        maxs = om[:,1,:]
        abs_mins = np.abs(mins)
        abs_maxs = np.abs(maxs)

        # Aggregate min and max solution values to obtain the absolute variation
        lower_variation = abs_svm_sol - abs_mins
        upper_variation = abs_maxs - abs_svm_sol
        variation = np.abs(lower_variation) + np.abs(upper_variation)

        # add up lower triangular matrix to upper one
        collapsed_variation = np.triu(variation)+np.tril(variation).T
        np.fill_diagonal(collapsed_variation, 0)
        #collapsed_variation = pd.DataFrame(collapsed_variation)

        # Create distance matrix
        dist_mat = np.triu(collapsed_variation).T + collapsed_variation
        # normalize
        dist_mat = 1- dist_mat/np.max(dist_mat)
        # get numpy array
        #dist_mat = dist_mat.values[:]
        # feature with itself has no distance
        np.fill_diagonal(dist_mat,0)

        # convert to squareform for scipy compat.
        dist_mat_square = squareform(dist_mat)

        # Execute clustering 
        link = linkage(dist_mat_square, method="ward")

        # Set cutoff at which threshold the linkage gets flattened (clustering)
        RATIO = cutoff_threshold
        threshold = RATIO * np.max(link[:,2]) # max of branch lengths (distances)

        feature_clustering = fcluster(link,threshold,criterion="distance")

        return feature_clustering, link

    def _get_relevance_mask(self,
                            upper_epsilon=0.1,
                            lower_epsilon=0.0323):
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
        if not self.feat_elim:
            rangevector = self.interval_
            prediction = np.zeros(rangevector.shape[0], dtype=np.bool)

            # Weakly relevant ones have high upper bounds
            prediction[rangevector[:, 1] > upper_epsilon] = True
            # Strongly relevant bigger than 0 + some epsilon
            prediction[rangevector[:, 0] > lower_epsilon] = True

            self.allrel_prediction_ = prediction
        else:
            if self.allrel_prediction_ is None:
                # Classify features
                best_fs = self._feature_elimination(
                    self.X_, self.y_, self._svm_clf, self.interval_)
                prediction = np.zeros(self.interval_.shape[0], dtype=np.bool)
                prediction[best_fs] = True
                self.allrel_prediction_ = prediction

        return self.allrel_prediction_

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

    @staticmethod
    def _opt_per_thread(bound):
        """
        Worker thread method for parallel computation
        """
        return bound.solve()

    def _main_opt(self, X, Y, svmloss, L1,
                  C, _hyper_epsilon, random_state, isRegression):
        """ Main calculation function.
            LP for each bound and distributes them depending on parallel flag.
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

        """
        Solver Parameters
        """
        '''
        kwargs = {"warm_start": False, "solver": "SCS",
                  "gpu": True, "verbose": False, "parallel": False}

        '''
        kwargs = {"verbose": False}

        """
        Create tasks for worker(s)
        """
        work = [self.LowerBound(di, d, n, kwargs,
                                L1, svmloss, C, X, Y,
                                regression=self.isRegression,
                                epsilon=_hyper_epsilon)
                for di in range(d)]
        work.extend([self.UpperBound(di, d, n, kwargs,
                                     L1, svmloss, C, X, Y,
                                     regression=self.isRegression,
                                     epsilon=_hyper_epsilon)
                     for di in range(d)])
        if self.shadow_features:
            for nr in range(self.n_resampling):
                work.extend([self.LowerBoundS(di, d, n, kwargs,
                                              L1, svmloss, C, X, Y,
                                              regression=isRegression,
                                              epsilon=_hyper_epsilon,
                                              random_state=random_state)
                             for di in range(d)])
                work.extend([self.UpperBoundS(di, d, n, kwargs,
                                              L1, svmloss, C, X, Y,
                                              regression=isRegression,
                                              epsilon=_hyper_epsilon,
                                              random_state=random_state)
                             for di in range(d)])

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
            prob_i = finished_bound.prob_instance
            if not hasattr(finished_bound, "isShadow"):
                rangevector[di, i] = np.abs(prob_i.problem.value)
                omegas[di, i] = prob_i.omega.value.reshape(d)
                biase[di, i] = prob_i.b.value
            else:
                # Get the mean of all shadow samples
                shadowrangevector[di, i] += (prob_i.problem.value / self.n_resampling)

        # rangevector = np.abs(rangevector)
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

        return rangevector, omegas, biase, shadowrangevector

    def _initEstimator(self, X, Y):
        pass

    def score(self, X, y):
        if self._svm_clf:
            return self._svm_clf.score(X, y)
        else:
            raise NotFittedError()


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

    def __init__(self, C=None, random_state=None,
                 shadow_features=False, parallel=False, n_resampling=3, feat_elim=False,**kwargs):
        """Initialize a solver for classification data
        Parameters
        ----------
        C : float , optional
            Regularization parameter, default obtains the hyperparameter
            through gridsearch optimizing accuracy
        random_state : object
            Set seed for random number generation.
        shadow_features : boolean, optional
            Enables noise reduction using feature permutation results.
        parallel : boolean, optional
            Enables parallel computation of feature intervals
        """
        super().__init__(isRegression=False,C=C, random_state=random_state,
                 shadow_features=shadow_features, parallel=parallel, feat_elim=False,**kwargs)

    def _initEstimator(self, X, Y):
        estimator = svm.LinearSVC(penalty='l2',
                                  loss="squared_hinge",
                                  dual=False,
                                  random_state=self.random_state)
        if self.C is None:
            # Hyperparameter Optimization over C, starting from minimal C
            min_c = svm.l1_min_c(X, Y)
            tuned_parameters = [{'C': min_c * np.logspace(1, 4)}]
            #tuned_parameters = [{'C': np.logspace(-6, 4, 11)}]
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
                                  scoring="average_precision",
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
        self._svm_loss = np.sum(np.maximum(0, 1 - Y_vector * prediction))

        self._svm_coef = self._svm_coef[0]

    def fit(self, X, y):
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

        super().fit(X, y)


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

    def __init__(self, epsilon=None, C=None, random_state=None,
                 shadow_features=False, parallel=False, feat_elim=False,**kwargs):
        super().__init__(isRegression=True,C=C, random_state=random_state,
                 shadow_features=shadow_features, parallel=parallel, feat_elim=False, **kwargs)
        self.epsilon = epsilon

    def _initEstimator(self, X, Y):
        #estimator = svm.SVR(kernel="linear",shrinking=False)
        
        estimator = svm.LinearSVR(
                                  loss="squared_epsilon_insensitive",
                                  dual=False,
                                  random_state=self.random_state)

        tuned_parameters = {'C': [self.C], 'epsilon': [self.epsilon]}
        if self.C is None:
            tuned_parameters["C"] = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        if self.epsilon is None:
            tuned_parameters["epsilon"] =  [0.0001, 0.001, 0.01, 0.1, 1, 2, 5]

        n = len(X)
        if n <= 20:
            cv = 3
        else:
            cv = 7

        gridsearch = GridSearchCV(estimator,
                                  tuned_parameters,
                                  scoring="r2",
                                  n_jobs=-1 if self.parallel else 1,
                                  cv=cv,
                                  verbose=0)
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


class EnsembleFRI(FRIBase):
    def __init__(self, model, n_bootstraps=10, random_state=None, n_jobs=1):
        self.random_state = check_random_state(random_state)
        self.n_bootstraps = n_bootstraps
        self.model = model
        self.n_jobs = n_jobs
        self.bs_seed = self.random_state.randint(10000000) # Seed for bootstraps rngs, constant for multiprocessing

        if isinstance(self.model, FRIClassification):
            isRegression = False
        else:
            isRegression = True

        model._ensemble = True
        model.random_state = self.random_state
        super().__init__(isRegression, random_state=self.random_state)

    def _fit_one_bootstrap(self, i):
        m = clone(self.model)
        m._ensemble = True

        X, y = self.X_, self.y_
        n = X.shape[0]
        n_samples = math.ceil(0.8 * n)

        # Get bootstrap set
        X_bs, y_bs = resample(X, y, replace=True,
                              n_samples=n_samples, random_state=self.bs_seed+i)

        m.fit(X_bs, y_bs)
        if self.model.shadow_features:
            return m.interval_, m._omegas, m._biase, m._shadowintervals
        else:
            return m.interval_, m._omegas, m._biase

    def fit(self, X, y):

        if isinstance(self.model, FRIClassification):
            self.isRegression = False
        else:
            self.isRegression = True

        if self.n_jobs > 1:
            def pmap(*args):
                with Pool(self.n_jobs) as p:
                    return p.map(*args)

            nmap = pmap
        else:
            nmap = map

        self.X_, self.y_ = X, y
        results = list(nmap(self._fit_one_bootstrap, range(self.n_bootstraps)))

        if self.model.shadow_features:
            rangevector, omegas, biase, shadowrangevector = zip(*results)
            self._shadowintervals = np.mean(shadowrangevector, axis=0)
        else:
            rangevector, omegas, biase = zip(*results)
        # Get average
        self.interval_ = np.mean(rangevector, axis=0)
        self._omegas = np.mean(omegas, axis=0)
        self._biase = np.mean(biase, axis=0)

        # Classify features
        self.model._initEstimator(X, y)
        self._svm_clf = self.model._svm_clf
        self._get_relevance_mask()

        return self

    def score(self, X, y):
        return self.model.score(X, y)
