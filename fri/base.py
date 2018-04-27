"""
    Abstract class providing base for classification and regression classes specific to data.

"""
import warnings
from abc import abstractmethod
from multiprocessing import Pool

import numpy as np
import scipy
from fri.utils import distance
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection.base import SelectorMixin
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .bounds import LowerBound, UpperBound, ShadowLowerBound, ShadowUpperBound


class NotFeasibleForParameters(Exception):
    """ Problem was infeasible with the current parameter set.
    """


class FRIBase(BaseEstimator, SelectorMixin):
    """Object for performing the feature relevance bound method.
    
    Parameters
    ----------
    C : float , optional
        Regularization parameter, default obtains the hyperparameter through gridsearch optimizing accuracy
    random_state : object
        Set seed for random number generation.
    shadow_features : boolean, optional
        Enables noise reduction using feature permutation results.
    n_resampling : integer ( Default = 3)
        Number of shadowfeature permutations used. 
    parallel : boolean, optional
        Enables parallel computation of feature intervals
    optimum_deviation : float, optional (Default = 0.01)
        Percentage of allowed deviation from the optimal solution (L1 norm of model weights).
        Default allows one percent deviation. 
        Allows for more relaxed optimization problems and leads to bigger intervals.
    debug : boolean
        Enable output of internal values for debugging purposes.

    --
    For Regression
    --
    epsilon : float
        Allowed epsilon wide tube around target.
    Attributes
    ----------
    allrel_prediction_ : array of booleans
        Truth value for each feature if it is relevant (weakly OR strongly).
    interval_ : array [[lower_Bound_0,UpperBound_0],...,]
        Relevance bounds in 2D array format.
    
    See Also
    --------
    :class:`FRIClassification`:
        Class for classification problems
    :class:`FRIRegression`:
        Class for regression problems
    
    """

    @abstractmethod
    def __init__(self, isRegression=False, C=None, optimum_deviation=0.1, random_state=None,
                 shadow_features=False, parallel=False, n_resampling=3, debug=False):
        self.random_state = random_state
        self.C = C
        self.optimum_deviation = optimum_deviation
        self.shadow_features = shadow_features
        self.parallel = parallel
        self.isRegression = isRegression
        self.n_resampling = n_resampling
        self.debug = debug


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

        self.optim_model_ = None
        self.optim_score_ = None
        self.optim_L1_ = None
        self.optim_loss_ = None
        self.tuned_epsilon_ = None
        self.tuned_C_ = None
        self.allrel_prediction_ = None
        self.feature_clusters_ = None
        self.linkage_ = None
        self.interval_ = None
        self.random_state = check_random_state(self.random_state)

        y = np.asarray(y)

        self.X_ = X
        self.y_ = y

        # Use SVM to get optimal solution
        self._initEstimator(X, y)
        if self.debug:
            print("loss", self.optim_loss_)
            print("L1", self.optim_L1_)
            print("offset", self._svm_bias)
            print("C", self.tuned_C_)
            print("score", self.optim_score_)
            print("coef:\n{}".format(self._svm_coef.T))

        if self.optim_score_ < 0.65:
            print("WARNING: Weak Model performance! score = {}".format(self.optim_score_))

        # Calculate bounds
        rangevector, omegas, biase, shadowrangevector = self._main_opt(X, y, self.optim_loss_,
                                                                       self.optim_L1_,
                                                                       self.random_state,
                                                                       self.shadow_features)
        # save unmodified intervals (without postprocessing
        self.unmod_interval_ = rangevector.copy()
        # Postprocess bounds
        rangevector, shadowrangevector = self._postprocessing(self.optim_L1_, rangevector, self.shadow_features,
                                                              shadowrangevector)

        self.interval_ = rangevector
        self._omegas = omegas
        self._biase = biase
        self._shadowintervals = shadowrangevector

        self._get_relevance_mask()

        # Return the classifier
        return self

    def community_detection(self, cutoff_threshold=0.55):
        X = self.X_
        y = self.y_
        # Do we have intervals?
        check_is_fitted(self, "interval_")
        interval = self.unmod_interval_
        d = len(interval)

        # Init arrays
        interval_constrained_to_min = np.zeros(
            (d, d, 2))  # Save ranges (d,2-dim) for every contrained run (d-times)
        absolute_delta_bounds_summed_min = np.zeros((d, d, 2))
        interval_constrained_to_max = np.zeros(
            (d, d, 2))  # Save ranges (d,2-dim) for every contrained run (d-times)
        absolute_delta_bounds_summed_max = np.zeros((d, d, 2))

        def run_with_single_dim_single_value_preset(i, preset_i, n_tries=10):
            """
            Method to run method once for one restricted feature
            Parameters
            ----------
            i restricted feature
            preset_i restricted range of feature i (set before optimization = preset)
            n_tries number of allowed relaxation steps for the L1 constraint in case of LP infeasible

            """
            constrained_ranges = np.zeros((d, 2))
            constrained_ranges_diff = np.zeros((d, 2))

            # Init empty preset
            preset = np.empty(shape=(d, 2))
            preset.fill(np.nan)

            # Add correct sign of this coef
            signed_preset_i = np.sign(self._svm_coef[0][i]) * preset_i
            preset[i] = signed_preset_i
            # Calculate all bounds with feature i set to min_i
            l1 = self.optim_L1_
            loss = self.optim_loss_
            for j in range(n_tries):
                # try several times if problem to stringent
                try:
                    kwargs = {"verbose": False, "solver": "ECOS"}
                    rangevector, _, _, _ = self._main_opt(X, y, loss,
                                                          l1,
                                                          self.random_state,
                                                          False, presetModel=preset,
                                                          solverargs=kwargs)
                except NotFeasibleForParameters:
                    preset[i] *= -1
                    # print("Community detection: Constrained run failed, swap sign".format)
                    continue
                else:
                    #print("solved constrained opt for ", i)
                    # problem was solvable
                    break
            else:
                raise NotFeasibleForParameters("Community detection failed.", "dim {}".format(i))

            # rangevector, _ = self._postprocessing(self.optim_L1_, rangevector, False,
            #                                      None)
            # Get differences for constrained intervals to normal intervals
            constrained_ranges_diff = self.unmod_interval_ - rangevector

            # Current dimension is not constrained, so these values are set accordingly
            rangevector[i] = preset_i
            constrained_ranges_diff[i] = 0

            return rangevector, constrained_ranges_diff

        # Set weight for each dimension to minimum and maximum possible value and run optimization of all others
        # We retrieve the relevance bounds and calculate the absolute difference between them and non-constrained bounds
        for i in range(d):
            # min
            ranges, diff = run_with_single_dim_single_value_preset(i, interval[i, 0])
            interval_constrained_to_min[i] = ranges
            absolute_delta_bounds_summed_min[i] = diff
            # max
            ranges, diff = run_with_single_dim_single_value_preset(i, interval[i, 1])
            interval_constrained_to_max[i] = ranges
            absolute_delta_bounds_summed_max[i] = diff

        feature_points = np.zeros((d, 2 * d * 2))
        for i in range(d):
            feature_points[i, :(2 * d)] = absolute_delta_bounds_summed_min[i].flatten()
            feature_points[i, (2 * d):] = absolute_delta_bounds_summed_max[i].flatten()

        # Calculate similarity using custom measure
        dist_mat = scipy.spatial.distance.pdist(feature_points, metric=distance)

        # Single Linkage clustering
        link = linkage(dist_mat, method="single")

        # Set cutoff at which threshold the linkage gets flattened (clustering)
        RATIO = cutoff_threshold
        threshold = RATIO * np.max(link[:, 2])  # max of branch lengths (distances)
        feature_clustering = fcluster(link, threshold, criterion="distance")

        # Max Clust
        # max_clusters = 2
        # feature_clustering = fcluster(link, max_clusters, criterion="maxclust")

        self.feature_clusters_, self.linkage_ = feature_clustering, link

        return feature_clustering, link, feature_points, dist_mat


    def _get_relevance_mask(self,
                            upper_epsilon=0.1,
                            lower_epsilon=0
                            ):
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
        prediction = np.zeros(rangevector.shape[0], dtype=np.int)

        # Weakly relevant ones have high upper bounds
        prediction[rangevector[:, 1] > upper_epsilon] = 1
        # Strongly relevant bigger than 0 + some epsilon
        prediction[rangevector[:, 0] > lower_epsilon] = 2

        self.relevance_classes_ = prediction
        self.allrel_prediction_ = prediction > 0


        return self.allrel_prediction_

    def _n_features(self):
        """

        Returns the number of selected features.
        -------

        """
        check_is_fitted(self,"allrel_prediction_")
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

    def _main_opt(self, X, Y, svmloss, L1, random_state, shadow_features, presetModel=None, solverargs=None):
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

        dims = np.arange(d)
        if presetModel is not None:
            # Exclude fixed (preset) dimensions from being run
            for di, preset in enumerate(presetModel):
                # Nans are unset and ignored
                if np.isnan(preset[0]):
                    continue
                else:
                    # Check for difference between upper and lower bound,
                    # when very small difference assume fixed value and skip computation later
                    if np.diff(np.abs(preset)) <= 0.0001:
                        np.delete(dims, di)

        """
        Solver Parameters
        """
        if solverargs is None:
            kwargs = {"verbose": False, "solver": "ECOS", "max_iters": 100}
        else:
            kwargs = solverargs


        # Create tasks for worker(s)
        #
        work = [LowerBound(problemClass=self, optim_dim=di, kwargs=kwargs, initLoss=svmloss, initL1=L1, X=X, Y=Y,
                           presetModel=presetModel)
                for di in dims]
        work.extend([UpperBound(problemClass=self, optim_dim=di, kwargs=kwargs, initLoss=svmloss, initL1=L1, X=X, Y=Y,
                                presetModel=presetModel)
                     for di in dims])
        if shadow_features:
            for nr in range(self.n_resampling):
                work.extend([ShadowLowerBound(problemClass=self, optim_dim=di, kwargs=kwargs, initLoss=svmloss,
                                              initL1=L1, X=X, Y=Y, sampleNum=nr, presetModel=presetModel)
                             for di in dims])
                work.extend([ShadowUpperBound(problemClass=self, optim_dim=di, kwargs=kwargs, initLoss=svmloss,
                                              initL1=L1, X=X, Y=Y, sampleNum=nr, presetModel=presetModel)
                             for di in dims])

        def pmap(*args):
            with Pool() as p:
                return p.map(*args)

        if self.parallel:
            newmap = pmap
        else:
            newmap = map
        #
        # Compute all bounds using redefined map function (parallel / non parallel)
        #
        done = newmap(self._opt_per_thread, work)

        # Retrieve results and aggregate values in arrays
        for finished_bound in done:
            di = finished_bound.optim_dim
            i = int(finished_bound.isUpperBound)
            # Handle shadow values differently (we discard useless values)
            if not hasattr(finished_bound, "isShadow"):
                prob_i = finished_bound.prob_instance
                rangevector[di, i] = np.abs(prob_i.problem.value)
                omegas[di, i] = prob_i.omega.value.reshape(d)
                biase[di, i] = prob_i.b.value
            else:
                # Get the mean of all shadow samples
                shadowrangevector[di, i] += (finished_bound.shadow_value / self.n_resampling)
        if presetModel is not None:
            for i, p in enumerate(presetModel):
                if np.all(np.isnan(p)):
                    continue
                else:
                    rangevector[i] = p

        return rangevector, omegas, biase, shadowrangevector

    def _postprocessing(self, L1, rangevector, shadow_features, shadowrangevector):
        #
        # Postprocessig intervals
        #
        # Correction through shadow features
        assert L1 > 0

        if shadow_features:
            shadow_variance = shadowrangevector[:, 1] - shadowrangevector[:, 0]
            rangevector[:, 0] -= shadow_variance
            rangevector[:, 1] -= shadow_variance
            rangevector[rangevector < 0] = 0
            shadowrangevector = shadowrangevector / L1

        # Scale to L1
        rangevector = rangevector / L1

        # round mins to zero
        rangevector[np.abs(rangevector) < 1 * 10 ** -4] = 0

        return rangevector, shadowrangevector

    def _initEstimator(self, X, Y):

        gridsearch = GridSearchCV(self.initModel(),
                                  self.tuned_parameters,
                                  n_jobs=-1 if self.parallel else 1,
                                  error_score=0,
                                  verbose=False)

        # Ignore warnings for extremely bad parameters (when precision=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gridsearch.fit(X, Y)

        # Legacy Code
        # TODO: remove legacy code
        ###
        self.tuned_C_ = gridsearch.best_params_['C']
        if self.isRegression:
            self.tuned_epsilon_ = gridsearch.best_params_['epsilon']
        ###

        # Save parameters for use in optimization
        self._best_params = gridsearch.best_params_
        self.optim_model_ = gridsearch.best_estimator_
        self.optim_score_ = self.optim_model_.score(X, Y)
        self._svm_coef = self.optim_model_.coef_
        self._svm_bias = self.optim_model_.intercept_
        self.optim_L1_ = np.linalg.norm(self._svm_coef[0], ord=1)
        self.optim_loss_ = np.abs(self.optim_model_.slack).sum()

        # Allow worse solutions (relaxation)
        self.optim_L1_ = self.optim_L1_ * (1 + self.optimum_deviation)


    def score(self, X, y):
        if self.optim_model_:
            return self.optim_model_.score(X, y)
        else:
            raise NotFittedError()

