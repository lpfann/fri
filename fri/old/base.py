"""
    Abstract class providing base for classification and regression classes specific to data.

"""
import warnings
from abc import abstractmethod

import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_selection.base import SelectorMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .bounds import LowerBound, UpperBound, ShadowUpperBound
from .l1models import L1OrdinalRegressor, ordinal_scores, L1HingeHyperplane


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
    n_resampling : integer ( Default = 40)
        Number of probe feature permutations used. 
    iter_psearch : integer ( Default = 10)
        Amount of samples used for parameter search.
        Trade off between finer tuned model performance and run time of parameter search.
    parallel : boolean, optional
        Enables parallel computation of feature intervals
    optimum_deviation : float, optional (Default = 0.001)
        Rate of allowed deviation from the optimal solution (L1 norm of model weights).
        Default allows one percent deviation. 
        Allows for more relaxed optimization problems and leads to bigger intervals which are easier to interpret.
        Setting to 0 allows the best feature selection accuracy.
    verbose : int ( Default = 0)
        Print out verbose messages. The higher the number, the more messages are printed.
    
    Attributes
    ----------
    allrel_prediction_ : array of booleans
        Truth value for each feature if it is relevant (weakly OR strongly).
    interval_ : array [[lower_Bound_0,UpperBound_0],...,]
        Relevance bounds in 2D array format.
    optim_L1_ : double
        L1 norm of baseline model.
    optim_loss_ : double
        Sum of slack (loss) of baseline model.
    optim_model_ : fri.l1models object
        Baseline model
    optim_score_ : double
        Score of baseline model
    relevance_classes_ : array like
        Array with classification of feature relevances: 2 denotes strongly relevant, 1 weakly relevant and 0 irrelevant.
    unmod_interval_ : array like
        Same as `interval_` but not scaled to L1.
    
    See Also
    --------
    :class:`FRIClassification`:
        Class for classification problems
    :class:`FRIRegression`:
        Class for regression problems
    
    """

    @abstractmethod
    def __init__(self, C=None, optimum_deviation=0.001, random_state=None, n_jobs=None, n_resampling=40, iter_psearch=30, verbose=0):
        self.random_state = random_state
        self.C = C
        self.optimum_deviation = optimum_deviation
        self.n_jobs = n_jobs
        self.n_resampling = n_resampling
        self.iter_psearch = 20 if iter_psearch is None else iter_psearch
        self.verbose = verbose

        self.optim_model_ = None
        self.optim_score_ = None
        self.optim_L1_ = None
        self.optim_loss_ = None
        self.allrel_prediction_ = None
        self.feature_clusters_ = None
        self.linkage_ = None
        self.interval_ = None
        self.tuned_parameters = None

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
        if self.verbose > 0:
            print("loss", self.optim_loss_)
            print("L1", self.optim_L1_)
            print("offset", self._svm_bias)
            print("best_parameters", self._best_params)
            print("score", self.optim_score_)
            if self.initModel is L1HingeHyperplane:
                print("Classification scores per class")
                print(self.classification_report)

            print("coef:\n{}".format(self._svm_coef.T))

        if 0 <self.optim_score_ < 0.65: # Only check positive scores, ordinal score function has its maximum at 0, we ignore that
            print("WARNING: Weak Model performance! score = {}".format(self.optim_score_))

        # Calculate bounds
        rangevector, omegas, biase = self._main_opt(X, y, self.optim_loss_,
                                                                       self.optim_L1_,
                                                                       self.random_state)
        # save unmodified intervals (without postprocessing
        self.unmod_interval_ = rangevector.copy()
        # Postprocess bounds
        rangevector = self._postprocessing(self.optim_L1_, rangevector)

        self.interval_ = rangevector
        self._omegas = omegas
        self._biase = biase

        self._get_relevance_mask()

        # Return the classifier
        return self



    @staticmethod
    def _opt_per_thread(bound):
        """
        Worker thread method for parallel computation
        """
        return bound.solve()

    def _main_opt(self, X, Y, svmloss, L1, random_state, presetModel=None, solverargs=None):
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
        self._shadow_values  = []
        omegas = np.zeros((d, 2, d))
        if self.classes_ is not None:
            class_thresholds = len(self.classes_) - 1
            biase = np.zeros((d, 2, class_thresholds))
        else:
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
            kwargs = {"verbose": False, "solver": "ECOS", "max_iters": 1000}
        else:
            kwargs = solverargs


        # Create tasks for worker(s)
        #

        def work_generator():
            for di in dims:
                yield LowerBound(problemClass=self, optim_dim=di, kwargs=kwargs, initLoss=svmloss, initL1=L1, X=X, Y=Y, presetModel=presetModel)
                yield UpperBound(problemClass=self, optim_dim=di, kwargs=kwargs, initLoss=svmloss, initL1=L1, X=X, Y=Y, presetModel=presetModel)
            # Random sample n_resampling shadow features by permuting real features and computing upper bound
            random_choice = random_state.choice(a=np.arange(d),size=self.n_resampling)
            for i,di in enumerate(random_choice):
                yield ShadowUpperBound(problemClass=self, optim_dim=di, kwargs=kwargs, initLoss=svmloss,initL1=L1, X=X, Y=Y, sampleNum=i, presetModel=presetModel)

        done = Parallel(n_jobs=self.n_jobs,verbose=self.verbose)(map(delayed(self._opt_per_thread), work_generator()))

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
                self._shadow_values.append(finished_bound.shadow_value)
        if presetModel is not None:
            for i, p in enumerate(presetModel):
                if np.all(np.isnan(p)):
                    continue
                else:
                    rangevector[i] = p

        return rangevector, omegas, biase

    def _initEstimator(self, X, Y):
        if self.initModel is L1OrdinalRegressor:
            # Use two scores for ordinal regression
            error = ordinal_scores
            mze = make_scorer(error, error_type="mze")
            mae = make_scorer(error, error_type="mae")
            mmae = make_scorer(error, error_type="mmae")
            scorer = {"mze": mze, "mae": mae, "mmae": mmae}
            refit = "mmae"
        else:
            scorer = None  # use default score from model
            refit = True
        print(self.tuned_parameters)
        gridsearch = RandomizedSearchCV(self.initModel(),
                                        self.tuned_parameters,
                                        scoring=scorer,
                                        random_state=self.random_state,
                                        refit=refit,
                                        n_iter=self.iter_psearch,
                                        n_jobs=self.n_jobs,
                                        error_score=np.nan,
                                        return_train_score=False,
                                        verbose=self.verbose)

        # Ignore warnings for extremely bad parameters (when precision=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gridsearch.fit(X, Y)

        # Save parameters for use in optimization
        self._best_params = gridsearch.best_params_
        self._cv_results = gridsearch.cv_results_
        self.optim_model_ = gridsearch.best_estimator_
        self.optim_score_ = self.optim_model_.score(X, Y)
        if self.verbose > 0 and self.initModel is L1HingeHyperplane:
            self.classification_report = self.optim_model_.score(X, Y, debug=True)
        self._svm_coef = self.optim_model_.coef_
        self._svm_bias = self.optim_model_.intercept_
        self.optim_L1_ = np.linalg.norm(self._svm_coef[0], ord=1)
        self.optim_loss_ = np.abs(self.optim_model_.slack).sum()

        # Allow worse solutions (relaxation)
        self.optim_L1_ = self.optim_L1_ * (1 + self.optimum_deviation)


    def _postprocessing(self, L1, rangevector):
        #
        # Postprocessig intervals
        #
        # Correction through shadow features
        assert L1 > 0

        # Scale to L1
        rangevector = rangevector / L1

        # round mins to zero
        rangevector[np.abs(rangevector) < 1 * 10 ** -4] = 0

        return rangevector

    def _get_relevance_mask(self,
                            fpr=0.01
                            ):
        """Determines relevancy using feature relevance interval values
        Parameters
        ----------
        fpr : float, optional
            false positive rate allowed under H_0
        Returns
        -------
        boolean array
            Relevancy prediction for each feature
        """

        rangevector = self.interval_
        prediction = np.zeros(rangevector.shape[0], dtype=np.int)
        maxs = self._shadow_values
        maxs = np.array(maxs)
        maxs = maxs / self.optim_L1_
        n = len(maxs)

        mean = maxs.mean()
        s = maxs.std()
        perc = fpr
        pos = mean + stats.t(df=n - 1).ppf(perc) * s * np.sqrt(1 + (1 / n))
        neg = mean - stats.t(df=n - 1).ppf(perc) * s * np.sqrt(1 + (1 / n))

        weakly = rangevector[:, 1] > neg
        strongly = rangevector[:, 0] > 0
        both = np.logical_and(weakly, strongly)

        prediction[weakly] = 1
        prediction[both] = 2

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


    def score(self, X, y):
        if self.optim_model_:
            return self.optim_model_.score(X, y)
        else:
            raise NotFittedError()


    def _run_with_single_dim_single_value_preset(self,i, preset_i, n_tries=10):
            """
            Method to run method once for one restricted feature
            Parameters
            ----------
            i:
                restricted feature
            preset_i:
                restricted range of feature i (set before optimization = preset)
            n_tries:
                number of allowed relaxation steps for the L1 constraint in case of LP infeasible

            """

            X = self.X_
            y = self.y_
            # Do we have intervals?
            check_is_fitted(self, "interval_")
            interval = self.unmod_interval_
            d = len(interval)

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
                    rangevector, _, _ = self._main_opt(X, y, loss,
                                                          l1,
                                                          self.random_state,
                                                          presetModel=preset,
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

    def constrained_intervals_(self, preset):
        """Method to return relevance intervals which are constrained using preset ranges or values.
        
        Parameters
        ----------
        preset : array like [[preset lower_Bound_0,preset upper_Bound_0],...,]
            An array where all entries which are not 'np.nan' are interpreted as constraint for that corresponding feature.
            
            Best created using 

            >>> np.full_like(fri_model.interval_, np.nan, dtype=np.double)

            Example: To set  feature 0 to a fixed value use 

            >>> preset[0] = fri_model.interval_[0, 0]
        
        Returns
        -------
        array like
            Relevance bounds with user constraints 
        """
        processed = preset*self.optim_L1_ # Revert scaling to L1 norm which is done for our output intervals (see postprocessing)
        return self._run_with_multiple_value_preset(preset=processed)

    def _run_with_multiple_value_preset(self, preset=None):
            """
            Method to run method with preset values
            """
            X = self.X_
            y = self.y_
            # Do we have intervals?
            check_is_fitted(self, "interval_")
            interval = self.unmod_interval_
            d = len(interval)

            constrained_ranges_diff = np.zeros((d, 2))

            # Add correct sign of this coef
            signed_presets = np.sign(self._svm_coef[0]) * preset.T
            signed_presets = signed_presets.T
            # Calculate all bounds with feature presets
            l1 = self.optim_L1_
            loss = self.optim_loss_
            sumofpreset = np.nansum(preset[:,1])
            if sumofpreset > l1:
                print("maximum L1 norm of presets: ",sumofpreset)
                print("L1 allowed:",l1)
                print("Presets are not feasible. Try lowering values.")
                return
            try:
                kwargs = {"verbose": False, "solver": "ECOS"}
                rangevector, _, _ = self._main_opt(X, y, loss,
                                                      l1,
                                                      self.random_state,
                                                      presetModel=signed_presets,
                                                      solverargs=kwargs)
            except NotFeasibleForParameters:
                print("Presets are not feasible")
                return


            constrained_ranges_diff = self.unmod_interval_ - rangevector

            # Current dimension is not constrained, so these values are set accordingly
            for i, p in enumerate(preset):
                if np.all(np.isnan(p)):
                    continue
                else:
                    rangevector[i] = p
            rangevector = self._postprocessing(self.optim_L1_, rangevector)
            return rangevector
