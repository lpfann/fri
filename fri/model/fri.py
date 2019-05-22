"""
    Abstract class providing base for classification and regression classes specific to data.

"""

import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .base import MLProblem
from .baseline import find_best_model
from .bound import compute_relevance_bounds


class NotFeasibleForParameters(Exception):
    """ Problem was infeasible with the current parameter set.
    """


class FRIBase(BaseEstimator, SelectorMixin):

    def __init__(self, problem_type: MLProblem, random_state=None, n_jobs=-1, verbose=0, **kwargs):

        # Init problem
        self.problem_type_ = problem_type(**kwargs)

        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, hsearch_iter=50, n_probe_features=40, **kwargs):

        # Preprocessing
        data = self.problem_type_.preprocessing((X, y))

        # Find an optimal, fitted model using hyperparemeter search
        optimal_model, best_score = find_best_model(self.problem_type_, data,
                                                    self.random_state, hsearch_iter, self.n_jobs,
                                                    self.verbose, **kwargs)

        relevance_bounds, probe_values = compute_relevance_bounds(data, optimal_model, self.problem_type_,
                                                                  self.random_state,
                                                                  n_resampling=n_probe_features, n_jobs=self.n_jobs,
                                                                  verbose=self.verbose)

        # Calculate bounds
        # save unmodified intervals (without postprocessing
        self.unmod_interval_ = relevance_bounds

        # Postprocess bounds
        norm_relevance_bounds = self._postprocessing(optimal_model.L1_factor, relevance_bounds)

        self.interval_ = norm_relevance_bounds
        # self._omegas = omegas # TODO: add model model_state to object
        # self._biase = biase

        norm_probe_values = self._postprocessing(optimal_model.L1_factor, probe_values)
        self._get_relevance_mask(norm_relevance_bounds, norm_probe_values)

        # Return the classifier
        return self

    def _postprocessing(self, L1, rangevector):
        #
        # Postprocessig intervals
        #
        assert L1 > 0

        # Scale to L1
        rangevector = rangevector.copy() / L1

        # round mins to zero
        rangevector[np.abs(rangevector) < 1 * 10 ** -4] = 0

        return rangevector

    def _get_relevance_mask(self,
                            relevance_bounds,
                            probe_values,

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

        n = len(probe_values)
        assert n > 0
        mean = probe_values.mean()
        s = probe_values.std()

        # We calculate only the upper prediction interval bound because the lower one should be smaller than 0 all the time
        perc = fpr
        ### lower_boundary = mean + stats.t(df=n - 1).ppf(perc) * s * np.sqrt(1 + (1 / n))
        upper_boundary = mean - stats.t(df=n - 1).ppf(perc) * s * np.sqrt(1 + (1 / n))

        weakly = relevance_bounds[:, 1] > upper_boundary
        strongly = relevance_bounds[:, 0] > 0
        both = np.logical_and(weakly, strongly)

        prediction = np.zeros(relevance_bounds.shape[0], dtype=np.int)
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
        check_is_fitted(self, "allrel_prediction_")
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
