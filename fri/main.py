"""
    Abstract class providing base for classification and regression classes specific to data.

"""
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from fri.compute import RelevanceBoundsIntervals
from fri.model.base_type import ProblemType
from parameter_searcher import find_best_model


class NotFeasibleForParameters(Exception):
    """ Problem was infeasible with the current parameter set.
    """


class FRIBase(BaseEstimator, SelectorMixin):

    def __init__(self, problem_type: ProblemType, random_state=None, n_jobs=1, verbose=0, n_param_search=50,
                 n_probe_features=50, **kwargs):

        # Init problem
        self.n_probe_features = n_probe_features
        self.n_param_search = n_param_search

        # assert issubclass(problem_type, ProblemType)
        self.problem_type_ = problem_type(**kwargs)

        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, lupi_features=None, **kwargs):
        self.lupi_features_ = lupi_features

        # Preprocessing
        data = self.problem_type_.preprocessing((X, y), lupi_features=lupi_features)

        # Get predefined template for our init. model
        init_model_template = self.problem_type_.get_initmodel_template
        # Get hyperparameters which are predefined to our model template and can be seleted by user choice
        hyperparameters = self.problem_type_.get_all_parameters()

        # Find an optimal, fitted model using hyperparemeter search
        optimal_model, best_score = find_best_model(init_model_template, hyperparameters, data,
                                                    self.random_state, self.n_param_search, self.n_jobs,
                                                    self.verbose, lupi_features=lupi_features, **kwargs)
        self.optim_model_ = optimal_model

        self._relevance_bounds_computer = RelevanceBoundsIntervals(data, self.problem_type_, optimal_model,
                                                                   self.random_state, self.n_probe_features,
                                                                   self.n_jobs, self.verbose)
        if lupi_features is None:
            self.interval_, feature_classes = self._relevance_bounds_computer.get_normalized_intervals()
        else:
            self.interval_, feature_classes = self._relevance_bounds_computer.get_normalized_lupi_intervals(
                lupi_features=lupi_features)
        self._get_relevance_mask(feature_classes)

        # Return the classifier
        return self

    def _get_relevance_mask(self,
                            prediction,
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

    def constrained_intervals_(self, preset: dict):
        """Method to return relevance intervals which are constrained using preset ranges or values.

        Parameters
        ----------
        preset : dict like, {i:float} or {i:[float,float]}
            Keys denote feature index, values represent a fixed single value (float) or a range of allowed values (lower and upper bound).

            Example: To set  feature 0 to a fixed value use

            >>> preset = {0: 0.1}

            or to use the minium releavance bound

            >>> preset[1] = self.interval_[1, 0]

        Returns
        -------
        array like
            Relevance bounds with user constraints
        """

        # Do we have intervals?
        check_is_fitted(self, "interval_")

        return self._relevance_bounds_computer.compute_multi_preset_relevance_bounds(preset=preset, normalized=True,
                                                                                     lupi_features=self.lupi_features_)
