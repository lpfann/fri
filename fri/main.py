from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

import warnings

try:
    # TODO: remove catch and try except on sklearn > 0.22 when regression is merged: https://github.com/scikit-learn/scikit-learn/pull/16132
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.feature_selection.base import SelectorMixin
except ModuleNotFoundError:
    from sklearn.feature_selection import SelectorMixin

from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from fri.compute import RelevanceBoundsIntervals
from fri.model.base_type import ProblemType
from fri.parameter_searcher import find_best_model

RELEVANCE_MAPPING = {0: "Irrelevant", 1: "Weak relevant", 2: "Strong relevant"}


class NotFeasibleForParameters(Exception):
    """ Problem was infeasible with the current parameter set.
    """


class FRIBase(BaseEstimator, SelectorMixin):
    def __init__(
        self,
        problem_type: ProblemType,
        random_state=None,
        n_jobs=1,
        verbose=0,
        n_param_search=30,
        n_probe_features=40,
        normalize=True,
        **kwargs,
    ):
        """

        Parameters
        ----------
        problem_type : abc.ABCMeta
        random_state : Union[mtrand.RandomState, int, None, None, None, None, None, None, None]
        n_jobs : int
        verbose : int
        n_param_search : int
        n_probe_features : int
        normalize : bool
        kwargs :

        Attributes
        ----------
        interval_ : array-like
         Feature relevance Intervals

        optim_model_ : `InitModel`
        Baseline model fitted on data

        relevance_classes_ : list(int)
        Classes of relevance encoded as int: 0 irrelevant, 1 weakly relevant, 2 strongly relevant

        relevance_classes_string_ : list(str)
        Classes of relevance encoded as string

        allrel_prediction_ : list(int)
        Relevance prediction encoded as boolean: 0 irrelevant, 1 relevant
        """

        self.problem_type = problem_type
        self.n_probe_features = n_probe_features
        self.n_param_search = n_param_search
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.normalize = normalize

        self.other_args = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.interval_ = None
        self.optim_model_ = None
        self.relevance_classes_ = None
        self.relevance_classes_string_ = None
        self.allrel_prediction_ = None

    def fit(self, X, y, lupi_features=0, **kwargs):
        """
        Method to fit model on data.

        Parameters
        ----------
        X : numpy.ndarray
        y : numpy.ndarray
        lupi_features : int
            Amount of features which are considered privileged information in `X`.
            The data is expected to be structured in a way that all lupi features are at the end of the set.
            For example `lupi_features=1` would denote the last column of `X` to be privileged.
        kwargs : dict
            Dictionary of additional keyword arguments depending on the `model`.

        Returns
        -------
        `FRIBase`
        """
        self.problem_object_ = self.problem_type(**self.other_args)
        self.lupi_features_ = lupi_features
        self.n_samples_ = X.shape[0]
        self.n_features_ = X.shape[1] - lupi_features

        self.optim_model_, best_score = self._fit_baseline(
            X, y, lupi_features, **kwargs
        )

        data = self.problem_object_.preprocessing((X, y), lupi_features=lupi_features)
        self._relevance_bounds_computer = RelevanceBoundsIntervals(
            data,
            self.problem_object_,
            self.optim_model_,
            self.random_state,
            self.n_probe_features,
            self.n_jobs,
            self.verbose,
            normalize=self.normalize,
        )
        if lupi_features == 0:
            (
                self.interval_,
                feature_classes,
            ) = self._relevance_bounds_computer.get_normalized_intervals()
        else:
            (
                self.interval_,
                feature_classes,
            ) = self._relevance_bounds_computer.get_normalized_lupi_intervals(
                lupi_features=lupi_features
            )
        self._get_relevance_mask(feature_classes)

        # Return the classifier
        return self

    def _fit_baseline(self, X, y, lupi_features=0, **kwargs):

        # Preprocessing
        data = self.problem_object_.preprocessing((X, y), lupi_features=lupi_features)
        # Get predefined template for our init. model
        init_model_template = self.problem_object_.get_initmodel_template
        # Get hyperparameters which are predefined to our model template and can be seleted by user choice
        hyperparameters = self.problem_object_.get_all_parameters()
        # search_samples = len(hyperparameters) * self.n_param_search # TODO: remove this
        search_samples = self.n_param_search
        # Find an optimal, fitted model using hyperparemeter search
        optimal_model, best_score = find_best_model(
            init_model_template,
            hyperparameters,
            data,
            self.random_state,
            search_samples,
            self.n_jobs,
            self.verbose,
            lupi_features=lupi_features,
            **kwargs,
        )
        return optimal_model, best_score

    def _get_relevance_mask(self, prediction):
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
        self.relevance_classes_string_ = [RELEVANCE_MAPPING[p] for p in prediction]
        self.allrel_prediction_ = prediction > 0

        self.allrel_prediction_nonpriv_ = self.allrel_prediction_[: self.n_features_]
        self.allrel_prediction_priv_ = self.allrel_prediction_[self.n_features_ :]
        self.relevance_classes_nonpriv_ = self.relevance_classes_[: self.n_features_]
        self.relevance_classes_priv_ = self.relevance_classes_[self.n_features_ :]

        return self.allrel_prediction_

    def _n_selected_features(self):
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
        """
        Using fitted model predict points for `X` and compare to truth `y`.

        Parameters
        ----------
        X : numpy.ndarray
        y : numpy.ndarray

        Returns
        -------
        Model specific score (0 is worst, 1 is best)
        """
        if self.optim_model_:
            return self.optim_model_.score(X, y)
        else:
            raise NotFittedError()

    def constrained_intervals(self, preset: dict):
        """
        Method to return relevance intervals which are constrained using preset ranges or values.

        Parameters
        ----------
        preset : dict like, {i:float} or {i:[float,float]}
            Keys denote feature index, values represent a fixed single value (float) or a range of allowed values (lower and upper bound).

            Example: To set  feature 0 to a fixed value use

            >>> preset = {0: 0.1}

            or to use the minimum relevance bound

            >>> preset[1] = self.interval_[1, 0]

        Returns
        -------
        array like
            Relevance bounds with user constraints
        """

        # Do we have intervals?
        check_is_fitted(self, "interval_")

        return self._relevance_bounds_computer.compute_multi_preset_relevance_bounds(
            preset=preset, lupi_features=self.lupi_features_
        )

    def print_interval_with_class(self):
        """

        Pretty print the relevance intervals and determined feature relevance class

        """
        output = ""
        if self.interval_ is None:
            output += "Model is not fitted."

        output += "############## Relevance bounds ##############\n"
        output += "feature: [LB -- UB], relevance class\n"
        for i in range(self.n_features_ + self.lupi_features_):
            if i == self.n_features_:
                output += "########## LUPI Relevance bounds\n"
            output += (
                f"{i:7}: [{self.interval_[i, 0]:1.1f} -- {self.interval_[i, 1]:1.1f}],"
            )
            output += f" {self.relevance_classes_string_[i]}\n"
        return output
