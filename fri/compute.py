import logging
from collections import defaultdict

import joblib
import numpy as np
from scipy import stats

from fri.baseline import InitModel
from fri.model.base import MLProblem
from fri.model.base import Relevance_CVXProblem
from fri.utils import permutate_feature_in_data

MIN_N_PROBE_FEATURES = 20  # Lower bound of probe features


def _start_solver_worker(bound: Relevance_CVXProblem):
    """
    Worker thread method for parallel computation
    """
    return bound.solve()


class RelevanceBoundsIntervals(object):
    def __init__(self, data, problem_type: MLProblem, best_init_model: InitModel, random_state, n_resampling, n_jobs,
                 verbose):
        self.data = data
        self.problem_type = problem_type
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_resampling = n_resampling
        self.random_state = random_state
        self.best_init_model = best_init_model
        self.best_hyperparameters = best_init_model.hyperparam

        # Relax constraints to improve stability
        relaxed_constraints = problem_type.get_relaxed_constraints(best_init_model.constraints)
        self.init_constraints = relaxed_constraints

    def get_normalized_lupi_intervals(self, lupi_features, presetModel=None):

        # We define a list of all the features we want to compute relevance bounds for
        X, _ = self.data  # TODO: handle other data formats
        all_d = X.shape[1]
        normal_d = all_d - lupi_features
        normal_dims = _get_necessary_dimensions(normal_d, presetModel)
        lupi_dims = _get_necessary_dimensions(all_d, presetModel, start=normal_d)

        # Compute relevance bounds and probes for normal features and LUPI
        with joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            relevance_bounds = self.compute_relevance_bounds(normal_dims, parallel=parallel)
            probe_values = self.compute_probe_values(normal_dims, parallel=parallel)
            relevance_bounds_priv = self.compute_relevance_bounds(lupi_dims, parallel=parallel)
            probe_values_priv = self.compute_probe_values(lupi_dims, parallel=parallel)

        # Normalize and classify features for normal features
        norm_bounds = _postprocessing(self.best_init_model.L1_factor, relevance_bounds)
        norm_probe_values = _postprocessing(self.best_init_model.L1_factor, probe_values)
        feature_classes = feature_classification(norm_probe_values, norm_bounds)

        # Normalize and classify features for lupi
        norm_bounds_priv = _postprocessing(self.best_init_model.constraints["w_priv_l1"], relevance_bounds_priv)
        norm_probe_values_priv = _postprocessing(self.best_init_model.L1_factor, probe_values_priv)
        feature_classes_priv = feature_classification(norm_probe_values_priv, norm_bounds_priv)

        self.unmod_interval_ = np.concatenate([relevance_bounds, relevance_bounds_priv])
        self.interval_ = np.concatenate([norm_bounds, norm_bounds_priv])
        feature_classes = np.concatenate([feature_classes, feature_classes_priv])
        return self.interval_, feature_classes

    def get_normalized_intervals(self, presetModel=None):
        # We define a list of all the features we want to compute relevance bounds for
        X, _ = self.data  # TODO: handle other data formats
        d = X.shape[1]
        # Depending on the preset model, we dont need to compute all bounds
        # e.g. in the case of fixed features we skip those
        dims = _get_necessary_dimensions(d, presetModel)

        with joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            relevance_bounds = self.compute_relevance_bounds(dims, parallel=parallel)
            probe_values = self.compute_probe_values(dims, parallel=parallel)

        # Postprocess bounds
        norm_bounds = _postprocessing(self.best_init_model.L1_factor, relevance_bounds)
        self.interval_ = norm_bounds
        norm_probe_values = _postprocessing(self.best_init_model.L1_factor, probe_values)
        feature_classes = feature_classification(norm_probe_values, self.interval_)

        self.unmod_interval_ = relevance_bounds
        return self.interval_, feature_classes

    def compute_relevance_bounds(self, dims, parallel=None, presetModel=None, solverargs=None):
        init_model_state = self.best_init_model.model_state

        work_queue = self._generate_relevance_bounds_tasks(dims, self.data, presetModel,
                                                           init_model_state)

        # Solve relevance bounds in parallel (when available)
        if parallel is None:
            parallel = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        bound_results = parallel(map(joblib.delayed(_start_solver_worker), work_queue))

        # Retrieve results and aggregate values in dict
        solved_bounds = defaultdict(list)
        for finished_bound in bound_results:

            # Only add bounds with feasible solutions
            if finished_bound.is_solved:
                solved_bounds[finished_bound.current_feature].append(finished_bound)

        # Initalize array for pair of bounds(= intervals)
        length = len(dims)
        intervals = np.zeros((length, 2))
        for abs_index, rel_index in zip(dims, range(length)):
            # Return interval for feature i (can be a fixed value when set beforehand)
            interval_i = _create_interval(abs_index, solved_bounds, presetModel)
            intervals[rel_index] = interval_i

        self.raw_intervals_ = intervals

        return intervals  # TODO: add model model_state (omega, bias) to return value

    def compute_probe_values(self, dims, n_resampling=50, parallel=None, presetModel=None, max_loops=3):
        # Get model parameters
        init_model_state = self.best_init_model.model_state

        # Prepare parallel framework
        if parallel is None:
            parallel = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        # Run loop until we got enough samples
        enough_samples = False
        i = 0
        probe_values = []
        while not enough_samples:
            # Genarate
            probe_queue = self._generate_probe_value_tasks(dims, self.data,
                                                           n_resampling,
                                                           self.random_state, presetModel, init_model_state)
            probe_results = parallel(map(joblib.delayed(_start_solver_worker), probe_queue))
            probe_values.extend([probe.objective.value for probe in probe_results if probe.is_solved])

            n_probes = len(probe_values)
            if n_resampling > MIN_N_PROBE_FEATURES > n_probes:
                print(f"Only {n_probes} probe features were feasible.")
                i += 1
            else:
                enough_samples = True
            if i == max_loops:
                break

        return probe_values

    def _generate_relevance_bounds_tasks(self, dims, data, preset_model=None,
                                         best_model_state=None):
        # Get problem type specific bound class (classification, regression, etc. ...)
        bound = self.problem_type.get_bound_model()

        # Instantiate objects for computation later
        for di in dims:
            # Add Lower Bound problem to work list
            isLowerBound = True
            yield bound(isLowerBound, di, data, self.best_hyperparameters, self.init_constraints,
                        preset_model=preset_model,
                        best_model_state=best_model_state)

            # Add two problems for Upper bound, we pick a feasible and maximal candidate later
            isLowerBound = False
            yield bound(isLowerBound, di, data, self.best_hyperparameters, self.init_constraints, sign=False,
                        preset_model=preset_model,
                        best_model_state=best_model_state)
            yield bound(isLowerBound, di, data, self.best_hyperparameters, self.init_constraints, sign=True,
                        preset_model=preset_model,
                        best_model_state=best_model_state)

    def _generate_probe_value_tasks(self, dims, data, n_resampling, random_state,
                                    preset_model=None, best_model_state=None):
        # Get problem type specific bound class (classification, regression, etc. ...)
        bound = self.problem_type.get_bound_model()

        # Random sample n_resampling shadow features by permuting real features and computing upper bound
        random_choice = random_state.choice(a=np.arange(len(dims)), size=n_resampling)

        # Instantiate objects
        for i, di in enumerate(random_choice):
            data_perm = permutate_feature_in_data(data, di, random_state)

            # We only use upper bounds as probe features
            isLowerBound = False
            yield bound(isLowerBound, di, data_perm, self.best_hyperparameters, self.init_constraints, sign=False,
                        preset_model=preset_model,
                        best_model_state=best_model_state)
            yield bound(isLowerBound, di, data_perm, self.best_hyperparameters, self.init_constraints, sign=True,
                        preset_model=preset_model,
                        best_model_state=best_model_state)

    def _compute_single_preset_relevance_bounds(self, i: int, signed_preset_i: [float, float]):
        """
        Method to run method once for one restricted feature
        Parameters
        ----------
        i:
            restricted feature
        signed_preset_i:
            restricted range of feature i (set before optimization = preset)

        """
        preset = {}
        preset[i] = signed_preset_i

        rangevector = self._compute_multi_preset_relevance_bounds(preset)

        return rangevector

    def _compute_multi_preset_relevance_bounds(self, preset=None, normalized=True):
        """
        Method to run method with preset values
        """
        X, y = self.data

        # The user is working with normalized values while we compute them unscaled
        if normalized:
            for k, v in preset.items():
                preset[k] = v * self.best_init_model.L1_factor

        # Add sign to presets
        preset = self._add_sign_to_preset(preset)

        # Calculate all bounds with feature i set to min_i
        rangevector = self.compute_relevance_bounds(presetModel=preset)

        # Current dimension is not constrained, so these values are set accordingly
        for i, p in enumerate(preset):
            if np.all(np.isnan(p)):
                continue
            else:
                rangevector[i] = p

        return rangevector

    def _add_sign_to_preset(self, unsigned_presets):
        """
        We need signed presets for our convex problem definition later.
        We reuse the coefficients of the optimal model for this

        Parameters
        ----------
        unsigned_presets : dict

        Returns
        -------
        dict
        """

        signed_presets = {}
        # Obtain optimal model parameters
        w = self.best_init_model.model_state["w"]
        sum = 0
        for i, preset in unsigned_presets.items():
            unsigned_preset_i = np.sign(w[i]) * preset
            # accumulate maximal feature  contribution
            sum += unsigned_preset_i[1]
            signed_presets[i] = unsigned_preset_i

        # Check if unsigned_presets makes sense
        l1 = self.init_constraints["w_l1"]
        if sum > l1:
            print("maximum L1 norm of presets: ", sum)
            print("L1 allowed:", l1)
            print("Presets are not feasible. Try lowering values.")
            return

        return signed_presets


def _get_necessary_dimensions(d: int, presetModel: dict, start=0):
    dims = np.arange(start, d)
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
    return dims


def _create_interval(feature: int, solved_bounds: dict, presetModel: dict = None):
    # Return preset values for fixed features
    if presetModel is not None:
        if feature in presetModel:
            return presetModel[feature]

    lower_bound = 0
    upper_bound = 0

    all_bounds = solved_bounds[feature]
    if len(all_bounds) < 2:
        logging.error(f"(Some) relevance bounds for feature {feature} were not solved.")
        raise Exception("Infeasible bound(s).")
    for bound in all_bounds:
        value = bound.objective.value
        if bound.isLowerBound:
            lower_bound = value
        else:
            if value > upper_bound:
                upper_bound = value
    return lower_bound, upper_bound


def _postprocessing(L1, rangevector):
    assert L1 > 0
    # Scale to L1
    rangevector = rangevector.copy() / L1
    # round mins to zero
    # rangevector[np.abs(rangevector) < 1 * 10 ** -4] = 0

    return rangevector


def feature_classification(probe_values, relevance_bounds, fpr=1e-5):
    n = len(probe_values)
    assert n > 0

    probe_values = np.asarray(probe_values)
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

    return prediction
