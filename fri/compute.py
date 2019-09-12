import logging
from collections import defaultdict

import joblib
import numpy as np
from scipy import stats

from fri.model.base_cvxproblem import Relevance_CVXProblem
from fri.model.base_initmodel import InitModel
from fri.model.base_type import ProblemType
from fri.utils import permutate_feature_in_data

MIN_N_PROBE_FEATURES = 20  # Lower bound of probe features


def _start_solver_worker(bound: Relevance_CVXProblem):
    """
    Worker thread method for parallel computation
    """
    return bound.solve()


class RelevanceBoundsIntervals(object):
    def __init__(
        self,
        data,
        problem_type: ProblemType,
        best_init_model: InitModel,
        random_state,
        n_resampling,
        n_jobs,
        verbose,
        normalize=True,
    ):
        self.data = data
        self.problem_type = problem_type
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_resampling = n_resampling
        self.random_state = random_state
        self.best_init_model = best_init_model
        self.best_hyperparameters = best_init_model.hyperparam
        self.normalize = normalize

        # Relax constraints to improve stability
        relaxed_constraints = problem_type.get_relaxed_constraints(
            best_init_model.constraints
        )
        self.init_constraints = relaxed_constraints

    def get_normalized_lupi_intervals(self, lupi_features, presetModel=None):

        # We define a list of all the features we want to compute relevance bounds for
        X, _ = self.data  # TODO: handle other data formats
        all_d = X.shape[1]
        normal_d = all_d - lupi_features

        # Compute relevance bounds and probes for normal features and LUPI
        with joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            d_n = _get_necessary_dimensions(normal_d, presetModel)
            rb = self.compute_relevance_bounds(d_n, parallel=parallel)
            probe_upper = self.compute_probe_values(d_n, True, parallel=parallel)
            probe_lower = self.compute_probe_values(d_n, False, parallel=parallel)

            d_l = _get_necessary_dimensions(all_d, presetModel, start=normal_d)
            rb_l = self.compute_relevance_bounds(d_l, parallel=parallel)
            probe_priv_upper = self.compute_probe_values(d_l, True, parallel=parallel)
            probe_priv_lower = self.compute_probe_values(d_l, False, parallel=parallel)
        probes = [probe_lower, probe_upper, probe_priv_lower, probe_priv_upper]
        #
        # Postprocess
        #

        # Get Scaling Parameters
        l1 = self.init_constraints["w_l1"]
        l1_priv = self.init_constraints["w_priv_l1"]
        l1 = l1 + l1_priv

        # Normalize Normal and Lupi features
        rb_norm = self._postprocessing(l1, rb)
        rb_l_norm = self._postprocessing(l1, rb_l)
        interval_ = np.concatenate([rb_norm, rb_l_norm])

        # Normalize Probes
        probe_lower = self._postprocessing(l1, probe_lower)
        probe_upper = self._postprocessing(l1, probe_upper)
        probe_priv_lower = self._postprocessing(l1, probe_priv_lower)
        probe_priv_upper = self._postprocessing(l1, probe_priv_upper)

        #
        #
        # Classify features
        fc = feature_classification(
            probe_lower, probe_upper, rb_norm, verbose=self.verbose
        )
        fc_l = feature_classification(
            probe_priv_lower, probe_priv_upper, rb_l_norm, verbose=self.verbose
        )
        fc_both = np.concatenate([fc, fc_l])

        return interval_, fc_both

    def get_normalized_intervals(self, presetModel=None):
        # We define a list of all the features we want to compute relevance bounds for
        X, _ = self.data  # TODO: handle other data formats
        d = X.shape[1]
        # Depending on the preset model, we dont need to compute all bounds
        # e.g. in the case of fixed features we skip those
        dims = _get_necessary_dimensions(d, presetModel)

        with joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose) as parallel:
            relevance_bounds = self.compute_relevance_bounds(
                dims, parallel=parallel, presetModel=presetModel
            )
            probe_values_upper = self.compute_probe_values(
                dims, isUpper=True, parallel=parallel, presetModel=presetModel
            )
            probe_values_lower = self.compute_probe_values(
                dims, isUpper=False, parallel=parallel, presetModel=presetModel
            )

        # Postprocess bounds
        norm_bounds = self._postprocessing(
            self.best_init_model.L1_factor, relevance_bounds
        )
        norm_probe_values_upper = self._postprocessing(
            self.best_init_model.L1_factor, probe_values_upper
        )
        norm_probe_values_lower = self._postprocessing(
            self.best_init_model.L1_factor, probe_values_lower
        )
        feature_classes = feature_classification(
            norm_probe_values_lower,
            norm_probe_values_upper,
            norm_bounds,
            verbose=self.verbose,
        )

        return norm_bounds, feature_classes

    def compute_relevance_bounds(
        self, dims, parallel=None, presetModel=None, solverargs=None
    ):
        init_model_state = self.best_init_model.model_state

        work_queue = self._generate_relevance_bounds_tasks(
            dims, self.data, presetModel, init_model_state
        )

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
            interval_i = self._create_interval(abs_index, solved_bounds, presetModel)
            intervals[rel_index] = interval_i

        return intervals  # TODO: add model model_state (omega, bias) to return value

    def compute_probe_values(self, dims, isUpper=True, parallel=None, presetModel=None):
        # Get model parameters
        init_model_state = self.best_init_model.model_state

        # Prepare parallel framework
        if parallel is None:
            parallel = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)

        # Generate
        probe_queue = self._generate_probe_value_tasks(
            self.data,
            dims,
            isUpper,
            self.n_resampling,
            self.random_state,
            presetModel,
            init_model_state,
        )
        # Compute solution
        probe_results = parallel(map(joblib.delayed(_start_solver_worker), probe_queue))
        # probe_values.extend([probe.objective.value for probe in probe_results if probe.is_solved])

        candidates = defaultdict(list)
        for candidate in probe_results:
            # Only add bounds with feasible solutions
            if candidate.is_solved:
                candidates[candidate.probeID].append(candidate)
        probe_values = []
        for probes_for_ID in candidates.values():
            if isUpper:
                probe_values.append(
                    self.problem_type.get_cvxproblem_template.aggregate_max_candidates(
                        probes_for_ID
                    )
                )
            else:
                probe_values.append(
                    self.problem_type.get_cvxproblem_template.aggregate_min_candidates(
                        probes_for_ID
                    )
                )

        return np.array(probe_values)

    def _generate_relevance_bounds_tasks(
        self, dims, data, preset_model=None, best_model_state=None
    ):
        # Do not compute bounds for fixed features
        if preset_model is not None:
            dims = [di for di in dims if di not in preset_model]

        # Instantiate objects for computation later
        for di in dims:
            # Add Lower Bound problem(s) to work list
            yield from self.problem_type.get_cvxproblem_template.generate_lower_bound_problem(
                self.best_hyperparameters,
                self.init_constraints,
                best_model_state,
                data,
                di,
                preset_model,
            )

            # Add problem(s) for Upper bound
            yield from self.problem_type.get_cvxproblem_template.generate_upper_bound_problem(
                self.best_hyperparameters,
                self.init_constraints,
                best_model_state,
                data,
                di,
                preset_model,
            )

    def _generate_probe_value_tasks(
        self,
        data,
        dims,
        isUpper,
        n_resampling,
        random_state,
        preset_model=None,
        best_model_state=None,
    ):
        if isUpper:
            factory = (
                self.problem_type.get_cvxproblem_template.generate_upper_bound_problem
            )
        else:
            factory = (
                self.problem_type.get_cvxproblem_template.generate_lower_bound_problem
            )

        # Random sample n_resampling shadow features by permuting real features and computing upper bound
        random_choice = random_state.choice(a=dims, size=n_resampling)

        # Instantiate objects
        for i, di in enumerate(random_choice):
            data_perm = permutate_feature_in_data(data, di, random_state)

            # We only use upper bounds as probe features
            yield from factory(
                self.best_hyperparameters,
                self.init_constraints,
                best_model_state,
                data_perm,
                di,
                preset_model,
                probeID=i,
            )

    def _create_interval(
        self, feature: int, solved_bounds: dict, presetModel: dict = None
    ):
        # Return preset values for fixed features
        if presetModel is not None:
            if feature in presetModel:
                return presetModel[feature].squeeze()

        all_bounds = solved_bounds[feature]
        min_problems_candidates = [p for p in all_bounds if p.isLowerBound]
        max_problems_candidates = [p for p in all_bounds if not p.isLowerBound]
        if len(all_bounds) < 2:
            logging.error(
                f"(Some) relevance bounds for feature {feature} were not solved."
            )
            raise Exception("Infeasible bound(s).")
        lower_bound = self.problem_type.get_cvxproblem_template.aggregate_min_candidates(
            min_problems_candidates
        )
        upper_bound = self.problem_type.get_cvxproblem_template.aggregate_max_candidates(
            max_problems_candidates
        )
        return lower_bound, upper_bound

    def compute_single_preset_relevance_bounds(
        self, i: int, signed_preset_i: [float, float]
    ):
        """
        Method to run method once for one restricted feature
        Parameters
        ----------
        i:
            restricted feature
        signed_preset_i:
            restricted range of feature i (set before optimization = preset)

        """
        preset = {i: signed_preset_i}

        rangevector = self.compute_multi_preset_relevance_bounds(preset)

        return rangevector

    def compute_multi_preset_relevance_bounds(self, preset, lupi_features=0):
        """
        Method to run method with preset values

        Parameters
        ----------
        lupi_features
        """

        # The user is working with normalized values while we compute them unscaled
        if self.normalize:
            normalized = {}
            for k, v in preset.items():
                normalized[k] = np.asarray(v) * self.best_init_model.L1_factor
            preset = normalized

        # Add sign to presets
        preset = self._add_sign_to_preset(preset)

        # Calculate all bounds with feature i set to min_i
        if lupi_features > 0:
            rangevector, f_classes = self.get_normalized_lupi_intervals(
                lupi_features, presetModel=preset
            )
        else:
            rangevector, f_classes = self.get_normalized_intervals(presetModel=preset)

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
            preset = np.array(preset)
            if preset.size == 1:
                preset = np.repeat(preset, 2)
            unsigned_preset_i = np.sign(w[i]) * preset
            # accumulate maximal feature  contribution
            sum += unsigned_preset_i[1]  # Take upper preset
            signed_presets[i] = unsigned_preset_i

        # Check if unsigned_presets makes sense
        l1 = self.init_constraints["w_l1"]
        if sum > l1:
            print("maximum L1 norm of presets: ", sum)
            print("L1 allowed:", l1)
            print("Presets are not feasible. Try lowering values.")
            return

        return signed_presets

    def _postprocessing(self, L1, rangevector, round_to_zero=True):
        if self.normalize:
            assert L1 > 0
            rangevector = rangevector.copy() / L1

        if round_to_zero:
            rangevector[rangevector <= 1e-11] = 0
        return rangevector


def _get_necessary_dimensions(d: int, presetModel: dict = None, start=0):
    dims = np.arange(start, d)

    # if presetModel is not None:
    #    # Exclude fixed (preset) dimensions from being redundantly computed
    #    dims = [di for di in dims if di not in presetModel.keys()]
    # TODO: check the removal of this block
    return dims


def feature_classification(
    probes_low, probes_up, relevance_bounds, fpr=1e-4, verbose=0
):
    logging.debug("**** Feature Selection ****")
    logging.debug("Generating Lower Probe Statistic")
    lower_stat = create_probe_statistic(probes_low, fpr, verbose=verbose)
    logging.debug("Generating Upper Probe Statistic")
    upper_stat = create_probe_statistic(probes_up, fpr, verbose=verbose)

    weakly = relevance_bounds[:, 1] > upper_stat[1]
    strongly = relevance_bounds[:, 0] > lower_stat[1]
    both = np.logical_and(weakly, strongly)
    prediction = np.zeros(relevance_bounds.shape[0], dtype=np.int)
    prediction[weakly] = 1
    prediction[both] = 2

    return prediction


def create_probe_statistic(probe_values, fpr, verbose=0):
    # Create prediction interval statistics based on randomly permutated probe features (based on real features)
    n = len(probe_values)

    if n == 0:
        if verbose > 0:
            logging.info(
                "All probes were infeasible. All features considered relevant."
            )
        #    # If all probes were infeasible we expect an empty list
        #    # If they are infeasible it also means that only strongly relevant features were in the data
        #    # As such we just set the prediction without considering the statistics
        mean = 0
    else:
        probe_values = np.asarray(probe_values)
        mean = probe_values.mean()

    if mean == 0:
        lower_threshold, upper_threshold = mean, mean
        s = 0
    else:
        s = probe_values.std()
        lower_threshold = mean + stats.t(df=n - 1).ppf(fpr) * s * np.sqrt(1 + (1 / n))
        upper_threshold = mean - stats.t(df=n - 1).ppf(fpr) * s * np.sqrt(1 + (1 / n))

    if verbose > 0:
        print(
            f"FS threshold: {lower_threshold}-{upper_threshold}, Mean:{mean}, Std:{s}, n_probes {n}"
        )

    return lower_threshold, upper_threshold
