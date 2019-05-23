import logging
from collections import defaultdict

import joblib
import numpy as np

from .baseline import InitModel
from .model.base import MLProblem
from .model.base import Relevance_CVXProblem


def compute_relevance_bounds(data, optimal_model, problem_type, random_state, n_resampling, n_jobs, verbose=None,
                             presetModel=None, solverargs=None):
    X, y = data  # TODO: handle other data formats
    n, d = X.shape

    # Depending on the preset model, we dont need to compute all bounds
    # e.g. in the case of fixed features we skip those
    dims = get_necessary_dimensions(d, presetModel)

    """
    Solver Parameters
    """
    # TODO: add custom solver arguments depending on baseline model
    if solverargs is not None:
        kwargs = solverargs
    else:
        kwargs = {"verbose": False, "solver": "ECOS", "max_iters": 1000}

    work_queue = generate_relevance_bounds_tasks(dims, data, optimal_model, problem_type)
    probe_queue = generate_probe_value_tasks(dims, data, optimal_model, problem_type, n_resampling, random_state)

    with joblib.Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:

        # Solve relevance bounds in parallel (when available)
        bound_results = parallel(map(joblib.delayed(_start_solver), work_queue))

        # Retrieve results and aggregate values in dict
        solved_bounds = defaultdict(list)
        for finished_bound in bound_results:

            # Only add bounds with feasible solutions
            if finished_bound.is_solved:
                solved_bounds[finished_bound.current_feature].append(finished_bound)

        # Initalize array for pair of bounds(= intervals)
        intervals = np.zeros((d, 2))
        for feature in dims:
            # Return interval for feature i (can be a fixed value when set beforehand)
            interval_i = _create_interval(feature, solved_bounds, presetModel)
            intervals[feature] = interval_i

        # Solve probe values for estimation of feature selection threshold
        probe_results = parallel(map(joblib.delayed(_start_solver), probe_queue))
        probe_values = [probe.objective.value for probe in probe_results if probe.is_solved]

    return intervals, probe_values  # TODO: add model model_state (omega, bias) to return value


def _start_solver(bound: Relevance_CVXProblem):
    """
    Worker thread method for parallel computation
    """
    return bound.solve()


def generate_relevance_bounds_tasks(dims, data, baseline_model: InitModel, problem_type: MLProblem):
    # Get problem type specific bound class (classification, regression, etc. ...)
    bound = problem_type.get_bound_model()
    # Instantiate objects for computation later
    for di in dims:
        # Add Lower Bound problem to work list
        isLowerBound = True
        yield bound(isLowerBound, di, data, baseline_model.hyperparam, baseline_model.constraints)

        # Add two problems for Upper bound, we pick a feasible and maximal candidate later
        isLowerBound = False
        yield bound(isLowerBound, di, data, baseline_model.hyperparam, baseline_model.constraints, sign=False)
        yield bound(isLowerBound, di, data, baseline_model.hyperparam, baseline_model.constraints, sign=True)


def permutate_feature_in_data(data, feature_i, random_state):
    X, y = data
    X_copy = np.copy(X)
    # Permute selected feature
    permutated_feature = random_state.permutation(X_copy[:, feature_i])
    # Add permutation back to dataset
    X_copy[:, feature_i] = permutated_feature
    return X_copy, y


def generate_probe_value_tasks(dims, data, baseline_model, problem_type, n_resampling, random_state):
    # Get problem type specific bound class (classification, regression, etc. ...)
    bound = problem_type.get_bound_model()
    # Random sample n_resampling shadow features by permuting real features and computing upper bound
    random_choice = random_state.choice(a=np.arange(len(dims)), size=n_resampling)
    # Instantiate objects
    for i, di in enumerate(random_choice):
        data_perm = permutate_feature_in_data(data, di, random_state)
        # We only use upper bounds as probe features
        isLowerBound = False
        yield bound(isLowerBound, di, data_perm, baseline_model.hyperparam, baseline_model.constraints, sign=False)
        yield bound(isLowerBound, di, data_perm, baseline_model.hyperparam, baseline_model.constraints, sign=True)


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


def get_necessary_dimensions(d, presetModel):
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
    return dims
