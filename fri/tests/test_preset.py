import numpy as np
import pytest
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state

from fri import FRI, ProblemName, quick_generate


@pytest.fixture(scope="module")
def randomstate():
    return check_random_state(1337)


@pytest.mark.parametrize(
    "problem",
    [ProblemName.REGRESSION, ProblemName.CLASSIFICATION, ProblemName.ORDINALREGRESSION],
)
def test__compute_single_preset_relevance_bounds(problem, randomstate):
    data = quick_generate(
        problem,
        n_samples=300,
        n_features=4,
        n_redundant=2,
        n_strel=2,
        random_state=randomstate,
    )

    X_orig, y = data
    X = scale(X_orig)

    model = FRI(problem, random_state=randomstate, n_jobs=1)
    model.fit(X, y)
    normal_range = model.interval_.copy()

    i = 0
    preset = [model.interval_[i, 0], model.interval_[i, 0]]
    range = model._relevance_bounds_computer.compute_single_preset_relevance_bounds(
        i, preset
    )
    assert normal_range.shape == range.shape
    assert range[i][0] == preset[0]
    assert range[i][1] == preset[1]


@pytest.mark.parametrize(
    "problem",
    [ProblemName.REGRESSION, ProblemName.CLASSIFICATION, ProblemName.ORDINALREGRESSION],
)
def test__compute_multi_preset_relevance_bounds(problem, randomstate):
    data = quick_generate(
        problem,
        n_samples=300,
        n_features=4,
        n_redundant=2,
        n_strel=2,
        random_state=randomstate,
    )

    X_orig, y = data
    X = scale(X_orig)

    model = FRI(problem, random_state=randomstate, n_jobs=1)
    model.fit(X, y)
    normal_range = model.interval_.copy()

    # We fix two feature values at the same time.
    # Note that we can not use the relevance bounds for multiple features, because of infeasibility
    # In the singular optimization case we exhaust the complete slack of our model constraints
    # If whe use multiple relevance bounds we double count this slack and the model optimization rightfully claims infeasible
    # Instead we use the mean value of the relevance bounds, which should be working in most cases
    i = 0
    mean_0 = np.mean([model.interval_[i, 0], model.interval_[i, 1]])
    preset_0 = [mean_0, mean_0]
    i = 1
    mean_1 = np.mean([model.interval_[i, 0], model.interval_[i, 1]])
    preset_1 = [mean_1, mean_1]
    presetModel = {0: preset_0, 1: preset_1}
    range = model._relevance_bounds_computer.compute_multi_preset_relevance_bounds(
        presetModel
    )

    assert normal_range.shape == range.shape
    i = 0
    assert range[i][0] == preset_0[0]
    assert range[i][1] == preset_0[1]
    i = 1
    assert range[i][0] == preset_1[0]
    assert range[i][1] == preset_1[1]


@pytest.mark.parametrize(
    "problem",
    [ProblemName.REGRESSION, ProblemName.CLASSIFICATION, ProblemName.ORDINALREGRESSION],
)
def test__compute_multi_preset_relevance_bounds(problem, randomstate):
    data = quick_generate(
        problem,
        n_samples=300,
        n_features=4,
        n_redundant=2,
        n_strel=2,
        random_state=randomstate,
    )

    X_orig, y = data
    X = scale(X_orig)

    model = FRI(problem, random_state=randomstate, n_jobs=1)
    model.fit(X, y)
    normal_range = model.interval_.copy()

    # We fix two feature values at the same time.
    # Note that we can not use the relevance bounds for multiple features, because of infeasibility
    # In the singular optimization case we exhaust the complete slack of our model constraints
    # If whe use multiple relevance bounds we double count this slack and the model optimization rightfully claims infeasible
    # Instead we use the mean value of the relevance bounds, which should be working in most cases
    i = 0
    mean_0 = np.mean([model.interval_[i, 0], model.interval_[i, 1]])
    preset_0 = [mean_0, mean_0]
    i = 1
    mean_1 = np.mean([model.interval_[i, 0], model.interval_[i, 1]])
    preset_1 = [mean_1, mean_1]
    presetModel = {0: preset_0, 1: preset_1}
    range = model.constrained_intervals(presetModel)

    assert normal_range.shape == range.shape
    i = 0
    assert range[i][0] == pytest.approx(preset_0[0])
    assert range[i][1] == pytest.approx(preset_0[1])
    i = 1
    assert range[i][0] == pytest.approx(preset_1[0])
    assert range[i][1] == pytest.approx(preset_1[1])


def test__compute_single_preset_relevance_bounds(randomstate):
    problem = ProblemName.CLASSIFICATION
    data = quick_generate(
        problem,
        n_samples=300,
        n_features=4,
        n_redundant=2,
        n_strel=2,
        random_state=randomstate,
    )

    X_orig, y = data
    X = scale(X_orig)

    model = FRI(problem, random_state=randomstate, n_jobs=1)
    model.fit(X, y)
    normal_range = model.interval_.copy()

    i = 0
    preset = [model.interval_[i, 0]]
    range = model._relevance_bounds_computer.compute_single_preset_relevance_bounds(
        i, preset
    )
    assert normal_range.shape == range.shape
    assert range[i][0] == preset[0]
    assert range[i][1] == preset

    preset = model.interval_[i, 0]
    range = model._relevance_bounds_computer.compute_single_preset_relevance_bounds(
        i, preset
    )
    assert normal_range.shape == range.shape
    assert range[i][0] == preset
    assert range[i][1] == preset
