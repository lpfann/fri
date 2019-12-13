"""
    In this class we use hyperparameter search to find parameters needed in our model.
    Depending on the input model we sample parameters from a random distribution.
    The sampling rate can be increased.
    The model with the best internally defined accuracy is picked.
    To increase robustness we use cross validation.
"""
import warnings

from sklearn.exceptions import FitFailedWarning

warnings.filterwarnings(action="ignore", category=FitFailedWarning)

from pprint import pprint
from typing import Tuple

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

from fri.model.base_initmodel import InitModel


def find_best_model(
    model_template: InitModel,
    hyperparameters: dict,
    data: Tuple[np.ndarray, np.ndarray],
    random_state: np.random.RandomState,
    n_iter: int,
    n_jobs: int,
    verbose: int = 0,
    lupi_features=None,
    kwargs: dict = None,
) -> Tuple[InitModel, float]:
    """
    Search function which wraps `sklearns`  `RandomizedSearchCV` function.
    We use distributions and parameters defined in the `model_template`.

    Parameters
    ----------
    model_template : InitModel
        A model template which is used to fit data.
    hyperparameters : dict
        Dictionary of hyperparameters.
    data : tuple
        Tuple of data (X,y)
    random_state : RandomState
        numpy RandomState object
    n_iter : int
        Amount of search samples.
    n_jobs : int
        Allows multiprocessing with `n_jobs` threads.
    verbose : int
        Allows verbose output when `verbose>0`.
    lupi_features : int
        Amount of lupi_features
    kwargs : dict
        Placeholder, dict to pass into fit functions.
    """
    if lupi_features > 0:
        model = model_template(lupi_features=lupi_features)
    else:
        model = model_template()

    scorer, metric = model.make_scorer()
    if scorer is None:
        refit = True
    else:
        refit = metric

    searcher = RandomizedSearchCV(
        model,
        hyperparameters,
        scoring=scorer,
        random_state=random_state,
        refit=refit,
        cv=3,
        n_iter=n_iter,
        n_jobs=n_jobs,
        error_score=np.nan,
        verbose=verbose,
    )

    X, y = data
    # Ignore warnings for extremely bad model_state (when precision=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        searcher.fit(X, y)

    best_model: InitModel = searcher.best_estimator_
    best_score = best_model.score(X, y)
    if verbose > 0:
        print("*" * 20, "Best found baseline model", "*" * 20)
        pprint(best_model)
        print("score: ", best_score)
        for k, v in best_model.constraints.items():
            pprint((f"{k}: {v}"))
        for k, v in best_model.model_state.items():
            if hasattr(v, "shape"):
                pprint((f"{k}: shape {v.shape}"))
            else:
                if "slack" in k:
                    continue
                pprint((f"{k}: {v}"))
        print("*" * 30)
    return best_model, best_score
