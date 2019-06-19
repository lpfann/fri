import warnings

from sklearn.exceptions import FitFailedWarning

warnings.filterwarnings(action="ignore", category=FitFailedWarning)

from pprint import pprint
from typing import Tuple

import numpy as np
from numpy.random import mtrand
from sklearn.model_selection import RandomizedSearchCV

from fri.model.base_initmodel import InitModel


def find_best_model(model_template: InitModel, hyperparameters: dict, data: Tuple[np.ndarray, np.ndarray],
                    random_state: mtrand.RandomState,
                    n_iter: int, n_jobs: int, verbose: int = 0, lupi_features=None, kwargs: dict = None) -> Tuple[
    InitModel, float]:
    model = model_template()

    scorer, metric = model.make_scorer()
    if scorer is None:
        refit = True
    else:
        refit = metric

    searcher = RandomizedSearchCV(model,
                                  hyperparameters,
                                  scoring=scorer,
                                  random_state=random_state,
                                  refit=refit,
                                  cv=5,
                                  n_iter=n_iter,
                                  n_jobs=n_jobs,
                                  error_score=np.nan,
                                  verbose=verbose)

    X, y = data
    # Ignore warnings for extremely bad model_state (when precision=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        searcher.fit(X, y, lupi_features=lupi_features)

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

