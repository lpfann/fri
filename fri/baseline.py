import warnings
from abc import ABC, abstractmethod
from pprint import pprint
from typing import Tuple

import numpy as np
from numpy.random import mtrand
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV


class InitModel(ABC, BaseEstimator):

    def __init__(self, **parameters):
        if parameters is None:
            parameters = {}
        self.hyperparam = parameters
        self._model_state = {}

    def _get_param_names(cls):
        return sorted(cls.hyperparameter())

    @classmethod
    @abstractmethod
    def hyperparameter(cls):
        raise NotImplementedError

    def get_params(self, deep=True):
        return self.hyperparam

    def set_params(self, **params):
        for p, value in params.items():
            self.hyperparam[p] = value
        return self

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y, **kwargs):
        pass

    @classmethod
    def make_scorer(self):
        return None, None

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        self._constraints = constraints

    @property
    def model_state(self):
        return self._model_state

    @model_state.setter
    def model_state(self, params):
        self._model_state = params

    @property
    def L1_factor(self):
        try:
            return self.constraints["w_l1"]
        except:
            raise NotImplementedError(
                "Baseline model does not provide (L1) normalization constant. Expected l1 norm of model weigts (e.g. w).")

    @property
    def solver_params(cls):
        return {"solver": "ECOS"}


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
                pprint((f"{k}: {v}"))
        print("*" * 30)
    return best_model, best_score

