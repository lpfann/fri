"""
    Base class for our initial baseline models which are used in Gridsearch.
    They store the constant parameters needed in the model
        and the dynamic instance attributes when fitted.
"""
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class InitModel(ABC, BaseEstimator):
    HYPERPARAMETER = {}
    SOLVER_PARAMS = {"solver": "ECOS"}

    def __init__(self, **parameters):
        self.model_state = {}
        self.constraints = {}

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y, **kwargs):
        pass

    def make_scorer(self):
        return None, None

    @property
    def L1_factor(self):
        try:
            return self.constraints["w_l1"]
        except:
            raise NotImplementedError(
                "Baseline model does not provide (L1) normalization constant. Expected l1 norm of model weights (e.g. w)."
            )


class LUPI_InitModel(InitModel):
    @property
    def L1_factor_priv(self):
        try:
            return self.constraints["w_priv_l1"]
        except:
            raise NotImplementedError(
                "Baseline model does not provide LUPI (L1) normalization constant. Expected l1 norm of LUPI model weights (e.g. w_priv)."
            )
