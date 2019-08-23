from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


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
                "Baseline model does not provide (L1) normalization constant. Expected l1 norm of model weights (e.g. w)."
            )

    @property
    def solver_params(cls):
        return {"solver": "ECOS"}


class LUPI_InitModel(InitModel):
    @property
    def L1_factor_priv(self):
        try:
            return self.constraints["w_priv_l1"]
        except:
            raise NotImplementedError(
                "Baseline model does not provide LUPI (L1) normalization constant. Expected l1 norm of LUPI model weights (e.g. w_priv)."
            )
