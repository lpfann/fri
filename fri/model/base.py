from abc import ABC, abstractmethod

import scipy


class MLProblem(ABC):

    def __init__(self, **kwargs):

        self.chosen_parameters_ = {}
        for p in self.parameters():
            if p in kwargs:
                self.chosen_parameters_[p] = kwargs[p]

    @classmethod
    @abstractmethod
    def parameters(cls):
        raise NotImplementedError

    def get_chosen_parameter(self, p):
        try:
            return self.chosen_parameters_[p]
        except:
            return scipy.stats.reciprocal(a=1e-3, b=1e3)

    def get_all_parameters(self):
        return {p: self.get_chosen_parameter(p) for p in self.parameters()}

    @classmethod
    @abstractmethod
    def get_init_model(cls):
        pass

    @classmethod
    @abstractmethod
    def get_bound_model(cls):
        pass

    @abstractmethod
    def preprocessing(self, data):
        return data

    def postprocessing(self, bounds):
        return bounds
