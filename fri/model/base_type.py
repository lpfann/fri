from abc import ABC, abstractmethod

import scipy.stats


class ProblemType(ABC):
    def __init__(self, **kwargs):

        self.chosen_parameters_ = {}

        for p in self.parameters():
            if p in kwargs:
                if kwargs[p] is not None:
                    self.chosen_parameters_[p] = kwargs[p]

        self.relax_factors_ = {}
        for p in self.relax_factors():
            if p in kwargs:
                if kwargs[p] is not None:
                    self.relax_factors_[p] = kwargs[p]

    @classmethod
    @abstractmethod
    def parameters(cls):
        raise NotImplementedError

    def get_chosen_parameter(self, p):
        try:
            return [
                self.chosen_parameters_[p]
            ]  # We return list for param search function
        except:
            # # TODO: rewrite the parameter logic
            # # TODO: move this to subclass
            if p == "scaling_lupi_w":
                # return [0.1, 1, 10, 100, 1000]
                return scipy.stats.reciprocal(a=1e-15, b=1e10)
            # if p == "scaling_lupi_loss":
            #    # value 0>p<1 causes standard svm solution
            #    # p>1 encourages usage of lupi function
            #    return scipy.stats.reciprocal(a=1e-15, b=1e15)
            if p == "C":
                return scipy.stats.reciprocal(a=1e-5, b=1e5)
            if p == "epsilon":
                return [0, 0.001, 0.01, 0.1, 1, 10, 100]
            else:
                return scipy.stats.reciprocal(a=1e-10, b=1e10)

    def get_all_parameters(self):
        return {p: self.get_chosen_parameter(p) for p in self.parameters()}

    @classmethod
    @abstractmethod
    def relax_factors(cls):
        raise NotImplementedError

    def get_chosen_relax_factors(self, p):
        try:
            factor = self.relax_factors_[p]
        except KeyError:
            try:
                factor = self.relax_factors_[p + "_slack"]
            except KeyError:
                factor = 0.1
        if factor < 0:
            raise ValueError("Slack Factor multiplier is positive!")
        return factor

    def get_all_relax_factors(self):
        return {p: self.get_chosen_relax_factors(p) for p in self.relax_factors()}

    @property
    @abstractmethod
    def get_initmodel_template(self):
        pass

    @property
    @abstractmethod
    def get_cvxproblem_template(self):
        pass

    @abstractmethod
    def preprocessing(self, data, lupi_features=None):
        return data

    def postprocessing(self, bounds):
        return bounds

    def get_relaxed_constraints(self, constraints):
        return {c: self.relax_constraint(c, v) for c, v in constraints.items()}

    def relax_constraint(self, key, value):
        return value * (1 + self.get_chosen_relax_factors(key))
