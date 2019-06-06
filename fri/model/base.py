from abc import ABC, abstractmethod

import cvxpy as cvx
import numpy as np
import scipy.stats
from cvxpy import SolverError


class MLProblem(ABC):

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
            return [self.chosen_parameters_[p]]  # We return list for param search function
        except:
            # TODO: rewrite the parameter logic
            # TODO: move this to subclass
            if p == "scaling_lupi_w":
                return scipy.stats.reciprocal(a=1e-10, b=1e1)
            if p == "scaling_lupi_loss":
                return scipy.stats.reciprocal(a=1e1, b=1e10)
            if p == "C":
                return scipy.stats.reciprocal(a=1e-10, b=1e3)
            else:
                return scipy.stats.reciprocal(a=1e-5, b=1e5)

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
                factor = 0.01
        assert factor > 0
        return factor

    def get_all_relax_factors(self):
        return {p: self.get_chosen_relax_factors(p) for p in self.relax_factors()}

    @classmethod
    @abstractmethod
    def get_init_model(cls):
        pass

    @classmethod
    @abstractmethod
    def get_bound_model(cls):
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


class Relevance_CVXProblem(ABC):

    def __str__(self) -> str:
        if self.isLowerBound:
            lower = "Lower"
        else:
            lower = "Upper"
        if self.sign == True:
            sign = "(+)"
        elif self.sign == False:
            sign = "(-)"
        else:
            sign = ""
        name = f"{sign}{lower}_{self.current_feature}_{self.__class__.__name__}"
        state = ""
        for s in self.init_hyperparameters.items():
            state += f"{s[0]}:{s[1]}, "
        for s in self.init_model_constraints.items():
            state += f"{s[0]}:{s[1]}, "
        state = "(" + state[:-2] + ")"
        if self.isProbe:
            prefix = "Probe_"
        else:
            prefix = ""
        return prefix + name + state

    def __init__(self, isLowerBound: bool, current_feature: int, data: tuple, hyperparameters, best_model_constraints,
                 sign: bool = False, preset_model=None, best_model_state=None, isProbe=False) -> None:
        self.isLowerBound = isLowerBound
        self.isProbe = isProbe

        # General data
        self.sign = sign
        self.current_feature = current_feature
        self.preset_model = preset_model
        self.best_model_state = best_model_state

        self.preprocessing_data(data, best_model_state)

        # Initialize constraints
        self._constraints = []
        self._objective = None
        self._is_solved = False
        self._init_constraints(hyperparameters, best_model_constraints)
        self._init_objective(isLowerBound)

        if self.preset_model is not None:
            self._add_preset_constraints(self.preset_model, self.best_model_state, best_model_constraints)

        self.init_hyperparameters = hyperparameters
        self.init_model_constraints = best_model_constraints

    def preprocessing_data(self, data, best_model_state):
        X, y = data
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X
        self.y = np.array(y)

    @property
    def constraints(self):
        return self._constraints

    def add_constraint(self, new):
        self._constraints.append(new)

    @property
    def objective(self):
        return self._objective

    @abstractmethod
    def _init_constraints(self, parameters, init_model_constraints):
        pass

    def _init_objective(self, isLowerBound):
        if isLowerBound:
            self._init_objective_LB()
        else:
            self._init_objective_UB()

    @abstractmethod
    def _init_objective_UB(self):
        pass

    @abstractmethod
    def _init_objective_LB(self):
        pass

    @property
    def cvx_problem(self):
        return self._cvx_problem

    @property
    def is_solved(self):
        return self._is_solved

    @property
    def accepted_status(self):
        return ["optimal", "optimal_inaccurate"]

    def solve(self) -> object:
        # We init cvx problem here because pickling LP solver objects is problematic
        # by deferring it to here, worker threads do the problem building themselves and we spare the serialization
        self._cvx_problem = cvx.Problem(objective=self.objective, constraints=self.constraints)
        try:
            self._cvx_problem.solve(**self.solver_kwargs)
        except SolverError:
            # We ignore Solver Errors, which are common with our framework:
            # We solve multiple problems per bound and choose a feasible solution later (see '_create_interval')
            pass

        status = self._cvx_problem.status
        if status in self.accepted_status:
            self._is_solved = True

        self._solver_status = status
        self._cvx_problem = None
        return self

    def _retrieve_result(self):
        return self.current_feature, self.objective

    @property
    def solver_kwargs(self):
        return {"verbose": False, "solver": "ECOS"}

    def _add_preset_constraints(self, preset_model: dict, best_model_state, best_model_constraints):
        w = best_model_state["w"]
        assert w is not None

        for feature, current_preset in preset_model.items():
            # Skip current feature
            if feature == self.current_feature:
                continue

            # Skip unset values
            if all(np.isnan(current_preset)):
                continue

            # a weight bigger than the optimal model L1 makes no sense
            assert abs(current_preset[0]) <= best_model_constraints["w_l1"]
            assert abs(current_preset[1]) <= best_model_constraints["w_l1"]

            # We add a pair of constraints depending on sign of known coefficient
            # this makes it possible to solve this as a convex problem
            if current_preset[0] >= 0:
                self.add_constraint(
                    w[feature] >= current_preset[0]
                )
                self.add_constraint(
                    w[feature] <= current_preset[1]
                )
            else:
                self.add_constraint(
                    w[feature] <= current_preset[0]
                )
                self.add_constraint(
                    w[feature] >= current_preset[1]
                )
