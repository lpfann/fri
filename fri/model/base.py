from abc import ABC, abstractmethod

import cvxpy as cvx
import numpy as np
import scipy
from cvxpy import SolverError


class MLProblem(ABC):

    def __init__(self, **kwargs):

        self.chosen_parameters_ = {}
        for p in self.parameters():
            if p in kwargs:
                if kwargs[p] is not None:
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


class BoundModel(ABC):

    @abstractmethod
    def __init__(self, cvxproblem, data, parameters, init_constraints):
        self.cvxproblem_ = cvxproblem
        self.data = data
        self.parameters = parameters
        self.init_constraints = init_constraints

    @abstractmethod
    def fit(self):
        pass

    @property
    def cvxproblem(self):
        return self.cvxproblem_

    @property
    def model_parameters(self):
        return self.model_parameters

    @property
    def value(self):
        return self.value

    @abstractmethod
    def _convert_to_shadow(self):
        return self

    @abstractmethod
    def _solve(self):
        pass


class Relevance_CVXProblem(ABC):

    def __init__(self, isLowerBound: bool, current_feature: int, data: tuple, parameters: object,
                 init_model_constraints: object,
                 sign: bool = None) -> None:
        self.isLowerBound = isLowerBound
        # General data model_state
        self.sign = sign
        self.current_feature = current_feature

        data = self.preprocessing(data)
        X, y = data
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X
        self.y = np.array(y)

        # Initalize constraints
        self._constraints = []
        self._objective = None
        self._is_solved = False

        self._init_constraints(parameters, init_model_constraints)
        self._init_objective(isLowerBound)

    def preprocessing(self, data):
        return data

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
        if status is "optimal":
            # TODO: add other stati
            self._is_solved = True
        self._cvx_problem = None
        return self

    def _retrieve_result(self):
        return self.current_feature, self.objective

    @property
    def solver_kwargs(self):
        return {"verbose": False, "solver": "ECOS", "max_iters": 1000}
