from abc import ABC, abstractmethod

import cvxpy as cvx
import numpy as np
from cvxpy import SolverError


class Relevance_CVXProblem(ABC):
    def __repr__(self) -> str:
        if self.isLowerBound:
            lower = "Lower"
        else:
            lower = "Upper"
        name = f"{lower}_{self.current_feature}_{self.__class__.__name__}"
        state = ""
        for s in self.init_hyperparameters.items():
            state += f"{s[0]}:{s[1]}, "
        for s in self.init_model_constraints.items():
            state += f"{s[0]}:{s[1]}, "
        state = "(" + state[:-2] + ")"
        if self.isProbe:
            prefix = f"Probe_{self.probeID}"
        else:
            prefix = ""
        return prefix + name + state

    def __init__(
        self,
        current_feature: int,
        data: tuple,
        hyperparameters,
        best_model_constraints,
        preset_model=None,
        best_model_state=None,
        probeID=-1,
        **kwargs,
    ) -> None:
        self._probeID = probeID
        self._feature_relevance = None
        self.isLowerBound = None

        # General data
        self.current_feature = current_feature
        self.preset_model = preset_model
        self.best_model_state = best_model_state

        self.preprocessing_data(data, best_model_state)

        # Initialize constraints
        self._constraints = []
        self._objective = None
        self.w = None
        self._init_constraints(hyperparameters, best_model_constraints)

        if self.preset_model is not None:
            self._add_preset_constraints(self.preset_model, best_model_constraints)

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

    @property
    def solved_relevance(self):
        if self.is_solved:
            return self.objective.value
        else:
            raise Exception("Problem not solved. No feature relevance computed.")

    @property
    def probeID(self):
        return self._probeID

    @property
    def isProbe(self):
        return self.probeID >= 0

    @abstractmethod
    def _init_constraints(self, parameters, init_model_constraints):
        pass

    @abstractmethod
    def init_objective_UB(self, **kwargs):
        pass

    @abstractmethod
    def init_objective_LB(self, **kwargs):
        pass

    @property
    def cvx_problem(self):
        return self._cvx_problem

    @property
    def is_solved(self):
        if self._solver_status in self.accepted_status:
            try:
                val = self.objective.value
            except ValueError:
                return False

            return True
        else:
            return False

    @property
    def accepted_status(self):
        return ["optimal", "optimal_inaccurate"]

    def solve(self) -> object:
        # We init cvx problem here because pickling LP solver objects is problematic
        # by deferring it to here, worker threads do the problem building themselves and we spare the serialization
        self._cvx_problem = cvx.Problem(
            objective=self.objective, constraints=self.constraints
        )
        try:
            # print("Solve", self)
            self._cvx_problem.solve(**self.solver_kwargs)
        except SolverError:
            # We ignore Solver Errors, which are common with our framework:
            # We solve multiple problems per bound and choose a feasible solution later (see '_create_interval')
            pass

        self._solver_status = self._cvx_problem.status
        # self._cvx_problem = None
        return self

    def _retrieve_result(self):
        return self.current_feature, self.objective

    @property
    def solver_kwargs(self):
        return {"verbose": False, "solver": "ECOS", "max_iters": 300}

    def _add_preset_constraints(self, preset_model: dict, best_model_constraints):

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
                self.add_constraint(self.w[feature] >= current_preset[0])
                self.add_constraint(self.w[feature] <= current_preset[1])
            else:
                self.add_constraint(self.w[feature] <= current_preset[0])
                self.add_constraint(self.w[feature] >= current_preset[1])

    @classmethod
    def generate_lower_bound_problem(
        cls,
        best_hyperparameters,
        init_constraints,
        best_model_state,
        data,
        di,
        preset_model,
        probeID=-1,
    ):
        problem = cls(
            di,
            data,
            best_hyperparameters,
            init_constraints,
            preset_model=preset_model,
            best_model_state=best_model_state,
            probeID=probeID,
        )
        problem.init_objective_LB()
        problem.isLowerBound = True
        yield problem

    @classmethod
    def generate_upper_bound_problem(
        cls,
        best_hyperparameters,
        init_constraints,
        best_model_state,
        data,
        di,
        preset_model,
        probeID=-1,
    ):
        for sign in [-1, 1]:
            problem = cls(
                di,
                data,
                best_hyperparameters,
                init_constraints,
                preset_model=preset_model,
                best_model_state=best_model_state,
                probeID=probeID,
            )
            problem.init_objective_UB(sign=sign)
            problem.isLowerBound = False
            yield problem

    @classmethod
    def aggregate_min_candidates(cls, min_problems_candidates):
        vals = [candidate.solved_relevance for candidate in min_problems_candidates]
        min_value = min(vals)
        return min_value

    @classmethod
    def aggregate_max_candidates(cls, max_problems_candidates):
        vals = [candidate.solved_relevance for candidate in max_problems_candidates]
        max_value = max(vals)
        return max_value
