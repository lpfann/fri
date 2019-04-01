import abc

import cvxpy as cvx
import numpy as np


class ProblemType(object, metaclass=abc.ABCMeta):
    # Decorator class to add problem type specific constraints and variables to the BaseProblem
    @abc.abstractmethod
    def add_type_specific(self, baseProblem):
        pass
#
# Minimal loss in constraints to mitigate numerical instabilities for solvers
#
MINLOSS = 0.01

class BaseProblem_lupi(object):
    def __init__(self, problemType: ProblemType, di: int = None, kwargs: dict = None,
                 X: np.ndarray = None, X_priv: np.ndarray = None, Y: np.ndarray = None, initLoss: float = None, initL1: float = None,
                 initL1_priv: float = None, parameters: dict = None,
                 presetModel: np.ndarray = None):
        # General data parameters
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X
        self.Y = Y
        self.X_priv = X_priv

        # InitModel Parameters
        self.initL1 = initL1
        self.initL1_priv = initL1_priv
        self.initLoss = max(MINLOSS, initLoss)
        self.parameters = parameters

        # Dimension specific data
        self.di = di

        # Solver parameters
        self._constraints = []
        self.xp = cvx.Variable(nonneg=True, name="currentDimWeight")  # x' , our opt. value
        self.omega = cvx.Variable(self.d, name="Omega")  # complete linear weight vector
        if kwargs is None:
            self.kwargs = {}  # Fix (unpacking)error when manually calling bounds without keyword arguments to the solver
        else:
            self.kwargs = kwargs

        # Check if some model values have a pre fixed value
        self.presetModel = presetModel
        if presetModel is not None:
            if np.all(np.isnan(presetModel)):
                self.presetModel = None
            else:
                assert presetModel.shape == (self.d, 2)
                for dim in range(self.d):
                    # Skip current dimension
                    if dim == di:
                        continue
                    current_preset = presetModel[dim]

                    # Skip unset values
                    if all(np.isnan(current_preset)):
                        continue

                    # a weight bigger than the optimal model L1 makes no sense
                    assert abs(current_preset[0]) <= self.initL1
                    assert abs(current_preset[1]) <= self.initL1
                    # We add a pair of constraints depending on sign of known coefficient
                    # this makes it possible to solve this as a convex problem
                    if current_preset[0] >= 0:
                        self._constraints.extend([
                            self.omega[dim] >= current_preset[0],
                            self.omega[dim] <= current_preset[1],
                        ])
                    else:
                        self._constraints.extend([
                            self.omega[dim] <= current_preset[0],
                            self.omega[dim] >= current_preset[1],
                        ])
        self.problem = None
        self._objective = None

        # Add problem type specific constraints and variables to cvxpy
        problemType().add_type_specific(self)

    def __str__(self):
        return '{self.__class__.__name__}(current_dim={self.di}, n={self.n}, d={self.d}, initL1={self.initL1}, initLoss={self.initLoss}, ' \
               'parameters={self.parameters}, kwargs={self.kwargs}, presetModel={self.presetModel})'.format(self=self)

    def solve(self):
        self.problem = cvx.Problem(self._objective, self._constraints)
        try:
            self.problem.solve(**self.kwargs)
        except Exception as e:
            return None
        return self


class MinProblem_lupi(BaseProblem_lupi):
    def __init__(self, problemType, di=None, kwargs=None, X=None, X_priv=None, Y=None, initLoss=None, initL1=None, initL1_priv=None, parameters=None, presetModel=None):
        super().__init__(problemType, di=di, kwargs=kwargs, X=X, X_priv=X_priv, Y=Y, initLoss=initLoss, initL1=initL1, initL1_priv=initL1_priv,
                         parameters=parameters, presetModel=presetModel)

        self._constraints.extend(
            [
                cvx.abs(self.omega[self.di]) <= self.xp
            ])

        self._objective = cvx.Minimize(self.xp)


class MaxProblem1_lupi(BaseProblem_lupi):
    def __init__(self, problemType, di=None, kwargs=None, X=None, X_priv=None, Y=None, initLoss=None, initL1=None, initL1_priv=None, parameters=None, presetModel=None):
        super().__init__(problemType, di=di, kwargs=kwargs, X=X, X_priv=X_priv, Y=Y, initLoss=initLoss, initL1=initL1, initL1_priv= initL1_priv,
                         parameters=parameters,presetModel=presetModel)

        self._constraints.extend(
            [
                self.xp <= self.omega[self.di]
            ])

        self._objective = cvx.Maximize(self.xp)


class MaxProblem2_lupi(BaseProblem_lupi):
    def __init__(self, problemType, di=None, kwargs=None, X=None, X_priv=None, Y=None, initLoss=None, initL1=None, initL1_priv=None, parameters=None, presetModel=None):
        super().__init__(problemType, di=di, kwargs=kwargs, X=X, X_priv=X_priv, Y=Y, initLoss=initLoss, initL1=initL1, initL1_priv=initL1_priv,
                         parameters=parameters,presetModel=presetModel)

        self._constraints.extend(
            [
                self.xp <= -(self.omega[self.di])
            ])

        self._objective = cvx.Maximize(self.xp)


class BaseLupiProblem(ProblemType):
    def add_type_specific(self, baseProblem):
        # Problem Specific variables and constraints
        baseProblem.b = cvx.Variable(name="offset")  # shift
        baseProblem.b_priv = cvx.Variable()
        baseProblem.omega_priv = cvx.Variable(baseProblem.X_priv.shape[1])



        baseProblem._constraints.extend(
            [
                # original conditions
                baseProblem.Y * (baseProblem.X * baseProblem.omega + baseProblem.b) >= 1 - (baseProblem.X_priv * baseProblem.omega_priv + baseProblem.b_priv),
                baseProblem.X_priv * baseProblem.omega_priv + baseProblem.b_priv >= 0,
                # bounds conditions
                cvx.norm(baseProblem.omega, 1) <= baseProblem.initL1,
                cvx.norm(baseProblem.omega_priv, 1) <= baseProblem.initL1_priv,
                cvx.sum(baseProblem.X_priv * baseProblem.omega_priv + baseProblem.b_priv) <= baseProblem.initLoss

            ])

